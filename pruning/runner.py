"""
Pruning Runner.

Orchestrates the full pruning pipeline.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
import json
import logging
from datetime import datetime
import copy
import gc

from config.schemas import (
    CollectorConfig, PruningConfig, RunnerConfig, FullConfig,
    CovarianceMode, PruneTarget, AttentionPruneMode,
)
from .collect import ActivationCollector, detect_model_structure, get_hook_points, detect_mlp_type, get_mlp_intermediate_dim
from .stats import RedundancyAnalyzer, RedundancyReport, QKDimReport
from .ranking import StructureRanker, RankingPolicy, QKDimRanker
from .compensate import AffineCompensator, CompensationResult, QKDimCompensator
from .apply_masks import MaskApplier, compare_model_sizes
from .diagnostics import DiagnosticsChecker, run_quick_validation
from .schedules import PruneSchedule, create_schedule, PruneStep
from .cache import compute_cache_key, save_stats_cache, load_stats_cache

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of a single pruning step."""
    layer_name: str
    prune_count: int
    survivor_count: int
    sparsity: float
    compensation_lambda: float
    compensation_cond: float
    target_type: str = "mlp"  # 'mlp' or 'attn'
    mse_before: Optional[float] = None
    mse_after: Optional[float] = None
    cosine_after: Optional[float] = None
    passed_diagnostics: bool = True


@dataclass
class RunResult:
    """Result of a full pruning run."""
    success: bool
    original_params: int
    pruned_params: int
    compression_ratio: float
    step_results: List[StepResult] = field(default_factory=list)
    final_cosine: Optional[float] = None
    error_message: Optional[str] = None
    output_dir: Optional[Path] = None
    pruned_model: Optional[nn.Module] = None  # The pruned model


class PruneRunner:
    """Orchestrates the full pruning pipeline."""

    def __init__(
        self,
        config: FullConfig,
    ):
        """Initialize the runner.

        Args:
            config: Full configuration
        """
        self.config = config
        self.collector_config = config.collector
        self.pruning_config = config.pruning
        self.runner_config = config.runner
        self.stats_cache_dir: Optional[Path] = None
        self.force_recollect: bool = False

        # Set up components
        self.analyzer = RedundancyAnalyzer()
        # Create rankers for MLP (uses min_channels) and attention (uses min_heads)
        self.ranker_mlp = StructureRanker.from_config(
            self.pruning_config.ranker,
            min_channels=self.pruning_config.min_channels,
            keep_topk_outliers=self.pruning_config.keep_topk_outliers,
        )
        self.ranker_attn = StructureRanker.from_config(
            self.pruning_config.ranker,
            min_channels=self.pruning_config.min_heads,
            keep_topk_outliers=self.pruning_config.keep_topk_outliers,
        )
        self.compensator = AffineCompensator(
            lambda_reg=self.pruning_config.lambda_reg,
            auto_shrinkage=self.pruning_config.auto_shrinkage,
        )
        self.qk_compensator = QKDimCompensator(
            lambda_reg=self.pruning_config.lambda_reg,
            auto_shrinkage=self.pruning_config.auto_shrinkage,
        )
        self.qk_dim_ranker = QKDimRanker(
            min_qk_dim=self.pruning_config.min_qk_dim,
        )
        self.mask_applier = MaskApplier(validate_shapes=True)
        self.diagnostics = DiagnosticsChecker()

        # Set up device and dtype
        self.device = self.runner_config.device
        self.dtype = getattr(torch, self.runner_config.dtype)

    def run(
        self,
        model: nn.Module,
        calib_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        skip_compensation: bool = False,
        model_name: Optional[str] = None,
    ) -> RunResult:
        """Run the full pruning pipeline.

        Args:
            model: Model to prune
            calib_loader: Calibration data loader
            val_loader: Optional validation data loader
            skip_compensation: If True, prune without applying compensation
            model_name: Model name for stats caching (required if stats_cache_dir is set)

        Returns:
            RunResult with pruning outcomes (includes pruned_model)
        """
        # Set up output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.runner_config.output_dir / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self._save_config(output_dir)

        # Handle compiled models
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod

        model = model.to(self.device)
        model.eval()

        # Detect model structure
        structure = detect_model_structure(model)
        logger.info(f"Model structure: {structure}")

        # Get layers to prune
        layer_dims = self._get_layer_dims(model, structure)
        logger.info(f"Layers to prune: {list(layer_dims.keys())}")

        # Create schedule
        schedule = create_schedule(self.pruning_config)

        # Track original parameters
        original_params = sum(p.numel() for p in model.parameters())

        # Make a working copy for pruning, then move original to CPU
        # to free ~4GB VRAM (original is only needed for final validation)
        pruned_model = copy.deepcopy(model)
        model.cpu()
        torch.cuda.empty_cache()

        step_results = []
        metrics_file = output_dir / "metrics.jsonl"

        try:
            # Get attention prune mode
            attn_mode = self.pruning_config.attn_prune_mode.value

            target = self.pruning_config.target

            # Try loading from stats cache
            cache_hit = False
            cache_key = None
            if self.stats_cache_dir is not None and model_name is not None:
                cache_key = compute_cache_key(
                    model_name=model_name,
                    calib_samples=self.runner_config.calib_samples,
                    attn_mode=attn_mode,
                    subsample_tokens=self.collector_config.subsample_tokens,
                )
                if not self.force_recollect:
                    cached = load_stats_cache(self.stats_cache_dir, cache_key)
                    if cached is not None:
                        all_stats, qk_reports = cached
                        cache_hit = True
                        logger.info("Using cached activation stats (skipping forward passes)")

            if not cache_hit:
                # When caching is enabled, always collect both MLP and attention
                # stats so the cache is reusable regardless of pruning target.
                collect_target_override = 'both' if self.stats_cache_dir is not None else None
                all_stats = self._collect_activations(
                    pruned_model, calib_loader, structure, attn_mode,
                    target_override=collect_target_override,
                )

                # Generate Q/K dimension reports if using dim pruning mode
                qk_reports = {}
                if attn_mode == 'dim-logit':
                    qk_reports = self.analyzer.generate_all_qk_dim_reports(all_stats)

                # Save to cache
                if self.stats_cache_dir is not None and cache_key is not None:
                    metadata = {
                        "model_name": model_name,
                        "calib_samples": self.runner_config.calib_samples,
                        "attn_mode": attn_mode,
                        "subsample_tokens": self.collector_config.subsample_tokens,
                    }
                    save_stats_cache(
                        self.stats_cache_dir, cache_key,
                        all_stats, qk_reports, metadata,
                    )

            # Derive lightweight reports from stats (no eigvalsh)
            reports = self.analyzer.reports_from_all_stats(all_stats)

            # Process each step in schedule
            current_round = -1

            for step in schedule.iterate(layer_dims, self.pruning_config.sparsity):
                # Check if new round - recalibrate if needed
                # Skip recalibration for Q/K dim pruning modes since different layers
                # will have different dimensions after pruning
                should_recalibrate = (
                    step.round_num > current_round
                    and schedule.requires_recalibration()
                    and attn_mode != 'dim-logit'  # Skip for Q/K dim pruning
                )
                if should_recalibrate:
                    if current_round >= 0:
                        logger.info(f"Recalibrating for round {step.round_num}")
                        # Free old stats and reports before collecting new ones
                        del all_stats
                        del reports
                        if qk_reports:
                            del qk_reports
                        gc.collect()
                        torch.cuda.empty_cache()
                        # Re-detect model structure since pruning may have changed dimensions
                        structure = detect_model_structure(pruned_model)
                        all_stats = self._collect_activations(pruned_model, calib_loader, structure, attn_mode)
                        reports = self.analyzer.reports_from_all_stats(all_stats)
                        if attn_mode == 'dim-logit':
                            qk_reports = self.analyzer.generate_all_qk_dim_reports(all_stats)
                        else:
                            qk_reports = {}
                current_round = step.round_num

                # Determine target type for this layer
                # Use value comparison to handle different enum imports
                target_value = target.value if hasattr(target, 'value') else target

                if step.layer_name.endswith('.mlp'):
                    target_type = 'mlp'
                elif step.layer_name.endswith('.attn'):
                    target_type = 'attn'
                elif target_value == 'mlp':
                    target_type = 'mlp'
                elif target_value == 'attn':
                    target_type = 'attn'
                else:
                    # For BOTH without suffix, skip (should have suffix)
                    logger.warning(f"Cannot determine target type for {step.layer_name}")
                    continue

                # Handle dim-logit attention separately (has its own per-head ranking)
                if target_type == 'attn' and attn_mode == 'dim-logit':
                    prune_per_head, surv_per_head = self._prune_attention_qk_dims(
                        pruned_model, step.layer_name, qk_reports, all_stats,
                        skip_compensation
                    )

                    if prune_per_head == 0:
                        logger.info(f"No Q/K dim pruning needed for {step.layer_name}")
                        continue

                    prune_count = prune_per_head
                    survivor_count = surv_per_head
                    comp_lambda = self.pruning_config.lambda_reg
                    comp_cond = 0.0

                else:
                    # MLP or head-mode attention: use report-based ranking
                    layer_key = self._find_layer_key(step.layer_name, reports, target_type)
                    if layer_key is None:
                        logger.warning(f"No report for layer {step.layer_name} (target={target_type}), skipping")
                        continue

                    report = reports[layer_key]
                    stats = all_stats[layer_key]

                    # Get module (needed for weight-based ranking)
                    if target_type == 'mlp':
                        module = self._get_mlp_module(pruned_model, step.layer_name)
                        module_type = 'mlp'
                    else:
                        module = self._get_attn_module(pruned_model, step.layer_name)
                        module_type = 'attn'

                    if module is None:
                        logger.warning(f"Could not find module for {step.layer_name}")
                        continue

                    ranker = self.ranker_mlp if target_type == 'mlp' else self.ranker_attn
                    prune_idx, surv_idx = ranker.select_for_sparsity(
                        report, step.sparsity,
                        module=module, module_type=module_type
                    )

                    if len(prune_idx) == 0:
                        logger.info(f"No pruning needed for {step.layer_name}")
                        continue

                    if target_type == 'mlp':
                        compensation = self.compensator.fit(stats, prune_idx, surv_idx)
                        self.mask_applier.prune_ffn_intermediate(
                            module, prune_idx,
                            compensation=None if skip_compensation else compensation
                        )
                        comp_lambda = compensation.lambda_used
                        comp_cond = compensation.condition_number
                        del compensation
                    else:
                        # Head-mode attention pruning
                        self.mask_applier.prune_attention_heads(module, prune_idx)
                        comp_lambda = 0.0
                        comp_cond = 0.0

                    prune_count = len(prune_idx)
                    survivor_count = len(surv_idx)

                # Update layer dims
                layer_dims[step.layer_name] = survivor_count

                # Record result
                result = StepResult(
                    layer_name=step.layer_name,
                    prune_count=prune_count,
                    survivor_count=survivor_count,
                    sparsity=step.sparsity,
                    compensation_lambda=comp_lambda,
                    compensation_cond=comp_cond,
                    target_type=target_type,
                )
                step_results.append(result)

                # Log to file
                self._log_step(metrics_file, step, result)

                logger.info(
                    f"Pruned {step.layer_name} ({target_type}): {result.prune_count} pruned, "
                    f"{result.survivor_count} remaining"
                )

                # Periodic memory cleanup
                torch.cuda.empty_cache()

            # Free activation stats before validation to reclaim GPU memory
            del all_stats, reports, qk_reports
            gc.collect()
            torch.cuda.empty_cache()

            # Final validation
            pruned_params = sum(p.numel() for p in pruned_model.parameters())

            # Quick validation
            sample_input = next(iter(calib_loader))
            if isinstance(sample_input, (list, tuple)):
                sample_input = sample_input[0]
            sample_input = sample_input[:4].to(self.device)

            passed, msg = run_quick_validation(model, pruned_model, sample_input, self.device)
            # Move original model back to CPU to free VRAM for downstream eval
            model.cpu()
            del sample_input
            torch.cuda.empty_cache()
            if passed:
                logger.info(f"Validation: {msg}")
            else:
                logger.warning(f"Validation: {msg}")

            # Save pruned model
            if self.runner_config.save_pruned_path:
                torch.save(pruned_model.state_dict(), self.runner_config.save_pruned_path)
                logger.info(f"Saved pruned model to {self.runner_config.save_pruned_path}")

            # Generate summary
            result = RunResult(
                success=True,
                original_params=original_params,
                pruned_params=pruned_params,
                compression_ratio=original_params / pruned_params,
                step_results=step_results,
                output_dir=output_dir,
                pruned_model=pruned_model,
            )

            self._save_summary(output_dir, result)

            return result

        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return RunResult(
                success=False,
                original_params=original_params,
                pruned_params=original_params,
                compression_ratio=1.0,
                error_message=str(e),
                output_dir=output_dir,
            )

    def _collect_activations(
        self,
        model: nn.Module,
        loader: DataLoader,
        structure: Dict,
        attn_prune_mode: str = 'head',
        target_override: Optional[str] = None,
    ) -> Dict:
        """Collect activations from model.

        Args:
            model: The model to collect activations from
            loader: Data loader for calibration data
            structure: Model structure info
            attn_prune_mode: Mode for attention pruning
            target_override: If set, override the pruning target for collection
                (e.g. 'both' to collect all stats for caching)
        """
        # Determine hook points
        target = target_override if target_override is not None else self.pruning_config.target.value
        hook_points = get_hook_points(model, target, attn_prune_mode)

        # Warn if dim-logit mode requires exact covariance but user specified otherwise
        if attn_prune_mode == 'dim-logit':
            user_cov_mode = self.collector_config.covariance_mode
            user_cov_value = user_cov_mode.value if hasattr(user_cov_mode, 'value') else user_cov_mode
            if user_cov_value != 'exact':
                logger.warning(
                    f"dim-logit mode requires exact covariance for Sylvester solver. "
                    f"Overriding covariance_mode from '{user_cov_value}' to 'exact'."
                )

        # Compensation uses covariance/mean, not raw activations
        config = CollectorConfig(
            target=self.pruning_config.target,
            covariance_mode=CovarianceMode.EXACT,
            store_raw=False,
            subsample_tokens=self.collector_config.subsample_tokens,
        )

        collector = ActivationCollector(model, config, device=self.device)
        collector.register_hooks(hook_points, model_structure=structure, attn_prune_mode=attn_prune_mode)

        n_samples = 0
        max_samples = self.runner_config.calib_samples

        with collector.collect():
            with torch.no_grad():
                for batch in loader:
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]
                    batch = batch.to(self.device)

                    model(batch)

                    n_samples += batch.shape[0]
                    if n_samples >= max_samples:
                        break

        collector.clear_hooks()
        logger.info(f"Collected activations from {n_samples} samples")

        return collector.get_all_stats()

    def _get_layer_dims(self, model: nn.Module, structure: Dict) -> Dict[str, int]:
        """Get dimensions of prunable layers based on target type.

        Args:
            model: The model to analyze
            structure: Model structure info

        Returns:
            For MLP target: Dict mapping block names to MLP hidden dimensions
            For ATTN target (head mode): Dict mapping block names to number of attention heads
            For ATTN target (dim-logit): Dict mapping block names to head_dim (Q/K dims per head)
            For BOTH target: Dict with both MLP and ATTN entries (suffixed)
        """
        layer_dims = {}
        target = self.pruning_config.target
        target_value = target.value if hasattr(target, 'value') else target

        # Get attention prune mode
        attn_mode = self.pruning_config.attn_prune_mode
        attn_mode_value = attn_mode.value if hasattr(attn_mode, 'value') else attn_mode

        for i in range(structure['num_blocks']):
            block_name = f"blocks.{i}"
            block = self._get_module(model, block_name)

            if block is None:
                continue

            # MLP dimensions
            if target_value in ('mlp', 'both'):
                if hasattr(block, 'mlp') and detect_mlp_type(block.mlp) != 'unknown':
                    key = f"{block_name}.mlp" if target_value == 'both' else block_name
                    layer_dims[key] = get_mlp_intermediate_dim(block.mlp)

            # Attention dimensions
            if target_value in ('attn', 'both'):
                if hasattr(block, 'attn') and hasattr(block.attn, 'num_heads'):
                    key = f"{block_name}.attn" if target_value == 'both' else block_name
                    # For dim-logit, use head_dim; for head mode, use num_heads
                    if attn_mode_value == 'dim-logit':
                        attn = block.attn
                        if hasattr(attn, 'head_dim'):
                            layer_dims[key] = attn.head_dim
                        else:
                            # DINOv2 MemEffAttention: compute from qkv weight
                            embed_dim = attn.qkv.weight.shape[1] if hasattr(attn, 'qkv') else attn.proj.weight.shape[0]
                            layer_dims[key] = embed_dim // attn.num_heads
                    else:
                        layer_dims[key] = block.attn.num_heads

        return layer_dims

    def _get_module(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """Get module by name."""
        parts = name.split('.')
        module = model

        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif hasattr(module, '__getitem__'):
                try:
                    module = module[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return module

    def _get_mlp_module(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get MLP module for a layer."""
        # Handle .mlp suffix for BOTH target
        if layer_name.endswith('.mlp'):
            base_name = layer_name[:-4]  # Remove '.mlp'
        else:
            base_name = layer_name

        module = self._get_module(model, base_name)
        if module is not None and hasattr(module, 'mlp'):
            return module.mlp
        return module

    def _get_attn_module(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get attention module for a layer."""
        # Handle .attn suffix for BOTH target
        if layer_name.endswith('.attn'):
            base_name = layer_name[:-5]  # Remove '.attn'
        else:
            base_name = layer_name

        module = self._get_module(model, base_name)
        if module is not None and hasattr(module, 'attn'):
            return module.attn
        return module

    def _find_layer_key(self, layer_name: str, reports: Dict, target_type: str = 'mlp') -> Optional[str]:
        """Find the report key for a layer name.

        Args:
            layer_name: Layer name to find
            reports: Dictionary of reports
            target_type: 'mlp' or 'attn'

        Returns:
            The matching key in reports, or None
        """
        # Handle .mlp or .attn suffix
        if layer_name.endswith('.mlp'):
            base_name = layer_name[:-4]
            target_type = 'mlp'
        elif layer_name.endswith('.attn'):
            base_name = layer_name[:-5]
            target_type = 'attn'
        else:
            base_name = layer_name

        if target_type == 'mlp':
            # Try with .mlp.act suffix (activations after GELU, standard MLP)
            for key in reports:
                if f"{base_name}.mlp.act" in key:
                    return key
            # Try SwiGLU hook points
            for key in reports:
                if f"{base_name}.mlp.w3" in key:
                    return key
            for key in reports:
                if f"{base_name}.mlp.down_proj" in key:
                    return key
            # Try with .mlp suffix
            for key in reports:
                if f"{base_name}.mlp" in key and ".act" not in key:
                    return key
        elif target_type == 'attn':
            # Try with .attn.proj suffix (attention output)
            for key in reports:
                if f"{base_name}.attn.proj" in key:
                    return key
            # Try with .attn suffix
            for key in reports:
                if f"{base_name}.attn" in key:
                    return key

        # Fallback: try exact match
        for key in reports:
            if base_name in key:
                return key

        return None

    def _prune_attention_qk_dims(
        self,
        model: nn.Module,
        layer_name: str,
        qk_reports: Dict[str, QKDimReport],
        all_stats: Dict,
        skip_compensation: bool = False,
    ) -> tuple:
        """Prune Q/K dimensions from an attention layer using dim-logit mode.

        Args:
            model: The model being pruned
            layer_name: Name of the attention layer (e.g., 'blocks.0' or 'blocks.0.attn')
            qk_reports: Dictionary of Q/K dimension reports
            all_stats: All activation statistics
            skip_compensation: If True, prune without compensation

        Returns:
            Tuple of (prune_count_per_head, survivor_count_per_head) for the Q/K dimensions.
            Returns (0, original_head_dim) if nothing was pruned.
        """
        attn = self._get_attn_module(model, layer_name)
        if attn is None:
            logger.warning(f"Could not find attention for {layer_name}")
            return (0, 0)

        # Get the base attention layer name
        if layer_name.endswith('.attn'):
            attn_name = layer_name
        else:
            attn_name = f"{layer_name}.attn"

        num_heads = attn.num_heads
        if hasattr(attn, 'head_dim'):
            orig_head_dim = attn.head_dim
        else:
            # DINOv2 MemEffAttention: compute from qkv weight
            embed_dim = attn.qkv.weight.shape[1] if hasattr(attn, 'qkv') else attn.proj.weight.shape[0]
            orig_head_dim = embed_dim // num_heads
        compensation_results = []
        total_pruned = 0
        total_survived = 0

        for h in range(num_heads):
            report_key = f"{attn_name}.attn.head_{h}"

            # Try alternate key formats
            if report_key not in qk_reports:
                # Try without double .attn
                report_key = f"{layer_name}.attn.head_{h}"
            if report_key not in qk_reports:
                # Try with blocks.X.attn format
                if layer_name.endswith('.attn'):
                    report_key = f"{layer_name}.head_{h}"
                else:
                    report_key = f"{layer_name}.attn.head_{h}"

            if report_key not in qk_reports:
                logger.warning(f"No Q/K report for {report_key}, skipping head {h}")
                continue

            report = qk_reports[report_key]

            # Select dimensions to prune
            prune_idx, surv_idx = self.qk_dim_ranker.select_for_sparsity(
                report, self.pruning_config.qk_sparsity
            )

            if len(prune_idx) == 0:
                logger.debug(f"No Q/K dims to prune for {layer_name} head {h}")

            total_pruned += len(prune_idx)
            total_survived += len(surv_idx)

            # Fit compensation using dim-logit (Sylvester-based)
            result = self.qk_compensator.fit_dim_logit(
                report.q_stats, report.k_stats,
                prune_idx, surv_idx,
                layer_name, h
            )

            if len(prune_idx) == 0:
                compensation_results.append(result)
                continue

            compensation_results.append(result)

            logger.debug(
                f"Q/K dim pruning {layer_name} head {h}: "
                f"{len(prune_idx)}/{report.head_dim} dims pruned"
            )

        # Apply pruning to attention module
        if compensation_results:
            self.mask_applier.prune_attention_qk_dims(
                attn, compensation_results,
                qk_compensator=None if skip_compensation else self.qk_compensator
            )

            logger.info(
                f"Pruned Q/K dims in {layer_name}: {num_heads} heads processed"
            )

        # Return per-head counts (all heads have the same survivor count)
        if num_heads > 0 and total_survived > 0:
            surv_per_head = total_survived // num_heads
        else:
            surv_per_head = orig_head_dim
        prune_per_head = orig_head_dim - surv_per_head

        return (prune_per_head, surv_per_head)

    def _save_config(self, output_dir: Path) -> None:
        """Save configuration to file."""
        config_file = output_dir / "config.json"
        config_dict = {
            'collector': {
                'target': self.collector_config.target.value,
                'subsample_tokens': self.collector_config.subsample_tokens,
                'covariance_mode': self.collector_config.covariance_mode.value,
            },
            'pruning': {
                'target': self.pruning_config.target.value,
                'schedule': self.pruning_config.schedule.value,
                'sparsity': self.pruning_config.sparsity,
                'ranker': self.pruning_config.ranker.value,
                'lambda_reg': self.pruning_config.lambda_reg,
                'min_channels': self.pruning_config.min_channels,
            },
            'runner': {
                'device': self.runner_config.device,
                'dtype': self.runner_config.dtype,
                'calib_samples': self.runner_config.calib_samples,
            },
        }
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def _log_step(self, file_path: Path, step: PruneStep, result: StepResult) -> None:
        """Log step to JSONL file."""
        entry = {
            'round': step.round_num,
            'step': step.step_num,
            'layer': result.layer_name,
            'target_type': result.target_type,
            'prune_count': result.prune_count,
            'survivor_count': result.survivor_count,
            'sparsity': result.sparsity,
            'lambda': result.compensation_lambda,
            'condition_number': result.compensation_cond,
        }
        with open(file_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def _save_summary(self, output_dir: Path, result: RunResult) -> None:
        """Save run summary."""
        summary_file = output_dir / "summary.md"
        with open(summary_file, 'w') as f:
            f.write("# Pruning Run Summary\n\n")
            f.write(f"- Success: {result.success}\n")
            f.write(f"- Target: {self.pruning_config.target.value}\n")
            f.write(f"- Original params: {result.original_params:,}\n")
            f.write(f"- Pruned params: {result.pruned_params:,}\n")
            f.write(f"- Compression ratio: {result.compression_ratio:.2f}x\n")
            f.write(f"- Steps completed: {len(result.step_results)}\n")

            if result.error_message:
                f.write(f"\n## Error\n{result.error_message}\n")

            f.write("\n## Step Details\n\n")
            for step in result.step_results:
                f.write(f"- {step.layer_name} [{step.target_type}]: {step.prune_count} pruned "
                       f"({step.sparsity*100:.1f}%)\n")
