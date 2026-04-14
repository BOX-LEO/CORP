#!/usr/bin/env python
"""
Activation-based structured pruning for OPT language models.

Supports OPT-125m, OPT-350m, and OPT-1.3b with MLP and/or attention head pruning.
Supports both layerwise and global (one-shot) pruning schedules.

Usage:
    # MLP only, layerwise
    python opt_prune/run_prune.py --model_size 125m --sparsity 0.3 --target mlp \
        --ranker active_energy --schedule layerwise --device cuda

    # MLP + Attention, global (one-shot)
    python opt_prune/run_prune.py --model_size 125m --sparsity 0.3 --target both \
        --ranker active_energy --schedule global --device cuda

    # Attention only
    python opt_prune/run_prune.py --model_size 350m --sparsity 0.3 --target attn \
        --ranker energy --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.schemas import (
    CollectorConfig, PruningConfig, RunnerConfig,
    PruneTarget, ScheduleType, RankerType, CovarianceMode,
)
from pruning.collect import ActivationCollector, LayerActivationStats
from pruning.stats import RedundancyAnalyzer, RedundancyReport
from pruning.ranking import StructureRanker
from pruning.compensate import AffineCompensator
from pruning.apply_masks import MaskApplier, get_model_parameter_count

from opt_prune.data import create_c4_calibration_loader, create_wikitext2_eval_loader
from opt_prune.evaluation import evaluate_perplexity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# OPT model dimensions reference
OPT_CONFIGS = {
    "125m": {"layers": 12, "embed_dim": 768, "mlp_hidden": 3072, "heads": 12, "head_dim": 64},
    "350m": {"layers": 24, "embed_dim": 1024, "mlp_hidden": 4096, "heads": 16, "head_dim": 64},
    "1.3b": {"layers": 24, "embed_dim": 2048, "mlp_hidden": 8192, "heads": 32, "head_dim": 64},
    "2.7b": {"layers": 32, "embed_dim": 2560, "mlp_hidden": 10240, "heads": 32, "head_dim": 80},
    "6.7b": {"layers": 32, "embed_dim": 4096, "mlp_hidden": 16384, "heads": 32, "head_dim": 128},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Activation-based structured pruning for OPT models'
    )

    parser.add_argument('--model_size', type=str, default='125m',
                        choices=['125m', '350m', '1.3b', '2.7b', '6.7b'],
                        help='OPT model size (default: 125m)')
    parser.add_argument('--sparsity', type=float, default=0.3,
                        help='Target sparsity (0.0 to 1.0, default: 0.3)')
    parser.add_argument('--target', type=str, default='both',
                        choices=['mlp', 'attn', 'both'],
                        help='Pruning target (default: both)')
    parser.add_argument('--schedule', type=str, default='layerwise',
                        choices=['layerwise', 'global'],
                        help='Pruning schedule (default: layerwise)')
    parser.add_argument('--ranker', type=str, default='active_energy',
                        choices=['energy', 'active', 'active_energy', 'weight_magnitude', 'energy_weightmagnitude', 'active_weightmagnitude', 'ensemble'],
                        help='Ranking policy (default: active_energy)')
    parser.add_argument('--outlier_safe', action='store_true', default=False,
                        help='Enable outlier-safe mode (boost importance of high-kurtosis channels)')
    parser.add_argument('--n_calib_segments', type=int, default=128,
                        help='Number of calibration segments (default: 128)')
    parser.add_argument('--seq_len', type=int, default=2048,
                        help='Sequence length (default: 2048)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--skip_compensation', action='store_true',
                        help='Skip compensation (prune without weight adjustment)')
    parser.add_argument('--lambda_reg', type=float, default=1e-3,
                        help='Ridge regularization strength (default: 1e-3)')
    parser.add_argument('--min_channels', type=int, default=64,
                        help='Minimum MLP channels to keep per layer (default: 64)')
    parser.add_argument('--min_heads', type=int, default=1,
                        help='Minimum attention heads to keep per layer (default: 1)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='./opt_logs',
                        help='Output directory (default: ./opt_logs)')
    parser.add_argument('--save_model', action='store_true',
                        help='Save pruned model checkpoint')

    return parser.parse_args()


def prune_opt_attention_heads(self_attn, prune_head_indices):
    """Prune attention heads from OPT's separate Q/K/V projections.

    OPT uses separate q_proj, k_proj, v_proj, out_proj (not fused QKV).

    Args:
        self_attn: OPT attention module with q_proj, k_proj, v_proj, out_proj
        prune_head_indices: Tensor of head indices to prune
    """
    n_heads = self_attn.num_heads
    head_dim = self_attn.head_dim

    n_prune = len(prune_head_indices)
    if n_prune == 0:
        return

    all_heads = set(range(n_heads))
    prune_set = set(prune_head_indices.tolist())
    survivor_heads = torch.tensor(sorted(all_heads - prune_set), dtype=torch.long)
    n_survivors = len(survivor_heads)

    logger.info(f"Pruning OPT attention heads: {n_heads} -> {n_survivors} ({n_prune} pruned)")

    # Compute feature indices for survivors
    survivor_rows = []
    for h in survivor_heads:
        start = h.item() * head_dim
        survivor_rows.extend(range(start, start + head_dim))
    survivor_rows = torch.tensor(survivor_rows, dtype=torch.long)

    # Prune q_proj, k_proj, v_proj: keep survivor rows (output dim)
    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
        proj = getattr(self_attn, proj_name)
        new_proj = nn.Linear(
            proj.in_features, n_survivors * head_dim,
            bias=proj.bias is not None,
            device=proj.weight.device, dtype=proj.weight.dtype,
        )
        with torch.no_grad():
            new_proj.weight.copy_(proj.weight[survivor_rows, :])
            if proj.bias is not None:
                new_proj.bias.copy_(proj.bias[survivor_rows])
        setattr(self_attn, proj_name, new_proj)

    # Prune out_proj: keep survivor columns (input dim)
    out_proj = self_attn.out_proj
    new_out_proj = nn.Linear(
        n_survivors * head_dim, out_proj.out_features,
        bias=out_proj.bias is not None,
        device=out_proj.weight.device, dtype=out_proj.weight.dtype,
    )
    with torch.no_grad():
        new_out_proj.weight.copy_(out_proj.weight[:, survivor_rows])
        if out_proj.bias is not None:
            new_out_proj.bias.copy_(out_proj.bias)
    self_attn.out_proj = new_out_proj

    # Update attention config
    self_attn.num_heads = n_survivors
    self_attn.embed_dim = n_survivors * head_dim


# ---------------------------------------------------------------------------
# Activation collection helpers
# ---------------------------------------------------------------------------

def collect_all_opt_stats(
    model, calib_loader, collector_config, num_layers, device, target,
) -> Dict[str, LayerActivationStats]:
    """Collect activation stats for all OPT layers in a single forward pass.

    Registers hooks on all target layers at once, runs calibration data through
    the model once, and returns all stats. Used by global schedule.

    Args:
        model: OPT CausalLM model
        calib_loader: Calibration DataLoader
        collector_config: CollectorConfig
        num_layers: Number of decoder layers
        device: Device string
        target: 'mlp', 'attn', or 'both'

    Returns:
        Dict mapping hook_name -> LayerActivationStats
    """
    prune_mlp = target in ('mlp', 'both')
    prune_attn = target in ('attn', 'both')

    collector = ActivationCollector(model, collector_config, device)

    for layer_idx in range(num_layers):
        layer = model.model.decoder.layers[layer_idx]

        if prune_mlp:
            hook_name = f"model.decoder.layers.{layer_idx}.fc2"
            hook_fn = collector._create_input_hook(hook_name)
            handle = layer.fc2.register_forward_hook(hook_fn)
            collector._hooks.append(handle)

        if prune_attn:
            hook_name = f"model.decoder.layers.{layer_idx}.self_attn.out_proj"
            num_heads = layer.self_attn.num_heads
            head_dim = layer.self_attn.head_dim
            hook_fn = collector._create_attn_head_hook(hook_name, num_heads, head_dim)
            handle = layer.self_attn.out_proj.register_forward_hook(hook_fn)
            collector._hooks.append(handle)

    logger.info(f"Collecting activations for all {num_layers} layers...")
    with collector.collect():
        for batch in calib_loader:
            input_ids = batch[0].to(device)
            with torch.no_grad():
                model(input_ids=input_ids)
    collector.clear_hooks()

    return collector.get_all_stats()


def collect_single_layer_stats(
    model, layer_idx, calib_loader, collector_config, device, target,
) -> Dict[str, LayerActivationStats]:
    """Collect activation stats for a single OPT layer.

    Used by layerwise schedule where we recollect after each layer is pruned.

    Returns:
        Dict mapping hook_name -> LayerActivationStats
    """
    prune_mlp = target in ('mlp', 'both')
    prune_attn = target in ('attn', 'both')
    layer = model.model.decoder.layers[layer_idx]

    collector = ActivationCollector(model, collector_config, device)

    if prune_mlp:
        hook_name = f"model.decoder.layers.{layer_idx}.fc2"
        hook_fn = collector._create_input_hook(hook_name)
        handle = layer.fc2.register_forward_hook(hook_fn)
        collector._hooks.append(handle)

    if prune_attn:
        hook_name = f"model.decoder.layers.{layer_idx}.self_attn.out_proj"
        num_heads = layer.self_attn.num_heads
        head_dim = layer.self_attn.head_dim
        hook_fn = collector._create_attn_head_hook(hook_name, num_heads, head_dim)
        handle = layer.self_attn.out_proj.register_forward_hook(hook_fn)
        collector._hooks.append(handle)

    with collector.collect():
        for batch in calib_loader:
            input_ids = batch[0].to(device)
            with torch.no_grad():
                model(input_ids=input_ids)
    collector.clear_hooks()

    return collector.get_all_stats()


# ---------------------------------------------------------------------------
# Core pruning logic
# ---------------------------------------------------------------------------

def apply_mlp_pruning(
    model, layer_idx, stats, report, ranker, compensator, mask_applier,
    sparsity, skip_compensation=False,
) -> dict:
    """Apply MLP pruning to a single layer using pre-collected stats/report."""
    layer = model.model.decoder.layers[layer_idx]

    prune_idx, surv_idx = ranker.select_for_sparsity(
        report, sparsity, module=layer, module_type='mlp',
    )
    n_prune = len(prune_idx)
    n_total = report.feature_dim

    if n_prune == 0:
        logger.info(f"  Layer {layer_idx} MLP: nothing to prune")
        return {"layer": layer_idx, "type": "mlp", "pruned": 0, "total": n_total}

    compensation = None
    if not skip_compensation and stats.covariance is not None:
        compensation = compensator.fit(stats, prune_idx, surv_idx)
        logger.info(f"  Layer {layer_idx} MLP: compensation λ={compensation.lambda_used:.2e}, "
                     f"cond={compensation.condition_number:.2e}")

    mask_applier.prune_ffn_intermediate(layer, prune_idx, compensation)
    logger.info(f"  Layer {layer_idx} MLP: {n_total} -> {n_total - n_prune} "
                f"({n_prune} pruned, {100*n_prune/n_total:.1f}%)")

    return {"layer": layer_idx, "type": "mlp", "pruned": n_prune, "total": n_total}


def apply_attn_pruning(
    model, layer_idx, report, ranker, sparsity, num_heads, min_heads=1,
) -> dict:
    """Apply attention head pruning to a single layer using pre-collected report."""
    layer = model.model.decoder.layers[layer_idx]

    target_prune_count = max(0, int(num_heads * sparsity))
    target_prune_count = min(target_prune_count, num_heads - min_heads)

    if target_prune_count <= 0:
        logger.info(f"  Layer {layer_idx} Attn: nothing to prune (min_heads={min_heads})")
        return {"layer": layer_idx, "type": "attn", "pruned": 0, "total": num_heads}

    # Use ranker.rank() for activation-only policies, select_for_sparsity
    # for weight-based policies that need the module.
    from pruning.ranking import RankingPolicy
    if ranker.policy in (RankingPolicy.WEIGHT_MAGNITUDE, RankingPolicy.ENERGY_WEIGHTMAGNITUDE,
                             RankingPolicy.ACTIVE_WEIGHTMAGNITUDE, RankingPolicy.ENSEMBLE):
        prune_idx, surv_idx = ranker.select_for_sparsity(
            report, sparsity, module=layer.self_attn, module_type='attn',
        )
    else:
        # Use ranker.rank() directly instead of select_prune_indices, because
        # select_prune_indices enforces min_channels (designed for MLP dims ~3072),
        # which blocks head pruning where feature_dim = num_heads (~12).
        # We already enforce min_heads above via target_prune_count.
        ranked = ranker.rank(report)  # ascending importance (most prunable first)
        prune_idx = ranked[:target_prune_count]
        surv_idx = ranked[target_prune_count:]

    prune_opt_attention_heads(layer.self_attn, prune_idx)
    n_prune = len(prune_idx)
    logger.info(f"  Layer {layer_idx} Attn: {num_heads} -> {num_heads - n_prune} heads "
                f"({n_prune} pruned, {100*n_prune/num_heads:.1f}%)")

    return {"layer": layer_idx, "type": "attn", "pruned": n_prune, "total": num_heads}


def run_opt_pruning(
    model,
    calib_loader,
    num_layers: int,
    target: str,
    schedule: str,
    sparsity: float,
    ranker: StructureRanker,
    analyzer: RedundancyAnalyzer,
    compensator: AffineCompensator,
    mask_applier: MaskApplier,
    collector_config: CollectorConfig,
    device: str,
    skip_compensation: bool = False,
    min_heads: int = 1,
    all_stats: Optional[Dict[str, LayerActivationStats]] = None,
    all_reports: Optional[Dict[str, RedundancyReport]] = None,
    attn_ranker: Optional[StructureRanker] = None,
) -> Tuple[List[dict], Optional[Dict[str, LayerActivationStats]], Optional[Dict[str, RedundancyReport]]]:
    """Run the full OPT pruning pipeline.

    Supports both layerwise and global schedules.

    Args:
        model: OPT CausalLM model (modified in-place)
        calib_loader: Calibration DataLoader
        num_layers: Number of decoder layers
        target: 'mlp', 'attn', or 'both'
        schedule: 'layerwise' or 'global'
        sparsity: Target sparsity ratio
        ranker, analyzer, compensator, mask_applier: Pruning components
        collector_config: CollectorConfig for activation collection
        device: Device string
        skip_compensation: Whether to skip compensation
        min_heads: Minimum attention heads to keep per layer
        all_stats: Pre-collected stats (optional, used for cache reuse)
        all_reports: Pre-generated reports (optional, used for cache reuse)
        attn_ranker: Optional separate ranker for attention heads (defaults to ranker)

    Returns:
        Tuple of (step_results, all_stats, all_reports)
    """
    prune_mlp = target in ('mlp', 'both')
    prune_attn = target in ('attn', 'both')
    attn_ranker = attn_ranker or ranker
    all_results = []

    if schedule == 'global':
        # Global (one-shot): collect all stats upfront, then prune all layers
        if all_stats is None:
            all_stats = collect_all_opt_stats(
                model, calib_loader, collector_config, num_layers, device, target,
            )
        if all_reports is None:
            all_reports = analyzer.reports_from_all_stats(all_stats)

        for layer_idx in range(num_layers):
            logger.info(f"Pruning layer {layer_idx}/{num_layers - 1}")

            if prune_mlp:
                hook_name = f"model.decoder.layers.{layer_idx}.fc2"
                if hook_name in all_stats:
                    result = apply_mlp_pruning(
                        model, layer_idx, all_stats[hook_name], all_reports[hook_name],
                        ranker, compensator, mask_applier, sparsity, skip_compensation,
                    )
                    all_results.append(result)

            if prune_attn:
                hook_name = f"model.decoder.layers.{layer_idx}.self_attn.out_proj"
                if hook_name in all_stats:
                    current_heads = model.model.decoder.layers[layer_idx].self_attn.num_heads
                    result = apply_attn_pruning(
                        model, layer_idx, all_reports[hook_name], attn_ranker,
                        sparsity, current_heads, min_heads,
                    )
                    all_results.append(result)

    else:
        # Layerwise: collect per-layer, prune, then move to next layer
        all_stats = {}
        all_reports = {}

        for layer_idx in range(num_layers):
            logger.info(f"Processing layer {layer_idx}/{num_layers - 1}")

            layer_stats = collect_single_layer_stats(
                model, layer_idx, calib_loader, collector_config, device, target,
            )
            layer_reports = analyzer.reports_from_all_stats(layer_stats)

            if prune_mlp:
                hook_name = f"model.decoder.layers.{layer_idx}.fc2"
                if hook_name in layer_stats:
                    result = apply_mlp_pruning(
                        model, layer_idx, layer_stats[hook_name], layer_reports[hook_name],
                        ranker, compensator, mask_applier, sparsity, skip_compensation,
                    )
                    all_results.append(result)

            if prune_attn:
                hook_name = f"model.decoder.layers.{layer_idx}.self_attn.out_proj"
                if hook_name in layer_stats:
                    current_heads = model.model.decoder.layers[layer_idx].self_attn.num_heads
                    result = apply_attn_pruning(
                        model, layer_idx, layer_reports[hook_name], attn_ranker,
                        sparsity, current_heads, min_heads,
                    )
                    all_results.append(result)

            # Merge into all_stats/all_reports for potential cache saving
            all_stats.update(layer_stats)
            all_reports.update(layer_reports)

            # Free memory
            del layer_stats, layer_reports
            torch.cuda.empty_cache()

    return all_results, all_stats, all_reports


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model_name = f"facebook/opt-{args.model_size}"
    opt_cfg = OPT_CONFIGS[args.model_size]
    num_layers = opt_cfg["layers"]

    logger.info("=" * 60)
    logger.info("OPT Activation-Based Structured Pruning")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Sparsity: {args.sparsity}")
    logger.info(f"Ranker: {args.ranker}")
    logger.info(f"Schedule: {args.schedule}")
    logger.info(f"Compensation: {'disabled' if args.skip_compensation else 'enabled'}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    original_params = get_model_parameter_count(model)
    logger.info(f"Parameters: {original_params:,}")

    # Create data loaders
    logger.info("Creating C4 calibration loader...")
    calib_loader = create_c4_calibration_loader(
        tokenizer,
        n_segments=args.n_calib_segments,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    logger.info(f"Calibration: {args.n_calib_segments} segments x {args.seq_len} tokens")

    logger.info("Creating WikiText-2 eval loader...")
    eval_loader = create_wikitext2_eval_loader(
        tokenizer, seq_len=args.seq_len, batch_size=args.batch_size,
    )

    # Evaluate original perplexity
    logger.info("Evaluating original model perplexity...")
    original_ppl = evaluate_perplexity(model, eval_loader, args.device)

    # Set up pruning components
    collector_config = CollectorConfig(
        target=PruneTarget(args.target),
        subsample_tokens=None,
        keep_cls_token=False,
        covariance_mode=CovarianceMode.EXACT,
        store_raw=False,
        batch_size=args.batch_size,
    )

    ranker = StructureRanker.from_config(
        RankerType(args.ranker),
        min_channels=args.min_channels,
    )
    analyzer = RedundancyAnalyzer()
    compensator = AffineCompensator(lambda_reg=args.lambda_reg)
    mask_applier = MaskApplier()

    # Run pruning
    logger.info("=" * 60)
    logger.info("Starting Pruning")
    logger.info("=" * 60)

    all_results, _, _ = run_opt_pruning(
        model=model,
        calib_loader=calib_loader,
        num_layers=num_layers,
        target=args.target,
        schedule=args.schedule,
        sparsity=args.sparsity,
        ranker=ranker,
        analyzer=analyzer,
        compensator=compensator,
        mask_applier=mask_applier,
        collector_config=collector_config,
        device=args.device,
        skip_compensation=args.skip_compensation,
        min_heads=args.min_heads,
    )

    # Evaluate pruned perplexity
    logger.info("=" * 60)
    logger.info("Evaluating pruned model...")
    pruned_ppl = evaluate_perplexity(model, eval_loader, args.device)

    pruned_params = get_model_parameter_count(model)
    compression_ratio = original_params / pruned_params if pruned_params > 0 else float('inf')

    # Report results
    logger.info("=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info(f"Original params:  {original_params:,}")
    logger.info(f"Pruned params:    {pruned_params:,}")
    logger.info(f"Compression:      {compression_ratio:.2f}x")
    logger.info(f"Params removed:   {100*(1 - pruned_params/original_params):.1f}%")
    logger.info(f"Original PPL:     {original_ppl:.2f}")
    logger.info(f"Pruned PPL:       {pruned_ppl:.2f}")
    logger.info(f"PPL change:       {pruned_ppl - original_ppl:+.2f}")

    # Per-layer summary
    logger.info("-" * 40)
    for r in all_results:
        if r["pruned"] > 0:
            logger.info(f"  Layer {r['layer']} {r['type']}: {r['total']} -> {r['total'] - r['pruned']} "
                        f"({100*r['pruned']/r['total']:.1f}% pruned)")

    # Save model if requested
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_model:
        save_path = output_dir / f"opt-{args.model_size}_pruned_{args.target}_s{args.sparsity}.pt"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved pruned model to {save_path}")

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
