#!/usr/bin/env python
"""
Run pruning experiments across ViT models and sparsity levels.

Usage:
    python vit_prune/run_experiments.py
    python vit_prune/run_experiments.py --models vit_base_patch16_224 vit_large_patch16_224
    python vit_prune/run_experiments.py --sparsity 0.3 0.5
    python vit_prune/run_experiments.py --force  # Re-run all experiments ignoring cache

    # Q/K dimension pruning with Sylvester-based compensation:
    python vit_prune/run_experiments.py --targets attn --attn_mode dim-logit --sparsity 0.3 0.5

    # Combined MLP + attention dimension pruning:
    python vit_prune/run_experiments.py --targets both --attn_mode dim-logit --sparsity 0.3
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default experiment configuration
DEFAULT_MODELS = [
    # "vit_base_patch16_224",
    "vit_large_patch16_224",
]

DEFAULT_SPARSITIES = [0.17,0.28]

# Default schedule configurations: list of (schedule_type, delta_per_round)
# delta=None means layerwise (no delta needed)
DEFAULT_SCHEDULES = [
    # ("layerwise", None),
    ("global", None),
]

# Default targets to prune
DEFAULT_TARGETS = ['both',
                    # 'attn','mlp'
                    ]  # Options: "mlp", "attn", "both"

# Large models that require memory optimizations
LARGE_MODELS = [
    "vit_large_patch16_224",
]

# Per-model batch sizes for large models to prevent OOM
# vit_large: 304M params, 24 blocks, 1024 embed_dim
LARGE_MODEL_BATCH_SIZE = {
    "vit_large_patch16_224": 32,
}

# Evaluation-only batch sizes (no covariance overhead, can be larger)
LARGE_MODEL_EVAL_BATCH_SIZE = {
    "vit_large_patch16_224": 128,
}

# Reduced num_workers for large models to save CPU memory
LARGE_MODEL_NUM_WORKERS = 2

# Paths for caching
EXPERIMENT_CACHE_PATH = Path(__file__).parent / "experiment_results_cache.json"


def is_large_model(model_name: str) -> bool:
    """Check if model requires memory optimizations."""
    return model_name in LARGE_MODELS


def _sdpa_flop_jit(inputs, outputs):
    """Count FLOPs for aten::scaled_dot_product_attention."""
    q = inputs[0]
    v = inputs[2]
    q_shape = q.type().sizes()
    v_shape = v.type().sizes()
    B, H, N, D_qk = q_shape
    D_v = v_shape[-1]
    return 2 * B * H * N * N * D_qk + 2 * B * H * N * N * D_v


def compute_model_flops(model: nn.Module, input_shape: tuple) -> int:
    """Compute model FLOPs in-place (no device transfer)."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)
    flops = FlopCountAnalysis(model, dummy_input)
    flops.set_op_handle("aten::scaled_dot_product_attention", _sdpa_flop_jit)
    total = flops.total()
    del flops, dummy_input
    torch.cuda.empty_cache()
    return total


def load_experiment_cache() -> Dict[str, Any]:
    """Load experiment results cache from JSON file."""
    if EXPERIMENT_CACHE_PATH.exists():
        with open(EXPERIMENT_CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_experiment_cache(cache: Dict[str, Any]) -> None:
    """Save experiment results cache to JSON file."""
    with open(EXPERIMENT_CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)
    logger.info(f"Saved experiment cache to {EXPERIMENT_CACHE_PATH}")


def get_cache_key(model: str, sparsity: float, val_path: str, schedule: str, delta: Optional[float], target: str, with_compensation: bool = True, attn_mode: str = 'head', ranker: str = 'active_energy') -> str:
    """Generate a unique cache key for an experiment.

    Note: target is a required parameter to prevent accidentally mixing cache entries
    from different pruning targets (mlp, attn, both).
    """
    comp_suffix = "comp" if with_compensation else "no_comp"
    delta_str = f"d{delta}" if delta is not None else "d_none"
    return f"{model}|{sparsity}|{val_path}|{schedule}|{delta_str}|{comp_suffix}|{target}|{attn_mode}|{ranker}"


def get_cached_result(model: str, sparsity: float, val_path: str, schedule: str, delta: Optional[float], target: str, attn_mode: str = 'head', ranker: str = 'active_energy') -> Optional[Dict[str, Any]]:
    """Get cached experiment result if available.

    Note: target is a required parameter to prevent accidentally loading
    cache entries from different pruning targets.
    """
    cache = load_experiment_cache()
    key = get_cache_key(model, sparsity, val_path, schedule, delta, target, attn_mode=attn_mode, ranker=ranker)
    return cache.get(key)


def save_experiment_result(model: str, sparsity: float, val_path: str, schedule: str, delta: Optional[float], target: str, result: Dict[str, Any], attn_mode: str = 'head', ranker: str = 'active_energy') -> None:
    """Save experiment result to cache.

    Note: target is a required parameter to ensure cache entries are
    properly separated by pruning target.
    """
    cache = load_experiment_cache()
    key = get_cache_key(model, sparsity, val_path, schedule, delta, target, attn_mode=attn_mode, ranker=ranker)
    result["schedule"] = schedule  # Store schedule in result for reference
    result["delta"] = delta  # Store delta in result for reference
    result["target"] = target  # Store target in result for reference
    result["attn_mode"] = attn_mode  # Store attn_mode in result for reference
    result["ranker"] = ranker  # Store ranker in result for reference
    result["cached_at"] = datetime.now().isoformat()
    cache[key] = result
    save_experiment_cache(cache)


def parse_args():
    parser = argparse.ArgumentParser(description='Run ViT pruning experiments')

    parser.add_argument(
        '--models', type=str, nargs='+', default=DEFAULT_MODELS,
        help='List of models to test'
    )
    parser.add_argument(
        '--sparsity', type=float, nargs='+', default=DEFAULT_SPARSITIES,
        help='List of sparsity levels to test (applied to both MLP and attention)'
    )
    parser.add_argument(
        '--calib_path', type=str,
        default='/home/boxiang/work/dao2/eff_vit/imagenet-mini/val',
        help='Path to calibration data'
    )
    parser.add_argument(
        '--val_path', type=str,
        default='/home/boxiang/work/ombs/imagenet-mini/imagenet-val',
        help='Path to validation data'
    )
    parser.add_argument(
        '--output_dir', type=str, default='vit_prune/experiment_results',
        help='Directory to save experiment results'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to run on'
    )
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='Batch size'
    )
    parser.add_argument(
        '--calib_samples', type=int, default=3923,
        help='Number of calibration samples'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force re-run all experiments, ignoring cache'
    )
    parser.add_argument(
        '--stats_cache_dir', type=str, default='vit_prune/activation_stats_cache',
        help='Directory for caching activation stats (default: activation_stats_cache)'
    )
    parser.add_argument(
        '--force_recollect', action='store_true',
        help='Bypass activation stats cache (still saves for future use)'
    )
    parser.add_argument(
        '--schedules', type=str, nargs='+', default=None,
        help='Schedule configs as "schedule:delta" (e.g., "layerwise global:0.15 global:0.1"). '
             'If not specified, uses DEFAULT_SCHEDULES.'
    )
    parser.add_argument(
        '--targets', type=str, nargs='+', default=DEFAULT_TARGETS,
        choices=['mlp', 'attn', 'both'],
        help='Pruning targets: mlp (MLP intermediate), attn (attention heads), both. '
             'Default: both'
    )
    parser.add_argument(
        '--attn_mode', type=str, default='dim-logit',
        choices=['head', 'dim-logit'],
        help='Attention pruning mode: '
             'head (prune entire heads), '
             'dim-logit (prune Q/K dims with Sylvester-based logit-matching compensation - '
             'applies different U/V transforms to Q/K). '
             'Uses E[q^2 * k^2] joint energy for ranking. Default: dim-logit'
    )
    parser.add_argument(
        '--min_qk_dim', type=int, default=8,
        help='Minimum Q/K dimensions to keep per head (for dim-logit mode). Default: 8'
    )
    parser.add_argument(
        '--ranker', type=str, default='energy_weightmagnitude',
        choices=['energy', 'active', 'active_energy', 'weight_magnitude', 'energy_weightmagnitude', 'active_weightmagnitude', 'ensemble'],
        help='Ranking policy for pruning. weight_magnitude uses weight L2 norms '
             'instead of activation statistics (baseline method). Default: energy_weightmagnitude'
    )
    parser.add_argument(
        '--no_comp_only', action='store_true',
        help='Run only non-compensation experiments (pruned model without compensation). '
             'By default, runs both with and without compensation for mlp/attn targets, '
             'and only with compensation for both target. '
             'This flag enables getting accuracy of pruned-only models for all targets.'
    )
    args = parser.parse_args()

    # Parse schedule configurations
    if args.schedules is None:
        args.schedule_configs = DEFAULT_SCHEDULES
    else:
        args.schedule_configs = []
        for s in args.schedules:
            if ':' in s:
                schedule, delta = s.split(':')
                args.schedule_configs.append((schedule, float(delta)))
            else:
                # layerwise doesn't need delta
                args.schedule_configs.append((s, None))

    return args


def run_single_experiment(
    model_name: str,
    sparsity: float,
    schedule: str,
    delta: Optional[float],
    target: str,
    args,
) -> Dict[str, Any]:
    """Run a single pruning experiment.

    Args:
        model_name: Name of the model to prune
        sparsity: Target sparsity level
        schedule: Pruning schedule ('layerwise' or 'global')
        delta: Delta per round for global schedule (None for layerwise)
        target: Pruning target ('mlp', 'attn', 'both')
        args: Command line arguments

    Returns:
        Dictionary with experiment results
    """
    from config.schemas import (
        CollectorConfig, PruningConfig, RunnerConfig, FullConfig,
        PruneTarget, ScheduleType, RankerType, CovarianceMode, AttentionPruneMode,
    )
    from pruning.runner import PruneRunner
    from vit_prune.run_prune import (
        load_model, load_validation_data, load_calibration_data,
        evaluate_accuracy, get_baseline_accuracy, save_baseline_accuracy,
    )

    delta_str = f"delta={delta}" if delta is not None else "no delta"
    logger.info(f"Running experiment: model={model_name}, sparsity={sparsity}, schedule={schedule}, {delta_str}, target={target}")

    result = {
        "model": model_name,
        "sparsity": sparsity,
        "schedule": schedule,
        "delta": delta,
        "target": target,
        "status": "failed",
    }

    try:
        # Create a namespace object for load functions
        class Args:
            pass

        exp_args = Args()
        exp_args.model = model_name
        exp_args.model_path = None
        exp_args.calib_path = args.calib_path
        exp_args.val_path = args.val_path
        # Use reduced batch size and num_workers for large models to prevent OOM
        if is_large_model(model_name):
            exp_args.batch_size = LARGE_MODEL_BATCH_SIZE.get(model_name, 16)
            exp_args.eval_batch_size = LARGE_MODEL_EVAL_BATCH_SIZE.get(model_name, exp_args.batch_size)
            exp_args.num_workers = LARGE_MODEL_NUM_WORKERS
            logger.info(f"Using batch_size={exp_args.batch_size}, eval_batch_size={exp_args.eval_batch_size}, num_workers={exp_args.num_workers} for large model {model_name}")
        else:
            exp_args.batch_size = args.batch_size
            exp_args.eval_batch_size = args.batch_size
            exp_args.num_workers = 4  # Default for normal models
        exp_args.device = args.device
        exp_args.calib_samples = args.calib_samples
        exp_args.target = target  # Use the target parameter
        exp_args.subsample_tokens = None
        exp_args.covariance_mode = 'exact'
        exp_args.sketch_dim = 256
        exp_args.schedule = schedule  # Use passed schedule
        exp_args.ranker = args.ranker  # Use CLI argument
        exp_args.lambda_reg = 1e-3
        exp_args.min_channels = 64
        exp_args.min_heads = 1  # Minimum attention heads
        exp_args.delta = delta  # Use passed delta (None = single round)
        exp_args.dtype = 'float32'
        exp_args.attn_mode = args.attn_mode  # Attention pruning mode
        # For attention dim modes, use the sparsity directly as qk_sparsity
        if target in ('attn', 'both') and args.attn_mode in ('dim-ac', 'dim-logit'):
            exp_args.qk_sparsity = sparsity  # Use attn_sparsity as qk_sparsity
        else:
            exp_args.qk_sparsity = sparsity  # Default (not used for head mode or MLP)
        exp_args.min_qk_dim = args.min_qk_dim  # Min Q/K dims for dim modes
        exp_args.output_dir = args.output_dir
        exp_args.save_pruned_path = None
        exp_args.seed = 42

        # Set seed
        torch.manual_seed(exp_args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(exp_args.seed)

        # Load validation data
        val_loader = load_validation_data(exp_args)

        # Load calibration data
        calib_loader = load_calibration_data(exp_args)

        # Get or compute baseline accuracy, params, and FLOPs (using original model)
        input_shape = (1, 3, 224, 224)
        cached_baseline = get_baseline_accuracy(model_name, args.val_path)
        if cached_baseline is not None and "params" in cached_baseline and "flops" in cached_baseline:
            baseline_top1 = cached_baseline["top1"]
            baseline_top5 = cached_baseline["top5"]
            baseline_params = cached_baseline["params"]
            baseline_flops = cached_baseline["flops"]
            logger.info(f"Using cached baseline: Top-1={baseline_top1:.2f}%, Top-5={baseline_top5:.2f}%, Params={baseline_params:,}, FLOPs={baseline_flops:,}")
        else:
            original_model = load_model(exp_args)
            baseline_params = sum(p.numel() for p in original_model.parameters())
            baseline_flops = compute_model_flops(original_model, input_shape)

            if cached_baseline is not None and "top1" in cached_baseline:
                baseline_top1 = cached_baseline["top1"]
                baseline_top5 = cached_baseline["top5"]
                logger.info(f"Using cached baseline accuracy: Top-1={baseline_top1:.2f}%, Top-5={baseline_top5:.2f}%")
            else:
                logger.info(f"Computing baseline accuracy for {model_name}...")
                baseline_top1, baseline_top5 = evaluate_accuracy(
                    original_model, val_loader, device=args.device, desc=f"Baseline {model_name}"
                )
                logger.info(f"Baseline: Top-1={baseline_top1:.2f}%, Top-5={baseline_top5:.2f}%")

            # Save baseline with params and FLOPs
            save_baseline_accuracy(model_name, args.val_path, {
                "top1": baseline_top1,
                "top5": baseline_top5,
                "params": baseline_params,
                "flops": baseline_flops,
            })
            logger.info(f"Baseline params: {baseline_params:,}, FLOPs: {baseline_flops:,}")

            # Free original model to save memory
            del original_model
            torch.cuda.empty_cache()

        result["baseline_top1"] = baseline_top1
        result["baseline_top5"] = baseline_top5
        result["baseline_params"] = baseline_params
        result["baseline_flops"] = baseline_flops

        # Process both with and without compensation (or only no-comp if --no_comp_only)
        compensation_modes = [False] if args.no_comp_only else [False, True]
        for with_compensation in compensation_modes:
            comp_label = "with compensation" if with_compensation else "without compensation"
            logger.info(f"\n--- Pruning {comp_label} ---")

            working_model = load_model(exp_args)

            # Create config
            collector = CollectorConfig(
                target=PruneTarget(exp_args.target),
                subsample_tokens=exp_args.subsample_tokens,
                covariance_mode=CovarianceMode(exp_args.covariance_mode),
                sketch_dim=exp_args.sketch_dim,
                store_raw=False,
                batch_size=exp_args.batch_size,
            )

            pruning_config = PruningConfig(
                target=PruneTarget(exp_args.target),
                schedule=ScheduleType(exp_args.schedule),
                sparsity=sparsity,
                ranker=RankerType(exp_args.ranker),
                lambda_reg=exp_args.lambda_reg,
                min_channels=exp_args.min_channels,
                min_heads=exp_args.min_heads,
                delta_per_round=exp_args.delta,
                attn_prune_mode=AttentionPruneMode(exp_args.attn_mode),
                qk_sparsity=exp_args.qk_sparsity,
                min_qk_dim=exp_args.min_qk_dim,
            )

            runner_config = RunnerConfig(
                device=exp_args.device,
                dtype=exp_args.dtype,
                output_dir=Path(exp_args.output_dir),
                save_pruned_path=None,
                calib_samples=exp_args.calib_samples,
                seed=exp_args.seed,
            )

            config = FullConfig(collector=collector, pruning=pruning_config, runner=runner_config)
            runner = PruneRunner(config)
            if args.stats_cache_dir:
                runner.stats_cache_dir = Path(args.stats_cache_dir)
                runner.force_recollect = args.force_recollect

            # Run pruning
            logger.info(f"Running pruning {comp_label}...")
            prune_result = runner.run(
                working_model, calib_loader,
                skip_compensation=not with_compensation,
                model_name=model_name,
            )

            # Evaluate pruned model
            pruned_top1, pruned_top5 = evaluate_accuracy(
                prune_result.pruned_model, val_loader, device=args.device,
                desc=f"Pruned ({'comp' if with_compensation else 'no comp'}) {model_name}"
            )
            logger.info(f"Pruned ({comp_label}): Top-1={pruned_top1:.2f}%, Top-5={pruned_top5:.2f}%")

            # Log parameter comparison
            pruned_params = prune_result.pruned_params
            param_reduction = baseline_params - pruned_params
            param_reduction_pct = (param_reduction / baseline_params) * 100 if baseline_params > 0 else 0
            logger.info(f"Parameters: {baseline_params:,} -> {pruned_params:,} "
                       f"(reduced {param_reduction:,}, {param_reduction_pct:.2f}%)")

            # Compute pruned FLOPs
            pruned_flops = compute_model_flops(prune_result.pruned_model, input_shape)

            # Store results
            if with_compensation:
                result["pruned_top1"] = pruned_top1
                result["pruned_top5"] = pruned_top5
                result["original_params"] = prune_result.original_params
                result["pruned_params"] = prune_result.pruned_params
                result["compression_ratio"] = prune_result.compression_ratio
                result["pruned_flops"] = pruned_flops
                result["flops_reduction"] = 1.0 - (pruned_flops / baseline_flops) if baseline_flops > 0 else 0.0
            else:
                result["pruned_no_comp_top1"] = pruned_top1
                result["pruned_no_comp_top5"] = pruned_top5

            # Clean up
            del prune_result, working_model
            torch.cuda.empty_cache()

        # Compute drops and recovery
        result["top1_drop_no_comp"] = baseline_top1 - result["pruned_no_comp_top1"]
        result["top5_drop_no_comp"] = baseline_top5 - result["pruned_no_comp_top5"]
        result["top1_drop"] = baseline_top1 - result["pruned_top1"]
        result["top5_drop"] = baseline_top5 - result["pruned_top5"]
        result["top1_recovery"] = result["pruned_top1"] - result["pruned_no_comp_top1"]
        result["top5_recovery"] = result["pruned_top5"] - result["pruned_no_comp_top5"]

        result["status"] = "success"

        # Save to cache
        save_experiment_result(model_name, sparsity, args.val_path, schedule, delta, target, result, args.attn_mode, args.ranker)

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


def run_single_experiment_both(
    model_name: str,
    mlp_sparsity: float,
    attn_sparsity: float,
    schedule: str,
    delta: Optional[float],
    args,
    target: str = 'both',
) -> Dict[str, Any]:
    """Run a single pruning experiment that prunes both MLP and attention.

    Args:
        model_name: Name of the model to prune
        mlp_sparsity: Target MLP sparsity level
        attn_sparsity: Target attention sparsity level
        schedule: Pruning schedule ('layerwise' or 'global')
        delta: Delta per round for global schedule (None for layerwise)
        args: Command line arguments
        target: Pruning target ('both')

    Returns:
        Dictionary with experiment results
    """
    from config.schemas import (
        CollectorConfig, PruningConfig, RunnerConfig, FullConfig,
        PruneTarget, ScheduleType, RankerType, CovarianceMode, AttentionPruneMode,
    )
    from pruning.runner import PruneRunner
    from vit_prune.run_prune import (
        load_model, load_validation_data, load_calibration_data,
        evaluate_accuracy, get_baseline_accuracy, save_baseline_accuracy,
    )

    delta_str = f"delta={delta}" if delta is not None else "no delta"
    logger.info(f"Running experiment: model={model_name}, mlp_sparsity={mlp_sparsity}, "
                f"attn_sparsity={attn_sparsity}, schedule={schedule}, {delta_str}, target={target}")

    result = {
        "model": model_name,
        "mlp_sparsity": mlp_sparsity,
        "attn_sparsity": attn_sparsity,
        "sparsity": f"mlp{mlp_sparsity}_attn{attn_sparsity}",  # Combined sparsity key
        "schedule": schedule,
        "delta": delta,
        "target": target,
        "status": "running",
    }

    try:
        # Setup experiment args
        exp_args = argparse.Namespace()
        exp_args.model = model_name
        exp_args.model_path = None
        exp_args.device = args.device
        exp_args.batch_size = LARGE_MODEL_BATCH_SIZE.get(model_name, args.batch_size) if is_large_model(model_name) else args.batch_size
        exp_args.eval_batch_size = LARGE_MODEL_EVAL_BATCH_SIZE.get(model_name, exp_args.batch_size) if is_large_model(model_name) else args.batch_size
        exp_args.num_workers = LARGE_MODEL_NUM_WORKERS if is_large_model(model_name) else 4
        exp_args.calib_path = args.calib_path
        exp_args.val_path = args.val_path
        exp_args.calib_samples = args.calib_samples
        exp_args.target = target
        exp_args.subsample_tokens = None
        exp_args.covariance_mode = 'exact'
        exp_args.sketch_dim = 256
        exp_args.schedule = schedule
        exp_args.ranker = args.ranker  # Use CLI argument
        exp_args.lambda_reg = 1e-3
        exp_args.min_channels = 64
        exp_args.min_heads = 1
        exp_args.delta = delta
        exp_args.dtype = 'float32'
        exp_args.attn_mode = args.attn_mode
        exp_args.qk_sparsity = attn_sparsity  # Use attn_sparsity for Q/K dim pruning
        exp_args.min_qk_dim = args.min_qk_dim
        exp_args.output_dir = args.output_dir
        exp_args.save_pruned_path = None
        exp_args.seed = 42

        # Set seed
        torch.manual_seed(exp_args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(exp_args.seed)

        # Load validation data
        val_loader = load_validation_data(exp_args)

        # Load calibration data
        calib_loader = load_calibration_data(exp_args)

        # Get or compute baseline accuracy, params, and FLOPs
        input_shape = (1, 3, 224, 224)
        cached_baseline = get_baseline_accuracy(model_name, args.val_path)
        if cached_baseline is not None and "params" in cached_baseline and "flops" in cached_baseline:
            baseline_top1 = cached_baseline["top1"]
            baseline_top5 = cached_baseline["top5"]
            baseline_params = cached_baseline["params"]
            baseline_flops = cached_baseline["flops"]
            logger.info(f"Using cached baseline: Top-1={baseline_top1:.2f}%, Top-5={baseline_top5:.2f}%, Params={baseline_params:,}, FLOPs={baseline_flops:,}")
        else:
            original_model = load_model(exp_args)
            baseline_params = sum(p.numel() for p in original_model.parameters())
            baseline_flops = compute_model_flops(original_model, input_shape)

            if cached_baseline is not None and "top1" in cached_baseline:
                baseline_top1 = cached_baseline["top1"]
                baseline_top5 = cached_baseline["top5"]
                logger.info(f"Using cached baseline accuracy: Top-1={baseline_top1:.2f}%, Top-5={baseline_top5:.2f}%")
            else:
                logger.info(f"Computing baseline accuracy for {model_name}...")
                baseline_top1, baseline_top5 = evaluate_accuracy(
                    original_model, val_loader, device=args.device, desc=f"Baseline {model_name}"
                )
                logger.info(f"Baseline: Top-1={baseline_top1:.2f}%, Top-5={baseline_top5:.2f}%")

            save_baseline_accuracy(model_name, args.val_path, {
                "top1": baseline_top1,
                "top5": baseline_top5,
                "params": baseline_params,
                "flops": baseline_flops,
            })
            logger.info(f"Baseline params: {baseline_params:,}, FLOPs: {baseline_flops:,}")
            del original_model
            torch.cuda.empty_cache()

        result["baseline_top1"] = baseline_top1
        result["baseline_top5"] = baseline_top5
        result["baseline_params"] = baseline_params
        result["baseline_flops"] = baseline_flops

        # For 'both' target, run only with compensation by default, or only no-comp if --no_comp_only
        compensation_modes = [False] if args.no_comp_only else [True]
        for with_compensation in compensation_modes:
            comp_label = "with compensation" if with_compensation else "without compensation"
            logger.info(f"\n--- Pruning {comp_label} ---")

            # Load fresh model for pruning
            logger.info(f"Pruning from scratch with MLP={mlp_sparsity}, Attn={attn_sparsity}")
            working_model = load_model(exp_args)

            # Create config - use MLP sparsity as the main sparsity for schedule iteration
            # The attn_sparsity is passed via qk_sparsity for dim-logit mode
            prune_target = PruneTarget(target)
            collector = CollectorConfig(
                target=prune_target,
                subsample_tokens=exp_args.subsample_tokens,
                covariance_mode=CovarianceMode(exp_args.covariance_mode),
                sketch_dim=exp_args.sketch_dim,
                store_raw=False,
                batch_size=exp_args.batch_size,
            )

            # For 'both' target, we need to handle MLP and attention sparsities separately
            # Use MLP sparsity as the main sparsity parameter
            pruning_config = PruningConfig(
                target=prune_target,
                schedule=ScheduleType(exp_args.schedule),
                sparsity=mlp_sparsity,  # MLP sparsity for schedule
                ranker=RankerType(exp_args.ranker),
                lambda_reg=exp_args.lambda_reg,
                min_channels=exp_args.min_channels,
                min_heads=exp_args.min_heads,
                delta_per_round=exp_args.delta,
                attn_prune_mode=AttentionPruneMode(exp_args.attn_mode),
                qk_sparsity=attn_sparsity,  # Attention sparsity for Q/K dims
                min_qk_dim=exp_args.min_qk_dim,
            )

            runner_config = RunnerConfig(
                device=exp_args.device,
                dtype=exp_args.dtype,
                output_dir=Path(exp_args.output_dir),
                save_pruned_path=None,
                calib_samples=exp_args.calib_samples,
                seed=exp_args.seed,
            )

            config = FullConfig(collector=collector, pruning=pruning_config, runner=runner_config)
            runner = PruneRunner(config)
            if args.stats_cache_dir:
                runner.stats_cache_dir = Path(args.stats_cache_dir)
                runner.force_recollect = args.force_recollect

            # Run pruning
            logger.info(f"Running pruning {comp_label}...")
            prune_result = runner.run(
                working_model, calib_loader,
                skip_compensation=not with_compensation,
                model_name=model_name,
            )

            # Evaluate pruned model
            pruned_top1, pruned_top5 = evaluate_accuracy(
                prune_result.pruned_model, val_loader, device=args.device,
                desc=f"Pruned ({'comp' if with_compensation else 'no comp'}) {model_name}"
            )
            logger.info(f"Pruned ({comp_label}): Top-1={pruned_top1:.2f}%, Top-5={pruned_top5:.2f}%")

            # Log parameter comparison
            pruned_params = prune_result.pruned_params
            param_reduction = baseline_params - pruned_params
            param_reduction_pct = (param_reduction / baseline_params) * 100 if baseline_params > 0 else 0
            logger.info(f"Parameters: {baseline_params:,} -> {pruned_params:,} "
                       f"(reduced {param_reduction:,}, {param_reduction_pct:.2f}%)")

            # Compute pruned FLOPs
            pruned_flops = compute_model_flops(prune_result.pruned_model, input_shape)

            # Store results based on compensation mode
            result["original_params"] = prune_result.original_params
            result["pruned_params"] = prune_result.pruned_params
            result["compression_ratio"] = prune_result.compression_ratio
            result["pruned_flops"] = pruned_flops
            result["flops_reduction"] = 1.0 - (pruned_flops / baseline_flops) if baseline_flops > 0 else 0.0
            if with_compensation:
                result["pruned_top1"] = pruned_top1
                result["pruned_top5"] = pruned_top5
                # Set no-comp results to same as comp if not run separately
                if "pruned_no_comp_top1" not in result:
                    result["pruned_no_comp_top1"] = pruned_top1
                    result["pruned_no_comp_top5"] = pruned_top5
            else:
                result["pruned_no_comp_top1"] = pruned_top1
                result["pruned_no_comp_top5"] = pruned_top5
                # Set comp results to same as no-comp if not running with comp
                if "pruned_top1" not in result:
                    result["pruned_top1"] = pruned_top1
                    result["pruned_top5"] = pruned_top5

            # Clean up
            del prune_result, working_model
            torch.cuda.empty_cache()

        # Compute drops (no recovery since we only run with compensation)
        result["top1_drop_no_comp"] = baseline_top1 - result["pruned_top1"]
        result["top5_drop_no_comp"] = baseline_top5 - result["pruned_top5"]
        result["top1_drop"] = baseline_top1 - result["pruned_top1"]
        result["top5_drop"] = baseline_top5 - result["pruned_top5"]
        result["top1_recovery"] = 0.0  # Not applicable for 'both' target
        result["top5_recovery"] = 0.0  # Not applicable for 'both' target

        result["status"] = "success"

        # Save to cache with combined sparsity key
        cache_key_sparsity = f"mlp{mlp_sparsity}_attn{attn_sparsity}"
        save_experiment_result(model_name, cache_key_sparsity, args.val_path, schedule, delta, target, result, args.attn_mode, args.ranker)

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


def format_results_table(results: List[Dict[str, Any]]) -> str:
    """Format results as a markdown table."""
    lines = []
    lines.append("| Model | Target | Schedule | Sparsity | Baseline Top-1 | Pruned (no comp) | Pruned (comp) | Drop | Recovery | Compression | FLOPs Red. |")
    lines.append("|-------|--------|----------|----------|----------------|------------------|---------------|------|----------|-------------|------------|")

    for r in results:
        delta = r.get('delta')
        if delta is not None:
            schedule_str = f"{r.get('schedule', 'layerwise')}:d{delta}"
        else:
            schedule_str = r.get('schedule', 'layerwise')
        target_str = r.get('target', 'mlp')
        # Format sparsity - handle both float and string (for 'both' target)
        sparsity = r['sparsity']
        if isinstance(sparsity, str):
            sparsity_str = sparsity
        else:
            sparsity_str = f"{sparsity:.1f}"
        if r["status"] == "success":
            flops_red = r.get('flops_reduction', 0.0)
            lines.append(
                f"| {r['model']} | {target_str} | {schedule_str} | {sparsity_str} | "
                f"{r['baseline_top1']:.2f}% | "
                f"{r['pruned_no_comp_top1']:.2f}% | "
                f"{r['pruned_top1']:.2f}% | "
                f"{r['top1_drop']:.2f}% | "
                f"+{r['top1_recovery']:.2f}% | "
                f"{r['compression_ratio']:.2f}x | "
                f"{flops_red:.1%} |"
            )
        else:
            lines.append(f"| {r['model']} | {target_str} | {schedule_str} | {sparsity_str} | FAILED | - | - | - | - | - | - |")

    return "\n".join(lines)


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sorted_sparsities = sorted(args.sparsity)

    # Format schedule configs for display
    schedule_strs = [f"{s}:d{d}" if d is not None else s for s, d in args.schedule_configs]

    logger.info("=" * 70)
    logger.info("ViT Pruning Experiments")
    logger.info("=" * 70)
    logger.info(f"Models: {args.models}")
    logger.info(f"Sparsities: {sorted_sparsities}")
    logger.info(f"Attn Mode: {args.attn_mode}")
    logger.info(f"Ranker: {args.ranker}")
    logger.info(f"Schedules: {schedule_strs}")
    logger.info(f"Targets: {args.targets}")
    logger.info(f"Compensation mode: {'no-comp only' if args.no_comp_only else 'default'}")
    logger.info(f"Stats cache dir: {args.stats_cache_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)

    all_results = []
    total_experiments = len(args.models) * len(args.targets) * len(sorted_sparsities) * len(args.schedule_configs)

    current = 0
    cached_count = 0
    computed_count = 0

    # Process each model
    for model in args.models:
        # Process each target
        for target in args.targets:
            for schedule, delta in args.schedule_configs:
                for sparsity in sorted_sparsities:
                    current += 1
                    delta_str = f"delta={delta}" if delta is not None else "no delta"
                    logger.info(f"\n{'='*70}")
                    logger.info(f"Experiment {current}/{total_experiments}: {model} @ {sparsity} sparsity [{schedule}, {delta_str}, target={target}]")
                    logger.info(f"{'='*70}")

                    # Check cache first (unless --force is specified)
                    if target == 'both':
                        cache_key_sparsity = f"mlp{sparsity}_attn{sparsity}"
                    else:
                        cache_key_sparsity = sparsity

                    if not args.force:
                        cached_result = get_cached_result(model, cache_key_sparsity, args.val_path, schedule, delta, target, args.attn_mode, args.ranker)
                        if cached_result is not None and cached_result.get("status") == "success":
                            logger.info(f"Using cached result (cached at: {cached_result.get('cached_at', 'unknown')})")
                            all_results.append(cached_result)
                            cached_count += 1
                            logger.info(f"Result (cached): Baseline={cached_result['baseline_top1']:.2f}%, "
                                       f"Pruned={cached_result['pruned_top1']:.2f}%, "
                                       f"Drop={cached_result['top1_drop']:.2f}%, "
                                       f"Compression={cached_result['compression_ratio']:.2f}x, "
                                       f"FLOPs Red.={cached_result.get('flops_reduction', 0.0):.1%}")
                            continue

                    # Run experiment
                    if target == 'both':
                        result = run_single_experiment_both(
                            model, sparsity, sparsity, schedule, delta, args,
                            target=target
                        )
                    else:
                        result = run_single_experiment(model, sparsity, schedule, delta, target, args)
                    all_results.append(result)
                    computed_count += 1

                    # Save intermediate results
                    results_file = output_dir / f"results_{timestamp}.json"
                    with open(results_file, 'w') as f:
                        json.dump(all_results, f, indent=2)

                    # Print current result
                    if result["status"] == "success":
                        logger.info(f"Result: Baseline={result['baseline_top1']:.2f}%, "
                                   f"Pruned={result['pruned_top1']:.2f}%, "
                                   f"Drop={result['top1_drop']:.2f}%, "
                                   f"Compression={result['compression_ratio']:.2f}x, "
                                   f"FLOPs Red.={result.get('flops_reduction', 0.0):.1%}")

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)

    table = format_results_table(all_results)
    logger.info("\n" + table)

    # Save final results
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")

    # Save markdown report
    report_file = output_dir / f"report_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write("# ViT Pruning Experiment Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Models:** {', '.join(args.models)}\n\n")
        f.write(f"**Sparsities:** {', '.join(map(str, sorted_sparsities))}\n\n")
        f.write(f"**Attn Mode:** {args.attn_mode}\n\n")
        f.write(f"**Ranker:** {args.ranker}\n\n")
        f.write(f"**Schedules:** {', '.join(schedule_strs)}\n\n")
        f.write(f"**Targets:** {', '.join(args.targets)}\n\n")
        f.write(f"**Compensation Mode:** {'No-comp only' if args.no_comp_only else 'Default'}\n\n")
        f.write("## Results\n\n")
        f.write(table + "\n\n")
        f.write("## Detailed Results\n\n")
        for r in all_results:
            delta = r.get('delta')
            if delta is not None:
                schedule_str = f"{r.get('schedule', 'layerwise')}:d{delta}"
            else:
                schedule_str = r.get('schedule', 'layerwise')
            target_str = r.get('target', 'mlp')
            f.write(f"### {r['model']} @ {r['sparsity']} sparsity [{schedule_str}, target={target_str}]\n\n")
            if r["status"] == "success":
                f.write(f"- **Target:** {target_str}\n")
                f.write(f"- **Schedule:** {schedule_str}\n")
                f.write(f"- **Baseline:** Top-1: {r['baseline_top1']:.2f}%, Top-5: {r['baseline_top5']:.2f}%\n")
                f.write(f"- **Pruned (no compensation):** Top-1: {r['pruned_no_comp_top1']:.2f}%, Top-5: {r['pruned_no_comp_top5']:.2f}%\n")
                f.write(f"- **Pruned (compensated):** Top-1: {r['pruned_top1']:.2f}%, Top-5: {r['pruned_top5']:.2f}%\n")
                f.write(f"- **Accuracy drop:** Top-1: {r['top1_drop']:.2f}%, Top-5: {r['top5_drop']:.2f}%\n")
                f.write(f"- **Compensation recovery:** Top-1: +{r['top1_recovery']:.2f}%, Top-5: +{r['top5_recovery']:.2f}%\n")
                f.write(f"- **Parameters:** {r['original_params']:,} -> {r['pruned_params']:,} ({r['compression_ratio']:.2f}x compression)\n")
                if 'baseline_flops' in r and 'pruned_flops' in r:
                    f.write(f"- **FLOPs:** {r['baseline_flops']:,} -> {r['pruned_flops']:,} ({r.get('flops_reduction', 0.0):.1%} reduction)\n")
            else:
                f.write(f"- **Status:** FAILED\n")
                f.write(f"- **Error:** {r.get('error', 'Unknown')}\n")
            f.write("\n")

    logger.info(f"Report saved to: {report_file}")

    # Print success/failure summary
    successful = sum(1 for r in all_results if r["status"] == "success")
    logger.info(f"\nCompleted: {successful}/{total_experiments} experiments successful")
    logger.info(f"  - From cache: {cached_count}")
    logger.info(f"  - Newly computed: {computed_count}")
    logger.info(f"\nExperiment cache: {EXPERIMENT_CACHE_PATH}")

    return 0 if successful == total_experiments else 1


if __name__ == '__main__':
    sys.exit(main())
