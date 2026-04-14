"""Batch experiment runner for DINOv2 pruning evaluation.

Evaluates pruning on downstream tasks (depth estimation, segmentation)
with support for task-specific vs shared-backbone pruning scopes.
"""

import argparse
import gc
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.schemas import (
    AttentionPruneMode,
    CollectorConfig,
    CovarianceMode,
    FullConfig,
    PruneTarget,
    PruningConfig,
    RankerType,
    RunnerConfig,
    ScheduleType,
)
from fvcore.nn import FlopCountAnalysis
from pruning.runner import PruneRunner

from dino_prune._datasets import create_calibration_loader, create_validation_loaders
from dino_prune.evaluation import (
    DepthEvaluator,
    FeatureEvaluator,
    SegmentationEvaluator,
    format_depth_metrics,
    format_feature_metrics,
    format_segmentation_metrics,
)
from dino_prune.models import count_parameters, load_model
from dino_prune.dinov2_models import create_dinov2_depth_model, create_dinov2_seg_model

logger = logging.getLogger(__name__)


def _sdpa_flop_jit(inputs, outputs):
    """Count FLOPs for aten::scaled_dot_product_attention.

    FLOPs = 2 * B * H * N * N * D_qk  (Q @ K^T)
          + 2 * B * H * N * N * D_v    (attn @ V)
    where Q is (B, H, N, D_qk), K is (B, H, N, D_qk), V is (B, H, N, D_v).
    """
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


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv2 pruning experiment runner")

    parser.add_argument("--model", type=str, default="huge",
                        choices=["small", "base", "large", "huge"])
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--target", type=str, default="both",
                        choices=["mlp", "attn", "both"])
    parser.add_argument("--ranker", type=str, default="weight_magnitude",
                        choices=["energy", "active", "active_energy", "weight_magnitude", "energy_weightmagnitude", "active_weightmagnitude", "ensemble"])
    parser.add_argument("--scope", type=str, default="shared-backbone",
                        choices=["task-specific", "shared-backbone"])

    # Data paths
    parser.add_argument("--nyu_path", type=str,
                        default="/home/boxiang/work/dao2/nyu_v2/data",
                        help="Path to NYU Depth V2 dataset")
    parser.add_argument("--use_ade", action="store_true", default=False,
                        help="Use ADE20K dataset (default True for shared-backbone)")

    # Data config
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--calib_samples", type=int, default=10000)
    parser.add_argument("--subsample_tokens", type=int, default=100)
    parser.add_argument("--nyu_samples", type=int, default=5000)
    parser.add_argument("--ade_samples", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=4)

    # Pruning config
    parser.add_argument("--qk_sparsity", type=float, default=0.1)
    parser.add_argument("--min_qk_dim", type=int, default=8)
    parser.add_argument("--lambda_reg", type=float, default=1e-3)

    # Runtime
    parser.add_argument("--output_dir", type=str, default="dino_prune/dino_experiments")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_compensation", action="store_true")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline computation if cache exists")
    parser.add_argument("--stats_cache_dir", type=str, default="dino_prune/activation_stats_cache",
                        help="Directory for caching activation stats (default: activation_stats_cache)")
    parser.add_argument("--force_recollect", action="store_true",
                        help="Bypass activation stats cache (still saves for future use)")

    args = parser.parse_args()

    # shared-backbone implies use_ade
    if args.scope == "shared-backbone":
        args.use_ade = True

    return args


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def get_cache_path(output_dir: str) -> Path:
    return Path(output_dir) / "experiment_cache.json"


def load_experiment_cache(output_dir: str) -> Dict[str, Any]:
    path = get_cache_path(output_dir)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_experiment_cache(output_dir: str, cache: Dict[str, Any]):
    path = get_cache_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def get_cache_key(args, task: str) -> str:
    """Build a cache key that captures all settings affecting experiment results."""
    parts = [
        "dinov2", args.model, str(args.sparsity), args.target,
        args.ranker, args.scope, task,
        f"qk{args.qk_sparsity}", f"mqk{args.min_qk_dim}",
        f"lam{args.lambda_reg}", f"calib{args.calib_samples}",
        f"sub{args.subsample_tokens}", f"img{args.image_size}",
        f"nyu{args.nyu_samples}", f"ade{args.ade_samples}",
        f"comp{int(not args.skip_compensation)}",
    ]
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------


def compute_baseline(args) -> Dict[str, Any]:
    """Compute baseline metrics for the unpruned model."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Store baseline cache in a fixed location shared across all runs
    cache_dir = Path(__file__).parent / "dino_experiments"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"baseline_dinov2_{args.model}_img{args.image_size}.json"

    if args.skip_baseline and cache_file.exists():
        logger.info(f"Loading cached baseline from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    logger.info(f"Computing baseline for dinov2/{args.model}...")
    model, model_config = load_model(args.model)
    model = model.to(args.device).eval()

    input_shape = (1, 3, args.image_size, args.image_size)
    original_params = count_parameters(model)
    original_flops = compute_model_flops(model, input_shape)

    baseline = {
        "model": args.model,
        "backbone": "dinov2",
        "original_params": original_params,
        "original_flops": original_flops,
        "calib_samples": args.calib_samples,
        "depth_metrics": None,
        "seg_metrics": None,
    }

    # Create validation loaders
    nyu_val_loader, ade_val_loader = create_validation_loaders(
        nyu_root=args.nyu_path,
        use_ade=args.use_ade,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # Depth evaluation — reload model from state_dict to avoid deepcopy on GPU
    if nyu_val_loader is not None:
        logger.info("Evaluating baseline depth on NYU...")
        depth_backbone, _ = load_model(args.model)
        depth_backbone.load_state_dict(model.state_dict())
        depth_model = create_dinov2_depth_model(depth_backbone, args.model)
        depth_model = depth_model.to(args.device).eval()
        depth_evaluator = DepthEvaluator()
        depth_metrics = depth_evaluator.evaluate(
            depth_model, nyu_val_loader, device=args.device, depth_head=None,
        )
        del depth_model, depth_backbone
        torch.cuda.empty_cache()
        baseline["depth_metrics"] = asdict(depth_metrics)
        logger.info(f"Baseline depth: {format_depth_metrics(depth_metrics)}")

    # Segmentation evaluation — reload model from state_dict to avoid deepcopy on GPU
    if ade_val_loader is not None:
        logger.info("Evaluating baseline segmentation on ADE20K...")
        seg_backbone, _ = load_model(args.model)
        seg_backbone.load_state_dict(model.state_dict())
        seg_model = create_dinov2_seg_model(seg_backbone, args.model)
        seg_model = seg_model.to(args.device).eval()
        seg_evaluator = SegmentationEvaluator()
        seg_metrics = seg_evaluator.evaluate(
            seg_model, ade_val_loader, device=args.device, seg_head=None,
        )
        del seg_model, seg_backbone
        torch.cuda.empty_cache()
        baseline["seg_metrics"] = asdict(seg_metrics)

    # Free the original backbone model — no longer needed
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Save cache
    with open(cache_file, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info(f"Baseline saved to {cache_file}")

    return baseline


# ---------------------------------------------------------------------------
# Calibration loader construction
# ---------------------------------------------------------------------------

def create_scoped_calib_loader(args, scope: str, task: str) -> torch.utils.data.DataLoader:
    """Create calibration loader based on scope and task."""
    if scope == "task-specific" and task == "depth":
        return create_calibration_loader(
            nyu_root=args.nyu_path, use_ade=False,
            nyu_samples=args.nyu_samples, ade_samples=0,
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size,
        )
    elif scope == "task-specific" and task == "seg":
        return create_calibration_loader(
            nyu_root=None, use_ade=True,
            nyu_samples=0, ade_samples=args.ade_samples,
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size,
        )
    else:  # shared-backbone
        return create_calibration_loader(
            nyu_root=args.nyu_path, use_ade=True,
            nyu_samples=args.nyu_samples, ade_samples=args.ade_samples,
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size,
        )


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------

def create_config(args) -> FullConfig:
    """Build FullConfig from experiment args."""
    collector = CollectorConfig(
        target=PruneTarget(args.target),
        subsample_tokens=args.subsample_tokens,
        covariance_mode=CovarianceMode.EXACT,
        store_raw=False,
        batch_size=args.batch_size,
    )

    pruning = PruningConfig(
        target=PruneTarget(args.target),
        schedule=ScheduleType.LAYERWISE,
        sparsity=args.sparsity,
        ranker=RankerType(args.ranker),
        lambda_reg=args.lambda_reg,
        attn_prune_mode=AttentionPruneMode.DIM_LOGIT,
        qk_sparsity=args.qk_sparsity,
        min_qk_dim=args.min_qk_dim,
    )

    runner = RunnerConfig(
        device=args.device,
        output_dir=Path(args.output_dir),
        save_pruned_path=None,
        calib_samples=args.calib_samples,
        seed=args.seed,
    )

    return FullConfig(collector=collector, pruning=pruning, runner=runner)


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    args,
    baseline: Dict[str, Any],
    task: str,
) -> Dict[str, Any]:
    """Run a single pruning experiment.

    Args:
        args: Parsed CLI args.
        baseline: Baseline metrics dict.
        task: "depth", "seg", or "both".

    Returns:
        Result dict with all metrics.
    """
    logger.info(f"Running experiment: model={args.model} sparsity={args.sparsity} "
                f"target={args.target} ranker={args.ranker} scope={args.scope} task={task}")

    # Load model — compute FLOPs first, then save state_dict to avoid
    # having both in CPU RAM simultaneously during JIT tracing
    model, model_config = load_model(args.model)
    model = model.to(args.device).eval()

    input_shape = (1, 3, args.image_size, args.image_size)
    original_params = count_parameters(model)
    original_flops = compute_model_flops(model, input_shape)

    # Save state_dict to CPU *after* FLOPs computation to avoid RAM pressure
    original_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Create scoped calibration loader
    calib_loader = create_scoped_calib_loader(args, args.scope, task)

    # Build config and run pruning
    config = create_config(args)
    runner = PruneRunner(config)
    if args.stats_cache_dir:
        runner.stats_cache_dir = Path(args.stats_cache_dir)
        runner.force_recollect = args.force_recollect
    prune_result = runner.run(
        model, calib_loader, skip_compensation=args.skip_compensation,
        model_name=f"dinov2_{args.model}",
    )

    # model was moved to CPU by the runner; free it now — we have original_state_dict
    del model
    gc.collect()

    if not prune_result.success:
        logger.error(f"Pruning failed: {prune_result.error_message}")
        return {
            "model": args.model,
            "sparsity": args.sparsity,
            "target": args.target,
            "ranker": args.ranker,
            "scope": args.scope,
            "task": task,
            "error": prune_result.error_message,
        }

    pruned_model = prune_result.pruned_model
    pruned_params = count_parameters(pruned_model)
    pruned_flops = compute_model_flops(pruned_model, input_shape)

    result = {
        "model": args.model,
        "sparsity": args.sparsity,
        "target": args.target,
        "ranker": args.ranker,
        "scope": args.scope,
        "task": task,
        "calib_samples": args.calib_samples,
        "original_params": original_params,
        "pruned_params": pruned_params,
        "compression_ratio": original_params / pruned_params if pruned_params > 0 else float("inf"),
        "original_flops": original_flops,
        "pruned_flops": pruned_flops,
        "flops_reduction": 1.0 - (pruned_flops / original_flops) if original_flops > 0 else 0.0,
        "depth_metrics": None,
        "seg_metrics": None,
        "feature_similarity": None,
        "baseline_depth": baseline.get("depth_metrics"),
        "baseline_seg": baseline.get("seg_metrics"),
    }

    # Create validation loaders
    eval_depth = task in ("depth", "both")
    eval_seg = task in ("seg", "both")

    nyu_val_loader, ade_val_loader = create_validation_loaders(
        nyu_root=args.nyu_path if eval_depth else None,
        use_ade=eval_seg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # Feature similarity (must run before depth/seg eval, which patch backbone.forward)
    logger.info("Computing feature similarity...")
    feat_loader = nyu_val_loader or ade_val_loader
    if feat_loader is not None:
        # Reconstruct original model from saved CPU state_dict
        original_model, _ = load_model(args.model)
        original_model.load_state_dict(original_state_dict)
        original_model = original_model.to(args.device).eval()
        feat_evaluator = FeatureEvaluator()
        feat_metrics = feat_evaluator.compare_models(
            original_model, pruned_model, feat_loader, device=args.device,
        )
        del original_model
        torch.cuda.empty_cache()
        result["feature_similarity"] = asdict(feat_metrics)
        logger.info(f"Feature similarity: {format_feature_metrics(feat_metrics)}")
    del original_state_dict

    # Depth evaluation
    if eval_depth and nyu_val_loader is not None:
        logger.info("Evaluating pruned model on NYU depth...")
        depth_model = create_dinov2_depth_model(pruned_model, args.model)
        depth_model = depth_model.to(args.device).eval()
        depth_evaluator = DepthEvaluator()
        depth_metrics = depth_evaluator.evaluate(
            depth_model, nyu_val_loader, device=args.device, depth_head=None,
        )
        del depth_model
        result["depth_metrics"] = asdict(depth_metrics)
        logger.info(f"Pruned depth: {format_depth_metrics(depth_metrics)}")

    # Segmentation evaluation
    if eval_seg and ade_val_loader is not None:
        logger.info("Evaluating pruned model on ADE20K segmentation...")
        seg_model = create_dinov2_seg_model(pruned_model, args.model)
        seg_model = seg_model.to(args.device).eval()
        seg_evaluator = SegmentationEvaluator()
        seg_metrics = seg_evaluator.evaluate(
            seg_model, ade_val_loader, device=args.device, seg_head=None,
        )
        del seg_model
        result["seg_metrics"] = asdict(seg_metrics)
        logger.info(f"Pruned seg: {format_segmentation_metrics(seg_metrics)}")

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(results: List[Dict[str, Any]], baseline: Dict[str, Any]) -> str:
    """Format results as a markdown summary table."""
    lines = []
    lines.append(f"# DINOv2 Pruning Experiment Results")
    lines.append("")
    lines.append(f"Model: {baseline.get('model', 'unknown')}")
    lines.append(f"Original params: {baseline.get('original_params', 0):,}")
    lines.append(f"Original FLOPs: {baseline.get('original_flops', 0):,}")
    lines.append("")

    # Table header
    header = "| Target | Ranker | Scope | Task | Params | Compression | FLOPs Red. |"
    sep = "|--------|--------|-------|------|--------|-------------|------------|"

    # Add task-specific metric columns
    has_depth = any(r.get("depth_metrics") for r in results)
    has_seg = any(r.get("seg_metrics") for r in results)

    if has_depth:
        header += " RMSE | delta1 | RMSE Delta |"
        sep += "------|--------|------------|"
    if has_seg:
        header += " mIoU | mIoU Delta |"
        sep += "------|------------|"

    lines.append(header)
    lines.append(sep)

    for r in results:
        if "error" in r:
            lines.append(f"| {r['target']} | {r['ranker']} | {r['scope']} | {r['task']} | ERROR | | |")
            continue

        row = (
            f"| {r['target']} "
            f"| {r['ranker']} "
            f"| {r['scope']} "
            f"| {r['task']} "
            f"| {r['pruned_params']:,} "
            f"| {r['compression_ratio']:.2f}x "
            f"| {r['flops_reduction']:.1%} "
        )

        if has_depth:
            dm = r.get("depth_metrics")
            bd = r.get("baseline_depth")
            if dm:
                rmse_delta = ""
                if bd:
                    rmse_delta = f"{dm['rmse'] - bd['rmse']:+.4f}"
                row += f"| {dm['rmse']:.4f} | {dm['delta1']:.4f} | {rmse_delta} "
            else:
                row += "| - | - | - "

        if has_seg:
            sm = r.get("seg_metrics")
            bs = r.get("baseline_seg")
            if sm:
                miou_delta = ""
                if bs:
                    miou_delta = f"{sm['mean_iou'] - bs['mean_iou']:+.4f}"
                row += f"| {sm['mean_iou']:.4f} | {miou_delta} "
            else:
                row += "| - | - "

        row += "|"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)

    # Load experiment cache
    cache = load_experiment_cache(args.output_dir)

    # Step 1: Compute or load baseline
    baseline = compute_baseline(args)
    logger.info(f"Baseline: params={baseline['original_params']:,} flops={baseline['original_flops']:,}")

    # Step 2: Determine sub-experiments based on scope
    all_results = []

    if args.scope == "task-specific":
        tasks = []
        # Always run depth for task-specific (NYU is required)
        tasks.append("depth")
        # Run seg if ADE is available
        if args.use_ade:
            tasks.append("seg")
    else:
        # shared-backbone: single experiment evaluating both
        tasks = ["both"]

    # Step 3: Run experiments
    for task in tasks:
        cache_key = get_cache_key(args, task)

        if cache_key in cache:
            logger.info(f"Cache hit for {cache_key}, loading cached result")
            all_results.append(cache[cache_key])
            continue

        result = run_single_experiment(args, baseline, task)
        all_results.append(result)

        # Save to cache incrementally
        cache[cache_key] = result
        save_experiment_cache(args.output_dir, cache)

    # Step 4: Save detailed results (merge with existing results)
    results_file = output_dir / f"results_{args.model}.json"
    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)
    else:
        existing_results = []

    # Build set of keys for new results to allow dedup/update
    def _result_key(r):
        return (r.get("sparsity"), r.get("target"), r.get("ranker"),
                r.get("scope"), r.get("task"))

    new_keys = {_result_key(r) for r in all_results}
    # Keep old results that don't overlap with new ones
    merged = [r for r in existing_results if _result_key(r) not in new_keys]
    merged.extend(all_results)

    with open(results_file, "w") as f:
        json.dump(merged, f, indent=2)
    logger.info(f"Results saved to {results_file} ({len(merged)} total entries)")

    # Step 5: Print report
    report = format_report(all_results, baseline)
    print("\n" + report + "\n")

    # Save report
    report_file = output_dir / f"report_{args.model}.md"
    with open(report_file, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()
