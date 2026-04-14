#!/usr/bin/env python
"""Batch experiment runner for OPT pruning evaluation.

Supports activation stats caching, baseline/experiment result caching,
FLOPs computation, and both layerwise and global pruning schedules.

Usage:
    # Basic experiment
    python opt_prune/run_experiments.py --model_size 125m --sparsity 0.3 \
        --target mlp --schedule global --device cuda

    # With stats caching and skip baseline
    python opt_prune/run_experiments.py --model_size 125m --sparsity 0.3 \
        --target both --schedule global --skip_baseline --device cuda

    # Force recollect stats
    python opt_prune/run_experiments.py --model_size 350m --sparsity 0.2 \
        --target mlp --force_recollect --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.schemas import (
    CollectorConfig,
    CovarianceMode,
    PruneTarget,
    RankerType,
)
from pruning.apply_masks import MaskApplier, get_model_parameter_count
from pruning.cache import compute_cache_key, load_stats_cache, save_stats_cache
from pruning.compensate import AffineCompensator
from pruning.ranking import StructureRanker
from pruning.stats import RedundancyAnalyzer

from opt_prune.data import create_c4_calibration_loader, create_wikitext2_eval_loader
from opt_prune.evaluation import evaluate_perplexity
from opt_prune.run_prune import (
    OPT_CONFIGS,
    collect_all_opt_stats,
    run_opt_pruning,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FLOPs computation for OPT models
# ---------------------------------------------------------------------------

def compute_opt_flops(model: nn.Module, seq_len: int = 2048, device: str = "cpu") -> int:
    """Compute FLOPs for an OPT model analytically.

    Inspects actual layer dimensions (works for both original and pruned models).
    Counts multiply-accumulate as 2 FLOPs.

    Per-layer FLOPs:
      Attention:
        Q/K/V projections: 3 * 2 * N * embed_dim * proj_out_dim
        Q @ K^T:           2 * num_heads * N * N * head_dim
        attn @ V:           2 * num_heads * N * N * head_dim
        out_proj:           2 * N * proj_in_dim * embed_dim
      MLP:
        fc1:               2 * N * embed_dim * mlp_hidden
        fc2:               2 * N * mlp_hidden * embed_dim
      LM head:
        lm_head:           2 * N * embed_dim * vocab_size

    Args:
        model: OPT CausalLM model
        seq_len: Sequence length
        device: Unused (kept for API compatibility)

    Returns:
        Total FLOPs (int)
    """
    N = seq_len
    total = 0

    decoder = model.model.decoder
    for layer in decoder.layers:
        attn = layer.self_attn

        # Attention projections: q_proj, k_proj, v_proj
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            proj = getattr(attn, proj_name)
            total += 2 * N * proj.in_features * proj.out_features

        # Attention matmuls: Q@K^T and attn@V
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        total += 2 * num_heads * N * N * head_dim  # Q @ K^T
        total += 2 * num_heads * N * N * head_dim  # attn @ V

        # out_proj
        total += 2 * N * attn.out_proj.in_features * attn.out_proj.out_features

        # MLP: fc1 and fc2
        total += 2 * N * layer.fc1.in_features * layer.fc1.out_features
        total += 2 * N * layer.fc2.in_features * layer.fc2.out_features

    # LM head
    if hasattr(model, "lm_head"):
        total += 2 * N * model.lm_head.in_features * model.lm_head.out_features

    return total


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def get_experiment_cache_path(output_dir: str) -> Path:
    return Path(output_dir) / "experiment_cache.json"


def load_experiment_cache(output_dir: str) -> Dict[str, Any]:
    path = get_experiment_cache_path(output_dir)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_experiment_cache(output_dir: str, cache: Dict[str, Any]):
    path = get_experiment_cache_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def get_experiment_cache_key(args) -> str:
    """Build a cache key that captures all settings affecting experiment results."""
    attn_ranker_str = args.attn_ranker or args.ranker
    parts = [
        "opt", args.model_size, str(args.sparsity), args.target,
        args.ranker, f"attn_{attn_ranker_str}", args.schedule,
        f"lam{args.lambda_reg}",
        f"calib{args.n_calib_segments}", f"seq{args.seq_len}",
        f"comp{int(not args.skip_compensation)}",
        f"minc{args.min_channels}", f"minh{args.min_heads}",
    ]
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------

def compute_baseline(args) -> Dict[str, Any]:
    """Compute baseline metrics for the unpruned OPT model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / f"baseline_opt_{args.model_size}_seq{args.eval_seq_len}.json"

    if args.skip_baseline and cache_file.exists():
        logger.info(f"Loading cached baseline from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    model_name = f"facebook/opt-{args.model_size}"
    logger.info(f"Computing baseline for {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(args.device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    original_params = get_model_parameter_count(model)

    logger.info("Computing baseline FLOPs...")
    original_flops = compute_opt_flops(model, seq_len=args.eval_seq_len, device=args.device)

    logger.info("Computing baseline perplexity...")
    eval_loader = create_wikitext2_eval_loader(
        tokenizer, seq_len=args.eval_seq_len, batch_size=args.batch_size,
    )
    original_ppl = evaluate_perplexity(model, eval_loader, args.device)

    baseline = {
        "model": args.model_size,
        "backbone": "opt",
        "original_params": original_params,
        "original_flops": original_flops,
        "original_ppl": original_ppl,
        "eval_seq_len": args.eval_seq_len,
    }

    with open(cache_file, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info(f"Baseline saved to {cache_file}")

    del model
    torch.cuda.empty_cache()

    return baseline


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    args,
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a single pruning experiment.

    Handles activation stats caching, pruning, perplexity evaluation,
    and FLOPs computation.

    Args:
        args: Parsed CLI args
        baseline: Baseline metrics dict

    Returns:
        Result dict with all metrics
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = f"facebook/opt-{args.model_size}"
    opt_cfg = OPT_CONFIGS[args.model_size]
    num_layers = opt_cfg["layers"]

    logger.info(f"Running experiment: model={args.model_size} sparsity={args.sparsity} "
                f"target={args.target} ranker={args.ranker} schedule={args.schedule}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(args.device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    original_params = get_model_parameter_count(model)

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
    attn_ranker = None
    if args.attn_ranker and args.attn_ranker != args.ranker:
        attn_ranker = StructureRanker.from_config(
            RankerType(args.attn_ranker),
            min_channels=args.min_channels,
        )
    analyzer = RedundancyAnalyzer()
    compensator = AffineCompensator(lambda_reg=args.lambda_reg)
    mask_applier = MaskApplier()

    # --- Activation stats caching (check before loading C4 data) ---
    all_stats = None
    all_reports = None
    cache_key = None
    stats_cache_dir = Path(args.stats_cache_dir)

    if args.stats_cache_dir:
        cache_key = compute_cache_key(
            model_name=f"opt_{args.model_size}_seq{args.seq_len}",
            calib_samples=args.n_calib_segments,
            attn_mode="head",
            subsample_tokens=None,
        )

        if not args.force_recollect:
            cached = load_stats_cache(stats_cache_dir, cache_key)
            if cached is not None:
                all_stats, _ = cached
                all_reports = analyzer.reports_from_all_stats(all_stats)
                logger.info("Using cached activation stats (skipping forward passes)")

    # Create calibration loader only when needed (stats not cached, or layerwise schedule)
    needs_calib = all_stats is None or args.schedule == 'layerwise'
    calib_loader = None
    if needs_calib:
        logger.info("Creating C4 calibration loader...")
        calib_loader = create_c4_calibration_loader(
            tokenizer,
            n_segments=args.n_calib_segments,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    # Collect stats if not cached (only for global schedule)
    if all_stats is None and args.schedule == 'global':
        # Always collect both MLP and attention stats so the cache is reusable
        # regardless of pruning target.
        logger.info("Collecting activation stats for all layers (both MLP and attention)...")
        all_stats = collect_all_opt_stats(
            model, calib_loader, collector_config, num_layers, args.device, 'both',
        )
        all_reports = analyzer.reports_from_all_stats(all_stats)

        # Save to cache
        if cache_key is not None:
            metadata = {
                "model_name": f"opt_{args.model_size}_seq{args.seq_len}",
                "calib_samples": args.n_calib_segments,
                "attn_mode": "head",
                "subsample_tokens": None,
            }
            save_stats_cache(stats_cache_dir, cache_key, all_stats, {}, metadata)

    # Run pruning
    logger.info(f"Running {args.schedule} pruning at {args.sparsity} sparsity...")
    step_results, final_stats, final_reports = run_opt_pruning(
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
        all_stats=all_stats,
        all_reports=all_reports,
        attn_ranker=attn_ranker,
    )

    # Save layerwise stats to cache if collected fresh
    if cache_key is not None and all_stats is None and args.schedule == 'layerwise':
        metadata = {
            "model_name": f"opt_{args.model_size}_seq{args.seq_len}",
            "calib_samples": args.n_calib_segments,
            "attn_mode": "head",
            "subsample_tokens": None,
        }
        save_stats_cache(stats_cache_dir, cache_key, final_stats, {}, metadata)

    # Evaluate pruned model
    pruned_params = get_model_parameter_count(model)

    logger.info("Computing pruned FLOPs...")
    pruned_flops = compute_opt_flops(model, seq_len=args.eval_seq_len, device=args.device)

    logger.info("Computing pruned perplexity...")
    eval_loader = create_wikitext2_eval_loader(
        tokenizer, seq_len=args.eval_seq_len, batch_size=args.batch_size,
    )
    pruned_ppl = evaluate_perplexity(model, eval_loader, args.device)

    original_flops = baseline.get("original_flops", 0)

    result = {
        "model": args.model_size,
        "sparsity": args.sparsity,
        "target": args.target,
        "ranker": args.ranker,
        "attn_ranker": args.attn_ranker or args.ranker,
        "schedule": args.schedule,
        "skip_compensation": args.skip_compensation,
        "lambda_reg": args.lambda_reg,
        "n_calib_segments": args.n_calib_segments,
        "seq_len": args.seq_len,
        "original_params": original_params,
        "pruned_params": pruned_params,
        "compression_ratio": original_params / pruned_params if pruned_params > 0 else float("inf"),
        "original_flops": original_flops,
        "pruned_flops": pruned_flops,
        "flops_reduction": 1.0 - (pruned_flops / original_flops) if original_flops > 0 else 0.0,
        "original_ppl": baseline.get("original_ppl"),
        "pruned_ppl": pruned_ppl,
        "ppl_change": pruned_ppl - baseline.get("original_ppl", 0),
    }

    del model
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(results: List[Dict[str, Any]], baseline: Dict[str, Any]) -> str:
    """Format results as a markdown summary table."""
    lines = []
    lines.append("# OPT Pruning Experiment Results")
    lines.append("")
    lines.append(f"Model: OPT-{baseline.get('model', 'unknown')}")
    lines.append(f"Original params: {baseline.get('original_params', 0):,}")
    lines.append(f"Original FLOPs: {baseline.get('original_flops', 0):,}")
    lines.append(f"Original PPL: {baseline.get('original_ppl', 0):.2f}")
    lines.append("")

    header = "| Sparsity | Target | Ranker | Schedule | Params | Compression | FLOPs Red. | PPL | PPL Change |"
    sep = "|----------|--------|--------|----------|--------|-------------|------------|-----|------------|"
    lines.append(header)
    lines.append(sep)

    for r in results:
        if "error" in r:
            lines.append(
                f"| {r.get('sparsity', '?')} | {r.get('target', '?')} "
                f"| {r.get('ranker', '?')} | {r.get('schedule', '?')} "
                f"| ERROR | | | | |"
            )
            continue

        row = (
            f"| {r['sparsity']} "
            f"| {r['target']} "
            f"| {r['ranker']} "
            f"| {r['schedule']} "
            f"| {r['pruned_params']:,} "
            f"| {r['compression_ratio']:.2f}x "
            f"| {r['flops_reduction']:.1%} "
            f"| {r['pruned_ppl']:.2f} "
            f"| {r['ppl_change']:+.2f} "
            f"|"
        )
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="OPT pruning experiment runner")

    # Model
    parser.add_argument("--model_size", type=str, default="1.3b",
                        choices=["125m", "350m", "1.3b", "2.7b", "6.7b"])

    # Pruning config
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--target", type=str, default="mlp",
                        choices=["mlp", "attn", "both"])
    parser.add_argument("--ranker", type=str, default="energy_weightmagnitude",
                        choices=["energy", "active", "active_energy", "weight_magnitude", "energy_weightmagnitude", "active_weightmagnitude", "ensemble"])
    parser.add_argument("--attn_ranker", type=str, default=None,
                        choices=["energy", "active", "active_energy", "weight_magnitude", "energy_weightmagnitude", "active_weightmagnitude", "ensemble"],
                        help="Separate ranker for attention heads (default: same as --ranker)")
    parser.add_argument("--schedule", type=str, default="global",
                        choices=["layerwise", "global"])
    parser.add_argument("--lambda_reg", type=float, default=1e-3)
    parser.add_argument("--min_channels", type=int, default=64)
    parser.add_argument("--min_heads", type=int, default=1)
    parser.add_argument("--skip_compensation", action="store_true")

    # Data config
    parser.add_argument("--n_calib_segments", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Calibration sequence length (default: 48)")
    parser.add_argument("--eval_seq_len", type=int, default=2048,
                        help="Evaluation sequence length for PPL (default: 2048)")
    parser.add_argument("--batch_size", type=int, default=4)

    # Caching
    parser.add_argument("--stats_cache_dir", type=str,
                        default="opt_prune/activation_stats_cache",
                        help="Directory for caching activation stats")
    parser.add_argument("--force_recollect", action="store_true",
                        help="Bypass activation stats cache (still saves for future use)")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline computation if cache exists")

    # Runtime
    parser.add_argument("--output_dir", type=str, default="opt_prune/opt_experiments")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_model", action="store_true",
                        help="Save pruned model checkpoint")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    # Load experiment cache
    cache = load_experiment_cache(args.output_dir)

    # Step 1: Compute or load baseline
    baseline = compute_baseline(args)
    logger.info(f"Baseline: params={baseline['original_params']:,} "
                f"flops={baseline['original_flops']:,} "
                f"ppl={baseline['original_ppl']:.2f}")

    # Step 2: Check experiment cache
    cache_key = get_experiment_cache_key(args)
    all_results = []

    if cache_key in cache:
        logger.info(f"Cache hit for experiment, loading cached result")
        all_results.append(cache[cache_key])
    else:
        result = run_single_experiment(args, baseline)
        all_results.append(result)

        # Save to cache incrementally
        cache[cache_key] = result
        save_experiment_cache(args.output_dir, cache)

    # Step 3: Save detailed results (merge with existing)
    results_file = output_dir / f"results_{args.model_size}.json"
    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)
    else:
        existing_results = []

    def _result_key(r):
        return (r.get("sparsity"), r.get("target"), r.get("ranker"),
                r.get("schedule"), r.get("skip_compensation"))

    new_keys = {_result_key(r) for r in all_results}
    merged = [r for r in existing_results if _result_key(r) not in new_keys]
    merged.extend(all_results)

    with open(results_file, "w") as f:
        json.dump(merged, f, indent=2)
    logger.info(f"Results saved to {results_file} ({len(merged)} total entries)")

    # Step 4: Print report
    report = format_report(all_results, baseline)
    print("\n" + report + "\n")

    report_file = output_dir / f"report_{args.model_size}.md"
    with open(report_file, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()
