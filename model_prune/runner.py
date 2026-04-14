"""Unified pruning orchestration.

Pipeline:
  YAML → (FullConfig, TaskConfig)
    → load_model()
    → build_loaders()
    → baseline_cache.get_or_compute()
    → PruneRunner.run() (generic)  OR  opt_pipeline.run_opt_prune() (hf_opt)
    → evaluate() pruned model
    → report.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from config.loader import load_yaml_config, TaskConfig
from config.schemas import FullConfig, PruneTarget
from dataset import build_loaders
from evaluation import evaluate as run_evaluation
from model import load_model
from pruning.runner import PruneRunner, RunResult

from . import baseline_cache
from .opt_pipeline import OPTRunResult, run_opt_prune

logger = logging.getLogger(__name__)


@dataclass
class PruneReport:
    task: str
    model_name: str
    baseline: Dict[str, Any]
    pruned_no_comp: Optional[Dict[str, Any]] = None
    pruned: Dict[str, Any] = field(default_factory=dict)
    original_params: int = 0
    pruned_params: int = 0
    compression_ratio: float = 1.0
    output_dir: Optional[Path] = None
    step_results: List[Any] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _val_descriptor(task_cfg: TaskConfig) -> str:
    """Identifier for the eval dataset, used in the baseline cache key."""
    ds = task_cfg.dataset
    for key in ("val_path", "nyu_path", "ade_path"):
        if ds.get(key):
            return str(ds[key])
    return task_cfg.task


def _compute_baseline(
    task: str,
    model,
    loaders: dict,
    device: str,
    **kw,
) -> Dict[str, Any]:
    return run_evaluation(task, model, loaders, device, **kw)


def _dinov2_factories(task_cfg: TaskConfig, meta: dict):
    """Return (depth_factory, seg_factory, heads_dir) for DINOv2 runs.

    Factories take a backbone and return the full head-attached model.
    If heads aren't available on disk the factories stay None and the
    evaluator silently skips that downstream metric.
    """
    from model import dinov2 as dinov2_mod

    heads_dir = task_cfg.model.get("heads_dir")
    model_size = meta.get("model_size")
    if model_size is None:
        return None, None, None

    def depth_factory(backbone):
        return dinov2_mod.create_depth_model(backbone, model_size, heads_dir=heads_dir)

    def seg_factory(backbone):
        return dinov2_mod.create_seg_model(backbone, model_size, heads_dir=heads_dir)

    return depth_factory, seg_factory, heads_dir


def _run_generic(
    full_cfg: FullConfig,
    task_cfg: TaskConfig,
    model,
    calib_loader,
    model_name: str,
    skip_compensation: bool = False,
) -> RunResult:
    runner = PruneRunner(full_cfg)
    runner.stats_cache_dir = task_cfg.cache.get("stats_dir")
    runner.force_recollect = bool(task_cfg.cache.get("force_recollect", False))
    return runner.run(
        model, calib_loader,
        skip_compensation=skip_compensation, model_name=model_name,
    )


def _run_opt(
    full_cfg: FullConfig,
    model,
    calib_loader,
    meta: dict,
    skip_compensation: bool = False,
    mlp_sparsity: Optional[float] = None,
    attn_sparsity: Optional[float] = None,
) -> OPTRunResult:
    num_layers = meta.get("layers")
    if num_layers is None:
        num_layers = len(model.model.decoder.layers)
    return run_opt_prune(
        model, calib_loader, full_cfg,
        num_layers=num_layers,
        skip_compensation=skip_compensation,
        mlp_sparsity=mlp_sparsity,
        attn_sparsity=attn_sparsity,
    )


def _apply_sparsity_split(
    full_cfg: FullConfig,
    task_cfg: TaskConfig,
    model,
    calib_loader,
    model_name: str,
    meta: dict,
    skip_compensation: bool = False,
):
    """Handle the mlp_sparsity / attn_sparsity YAML form for target=both.

    Generic runner: run it twice (once for MLP, once for attention), passing
    the working model through. OPT runner: accepts both in one call.
    """
    split = task_cfg.sparsity_split or {}
    mlp_s = split.get("mlp")
    attn_s = split.get("attn")

    if meta.get("source") == "hf_opt":
        return _run_opt(
            full_cfg, model, calib_loader, meta,
            skip_compensation=skip_compensation,
            mlp_sparsity=mlp_s, attn_sparsity=attn_s,
        )

    # Generic: two passes so the core algorithm doesn't need to know about splits.
    from dataclasses import replace

    current = model
    first_result: Optional[RunResult] = None
    if mlp_s is not None:
        cfg_mlp = replace(
            full_cfg,
            pruning=replace(full_cfg.pruning, target=PruneTarget.MLP, sparsity=mlp_s),
            collector=replace(full_cfg.collector, target=PruneTarget.MLP),
        )
        first_result = _run_generic(cfg_mlp, task_cfg, current, calib_loader, model_name, skip_compensation)
        current = first_result.pruned_model
    if attn_s is not None:
        cfg_attn = replace(
            full_cfg,
            pruning=replace(full_cfg.pruning, target=PruneTarget.ATTN, sparsity=attn_s),
            collector=replace(full_cfg.collector, target=PruneTarget.ATTN),
        )
        second = _run_generic(cfg_attn, task_cfg, current, calib_loader, model_name, skip_compensation)
        if first_result is not None:
            second.step_results = first_result.step_results + second.step_results
        return second
    return first_result


def run_prune(
    config_path: str | Path,
    overrides: Optional[List[str]] = None,
    eval_no_comp: bool = False,
) -> PruneReport:
    """Run one pruning experiment from a YAML config.

    Args:
        config_path: path to YAML.
        overrides: CLI overrides (``key.path=value``).
        eval_no_comp: if True, also run the pipeline once with compensation
            disabled and evaluate that model (slow; use only for ablations).
    """
    full_cfg, task_cfg = load_yaml_config(config_path, overrides=overrides)
    _set_seed(full_cfg.runner.seed)

    device = full_cfg.runner.device

    # Load model + tokenizer + meta.
    model_cfg = task_cfg.model
    source = model_cfg.get("source")
    name = model_cfg.get("name")
    if not source or not name:
        raise ValueError("YAML must specify model.source and model.name")

    logger.info("=" * 60)
    logger.info(f"Loading model: source={source} name={name}")
    model, meta = load_model(
        source, name,
        checkpoint=model_cfg.get("checkpoint"),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dtype=full_cfg.runner.dtype,
        device=device if source == "hf_opt" else None,
        heads_dir=model_cfg.get("heads_dir"),
    )
    if source != "hf_opt":
        model = model.to(device)
    model.eval()

    tokenizer = meta.get("tokenizer")

    # Data loaders.
    logger.info(f"Building dataset loaders for task={task_cfg.task}")
    loaders = build_loaders(task_cfg.task, task_cfg.dataset, tokenizer=tokenizer, meta=meta)

    # DINOv2 task needs factory callbacks for depth/seg head assembly during eval.
    eval_kwargs: Dict[str, Any] = {}
    if task_cfg.task == "dinov2_vision":
        depth_factory, seg_factory, _ = _dinov2_factories(task_cfg, meta)
        eval_kwargs["depth_model_factory"] = depth_factory
        eval_kwargs["seg_model_factory"] = seg_factory

    # Baseline.
    val_desc = _val_descriptor(task_cfg)
    baseline = baseline_cache.get_or_compute(
        baseline_dir=task_cfg.cache["baseline_dir"],
        model_name=name,
        val_descriptor=val_desc,
        task=task_cfg.task,
        compute_fn=lambda: _compute_baseline(
            task_cfg.task, model, loaders, device,
            **{k: v for k, v in eval_kwargs.items() if k != "original_model"},
        ),
    )
    logger.info(f"Baseline: {baseline}")

    # Run pruning.
    logger.info("=" * 60)
    logger.info("Pruning...")
    if task_cfg.sparsity_split:
        result = _apply_sparsity_split(
            full_cfg, task_cfg, model, loaders["calib"], name, meta,
            skip_compensation=False,
        )
    elif meta.get("source") == "hf_opt":
        result = _run_opt(full_cfg, model, loaders["calib"], meta, skip_compensation=False)
    else:
        result = _run_generic(full_cfg, task_cfg, model, loaders["calib"], name, skip_compensation=False)

    if not getattr(result, "success", True):
        logger.error(f"Pruning failed: {getattr(result, 'error_message', 'unknown')}")
        return PruneReport(
            task=task_cfg.task, model_name=name,
            baseline=baseline,
            original_params=getattr(result, "original_params", 0),
            pruned_params=getattr(result, "pruned_params", 0),
            compression_ratio=getattr(result, "compression_ratio", 1.0),
            success=False,
            error_message=getattr(result, "error_message", None),
            output_dir=getattr(result, "output_dir", None),
        )

    pruned_model = result.pruned_model

    # Evaluate.
    logger.info("Evaluating pruned model...")
    if task_cfg.task == "dinov2_vision":
        # For feature similarity evaluation we need the unpruned backbone alive.
        eval_kwargs["original_model"] = model
    pruned_metrics = run_evaluation(
        task_cfg.task, pruned_model, loaders, device, **eval_kwargs,
    )
    logger.info(f"Pruned metrics: {pruned_metrics}")

    # Optional: run again without compensation for an ablation comparison.
    pruned_no_comp_metrics = None
    if eval_no_comp and meta.get("source") != "hf_opt":
        logger.info("Re-running pruning with compensation disabled (ablation)...")
        if task_cfg.sparsity_split:
            r2 = _apply_sparsity_split(
                full_cfg, task_cfg, copy.deepcopy(model).to(device), loaders["calib"],
                name, meta, skip_compensation=True,
            )
        else:
            r2 = _run_generic(
                full_cfg, task_cfg,
                copy.deepcopy(model).to(device), loaders["calib"], name,
                skip_compensation=True,
            )
        pruned_no_comp_metrics = run_evaluation(
            task_cfg.task, r2.pruned_model, loaders, device, **eval_kwargs,
        )

    report = PruneReport(
        task=task_cfg.task,
        model_name=name,
        baseline=baseline,
        pruned=pruned_metrics,
        pruned_no_comp=pruned_no_comp_metrics,
        original_params=result.original_params,
        pruned_params=result.pruned_params,
        compression_ratio=result.compression_ratio,
        output_dir=getattr(result, "output_dir", None),
        step_results=list(getattr(result, "step_results", []) or []),
        success=True,
    )
    _print_summary(report)
    return report


def _print_summary(r: PruneReport) -> None:
    logger.info("=" * 60)
    logger.info("Final Summary")
    logger.info("=" * 60)
    logger.info(f"Task: {r.task}  Model: {r.model_name}")
    logger.info(f"Original params: {r.original_params:,}")
    logger.info(f"Pruned params:   {r.pruned_params:,}")
    logger.info(f"Compression:     {r.compression_ratio:.2f}x")
    logger.info(f"Baseline:        {r.baseline}")
    if r.pruned_no_comp is not None:
        logger.info(f"Pruned (no comp): {r.pruned_no_comp}")
    logger.info(f"Pruned:          {r.pruned}")
    if r.output_dir is not None:
        logger.info(f"Output dir:      {r.output_dir}")
