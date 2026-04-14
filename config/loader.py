"""YAML → (FullConfig, TaskConfig) loader with CLI overrides.

Single YAML schema — see configs/*.yaml for examples. Paths inside the YAML
are resolved relative to the YAML file's directory unless absolute.

CLI overrides are dotted keys: `--set pruning.sparsity=0.4 model.name=vit_base_patch16_224`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .schemas import (
    CollectorConfig,
    CovarianceMode,
    FullConfig,
    PruneTarget,
    PruningConfig,
    RankerType,
    RunnerConfig,
    ScheduleType,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TaskConfig:
    """Everything the YAML describes that isn't the pruning algorithm itself."""
    task: str
    model: dict
    dataset: dict
    cache: dict
    evaluation: dict = field(default_factory=dict)
    experiment: Optional[dict] = None
    # Optional per-target sparsity split (set by loader when YAML uses mlp_sparsity/attn_sparsity).
    # When non-empty, the orchestrator is expected to run MLP and attention passes with
    # different sparsities. None means "use full_cfg.pruning.sparsity for everything".
    sparsity_split: Optional[dict] = None
    raw: dict = field(default_factory=dict)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `key.dotted.path=value` overrides. `value` parsed as YAML scalar."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, raw_val = item.split("=", 1)
        try:
            val = yaml.safe_load(raw_val)
        except yaml.YAMLError:
            val = raw_val
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = val
    return out


def _resolve_path(p: Optional[str], base_dir: Path) -> Optional[Path]:
    """Resolve ``p``. Absolute paths are returned as-is. Relative paths
    resolve against the repository root (so ``data/imagenet`` in any YAML
    means ``<repo>/data/imagenet``, regardless of where the YAML lives).
    """
    if p is None:
        return None
    q = Path(p)
    if q.is_absolute():
        return q
    return (_REPO_ROOT / q).resolve()


def _enum(value: Any, enum_cls):
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def load_yaml_config(
    path: str | Path,
    overrides: Optional[list[str]] = None,
) -> tuple[FullConfig, TaskConfig]:
    """Load a YAML config file and build (FullConfig, TaskConfig).

    Args:
        path: path to YAML file.
        overrides: list of dotted-path `key=value` strings applied after loading.

    Returns:
        (full_cfg, task_cfg).
    """
    cfg_path = Path(path).resolve()
    base_dir = cfg_path.parent

    with open(cfg_path) as f:
        raw = yaml.safe_load(f) or {}

    raw = _apply_overrides(raw, overrides or [])

    # --- collector ---
    c = raw.get("collector", {}) or {}
    collector = CollectorConfig(
        target=_enum(raw.get("pruning", {}).get("target", "mlp"), PruneTarget),
        subsample_tokens=c.get("subsample_tokens"),
        covariance_mode=_enum(c.get("covariance_mode", "exact"), CovarianceMode),
        store_raw=bool(c.get("store_raw", False)),
        keep_cls_token=bool(c.get("keep_cls_token", True)),
        sketch_dim=int(c.get("sketch_dim", 256)),
        batch_size=int(raw.get("dataset", {}).get("batch_size", 32)),
    )

    # --- pruning ---
    p = raw.get("pruning", {}) or {}
    sparsity_split = None
    if "sparsity" in p and p["sparsity"] is not None:
        sparsity = float(p["sparsity"])
    elif "mlp_sparsity" in p or "attn_sparsity" in p:
        mlp_s = float(p["mlp_sparsity"]) if p.get("mlp_sparsity") is not None else None
        attn_s = float(p["attn_sparsity"]) if p.get("attn_sparsity") is not None else None
        if mlp_s is None and attn_s is None:
            raise ValueError("pruning.mlp_sparsity and pruning.attn_sparsity are both null")
        # Seed `sparsity` with either value; orchestrator will switch between passes.
        sparsity = mlp_s if mlp_s is not None else attn_s
        sparsity_split = {"mlp": mlp_s, "attn": attn_s}
    else:
        raise ValueError(
            "pruning config must specify `sparsity` or `mlp_sparsity`/`attn_sparsity`"
        )

    pruning = PruningConfig(
        target=_enum(p.get("target", "mlp"), PruneTarget),
        schedule=_enum(p.get("schedule", "layerwise"), ScheduleType),
        sparsity=sparsity,
        ranker=_enum(p.get("ranker", "active_energy"), RankerType),
        lambda_reg=float(p.get("lambda_reg", 1e-3)),
        auto_shrinkage=bool(p.get("auto_shrinkage", True)),
        min_channels=int(p.get("min_channels", 64)),
        min_heads=int(p.get("min_heads", 1)),
        min_qk_dim=int(p.get("min_qk_dim", 8)),
        qk_sparsity=float(p.get("qk_sparsity", 0.3)),
        keep_topk_outliers=int(p.get("keep_topk_outliers", 0)),
    )

    # --- runner ---
    r = raw.get("runner", {}) or {}
    runner = RunnerConfig(
        device=str(r.get("device", "cuda")),
        output_dir=_resolve_path(r.get("output_dir", "logs"), base_dir),
        save_pruned_path=_resolve_path(r.get("save_pruned_path"), base_dir),
        results_log=_resolve_path(r.get("results_log", "cache/experiments/results_log.json"), base_dir),
        calib_samples=int(r.get("calib_samples", raw.get("dataset", {}).get("calib_samples", 1024))),
        dtype=str(r.get("dtype", "float32")),
        seed=int(r.get("seed", 42)),
    )

    full_cfg = FullConfig(collector=collector, pruning=pruning, runner=runner)

    # --- task config ---
    ds_raw = raw.get("dataset", {}) or {}
    ds = dict(ds_raw)
    for key in ("calib_path", "val_path", "nyu_path", "ade_path", "tokenizer_path"):
        if key in ds and ds[key] is not None:
            ds[key] = _resolve_path(ds[key], base_dir)

    cache_raw = raw.get("cache", {}) or {}
    cache = {
        "baseline_dir": _resolve_path(cache_raw.get("baseline_dir", "cache/baselines"), base_dir),
        "stats_dir": _resolve_path(cache_raw.get("stats_dir", "cache/stats"), base_dir),
        "experiments_dir": _resolve_path(
            cache_raw.get("experiments_dir", "cache/experiments"), base_dir
        ),
        "force_recollect": bool(cache_raw.get("force_recollect", False)),
    }

    task_cfg = TaskConfig(
        task=str(raw.get("task", "imagenet_classification")),
        model=dict(raw.get("model", {})),
        dataset=ds,
        cache=cache,
        evaluation=dict(raw.get("evaluation", {}) or {}),
        experiment=raw.get("experiment"),
        sparsity_split=sparsity_split,
        raw=raw,
    )

    return full_cfg, task_cfg


def repo_root() -> Path:
    return _REPO_ROOT
