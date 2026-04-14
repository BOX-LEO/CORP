"""YAML-driven experiment sweeps.

Example ``experiment:`` block in a YAML config::

    experiment:
      sweep:
        pruning.sparsity: [0.1, 0.2, 0.3]
        pruning.ranker: [energy, active_energy]
      result_cache: cache/experiments/deit_sweep.json

The sweep is the cartesian product of listed values, applied as CLI-style
overrides on top of the base YAML. Each run's summary is cached by the
canonical override signature so re-running only executes missing cells.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .runner import run_prune, PruneReport

logger = logging.getLogger(__name__)


def _format_value(v) -> str:
    """Serialize a sweep value back into something the override parser can re-read.

    yaml.safe_dump adds doc-end markers on scalars; `json.dumps` gives us clean
    scalars and still round-trips through yaml.safe_load in the override parser.
    """
    import json as _json
    return _json.dumps(v)


def _expand_sweep(sweep: Dict[str, list]) -> List[List[str]]:
    """Return a list of override lists (one per cartesian product cell)."""
    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        overrides = [f"{k}={_format_value(v)}" for k, v in zip(keys, combo)]
        combos.append(overrides)
    return combos


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _run_signature(overrides: List[str]) -> str:
    return "|".join(sorted(overrides))


def _serialize_report(r: PruneReport) -> dict:
    d = asdict(r)
    d["output_dir"] = str(d["output_dir"]) if d.get("output_dir") is not None else None
    # step_results may contain non-serializable dataclasses — coerce to str.
    d["step_results"] = [str(s) for s in d.get("step_results", [])]
    return d


def run_experiments(
    config_path: str | Path,
    extra_overrides: Optional[List[str]] = None,
    force: bool = False,
) -> Dict[str, dict]:
    """Run a YAML-defined sweep. Returns ``{signature: serialized_report}``."""
    config_path = Path(config_path)
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    exp = raw.get("experiment") or {}
    sweep = exp.get("sweep") or {}
    result_cache_path = exp.get("result_cache") or "cache/experiments/results.json"
    # Resolve relative to YAML file.
    result_cache_path = Path(result_cache_path)
    if not result_cache_path.is_absolute():
        result_cache_path = (config_path.parent / result_cache_path).resolve()

    cache = _load_cache(result_cache_path)

    combos = _expand_sweep(sweep) if sweep else [[]]
    logger.info(f"Running {len(combos)} experiment(s) from {config_path}")

    for overrides in combos:
        full_overrides = (extra_overrides or []) + overrides
        sig = _run_signature(full_overrides)
        if not force and sig in cache:
            logger.info(f"[cached] {sig}")
            continue
        logger.info(f"[run]    {sig}")
        try:
            report = run_prune(config_path, overrides=full_overrides)
            cache[sig] = _serialize_report(report)
        except Exception as e:
            logger.exception(f"Run failed for {sig}: {e}")
            cache[sig] = {"error": str(e), "overrides": full_overrides}
        _save_cache(result_cache_path, cache)

    return cache
