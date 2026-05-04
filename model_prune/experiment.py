"""YAML-driven experiment sweeps.

Example ``experiment:`` block in a YAML config::

    experiment:
      sweep:
        pruning.sparsity: [0.1, 0.2, 0.3]
        pruning.ranker: [energy, active_energy]
      result_cache: cache/experiments/deit_sweep.json

The sweep is the cartesian product of listed values, applied as CLI-style
overrides on top of the base YAML. Each cell runs in a **fresh subprocess**
so that activation-stats caches, model weights, and allocator fragmentation
from one cell can't bleed into the next — critical for OPT-1.3B sweeps on
machines that can't fit two cells' worth of state at once. The parent
process only does scheduling and result aggregation.
"""

from __future__ import annotations

import itertools
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def _format_value(v) -> str:
    """Serialize a sweep value back into something the override parser can re-read.

    json.dumps gives clean scalars that round-trip through yaml.safe_load in
    the override parser.
    """
    return json.dumps(v)


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


def _run_cell_subprocess(
    config_path: Path,
    overrides: List[str],
) -> Dict[str, Any]:
    """Spawn `run_prune.py` as a subprocess for one sweep cell.

    Returns the serialized PruneReport (parsed from the JSON the child writes
    to a temp file). On non-zero exit or missing report, returns an error dict
    in the same shape used by the in-process error path.
    """
    project_root = Path(__file__).resolve().parent.parent
    runner_script = project_root / "run_prune.py"

    # Use NamedTemporaryFile only to reserve a unique path; the child writes
    # to it. delete=False because the child opens it by name, not by fd.
    with tempfile.NamedTemporaryFile(
        prefix="prune_report_", suffix=".json", delete=False,
    ) as tf:
        report_path = Path(tf.name)

    cmd: List[str] = [sys.executable, str(runner_script),
                      "--config", str(config_path),
                      "--report-out", str(report_path)]
    for ov in overrides:
        cmd.extend(["--set", ov])

    try:
        # Inherit stdout/stderr so the child's logging streams to the user's
        # terminal in real time. No timeout — pruning + eval can take a while.
        result = subprocess.run(cmd, cwd=str(project_root), check=False)
        if result.returncode != 0:
            return {
                "error": f"subprocess exited with code {result.returncode}",
                "overrides": overrides,
            }
        if not report_path.exists() or report_path.stat().st_size == 0:
            return {
                "error": "subprocess completed but wrote no report",
                "overrides": overrides,
            }
        with open(report_path) as f:
            return json.load(f)
    finally:
        try:
            report_path.unlink()
        except FileNotFoundError:
            pass


def run_experiments(
    config_path: str | Path,
    extra_overrides: Optional[List[str]] = None,
    force: bool = False,
) -> Dict[str, dict]:
    """Run a YAML-defined sweep. Returns ``{signature: serialized_report}``."""
    config_path = Path(config_path).resolve()
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
            cache[sig] = _run_cell_subprocess(config_path, full_overrides)
        except Exception as e:
            logger.exception(f"Failed to spawn cell for {sig}: {e}")
            cache[sig] = {"error": str(e), "overrides": full_overrides}
        _save_cache(result_cache_path, cache)

    return cache
