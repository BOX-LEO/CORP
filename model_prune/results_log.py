"""Append-only JSON log of per-run experiment results.

Every call to `model_prune.runner.run_prune` appends (or overwrites, keyed by
signature) one flat entry to a single JSON file. Sweeps driven by
`run_experiments.py` contribute one entry per cell through the same code path.

Schema (one entry)::

    {
      "task": "classification",
      "model_name": "deit_tiny_patch16_224",
      "config": { ... effective config ... },
      "baseline":       {"metrics": ..., "num_params": N, "flops": F},
      "pruned":         {"metrics": ..., "num_params": N, "flops": F},
      "pruned_no_comp": null | {"metrics": ...},
      "compression_ratio": 2.31,
      "flops_reduction":  1.67,
      "step_results":   [ "...", "..." ],
      "success": true,
      "error_message": null,
      "recorded_at": "2026-04-14T12:00:00Z"
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def signature_from_overrides(overrides: Optional[List[str]]) -> Optional[str]:
    """Canonical signature for sweep cells (mirrors experiment._run_signature)."""
    if not overrides:
        return None
    return "|".join(sorted(overrides))


def signature_from_report(report_fields: Dict[str, Any]) -> str:
    """Fallback signature for single-shot runs."""
    return (
        f"{report_fields.get('model_name')}|"
        f"{report_fields.get('task')}|"
        f"sparsity={report_fields.get('sparsity')}|"
        f"schedule={report_fields.get('schedule')}|"
        f"ranker={report_fields.get('ranker')}|"
        f"target={report_fields.get('target')}"
    )


def append(
    path: Path,
    signature: str,
    entry: Dict[str, Any],
) -> None:
    """Write or overwrite a single entry keyed by `signature`."""
    path = Path(path)
    data = _load(path)
    entry = dict(entry)
    entry.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
    data[signature] = entry
    _save(path, data)
    logger.info(f"Appended results_log entry {signature} -> {path}")


def build_entry(
    report,
    effective_config: Dict[str, Any],
    orig_num_params: int,
    orig_flops: Optional[int],
    pruned_num_params: int,
    pruned_flops: Optional[int],
) -> Dict[str, Any]:
    """Assemble the flat entry dict from a PruneReport + counts."""
    baseline_metrics = report.baseline.get("metrics", report.baseline) \
        if isinstance(report.baseline, dict) else report.baseline

    flops_reduction = None
    if orig_flops and pruned_flops:
        flops_reduction = orig_flops / pruned_flops

    return {
        "task": report.task,
        "model_name": report.model_name,
        "config": effective_config,
        "baseline": {
            "metrics": baseline_metrics,
            "num_params": orig_num_params,
            "flops": orig_flops,
        },
        "pruned": {
            "metrics": report.pruned,
            "num_params": pruned_num_params,
            "flops": pruned_flops,
        },
        "pruned_no_comp": (
            {"metrics": report.pruned_no_comp} if report.pruned_no_comp is not None else None
        ),
        "compression_ratio": report.compression_ratio,
        "flops_reduction": flops_reduction,
        "step_results": [str(s) for s in (report.step_results or [])],
        "success": report.success,
        "error_message": report.error_message,
    }
