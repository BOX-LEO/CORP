"""JSON file-backed baseline cache shared across all tasks.

Cached value schema::

    {
        "metrics": { ... task-specific evaluation metrics ... },
        "num_params": <int>,
        "flops": <int or null>,
    }

Old entries (flat metrics dict only) are transparently upgraded on read:
missing ``num_params`` / ``flops`` are filled in the next time a run passes
the corresponding counting callbacks.

Cache key: ``{model_name}|{val_descriptor}|{task}``.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def _cache_file(baseline_dir: Path) -> Path:
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    return baseline_dir / "baseline.json"


def _load(baseline_dir: Path) -> dict:
    path = _cache_file(baseline_dir)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save(baseline_dir: Path, data: dict) -> None:
    path = _cache_file(baseline_dir)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _make_key(model_name: str, val_descriptor: str, task: str) -> str:
    return f"{model_name}|{val_descriptor}|{task}"


def _normalize(entry: Any) -> Dict[str, Any]:
    """Accept either the new schema or the legacy flat-metrics format."""
    if isinstance(entry, dict) and "metrics" in entry:
        return {
            "metrics": entry.get("metrics") or {},
            "num_params": entry.get("num_params"),
            "flops": entry.get("flops"),
        }
    # Legacy: the whole dict *is* the metrics payload.
    return {"metrics": entry or {}, "num_params": None, "flops": None}


def get(
    baseline_dir: Path,
    model_name: str,
    val_descriptor: str,
    task: str,
) -> Optional[Dict[str, Any]]:
    raw = _load(baseline_dir).get(_make_key(model_name, val_descriptor, task))
    if raw is None:
        return None
    return _normalize(raw)


def put(
    baseline_dir: Path,
    model_name: str,
    val_descriptor: str,
    task: str,
    metrics: Dict[str, Any],
    num_params: Optional[int] = None,
    flops: Optional[int] = None,
) -> None:
    data = _load(baseline_dir)
    data[_make_key(model_name, val_descriptor, task)] = {
        "metrics": metrics,
        "num_params": num_params,
        "flops": flops,
    }
    _save(baseline_dir, data)
    logger.info(f"Saved baseline entry for {model_name}")


def get_or_compute(
    baseline_dir: Path,
    model_name: str,
    val_descriptor: str,
    task: str,
    compute_fn: Callable[[], Dict[str, Any]],
    count_params_fn: Optional[Callable[[], int]] = None,
    count_flops_fn: Optional[Callable[[], Optional[int]]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Return the full cached entry or compute + cache it now.

    Shape: ``{"metrics": ..., "num_params": ..., "flops": ...}``.

    If the cached entry is missing ``num_params`` or ``flops`` and the
    corresponding callback is supplied, the missing fields are computed
    and the entry is upgraded in place.
    """
    cached = None if force else get(baseline_dir, model_name, val_descriptor, task)

    if cached is None:
        metrics = compute_fn()
        num_params = count_params_fn() if count_params_fn is not None else None
        flops = count_flops_fn() if count_flops_fn is not None else None
        put(baseline_dir, model_name, val_descriptor, task, metrics, num_params, flops)
        return {"metrics": metrics, "num_params": num_params, "flops": flops}

    logger.info(f"Using cached baseline for {model_name}: {cached['metrics']}")

    # Upgrade legacy / partial entries in place.
    upgraded = False
    if cached["num_params"] is None and count_params_fn is not None:
        cached["num_params"] = count_params_fn()
        upgraded = True
    if cached["flops"] is None and count_flops_fn is not None:
        cached["flops"] = count_flops_fn()
        upgraded = True
    if upgraded:
        put(
            baseline_dir, model_name, val_descriptor, task,
            cached["metrics"], cached["num_params"], cached["flops"],
        )

    return cached
