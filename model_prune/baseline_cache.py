"""JSON file-backed baseline metric cache shared across all tasks.

Replaces the four copies previously in ``*_prune/baseline_accuracy_cache.json``.
Cache key is ``{model_name}|{val_descriptor}|{task}`` so the same cache file
serves multiple model / dataset combinations.
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


def get(
    baseline_dir: Path,
    model_name: str,
    val_descriptor: str,
    task: str,
) -> Optional[Dict[str, Any]]:
    return _load(baseline_dir).get(_make_key(model_name, val_descriptor, task))


def put(
    baseline_dir: Path,
    model_name: str,
    val_descriptor: str,
    task: str,
    metrics: Dict[str, Any],
) -> None:
    data = _load(baseline_dir)
    data[_make_key(model_name, val_descriptor, task)] = metrics
    _save(baseline_dir, data)
    logger.info(f"Saved baseline metrics for {model_name}")


def get_or_compute(
    baseline_dir: Path,
    model_name: str,
    val_descriptor: str,
    task: str,
    compute_fn: Callable[[], Dict[str, Any]],
    force: bool = False,
) -> Dict[str, Any]:
    """Return cached baseline metrics or compute + cache them now."""
    if not force:
        cached = get(baseline_dir, model_name, val_descriptor, task)
        if cached is not None:
            logger.info(f"Using cached baseline metrics for {model_name}: {cached}")
            return cached
    metrics = compute_fn()
    put(baseline_dir, model_name, val_descriptor, task, metrics)
    return metrics
