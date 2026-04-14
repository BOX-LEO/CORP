"""
Activation Stats Cache.

Caches collected activation statistics so that experiments differing only in
ranker, target, or sparsity can skip the expensive forward-pass collection.

Only raw stats are cached — lightweight reports are derived on the fly.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .collect import LayerActivationStats
from .stats import QKDimReport

logger = logging.getLogger(__name__)

CACHE_VERSION = 2


def compute_cache_key(
    model_name: str,
    calib_samples: int,
    subsample_tokens: Optional[int],
) -> str:
    """Compute a deterministic cache key for the given configuration.

    Cache depends only on factors that affect which activations are collected:
    model, data size, and token subsampling. Independent of ranker/target so
    the same stats can be reused across experiments. Activations are always
    collected for both MLP and attention layers.
    """
    parts = [
        str(CACHE_VERSION),
        model_name,
        str(calib_samples),
        str(subsample_tokens),
    ]
    raw = "|".join(parts)
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    safe_name = model_name.replace("/", "_")
    return f"{safe_name}_{h}"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_layer_stats(stats: LayerActivationStats) -> Dict:
    """Convert LayerActivationStats to a serializable dict of tensors."""
    d = {
        "layer_name": stats.layer_name,
        "feature_dim": stats.feature_dim,
        "n_samples": stats.n_samples,
    }
    for attr in ("sum_x", "sum_x2", "covariance", "active_count", "sum_q2k2", "_cov_accum"):
        val = getattr(stats, attr, None)
        if val is not None:
            d[attr] = val.cpu()
    return d


def _deserialize_layer_stats(d: Dict) -> LayerActivationStats:
    """Reconstruct LayerActivationStats from a serialized dict."""
    stats = LayerActivationStats(
        layer_name=d["layer_name"],
        feature_dim=d["feature_dim"],
        n_samples=d["n_samples"],
    )
    for attr in ("sum_x", "sum_x2", "covariance", "active_count", "sum_q2k2", "_cov_accum"):
        if attr in d:
            setattr(stats, attr, d[attr])
    return stats


def _serialize_qk_report(report: QKDimReport) -> Dict:
    """Convert QKDimReport to a serializable dict."""
    d = {
        "layer_name": report.layer_name,
        "head_idx": report.head_idx,
        "head_dim": report.head_dim,
        "n_samples": report.n_samples,
    }
    for attr in ("q_energy", "k_energy", "joint_score", "qk_energy"):
        val = getattr(report, attr, None)
        if val is not None:
            d[attr] = val.cpu()
    if report.q_stats is not None:
        d["q_stats"] = _serialize_layer_stats(report.q_stats)
    if report.k_stats is not None:
        d["k_stats"] = _serialize_layer_stats(report.k_stats)
    if report.qk_stats is not None:
        d["qk_stats"] = _serialize_layer_stats(report.qk_stats)
    return d


def _deserialize_qk_report(d: Dict) -> QKDimReport:
    """Reconstruct QKDimReport from a serialized dict."""
    q_stats = _deserialize_layer_stats(d["q_stats"]) if "q_stats" in d else None
    k_stats = _deserialize_layer_stats(d["k_stats"]) if "k_stats" in d else None
    qk_stats = _deserialize_layer_stats(d["qk_stats"]) if "qk_stats" in d else None
    return QKDimReport(
        layer_name=d["layer_name"],
        head_idx=d["head_idx"],
        head_dim=d["head_dim"],
        n_samples=d["n_samples"],
        q_energy=d.get("q_energy"),
        k_energy=d.get("k_energy"),
        joint_score=d.get("joint_score"),
        q_stats=q_stats,
        k_stats=k_stats,
        qk_energy=d.get("qk_energy"),
        qk_stats=qk_stats,
    )


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_stats_cache(
    cache_dir: Path,
    cache_key: str,
    all_stats: Dict[str, LayerActivationStats],
    qk_reports: Dict[str, QKDimReport],
    metadata: Dict,
) -> Path:
    """Save activation stats and QK reports to a cache file.

    Returns the path to the saved file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key}.pt"

    payload = {
        "cache_version": CACHE_VERSION,
        "metadata": metadata,
        "all_stats": {k: _serialize_layer_stats(v) for k, v in all_stats.items()},
        "qk_reports": {k: _serialize_qk_report(v) for k, v in qk_reports.items()},
    }

    torch.save(payload, path)
    logger.info(f"Saved activation stats cache to {path}")
    return path


def load_stats_cache(
    cache_dir: Path,
    cache_key: str,
) -> Optional[Tuple[Dict[str, LayerActivationStats], Dict[str, QKDimReport]]]:
    """Load cached activation stats and QK reports.

    Returns None if the cache file doesn't exist or has a version mismatch.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{cache_key}.pt"

    if not path.exists():
        logger.info(f"No stats cache found at {path}")
        return None

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning(f"Failed to load stats cache {path}: {e}")
        return None

    if payload.get("cache_version") != CACHE_VERSION:
        logger.warning(
            f"Stats cache version mismatch: expected {CACHE_VERSION}, "
            f"got {payload.get('cache_version')}. Discarding cache."
        )
        return None

    all_stats = {k: _deserialize_layer_stats(v) for k, v in payload["all_stats"].items()}
    qk_reports = {k: _deserialize_qk_report(v) for k, v in payload.get("qk_reports", {}).items()}

    logger.info(f"Loaded activation stats cache from {path} "
                f"({len(all_stats)} layers, {len(qk_reports)} qk_reports)")
    return all_stats, qk_reports
