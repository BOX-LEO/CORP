"""Dataset loading registry keyed by task name.

`build_loaders(task, cfg, tokenizer=None, meta=None)` returns a dict like:

    {"calib": <DataLoader>, "val": <DataLoader | dict[str, DataLoader]>}

Each task loader pulls whatever it needs from ``cfg`` (the YAML ``dataset``
block, already path-resolved) and optionally from ``meta`` (model metadata,
e.g. image size, normalization).
"""

from typing import Callable, Dict, Any, Optional

from . import imagenet, C4, nyu_dataset, ade20k_dataset
from . import WikiText_2 as WikiText2


BuilderFn = Callable[..., Dict[str, Any]]


def _build_imagenet_classification(cfg, tokenizer=None, meta=None):
    return imagenet.build_loaders(cfg, meta=meta)


def _build_language_modeling(cfg, tokenizer=None, meta=None):
    if tokenizer is None:
        raise ValueError("language_modeling task requires tokenizer in model meta")
    calib = C4.build_calibration(tokenizer=tokenizer, cfg=cfg)
    val = WikiText2.build_eval(tokenizer=tokenizer, cfg=cfg)
    return {"calib": calib, "val": val}


def _build_dinov2_vision(cfg, tokenizer=None, meta=None):
    from .dinov2_utils import build_dinov2_loaders
    return build_dinov2_loaders(cfg, meta=meta)


REGISTRY: Dict[str, BuilderFn] = {
    "imagenet_classification": _build_imagenet_classification,
    "language_modeling": _build_language_modeling,
    "dinov2_vision": _build_dinov2_vision,
}


def build_loaders(
    task: str,
    cfg: dict,
    tokenizer=None,
    meta: Optional[dict] = None,
) -> Dict[str, Any]:
    if task not in REGISTRY:
        raise ValueError(
            f"Unknown dataset task '{task}'. Registered: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[task](cfg, tokenizer=tokenizer, meta=meta)


__all__ = ["build_loaders", "REGISTRY"]
