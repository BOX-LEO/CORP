"""Evaluation registry keyed by task name.

`evaluate(task, model, loaders, device, **kw) -> dict[str, float]`.

Every evaluator returns a flat ``{metric_name: value}`` dict so the runner
can merge results from multiple tasks uniformly.
"""

from typing import Callable, Dict, Any

import torch.nn as nn

from . import classification, perplexity, depth_estimation, segmentation, feature_similarity


EvalFn = Callable[..., Dict[str, Any]]


def _eval_imagenet_classification(model, loaders, device, **kw):
    top1, top5 = classification.evaluate(
        model, loaders.get("val"), device=device, desc=kw.get("desc", "Evaluating"),
    )
    return {"top1": top1, "top5": top5}


def _eval_language_modeling(model, loaders, device, **kw):
    ppl = perplexity.evaluate(model, loaders["val"], device=device)
    return {"perplexity": ppl}


def _eval_dinov2_vision(model, loaders, device, **kw):
    """Runs depth + seg + feature similarity when their loaders / heads are available.

    Looks for:
      - ``loaders['val']['nyu']`` → depth via model + DPT head from kw['depth_model_factory'].
      - ``loaders['val']['ade']`` → seg via model + linear head from kw['seg_model_factory'].
      - ``kw['original_model']`` + any calib batch → feature similarity.
    """
    results: Dict[str, Any] = {}
    val = loaders.get("val") or {}
    depth_loader = val.get("nyu") if isinstance(val, dict) else None
    seg_loader = val.get("ade") if isinstance(val, dict) else None

    depth_factory = kw.get("depth_model_factory")
    if depth_loader is not None and depth_factory is not None:
        depth_model = depth_factory(model).to(device).eval()
        metrics = depth_estimation.evaluate(depth_model, depth_loader, device=device)
        for k, v in metrics.items():
            results[f"depth.{k}"] = v
        del depth_model

    seg_factory = kw.get("seg_model_factory")
    if seg_loader is not None and seg_factory is not None:
        seg_model = seg_factory(model).to(device).eval()
        metrics = segmentation.evaluate(seg_model, seg_loader, device=device)
        for k, v in metrics.items():
            results[f"seg.{k}"] = v
        del seg_model

    original_model = kw.get("original_model")
    calib_loader = loaders.get("calib")
    if original_model is not None and calib_loader is not None:
        sim = feature_similarity.evaluate(
            original_model, model, calib_loader, device=device,
            max_samples=int(kw.get("feature_max_samples", 1000)),
        )
        for k, v in sim.items():
            results[f"feature.{k}"] = v
    return results


REGISTRY: Dict[str, EvalFn] = {
    "imagenet_classification": _eval_imagenet_classification,
    "language_modeling": _eval_language_modeling,
    "dinov2_vision": _eval_dinov2_vision,
}


def evaluate(task: str, model: nn.Module, loaders: dict, device: str, **kw) -> Dict[str, Any]:
    if task not in REGISTRY:
        raise ValueError(
            f"Unknown evaluation task '{task}'. Registered: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[task](model, loaders, device, **kw)


__all__ = ["evaluate", "REGISTRY"]
