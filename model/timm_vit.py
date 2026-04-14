"""ViT/DeiT loader via timm (falls back to torch.hub for DeiT)."""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn


def load(
    name: str,
    checkpoint: Optional[str] = None,
    pretrained: bool = True,
    **_,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a timm ViT/DeiT by name.

    Args:
        name: timm model name (e.g. ``deit_tiny_patch16_224``, ``vit_base_patch16_224``).
        checkpoint: optional local state_dict path.
        pretrained: whether to fetch timm pretrained weights.

    Returns:
        (model, meta). `meta` contains image_size and num_classes inferred
        from the model's default_cfg when available.
    """
    try:
        import timm
        model = timm.create_model(name, pretrained=pretrained)
    except ImportError:
        # Fallback for DeiT
        if name.startswith("deit"):
            model = torch.hub.load("facebookresearch/deit:main", name, pretrained=pretrained)
        else:
            raise RuntimeError(
                "timm is required to load ViT models. Install with: pip install timm"
            )

    if checkpoint:
        state = torch.load(Path(checkpoint), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)

    meta: Dict[str, Any] = {"source": "timm", "name": name}
    default_cfg = getattr(model, "default_cfg", None)
    if default_cfg:
        input_size = default_cfg.get("input_size")
        if input_size:
            meta["image_size"] = input_size[-1]
        mean = default_cfg.get("mean")
        std = default_cfg.get("std")
        if mean:
            meta["normalize_mean"] = list(mean)
        if std:
            meta["normalize_std"] = list(std)
    num_classes = getattr(model, "num_classes", None)
    if num_classes is not None:
        meta["num_classes"] = num_classes

    meta.setdefault("image_size", 224)
    meta.setdefault("normalize_mean", [0.485, 0.456, 0.406])
    meta.setdefault("normalize_std", [0.229, 0.224, 0.225])
    return model, meta
