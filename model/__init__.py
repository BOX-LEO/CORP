"""Model loading registry.

Each loader returns `(model, meta)`. `meta` is a dict that downstream
components (dataset builders, evaluators) may read, e.g.:

    {
        "image_size": 224,
        "num_classes": 1000,
        "tokenizer": <tokenizer>,    # for language models
        "model_size": "vitb14",       # for DINOv2
        "num_heads": 12,
        "embed_dim": 768,
        ...
    }

To add a new source: write a loader module, import + register here.
"""

from typing import Callable, Tuple, Dict, Any
import torch.nn as nn

from . import timm_vit, dinov2, opt


LoaderFn = Callable[..., Tuple[nn.Module, Dict[str, Any]]]

REGISTRY: Dict[str, LoaderFn] = {
    "timm": timm_vit.load,
    "torch_hub_dinov2": dinov2.load,
    "hf_opt": opt.load,
}


def load_model(source: str, name: str, **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
    if source not in REGISTRY:
        raise ValueError(
            f"Unknown model source '{source}'. Registered: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[source](name, **kwargs)


__all__ = ["load_model", "REGISTRY"]
