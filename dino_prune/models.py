"""
DINOv2 model loading and utilities.

Provides:
- Model loading through torch.hub (DINOv2 backbones)
- Parameter counting
"""

import torch.nn as nn
from typing import Dict, Tuple

from dinov2_models import load_dinov2_backbone


def load_model(
    model_name: str = "base",
    pretrained: bool = True,
) -> Tuple[nn.Module, Dict]:
    """Load a DINOv2 backbone model.

    Args:
        model_name: Model size name (small, base, large, huge).
        pretrained: Whether to load pretrained weights.

    Returns:
        Tuple of (model, config_dict).
    """
    return load_dinov2_backbone(model_name, pretrained=pretrained)


def count_parameters(model: nn.Module) -> int:
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())
