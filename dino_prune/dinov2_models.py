"""
DINOv2 model loading with backbone and task-specific heads.

Supports:
- DINOv2 backbones (ViT-S/B/L/G) via torch.hub
- Depth estimation heads (DPT) loaded from local .pth files
- Segmentation heads (linear BN+Conv) loaded from local .pth files

Head weights are mmseg-format state_dicts stored in the heads/ directory.
Architecture is reconstructed from the cached dinov2 hub code.
"""

import os
import sys
from functools import partial
from typing import Dict, Tuple

import torch
import torch.nn as nn

HEADS_DIR = os.path.join(os.path.dirname(__file__), "heads")

# Map CLI names (small/base/large/huge) to DINOv2 internal size names
SIZE_NAME_MAP = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "huge": "vitg14",
}

# Map internal size names to hub arch names used by dinov2 code
ARCH_NAME_MAP = {
    "vits14": "vit_small",
    "vitb14": "vit_base",
    "vitl14": "vit_large",
    "vitg14": "vit_giant2",
}

EMBED_DIMS = {
    "vits14": 384,
    "vitb14": 768,
    "vitl14": 1024,
    "vitg14": 1536,
}

OUT_INDEX = {
    "vit_small": [2, 5, 8, 11],
    "vit_base": [2, 5, 8, 11],
    "vit_large": [4, 11, 17, 23],
    "vit_giant2": [9, 19, 29, 39],
}

NUM_BLOCKS = {
    "vits14": 12,
    "vitb14": 12,
    "vitl14": 24,
    "vitg14": 40,
}


def _get_hub_module():
    """Import depth/encoder-decoder classes from the cached dinov2 hub code."""
    hub_dir = os.path.expanduser(
        "~/.cache/torch/hub/facebookresearch_dinov2_main"
    )
    if hub_dir not in sys.path:
        sys.path.insert(0, hub_dir)
    from dinov2.hub.depth import BNHead, DepthEncoderDecoder, DPTHead
    from dinov2.hub.utils import CenterPadding
    return BNHead, DepthEncoderDecoder, DPTHead, CenterPadding


def _resolve_model_size(model_name: str) -> str:
    """Resolve CLI name to internal size (e.g. 'small' -> 'vits14')."""
    if model_name in SIZE_NAME_MAP:
        return SIZE_NAME_MAP[model_name]
    if model_name in EMBED_DIMS:
        return model_name
    raise ValueError(
        f"Unknown model: {model_name}. Use one of "
        f"{list(SIZE_NAME_MAP.keys())} or {list(EMBED_DIMS.keys())}"
    )


def _load_head_state_dict(path: str, prefix: str = "decode_head.") -> dict:
    """Load a .pth checkpoint and strip the prefix from state_dict keys."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def get_dinov2_head_path(model_size: str, task: str) -> str:
    """Get path to a local head .pth file.

    Args:
        model_size: Internal size name (vits14/vitb14/vitl14/vitg14).
        task: 'depth' or 'seg'.
    """
    if task == "depth":
        filename = f"dinov2_{model_size}_nyu_dpt_head.pth"
    elif task == "seg":
        filename = f"dinov2_{model_size}_ade20k_linear_head.pth"
    else:
        raise ValueError(f"Unknown task: {task}")
    path = os.path.join(HEADS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Head file not found: {path}")
    return path


def load_backbone(model_size: str = "vitb14") -> nn.Module:
    """Load a DINOv2 backbone from torch.hub.

    Uses the non-register variant because the pretrained depth/seg heads
    were trained with non-register backbones.
    """
    model_size = _resolve_model_size(model_size)
    hub_name = f"dinov2_{model_size}"
    return torch.hub.load("facebookresearch/dinov2", hub_name)


def load_dinov2_backbone(
    model_name: str = "base",
    pretrained: bool = True,
) -> Tuple[nn.Module, Dict]:
    """Load a DINOv2 backbone, returning (model, config_dict)."""
    model_size = _resolve_model_size(model_name)
    arch_name = ARCH_NAME_MAP[model_size]
    embed_dim = EMBED_DIMS[model_size]

    backbone = load_backbone(model_size)

    config = {
        "backbone": "dinov2",
        "model_size": model_size,
        "arch_name": arch_name,
        "embed_dim": embed_dim,
        "num_heads": backbone.num_heads if hasattr(backbone, "num_heads") else 0,
        "num_blocks": NUM_BLOCKS[model_size],
        "patch_size": 14,
        "image_size": 518,
        "block_prefix": "blocks",
    }

    return backbone, config


def create_dinov2_depth_model(backbone: nn.Module, model_size: str) -> nn.Module:
    """Create a full depth encoder-decoder with the given backbone + DPT head.

    Loads DPT head weights from local .pth file and wires up the backbone
    forward to use get_intermediate_layers.

    Args:
        backbone: A DINOv2 backbone (may be pruned).
        model_size: Internal size name (vits14/vitb14/vitl14/vitg14)
                    or CLI name (small/base/large/huge).

    Returns:
        DepthEncoderDecoder that takes images and outputs depth maps.
    """
    model_size = _resolve_model_size(model_size)
    arch_name = ARCH_NAME_MAP[model_size]
    embed_dim = EMBED_DIMS[model_size]
    out_index = OUT_INDEX[arch_name]

    _, DepthEncoderDecoder, DPTHead, CenterPadding = _get_hub_module()

    # Construct DPT head architecture
    head = DPTHead(
        in_channels=[embed_dim] * 4,
        channels=256,
        embed_dims=embed_dim,
        post_process_channels=[embed_dim // 2 ** (3 - i) for i in range(4)],
        readout_type="project",
        min_depth=0.001,
        max_depth=10.0,
        loss_decode=(),
    )

    head_state = _load_head_state_dict(get_dinov2_head_path(model_size, "depth"))
    head.load_state_dict(head_state, strict=False)

    # Build encoder-decoder
    enc_dec = DepthEncoderDecoder(backbone=backbone, decode_head=head)

    # Patch backbone forward to use intermediate layers
    patch_size = backbone.patch_size if hasattr(backbone, "patch_size") else 14
    enc_dec.backbone.forward = partial(
        backbone.get_intermediate_layers,
        n=out_index,
        reshape=True,
        return_class_token=True,
        norm=False,
    )
    enc_dec.backbone.register_forward_pre_hook(
        lambda _, x: CenterPadding(patch_size)(x[0])
    )

    # Wrap so model(images) returns depth directly (evaluator expects this)
    return _DepthModelWrapper(enc_dec)


class _DepthModelWrapper(nn.Module):
    """Wraps DepthEncoderDecoder so forward(images) returns depth maps."""

    def __init__(self, enc_dec):
        super().__init__()
        self.enc_dec = enc_dec

    def forward(self, img):
        return self.enc_dec.encode_decode(img, img_metas=None, rescale=True)


class _LinearSegHead(nn.Module):
    """BN + 1x1 Conv segmentation head matching DINOv2 linear head format.

    Matches the mmseg BNHead used in DINOv2 segmentation evaluation:
    SyncBatchNorm(embed_dim) -> Conv2d(embed_dim, num_classes, 1).
    """

    def __init__(self, embed_dim: int, num_classes: int = 150):
        super().__init__()
        self.bn = nn.SyncBatchNorm(embed_dim)
        self.conv_seg = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # x is a list of feature tensors from get_intermediate_layers
        # Use the last layer's patch features
        if isinstance(x, (list, tuple)):
            features = x[-1]
            if isinstance(features, (list, tuple)):
                features = features[0]  # patch tokens from (patch, cls) tuple
        else:
            features = x
        out = self.bn(features)
        out = self.conv_seg(out)
        return out


class DINOv2SegModel(nn.Module):
    """DINOv2 backbone + linear segmentation head, end-to-end."""

    def __init__(self, backbone: nn.Module, seg_head: nn.Module,
                 out_index: list, patch_size: int = 14):
        super().__init__()
        self.backbone = backbone
        self.seg_head = seg_head
        self.out_index = out_index
        self.patch_size = patch_size

        _, _, _, CenterPadding = _get_hub_module()
        self._center_padding = CenterPadding(patch_size)

    def forward(self, x):
        x = self._center_padding(x)
        features = self.backbone.get_intermediate_layers(
            x, n=self.out_index, reshape=True,
        )
        logits = self.seg_head(features)
        return logits


def create_dinov2_seg_model(backbone: nn.Module, model_size: str) -> nn.Module:
    """Create a segmentation model with the given backbone + linear seg head.

    Args:
        backbone: A DINOv2 backbone (may be pruned).
        model_size: Internal size name or CLI name.

    Returns:
        Model that takes images and outputs segmentation logits (B, 150, H, W).
    """
    model_size = _resolve_model_size(model_size)
    arch_name = ARCH_NAME_MAP[model_size]
    embed_dim = EMBED_DIMS[model_size]
    out_index = OUT_INDEX[arch_name]
    patch_size = backbone.patch_size if hasattr(backbone, "patch_size") else 14

    # Build head
    seg_head = _LinearSegHead(embed_dim, num_classes=150)

    head_state = _load_head_state_dict(get_dinov2_head_path(model_size, "seg"))
    seg_head.load_state_dict(head_state, strict=True)

    return DINOv2SegModel(backbone, seg_head, out_index, patch_size)
