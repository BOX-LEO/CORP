"""DINOv2 backbone loader + depth/seg head wrappers.

Ported from the old dino_prune/dinov2_models.py. Head .pth files live under
``model/heads/`` by default (configurable via ``heads_dir`` kwarg / YAML).
"""

import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn


_DEFAULT_HEADS_DIR = Path(__file__).resolve().parent / "heads"


SIZE_NAME_MAP = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "huge": "vitg14",
}

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
    hub_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    if hub_dir not in sys.path:
        sys.path.insert(0, hub_dir)
    from dinov2.hub.depth import BNHead, DepthEncoderDecoder, DPTHead
    from dinov2.hub.utils import CenterPadding
    return BNHead, DepthEncoderDecoder, DPTHead, CenterPadding


def resolve_model_size(model_name: str) -> str:
    if model_name in SIZE_NAME_MAP:
        return SIZE_NAME_MAP[model_name]
    if model_name in EMBED_DIMS:
        return model_name
    raise ValueError(
        f"Unknown DINOv2 model: {model_name}. Use one of "
        f"{list(SIZE_NAME_MAP)} or {list(EMBED_DIMS)}"
    )


def _load_head_state_dict(path: Path, prefix: str = "decode_head.") -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _head_path(heads_dir: Path, model_size: str, task: str) -> Path:
    if task == "depth":
        filename = f"dinov2_{model_size}_nyu_dpt_head.pth"
    elif task == "seg":
        filename = f"dinov2_{model_size}_ade20k_linear_head.pth"
    else:
        raise ValueError(f"Unknown task: {task}")
    p = Path(heads_dir) / filename
    if not p.exists():
        raise FileNotFoundError(f"Head file not found: {p}")
    return p


def load(
    name: str = "base",
    pretrained: bool = True,
    **_,
) -> Tuple[nn.Module, Dict]:
    """Load a DINOv2 backbone, returning (model, meta)."""
    model_size = resolve_model_size(name)
    arch_name = ARCH_NAME_MAP[model_size]
    embed_dim = EMBED_DIMS[model_size]

    hub_name = f"dinov2_{model_size}"
    backbone = torch.hub.load("facebookresearch/dinov2", hub_name)

    meta = {
        "source": "torch_hub_dinov2",
        "name": name,
        "model_size": model_size,
        "arch_name": arch_name,
        "embed_dim": embed_dim,
        "num_heads": getattr(backbone, "num_heads", 0),
        "num_blocks": NUM_BLOCKS[model_size],
        "patch_size": 14,
        "image_size": 518,
        "block_prefix": "blocks",
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }
    return backbone, meta


class _DepthModelWrapper(nn.Module):
    def __init__(self, enc_dec):
        super().__init__()
        self.enc_dec = enc_dec

    def forward(self, img):
        return self.enc_dec.encode_decode(img, img_metas=None, rescale=True)


def create_depth_model(
    backbone: nn.Module,
    model_size: str,
    heads_dir: Optional[Path] = None,
) -> nn.Module:
    """Attach a DPT depth head to a DINOv2 backbone."""
    heads_dir = Path(heads_dir) if heads_dir else _DEFAULT_HEADS_DIR
    model_size = resolve_model_size(model_size)
    arch_name = ARCH_NAME_MAP[model_size]
    embed_dim = EMBED_DIMS[model_size]
    out_index = OUT_INDEX[arch_name]

    _, DepthEncoderDecoder, DPTHead, CenterPadding = _get_hub_module()

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

    head_state = _load_head_state_dict(_head_path(heads_dir, model_size, "depth"))
    head.load_state_dict(head_state, strict=False)

    enc_dec = DepthEncoderDecoder(backbone=backbone, decode_head=head)
    patch_size = getattr(backbone, "patch_size", 14)
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
    return _DepthModelWrapper(enc_dec)


class _LinearSegHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 150):
        super().__init__()
        self.bn = nn.SyncBatchNorm(embed_dim)
        self.conv_seg = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            features = x[-1]
            if isinstance(features, (list, tuple)):
                features = features[0]
        else:
            features = x
        return self.conv_seg(self.bn(features))


class DINOv2SegModel(nn.Module):
    def __init__(self, backbone, seg_head, out_index, patch_size=14):
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
        return self.seg_head(features)


def create_seg_model(
    backbone: nn.Module,
    model_size: str,
    heads_dir: Optional[Path] = None,
) -> nn.Module:
    """Attach a linear segmentation head to a DINOv2 backbone."""
    heads_dir = Path(heads_dir) if heads_dir else _DEFAULT_HEADS_DIR
    model_size = resolve_model_size(model_size)
    arch_name = ARCH_NAME_MAP[model_size]
    embed_dim = EMBED_DIMS[model_size]
    out_index = OUT_INDEX[arch_name]
    patch_size = getattr(backbone, "patch_size", 14)

    seg_head = _LinearSegHead(embed_dim, num_classes=150)
    head_state = _load_head_state_dict(_head_path(heads_dir, model_size, "seg"))
    seg_head.load_state_dict(head_state, strict=True)
    return DINOv2SegModel(backbone, seg_head, out_index, patch_size)
