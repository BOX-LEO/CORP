"""Calibration / validation loaders for the DINOv2 vision task.

Combines NYU Depth V2 and ADE20k into a single calibration stream
(images only) and returns separate validation loaders.
"""

from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .nyu_dataset import NYUDepthV2Dataset
from .ade20k_dataset import ADE20kDataset


class _CombinedCalib(Dataset):
    """Yields ``image`` tensors from NYU + ADE20k in sequence (no labels)."""

    def __init__(self, nyu: Optional[NYUDepthV2Dataset], ade: Optional[ADE20kDataset]):
        self.nyu = nyu
        self.ade = ade
        self.nyu_len = len(nyu) if nyu is not None else 0
        self.ade_len = len(ade) if ade is not None else 0

    def __len__(self) -> int:
        return self.nyu_len + self.ade_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < self.nyu_len:
            return self.nyu[idx]["image"]
        return self.ade[idx - self.nyu_len]["image"]


def _get_transforms(image_size: int, mean, std):
    from torchvision import transforms

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    def depth_transform(depth):
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth).float()
        if depth.max() > 20.0:
            depth = depth / 1000.0
        return torch.nn.functional.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=(image_size, image_size), mode="bilinear", align_corners=False,
        ).squeeze()

    def mask_transform(mask):
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        return torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(image_size, image_size), mode="nearest",
        ).squeeze().long()

    return image_transform, depth_transform, mask_transform


def build_dinov2_loaders(cfg: dict, meta: Optional[dict] = None) -> Dict[str, Any]:
    """Build calibration + per-dataset validation loaders for DINOv2 tasks."""
    image_size = int((meta or {}).get("image_size", cfg.get("image_size", 518)))
    mean = (meta or {}).get("normalize_mean", [0.485, 0.456, 0.406])
    std = (meta or {}).get("normalize_std", [0.229, 0.224, 0.225])
    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 4))

    image_tf, depth_tf, mask_tf = _get_transforms(image_size, mean, std)

    nyu_root = cfg.get("nyu_path")
    use_ade = bool(cfg.get("use_ade", False))
    ade_root = cfg.get("ade_path")
    nyu_samples = int(cfg.get("nyu_samples", 3000))
    ade_samples = int(cfg.get("ade_samples", 3000))

    nyu_train = nyu_val = None
    if nyu_root and Path(nyu_root).exists():
        nyu_train = NYUDepthV2Dataset(
            str(nyu_root), split="train",
            transform=image_tf, depth_transform=depth_tf,
            max_samples=nyu_samples,
        )
        nyu_val = NYUDepthV2Dataset(
            str(nyu_root), split="val",
            transform=image_tf, depth_transform=depth_tf,
        )

    ade_train = ade_val = None
    if use_ade and ade_root and Path(ade_root).exists():
        ade_train = ADE20kDataset(
            str(ade_root), split="train",
            transform=image_tf, target_transform=mask_tf,
            max_samples=ade_samples,
        )
        ade_val = ADE20kDataset(
            str(ade_root), split="validation",
            transform=image_tf, target_transform=mask_tf,
        )

    if nyu_train is None and ade_train is None:
        raise ValueError(
            "dinov2_vision task requires at least one of nyu_path or ade_path "
            "(and use_ade: true for ADE20k)"
        )

    combined = _CombinedCalib(nyu_train, ade_train)
    calib_loader = DataLoader(
        combined, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    val: Dict[str, DataLoader] = {}
    if nyu_val is not None:
        val["nyu"] = DataLoader(
            nyu_val, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    if ade_val is not None:
        val["ade"] = DataLoader(
            ade_val, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    return {"calib": calib_loader, "val": val}
