"""ImageNet calibration + validation loaders (ViT/DeiT classification)."""

from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader, TensorDataset


def _build_transform(image_size: int, meta: Optional[dict]):
    from torchvision import transforms
    mean = (meta or {}).get("normalize_mean", [0.485, 0.456, 0.406])
    std = (meta or {}).get("normalize_std", [0.229, 0.224, 0.225])
    resize = max(256, int(round(image_size * 256 / 224)))
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _build_calibration_loader(
    calib_path: Path,
    batch_size: int,
    transform,
    num_workers: int,
) -> DataLoader:
    """ImageFolder or a .pt/.pth tensor file."""
    calib_path = Path(calib_path)
    if calib_path.suffix in (".pt", ".pth"):
        data = torch.load(calib_path)
        if isinstance(data, dict):
            images = data.get("images", data.get("data"))
        else:
            images = data
        dataset = TensorDataset(images)
    elif calib_path.is_dir():
        from torchvision import datasets
        dataset = datasets.ImageFolder(calib_path, transform=transform)
    else:
        raise ValueError(f"Unknown calibration data format: {calib_path}")

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )


def _build_val_loader(
    val_path: Path,
    batch_size: int,
    transform,
    num_workers: int,
) -> DataLoader:
    from torchvision import datasets
    val_path = Path(val_path)
    if not val_path.is_dir():
        raise ValueError(f"val_path must be a directory: {val_path}")
    dataset = datasets.ImageFolder(val_path, transform=transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


def build_loaders(cfg: dict, meta: Optional[dict] = None) -> Dict[str, Any]:
    image_size = int((meta or {}).get("image_size", cfg.get("image_size", 224)))
    transform = _build_transform(image_size, meta)
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 4))

    loaders: Dict[str, Any] = {}
    if cfg.get("calib_path"):
        loaders["calib"] = _build_calibration_loader(
            cfg["calib_path"], batch_size, transform, num_workers
        )
    if cfg.get("val_path"):
        eval_bs = int(cfg.get("eval_batch_size") or batch_size)
        val_transform = _build_transform(image_size, meta)
        loaders["val"] = _build_val_loader(
            cfg["val_path"], eval_bs, val_transform, num_workers
        )
    return loaders
