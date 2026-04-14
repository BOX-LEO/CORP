"""
Dataset loaders for NYU Depth V2 and ADE20k.

Provides:
- NYUDepthV2Dataset: Depth estimation dataset
- ADE20kDataset: Semantic segmentation dataset
- Combined calibration loader for pruning
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Callable
import logging

from dino_prune.nyu_dataset import NYUDepthV2Dataset
from dino_prune.ade20k_dataset import ADE20kDataset

logger = logging.getLogger(__name__)

# Re-export for backwards compatibility
__all__ = [
    "NYUDepthV2Dataset",
    "ADE20kDataset",
    "CombinedCalibrationDataset",
    "get_transforms",
    "create_calibration_loader",
    "create_validation_loaders",
]


class CombinedCalibrationDataset(Dataset):
    """Combined dataset from NYU Depth and ADE20k for calibration.

    Returns only images (no labels) for activation collection.
    """

    def __init__(
        self,
        nyu_dataset: Optional[NYUDepthV2Dataset],
        ade_dataset: Optional[ADE20kDataset],
    ):
        """
        Args:
            nyu_dataset: NYU Depth dataset instance
            ade_dataset: ADE20k dataset instance
        """
        self.nyu_dataset = nyu_dataset
        self.ade_dataset = ade_dataset

        self.nyu_len = len(nyu_dataset) if nyu_dataset is not None else 0
        self.ade_len = len(ade_dataset) if ade_dataset is not None else 0

        logger.info(f"Combined calibration: {self.nyu_len} NYU + {self.ade_len} ADE20k = {len(self)} total")

    def __len__(self) -> int:
        return self.nyu_len + self.ade_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return only the image tensor for calibration."""
        if idx < self.nyu_len:
            sample = self.nyu_dataset[idx]
            return sample['image']
        else:
            sample = self.ade_dataset[idx - self.nyu_len]
            return sample['image']


def get_transforms(image_size: int = 518, model_type: str = "dinov2"):
    """Get standard transforms for DINOv2 models.

    Args:
        image_size: Target image size (default 518 for DINOv2/v3 large models)
        model_type: Model type for normalization stats

    Returns:
        Tuple of (image_transform, depth_transform, mask_transform)
    """
    from torchvision import transforms

    # DINOv2/v3 normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    def depth_transform(depth):
        """Transform depth map (convert mm to meters and resize)."""
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth).float()
        # NYU Depth V2 stores depth in millimeters — convert to meters
        if depth.max() > 20.0:
            depth = depth / 1000.0
        # Resize depth to match image size
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        return depth

    def mask_transform(mask):
        """Transform segmentation mask."""
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        # Resize mask with nearest neighbor
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(image_size, image_size),
            mode='nearest'
        ).squeeze().long()
        return mask

    return image_transform, depth_transform, mask_transform


def create_calibration_loader(
    nyu_root: Optional[str] = None,
    use_ade: bool = False,
    nyu_samples: int = 3000,
    ade_samples: int = 3000,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 518,
) -> DataLoader:
    """Create a combined calibration data loader.

    Args:
        nyu_root: Path to NYU Depth V2 dataset
        use_ade: Whether to include ADE20k (loaded from HuggingFace)
        nyu_samples: Number of samples from NYU (default 3000)
        ade_samples: Number of samples from ADE20k (default 3000)
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size for transforms

    Returns:
        DataLoader for calibration
    """
    image_transform, depth_transform, mask_transform = get_transforms(image_size)

    nyu_dataset = None
    ade_dataset = None

    if nyu_root is not None and Path(nyu_root).exists():
        nyu_dataset = NYUDepthV2Dataset(
            root=nyu_root,
            split="train",
            transform=image_transform,
            depth_transform=depth_transform,
            max_samples=nyu_samples,
        )

    if use_ade:
        ade_dataset = ADE20kDataset(
            split="train",
            transform=image_transform,
            target_transform=mask_transform,
            max_samples=ade_samples,
        )

    if nyu_dataset is None and ade_dataset is None:
        raise ValueError("At least one of nyu_root or use_ade must be provided")

    combined_dataset = CombinedCalibrationDataset(nyu_dataset, ade_dataset)

    loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader


def create_validation_loaders(
    nyu_root: Optional[str] = None,
    use_ade: bool = False,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 518,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """Create validation data loaders for evaluation.

    Args:
        nyu_root: Path to NYU Depth V2 dataset
        use_ade: Whether to include ADE20k (loaded from HuggingFace)
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size for transforms

    Returns:
        Tuple of (nyu_val_loader, ade_val_loader)
    """
    image_transform, depth_transform, mask_transform = get_transforms(image_size)

    nyu_val_loader = None
    ade_val_loader = None

    if nyu_root is not None and Path(nyu_root).exists():
        nyu_val = NYUDepthV2Dataset(
            root=nyu_root,
            split="val",
            transform=image_transform,
            depth_transform=depth_transform,
        )
        nyu_val_loader = DataLoader(
            nyu_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info(f"Created NYU validation loader: {len(nyu_val)} samples")

    if use_ade:
        ade_val = ADE20kDataset(
            split="validation",
            transform=image_transform,
            target_transform=mask_transform,
        )
        ade_val_loader = DataLoader(
            ade_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info(f"Created ADE20k validation loader: {len(ade_val)} samples")

    return nyu_val_loader, ade_val_loader
