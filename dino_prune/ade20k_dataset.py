"""ADE20k semantic segmentation dataset loader."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Callable, List
from PIL import Image
import logging

logger = logging.getLogger(__name__)

ADE20K_ROOT = "/home/boxiang/work/dao2/ade20k/ADEChallengeData2016"


class ADE20kDataset(Dataset):
    """ADE20k semantic segmentation dataset loaded from local directory.

    Expects the ADEChallengeData2016 layout:
        images/{training,validation}/ADE_*.jpg
        annotations/{training,validation}/ADE_*.png

    Annotation PNGs are single-channel: pixel value = class index
    (0 = background/unlabeled, 1-150 = semantic classes).
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            split: 'train' or 'validation'
            transform: Transform for RGB images
            target_transform: Transform for segmentation masks
            max_samples: Maximum number of samples to use (for calibration)
        """
        self.transform = transform
        self.target_transform = target_transform

        split_dir = "training" if split == "train" else "validation"
        self.image_dir = os.path.join(ADE20K_ROOT, "images", split_dir)
        self.anno_dir = os.path.join(ADE20K_ROOT, "annotations", split_dir)

        self.filenames: List[str] = sorted([
            f for f in os.listdir(self.image_dir) if f.endswith(".jpg")
        ])

        if max_samples is not None and max_samples < len(self.filenames):
            indices = torch.randperm(len(self.filenames))[:max_samples].tolist()
            self.filenames = [self.filenames[i] for i in indices]

        logger.info(f"ADE20k {split}: {len(self.filenames)} samples from {self.image_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_name = self.filenames[idx]
        anno_name = img_name.replace(".jpg", ".png")

        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        annotation = np.array(Image.open(os.path.join(self.anno_dir, anno_name))).astype(np.int64)
        # ADE20k labels: 0=unlabeled, 1-150=classes. Shift to 0-149; mark unlabeled as 255.
        annotation[annotation == 0] = 256  # temporary
        annotation -= 1
        annotation[annotation == 255] = 255  # unlabeled stays 255 (ignore_index)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        if self.target_transform is not None:
            mask = self.target_transform(annotation)
        else:
            mask = torch.from_numpy(annotation).long()

        return {
            'image': image,
            'mask': mask,
            'index': idx,
        }
