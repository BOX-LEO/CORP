"""ADE20k semantic segmentation dataset (ADEChallengeData2016 layout)."""

from pathlib import Path
from typing import Callable, Dict, List, Optional
import logging

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ADE20kDataset(Dataset):
    """ADE20k semantic segmentation, loaded from a local ADEChallengeData2016 root.

    Expected layout::

        <root>/images/{training,validation}/ADE_*.jpg
        <root>/annotations/{training,validation}/ADE_*.png

    Annotation PNGs are single-channel: pixel value = class index
    (0 = unlabeled, 1-150 = semantic classes; shifted to 0-149 / 255=ignore).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.transform = transform
        self.target_transform = target_transform
        split_dir = "training" if split == "train" else "validation"
        self.image_dir = Path(root) / "images" / split_dir
        self.anno_dir = Path(root) / "annotations" / split_dir

        self.filenames: List[str] = sorted(
            f.name for f in self.image_dir.iterdir() if f.suffix == ".jpg"
        )
        if max_samples is not None and max_samples < len(self.filenames):
            indices = torch.randperm(len(self.filenames))[:max_samples].tolist()
            self.filenames = [self.filenames[i] for i in indices]

        logger.info(f"ADE20k {split}: {len(self.filenames)} samples from {self.image_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_name = self.filenames[idx]
        anno_name = img_name.replace(".jpg", ".png")
        image = Image.open(self.image_dir / img_name).convert("RGB")
        annotation = np.array(Image.open(self.anno_dir / anno_name)).astype(np.int64)
        annotation[annotation == 0] = 256
        annotation -= 1
        annotation[annotation == 255] = 255

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        if self.target_transform is not None:
            mask = self.target_transform(annotation)
        else:
            mask = torch.from_numpy(annotation).long()
        return {"image": image, "mask": mask, "index": idx}
