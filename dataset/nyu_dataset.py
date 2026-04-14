"""NYU Depth V2 dataset."""

from pathlib import Path
from typing import Callable, Dict, Optional
import logging

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class NYUDepthV2Dataset(Dataset):
    """NYU Depth V2 for depth estimation (paired .jpg + depth .png/.npy)."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        depth_transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.depth_transform = depth_transform
        self.max_samples = max_samples
        self._load_directory()
        if max_samples is not None and max_samples < len(self.samples):
            indices = torch.randperm(len(self.samples))[:max_samples].tolist()
            self.samples = [self.samples[i] for i in indices]

    def _load_directory(self):
        self.samples = []
        split_map = {"train": "nyu2_train", "val": "nyu2_test"}
        split_dir_name = split_map.get(self.split, self.split)
        split_dir = self.root / split_dir_name

        if split_dir.exists():
            subdirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
            if subdirs:
                for scene_dir in subdirs:
                    for jpg in sorted(scene_dir.glob("*.jpg")):
                        png = scene_dir / (jpg.stem + ".png")
                        if png.exists():
                            self.samples.append((jpg, png))
            else:
                color_files = sorted(split_dir.glob("*_colors.png"))
                if color_files:
                    for color in color_files:
                        depth = split_dir / color.name.replace("_colors.png", "_depth.png")
                        if depth.exists():
                            self.samples.append((color, depth))
                else:
                    for jpg in sorted(split_dir.glob("*.jpg")):
                        png = split_dir / (jpg.stem + ".png")
                        if png.exists():
                            self.samples.append((jpg, png))

        if not self.samples:
            for img_dir_name, depth_dir_name in [("images", "depth"), ("rgb", "depth")]:
                image_dir = self.root / self.split / img_dir_name
                depth_dir = self.root / self.split / depth_dir_name
                if image_dir.exists() and depth_dir.exists():
                    for ext in (".png", ".jpg", ".jpeg"):
                        for img in sorted(image_dir.glob(f"*{ext}")):
                            depth = depth_dir / (img.stem + ".png")
                            if not depth.exists():
                                depth = depth_dir / (img.stem + ".npy")
                            if depth.exists():
                                self.samples.append((img, depth))
                    if self.samples:
                        break

        if not self.samples:
            raise ValueError(
                f"Could not find NYU data in {self.root}. "
                f"Expected nyu2_train/nyu2_test subdirs or {{split}}/images/ + {{split}}/depth/ layout."
            )
        logger.info(f"Loaded NYU Depth V2 {self.split}: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, depth_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if depth_path.suffix == ".npy":
            depth = np.load(depth_path)
        else:
            depth = np.array(Image.open(depth_path))

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        if self.depth_transform is not None:
            depth_tensor = self.depth_transform(depth)
        else:
            depth_tensor = torch.from_numpy(depth).float()

        return {"image": image, "depth": depth_tensor, "index": idx}
