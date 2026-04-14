"""NYU Depth V2 dataset loader."""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Dict, Callable
import logging

logger = logging.getLogger(__name__)


class NYUDepthV2Dataset(Dataset):
    """NYU Depth V2 dataset for depth estimation.

    Supports directory format with paired image (.jpg) and depth (.png) files.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        depth_transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            root: Path to NYU Depth V2 dataset
            split: 'train' or 'val'
            transform: Transform for RGB images
            depth_transform: Transform for depth maps
            max_samples: Maximum number of samples to use (for calibration)
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.depth_transform = depth_transform
        self.max_samples = max_samples

        self._load_directory()

        # Apply max_samples limit
        if self.max_samples is not None and self.max_samples < len(self.samples):
            indices = torch.randperm(len(self.samples))[:self.max_samples].tolist()
            self.samples = [self.samples[i] for i in indices]

    def _load_directory(self):
        """Load data from directory structure.

        Supports layouts:
        - nyu2_train/<scene>/*.jpg + *.png (paired by filename stem)
        - nyu2_test/*.jpg / *_colors.png + *_depth.png
        - {split}/images/ + {split}/depth/
        """
        self.samples = []  # list of (image_path, depth_path)

        # Map split name to directory name
        split_map = {"train": "nyu2_train", "val": "nyu2_test"}
        split_dir_name = split_map.get(self.split, self.split)

        # Try nyu2_train / nyu2_test layout first
        split_dir = self.root / split_dir_name
        if split_dir.exists():
            # Check if it has scene subdirectories (nyu2_train style)
            subdirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
            if subdirs:
                # nyu2_train layout: each subdir has paired .jpg (rgb) and .png (depth)
                for scene_dir in subdirs:
                    jpgs = sorted(scene_dir.glob('*.jpg'))
                    for jpg_path in jpgs:
                        png_path = scene_dir / (jpg_path.stem + '.png')
                        if png_path.exists():
                            self.samples.append((jpg_path, png_path))
            else:
                # nyu2_test layout: flat dir with *_colors.png and *_depth.png
                color_files = sorted(split_dir.glob('*_colors.png'))
                if color_files:
                    for color_path in color_files:
                        depth_path = split_dir / color_path.name.replace('_colors.png', '_depth.png')
                        if depth_path.exists():
                            self.samples.append((color_path, depth_path))
                else:
                    # Fallback: paired .jpg and .png by stem
                    jpgs = sorted(split_dir.glob('*.jpg'))
                    for jpg_path in jpgs:
                        png_path = split_dir / (jpg_path.stem + '.png')
                        if png_path.exists():
                            self.samples.append((jpg_path, png_path))

        if not self.samples:
            # Try {split}/images + {split}/depth layout
            for img_dir_name, depth_dir_name in [
                ("images", "depth"), ("rgb", "depth"),
            ]:
                image_dir = self.root / self.split / img_dir_name
                depth_dir = self.root / self.split / depth_dir_name
                if image_dir.exists() and depth_dir.exists():
                    for ext in ['.png', '.jpg', '.jpeg']:
                        for img_path in sorted(image_dir.glob(f'*{ext}')):
                            depth_path = depth_dir / (img_path.stem + '.png')
                            if not depth_path.exists():
                                depth_path = depth_dir / (img_path.stem + '.npy')
                            if depth_path.exists():
                                self.samples.append((img_path, depth_path))
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

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Load depth
        if depth_path.suffix == '.npy':
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

        return {
            'image': image,
            'depth': depth_tensor,
            'index': idx,
        }
