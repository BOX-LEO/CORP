"""ADE20k-style segmentation metrics (pixel acc, mean acc, mIoU, fw_iou)."""

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def evaluate(
    seg_model,
    dataloader,
    device: str = "cuda",
    num_classes: int = 150,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """Accumulate a confusion matrix and compute accuracy/IoU metrics."""
    seg_model.eval().to(device)
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    for batch in tqdm(dataloader, desc="Evaluating segmentation"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        pred = seg_model(images)
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.ndim == 4:
            pred = pred.argmax(dim=1)
        if pred.shape[-2:] != masks.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(1).float(),
                size=masks.shape[-2:], mode="nearest",
            ).squeeze(1).long()
        valid = (masks != ignore_index) & (masks >= 0) & (masks < num_classes)
        p = pred[valid].clamp(0, num_classes - 1)
        m = masks[valid]
        idx = m * num_classes + p
        conf += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    conf_np = conf.cpu().numpy()
    pixel_acc = np.diag(conf_np).sum() / max(conf_np.sum(), 1)
    class_acc = np.diag(conf_np) / conf_np.sum(axis=1).clip(min=1)
    valid_classes = conf_np.sum(axis=1) > 0
    mean_acc = class_acc[valid_classes].mean() if valid_classes.any() else 0.0
    intersection = np.diag(conf_np)
    union = conf_np.sum(axis=1) + conf_np.sum(axis=0) - np.diag(conf_np)
    iou = intersection / union.clip(min=1)
    mean_iou = iou[valid_classes].mean() if valid_classes.any() else 0.0
    freq = conf_np.sum(axis=1) / max(conf_np.sum(), 1)
    fw_iou = (freq[valid_classes] * iou[valid_classes]).sum() if valid_classes.any() else 0.0
    return {
        "pixel_accuracy": float(pixel_acc),
        "mean_accuracy": float(mean_acc),
        "mean_iou": float(mean_iou),
        "fw_iou": float(fw_iou),
    }
