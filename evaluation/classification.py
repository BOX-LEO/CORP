"""ImageNet top-1 / top-5 accuracy."""

from typing import Tuple
import logging

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device: str = "cuda",
    desc: str = "Evaluating",
) -> Tuple[float, float]:
    """Return (top-1, top-5) accuracy percentages on ``dataloader``."""
    model.eval()
    model.to(device)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for batch in tqdm(dataloader, desc=desc):
        if isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1]
        else:
            images, labels = batch, None
        if labels is None:
            logger.warning("No labels found in batch, skipping accuracy evaluation")
            return 0.0, 0.0

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, pred_top1 = outputs.topk(1, dim=1)
        correct_top1 += (pred_top1.squeeze() == labels).sum().item()
        _, pred_top5 = outputs.topk(5, dim=1)
        correct_top5 += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    top1 = 100.0 * correct_top1 / total
    top5 = 100.0 * correct_top5 / total
    return top1, top5
