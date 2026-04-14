"""Cosine / MSE / correlation between features of two models (original vs pruned)."""

from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def evaluate(
    model_a,
    model_b,
    dataloader,
    device: str = "cuda",
    max_samples: int = 1000,
) -> Dict[str, float]:
    """Compare features produced by two models for the same inputs."""
    model_a.eval().to(device)
    model_b.eval().to(device)
    total_cos = 0.0
    total_mse = 0.0
    total_corr = 0.0
    n = 0
    for batch in tqdm(dataloader, desc="Comparing features"):
        if isinstance(batch, dict):
            images = batch["image"].to(device)
        elif isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
        else:
            images = batch.to(device)
        f1 = model_a(images)
        f2 = model_b(images)
        if isinstance(f1, tuple):
            f1 = f1[0]
        if isinstance(f2, tuple):
            f2 = f2[0]
        a = f1.reshape(-1)
        b = f2.reshape(-1)
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        mse = F.mse_loss(a, b).item()
        ac = a - a.mean()
        bc = b - b.mean()
        corr = ((ac * bc).sum() / (ac.norm() * bc.norm() + 1e-8)).item()
        bs = images.shape[0]
        total_cos += cos * bs
        total_mse += mse * bs
        total_corr += corr * bs
        n += bs
        if n >= max_samples:
            break
    return {
        "cosine": total_cos / max(n, 1),
        "mse": total_mse / max(n, 1),
        "correlation": total_corr / max(n, 1),
    }
