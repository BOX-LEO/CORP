"""NYU Depth V2-style depth metrics (RMSE, AbsRel, delta1/2/3)."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class DepthMetrics:
    rmse: float
    rmse_log: float
    abs_rel: float
    sq_rel: float
    delta1: float
    delta2: float
    delta3: float


def compute(
    pred: torch.Tensor,
    target: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
) -> DepthMetrics:
    pred = pred.squeeze().reshape(-1)
    target = target.squeeze().reshape(-1)
    valid = (target > min_depth) & (target < max_depth) & (pred > min_depth)
    pred = pred[valid]
    target = target[valid]
    if len(pred) == 0:
        return DepthMetrics(float("inf"), float("inf"), float("inf"), float("inf"), 0, 0, 0)

    thresh = torch.max(pred / target, target / pred)
    return DepthMetrics(
        rmse=torch.sqrt(((pred - target) ** 2).mean()).item(),
        rmse_log=torch.sqrt(((torch.log(pred) - torch.log(target)) ** 2).mean()).item(),
        abs_rel=((pred - target).abs() / target).mean().item(),
        sq_rel=(((pred - target) ** 2) / target).mean().item(),
        delta1=(thresh < 1.25).float().mean().item(),
        delta2=(thresh < 1.25 ** 2).float().mean().item(),
        delta3=(thresh < 1.25 ** 3).float().mean().item(),
    )


@torch.no_grad()
def evaluate(
    depth_model,
    dataloader,
    device: str = "cuda",
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
    scale_invariant: bool = True,
) -> Dict[str, float]:
    """Run a depth model over the loader, return averaged metrics."""
    depth_model.eval().to(device)
    metrics_list: List[DepthMetrics] = []

    for batch in tqdm(dataloader, desc="Evaluating depth"):
        images = batch["image"].to(device)
        depths = batch["depth"].to(device)
        pred = depth_model(images)
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if pred.shape[-2:] != depths.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(1) if pred.ndim == 3 else pred,
                size=depths.shape[-2:], mode="bilinear", align_corners=False,
            ).squeeze(1)
        if scale_invariant:
            valid = (depths > min_depth) & (depths < max_depth)
            for i in range(pred.shape[0]):
                m = valid[i] if valid.ndim == 3 else valid
                if m.sum() > 0:
                    scale = (depths[i][m] / pred[i][m].clamp(min=1e-8)).median()
                    pred[i] = pred[i] * scale
        metrics_list.append(compute(pred, depths, min_depth=min_depth, max_depth=max_depth))

    n = len(metrics_list)
    if n == 0:
        return {"rmse": 0.0, "abs_rel": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
    return {
        "rmse": sum(m.rmse for m in metrics_list) / n,
        "rmse_log": sum(m.rmse_log for m in metrics_list) / n,
        "abs_rel": sum(m.abs_rel for m in metrics_list) / n,
        "sq_rel": sum(m.sq_rel for m in metrics_list) / n,
        "delta1": sum(m.delta1 for m in metrics_list) / n,
        "delta2": sum(m.delta2 for m in metrics_list) / n,
        "delta3": sum(m.delta3 for m in metrics_list) / n,
    }
