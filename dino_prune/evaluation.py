"""
Evaluation metrics for depth estimation and semantic segmentation.

Provides:
- Depth estimation metrics (RMSE, AbsRel, delta accuracies)
- Semantic segmentation metrics (mIoU, pixel accuracy)
- Feature similarity metrics for comparing models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


@dataclass
class DepthMetrics:
    """Depth estimation metrics."""
    rmse: float  # Root Mean Square Error
    rmse_log: float  # RMSE of log depth
    abs_rel: float  # Absolute Relative Error
    sq_rel: float  # Squared Relative Error
    delta1: float  # % of pixels with max(pred/gt, gt/pred) < 1.25
    delta2: float  # % of pixels with max(pred/gt, gt/pred) < 1.25^2
    delta3: float  # % of pixels with max(pred/gt, gt/pred) < 1.25^3


@dataclass
class SegmentationMetrics:
    """Semantic segmentation metrics."""
    pixel_accuracy: float  # Overall pixel accuracy
    mean_accuracy: float  # Mean class accuracy
    mean_iou: float  # Mean Intersection over Union
    fw_iou: float  # Frequency weighted IoU
    per_class_iou: Optional[Dict[int, float]] = None


@dataclass
class FeatureSimilarityMetrics:
    """Feature similarity metrics between two models."""
    cosine_similarity: float
    mse: float
    correlation: float


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
) -> DepthMetrics:
    """Compute depth estimation metrics.

    Args:
        pred: Predicted depth map (B, H, W) or (B, 1, H, W)
        target: Ground truth depth map (B, H, W) or (B, 1, H, W)
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth

    Returns:
        DepthMetrics dataclass
    """
    # Flatten and remove invalid pixels
    pred = pred.squeeze()
    target = target.squeeze()

    if pred.ndim == 3:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
    elif pred.ndim == 2:
        pred = pred.reshape(-1)
        target = target.reshape(-1)

    # Create valid mask
    valid_mask = (target > min_depth) & (target < max_depth) & (pred > min_depth)
    pred = pred[valid_mask]
    target = target[valid_mask]

    if len(pred) == 0:
        return DepthMetrics(
            rmse=float('inf'), rmse_log=float('inf'),
            abs_rel=float('inf'), sq_rel=float('inf'),
            delta1=0.0, delta2=0.0, delta3=0.0
        )

    # Compute metrics
    thresh = torch.max(pred / target, target / pred)

    delta1 = (thresh < 1.25).float().mean().item()
    delta2 = (thresh < 1.25 ** 2).float().mean().item()
    delta3 = (thresh < 1.25 ** 3).float().mean().item()

    rmse = torch.sqrt(((pred - target) ** 2).mean()).item()
    rmse_log = torch.sqrt(((torch.log(pred) - torch.log(target)) ** 2).mean()).item()
    abs_rel = ((pred - target).abs() / target).mean().item()
    sq_rel = (((pred - target) ** 2) / target).mean().item()

    return DepthMetrics(
        rmse=rmse,
        rmse_log=rmse_log,
        abs_rel=abs_rel,
        sq_rel=sq_rel,
        delta1=delta1,
        delta2=delta2,
        delta3=delta3,
    )


def compute_segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 150,
    ignore_index: int = 255,
) -> SegmentationMetrics:
    """Compute semantic segmentation metrics.

    Args:
        pred: Predicted segmentation logits (B, C, H, W) or labels (B, H, W)
        target: Ground truth labels (B, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore in evaluation

    Returns:
        SegmentationMetrics dataclass
    """
    # Get predictions if logits
    if pred.ndim == 4:
        pred = pred.argmax(dim=1)

    # Flatten
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    # Create valid mask
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]

    if len(pred) == 0:
        return SegmentationMetrics(
            pixel_accuracy=0.0, mean_accuracy=0.0,
            mean_iou=0.0, fw_iou=0.0
        )

    # Compute confusion matrix (vectorized)
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    target = target[valid]
    pred = pred[valid]
    indices = target * num_classes + pred
    conf_matrix = torch.bincount(
        indices, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)

    # Convert to numpy for easier computation
    conf_matrix = conf_matrix.cpu().numpy()

    # Pixel accuracy
    pixel_accuracy = np.diag(conf_matrix).sum() / conf_matrix.sum()

    # Mean class accuracy
    class_accuracy = np.diag(conf_matrix) / conf_matrix.sum(axis=1).clip(min=1)
    valid_classes = conf_matrix.sum(axis=1) > 0
    mean_accuracy = class_accuracy[valid_classes].mean()

    # IoU per class
    intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    iou = intersection / union.clip(min=1)

    # Mean IoU (only for present classes)
    mean_iou = iou[valid_classes].mean()

    # Frequency weighted IoU
    freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
    fw_iou = (freq[valid_classes] * iou[valid_classes]).sum()

    # Per-class IoU
    per_class_iou = {i: iou[i] for i in range(num_classes) if valid_classes[i]}

    return SegmentationMetrics(
        pixel_accuracy=float(pixel_accuracy),
        mean_accuracy=float(mean_accuracy),
        mean_iou=float(mean_iou),
        fw_iou=float(fw_iou),
        per_class_iou=per_class_iou,
    )


def compute_feature_similarity(
    features1: torch.Tensor,
    features2: torch.Tensor,
) -> FeatureSimilarityMetrics:
    """Compute similarity metrics between two feature tensors.

    Args:
        features1: First feature tensor (B, D) or (B, N, D)
        features2: Second feature tensor (same shape as features1)

    Returns:
        FeatureSimilarityMetrics dataclass
    """
    # Flatten if needed
    f1 = features1.reshape(-1)
    f2 = features2.reshape(-1)

    # Cosine similarity
    cosine = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()

    # MSE
    mse = F.mse_loss(f1, f2).item()

    # Correlation
    f1_centered = f1 - f1.mean()
    f2_centered = f2 - f2.mean()
    correlation = (f1_centered * f2_centered).sum() / (f1_centered.norm() * f2_centered.norm() + 1e-8)
    correlation = correlation.item()

    return FeatureSimilarityMetrics(
        cosine_similarity=cosine,
        mse=mse,
        correlation=correlation,
    )


class DepthEvaluator:
    """Evaluator for depth estimation models."""

    def __init__(
        self,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        scale_invariant: bool = True,
    ):
        """
        Args:
            min_depth: Minimum valid depth
            max_depth: Maximum valid depth
            scale_invariant: Whether to use scale-invariant evaluation
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scale_invariant = scale_invariant

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader,
        device: str = "cuda",
        depth_head: Optional[nn.Module] = None,
    ) -> DepthMetrics:
        """Evaluate depth estimation model.

        Args:
            model: Feature extraction model
            dataloader: Data loader returning {'image': ..., 'depth': ...}
            device: Device to run evaluation on
            depth_head: Optional depth prediction head

        Returns:
            Aggregated DepthMetrics
        """
        model.eval()
        model = model.to(device)
        if depth_head is not None:
            depth_head.eval()
            depth_head = depth_head.to(device)

        all_metrics = []

        for batch in tqdm(dataloader, desc="Evaluating depth"):
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)

            # Get predictions
            features = model(images)

            if depth_head is not None:
                pred_depths = depth_head(features)
            else:
                # If no depth head, assume model outputs depth directly
                if isinstance(features, tuple):
                    pred_depths = features[0]
                else:
                    pred_depths = features

            # Squeeze channel dim if present: (B, 1, H, W) -> (B, H, W)
            if pred_depths.ndim == 4 and pred_depths.shape[1] == 1:
                pred_depths = pred_depths.squeeze(1)

            # Resize predictions to match target if needed
            if pred_depths.shape[-2:] != depths.shape[-2:]:
                pred_depths = F.interpolate(
                    pred_depths.unsqueeze(1) if pred_depths.ndim == 3 else pred_depths,
                    size=depths.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1)

            # Scale-invariant evaluation
            if self.scale_invariant:
                # Compute optimal scale
                valid_mask = (depths > self.min_depth) & (depths < self.max_depth)
                for i in range(pred_depths.shape[0]):
                    mask = valid_mask[i] if valid_mask.ndim == 3 else valid_mask
                    if mask.sum() > 0:
                        scale = (depths[i][mask] / pred_depths[i][mask].clamp(min=1e-8)).median()
                        pred_depths[i] = pred_depths[i] * scale

            # Compute metrics
            metrics = compute_depth_metrics(
                pred_depths, depths,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
            )
            all_metrics.append(metrics)

        # Aggregate metrics
        return self._aggregate_metrics(all_metrics)

    def _aggregate_metrics(self, metrics_list: List[DepthMetrics]) -> DepthMetrics:
        """Aggregate metrics from multiple batches."""
        n = len(metrics_list)
        if n == 0:
            return DepthMetrics(0, 0, 0, 0, 0, 0, 0)

        return DepthMetrics(
            rmse=sum(m.rmse for m in metrics_list) / n,
            rmse_log=sum(m.rmse_log for m in metrics_list) / n,
            abs_rel=sum(m.abs_rel for m in metrics_list) / n,
            sq_rel=sum(m.sq_rel for m in metrics_list) / n,
            delta1=sum(m.delta1 for m in metrics_list) / n,
            delta2=sum(m.delta2 for m in metrics_list) / n,
            delta3=sum(m.delta3 for m in metrics_list) / n,
        )


class SegmentationEvaluator:
    """Evaluator for semantic segmentation models."""

    def __init__(
        self,
        num_classes: int = 150,
        ignore_index: int = 255,
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Index to ignore in evaluation
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader,
        device: str = "cuda",
        seg_head: Optional[nn.Module] = None,
    ) -> SegmentationMetrics:
        """Evaluate segmentation model.

        Args:
            model: Feature extraction model
            dataloader: Data loader returning {'image': ..., 'mask': ...}
            device: Device to run evaluation on
            seg_head: Optional segmentation prediction head

        Returns:
            Aggregated SegmentationMetrics
        """
        model.eval()
        model = model.to(device)
        if seg_head is not None:
            seg_head.eval()
            seg_head = seg_head.to(device)

        # Accumulate confusion matrix
        conf_matrix = torch.zeros(
            self.num_classes, self.num_classes,
            dtype=torch.long, device=device
        )

        for batch in tqdm(dataloader, desc="Evaluating segmentation"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # Get predictions
            features = model(images)

            if seg_head is not None:
                pred = seg_head(features)
            else:
                if isinstance(features, tuple):
                    pred = features[0]
                else:
                    pred = features

            # Get class predictions
            if pred.ndim == 4:
                pred = pred.argmax(dim=1)

            # Resize predictions to match target if needed
            if pred.shape[-2:] != masks.shape[-2:]:
                pred = F.interpolate(
                    pred.unsqueeze(1).float(),
                    size=masks.shape[-2:],
                    mode='nearest',
                ).squeeze(1).long()

            # Update confusion matrix (vectorized)
            valid_mask = (masks != self.ignore_index) & (masks >= 0) & (masks < self.num_classes)
            pred_valid = pred[valid_mask]
            mask_valid = masks[valid_mask]
            pred_valid = pred_valid.clamp(0, self.num_classes - 1)
            indices = mask_valid * self.num_classes + pred_valid
            conf_matrix += torch.bincount(
                indices, minlength=self.num_classes * self.num_classes
            ).reshape(self.num_classes, self.num_classes)

        # Compute metrics from confusion matrix
        conf_matrix = conf_matrix.cpu().numpy()

        pixel_accuracy = np.diag(conf_matrix).sum() / conf_matrix.sum()
        class_accuracy = np.diag(conf_matrix) / conf_matrix.sum(axis=1).clip(min=1)
        valid_classes = conf_matrix.sum(axis=1) > 0
        mean_accuracy = class_accuracy[valid_classes].mean()

        intersection = np.diag(conf_matrix)
        union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix)
        iou = intersection / union.clip(min=1)
        mean_iou = iou[valid_classes].mean()

        freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
        fw_iou = (freq[valid_classes] * iou[valid_classes]).sum()

        per_class_iou = {i: iou[i] for i in range(self.num_classes) if valid_classes[i]}

        return SegmentationMetrics(
            pixel_accuracy=float(pixel_accuracy),
            mean_accuracy=float(mean_accuracy),
            mean_iou=float(mean_iou),
            fw_iou=float(fw_iou),
            per_class_iou=per_class_iou,
        )


class FeatureEvaluator:
    """Evaluator for comparing features between models."""

    @torch.no_grad()
    def compare_models(
        self,
        model1: nn.Module,
        model2: nn.Module,
        dataloader,
        device: str = "cuda",
        max_samples: int = 1000,
    ) -> FeatureSimilarityMetrics:
        """Compare feature similarity between two models.

        Args:
            model1: First model (e.g., original)
            model2: Second model (e.g., pruned)
            dataloader: Data loader
            device: Device to run on
            max_samples: Maximum number of samples to use

        Returns:
            Aggregated FeatureSimilarityMetrics
        """
        model1.eval()
        model2.eval()
        model1 = model1.to(device)
        model2 = model2.to(device)

        total_cosine = 0.0
        total_mse = 0.0
        total_correlation = 0.0
        n_samples = 0

        for batch in tqdm(dataloader, desc="Comparing features"):
            if isinstance(batch, dict):
                images = batch['image'].to(device)
            elif isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
            else:
                images = batch.to(device)

            # Get features from both models
            features1 = model1(images)
            features2 = model2(images)

            if isinstance(features1, tuple):
                features1 = features1[0]
            if isinstance(features2, tuple):
                features2 = features2[0]

            # Compute batch metrics
            metrics = compute_feature_similarity(features1, features2)
            total_cosine += metrics.cosine_similarity * images.shape[0]
            total_mse += metrics.mse * images.shape[0]
            total_correlation += metrics.correlation * images.shape[0]
            n_samples += images.shape[0]

            if n_samples >= max_samples:
                break

        return FeatureSimilarityMetrics(
            cosine_similarity=total_cosine / n_samples,
            mse=total_mse / n_samples,
            correlation=total_correlation / n_samples,
        )


def format_depth_metrics(metrics: DepthMetrics) -> str:
    """Format depth metrics as a readable string."""
    return (
        f"Depth Metrics:\n"
        f"  RMSE: {metrics.rmse:.4f}\n"
        f"  RMSE (log): {metrics.rmse_log:.4f}\n"
        f"  AbsRel: {metrics.abs_rel:.4f}\n"
        f"  SqRel: {metrics.sq_rel:.4f}\n"
        f"  delta1 (<1.25): {metrics.delta1*100:.2f}%\n"
        f"  delta2 (<1.25^2): {metrics.delta2*100:.2f}%\n"
        f"  delta3 (<1.25^3): {metrics.delta3*100:.2f}%"
    )


def format_segmentation_metrics(metrics: SegmentationMetrics) -> str:
    """Format segmentation metrics as a readable string."""
    return (
        f"Segmentation Metrics:\n"
        f"  Pixel Accuracy: {metrics.pixel_accuracy*100:.2f}%\n"
        f"  Mean Accuracy: {metrics.mean_accuracy*100:.2f}%\n"
        f"  Mean IoU: {metrics.mean_iou*100:.2f}%\n"
        f"  FW IoU: {metrics.fw_iou*100:.2f}%"
    )


def format_feature_metrics(metrics: FeatureSimilarityMetrics) -> str:
    """Format feature similarity metrics as a readable string."""
    return (
        f"Feature Similarity:\n"
        f"  Cosine Similarity: {metrics.cosine_similarity:.4f}\n"
        f"  MSE: {metrics.mse:.6f}\n"
        f"  Correlation: {metrics.correlation:.4f}"
    )
