"""
Ranking and Selection for Structured Pruning.

Provides ranking policies and structure selection based on redundancy metrics.
"""

import torch
import torch.nn as nn
from enum import Enum
from typing import Tuple, Optional
import logging

from .stats import RedundancyReport, QKDimReport
from .collect import detect_mlp_type
from config.schemas import RankerType

logger = logging.getLogger(__name__)


class RankingPolicy(Enum):
    """Ranking policy for pruning importance."""
    ENERGY = 'energy'           # Prune lowest E[x^2]
    ACTIVE = 'active'           # Prune lowest P(|x| > eps)
    ACTIVE_ENERGY = 'active_energy'  # Prune lowest energy * active_rate
    WEIGHT_MAGNITUDE = 'weight_magnitude'  # Prune by weight L2 norms (baseline)
    ENERGY_WEIGHTMAGNITUDE = 'energy_weightmagnitude'  # energy * weight_magnitude
    ACTIVE_WEIGHTMAGNITUDE = 'active_weightmagnitude'  # active_rate * weight_magnitude
    ENSEMBLE = 'ensemble'       # active_rate * energy * weight_magnitude


def compute_mlp_weight_importance(mlp: nn.Module) -> torch.Tensor:
    """Compute importance for MLP channels based on weight magnitude.

    Dispatches by MLP type:
    - standard: ||fc1[i]||_2 * ||fc2[:,i]||_2
    - swiglu_fused: (||w12_gate[i]|| + ||w12_up[i]||) * ||w3[:,i]||
    - swiglu_split: (||gate_proj[i]|| + ||up_proj[i]||) * ||down_proj[:,i]||

    Args:
        mlp: MLP module

    Returns:
        Importance tensor of shape (intermediate_dim,)
    """
    mlp_type = detect_mlp_type(mlp)

    if mlp_type == 'standard':
        fc1_norms = mlp.fc1.weight.data.norm(dim=1)  # (intermediate_dim,)
        fc2_norms = mlp.fc2.weight.data.norm(dim=0)  # (intermediate_dim,)
        return fc1_norms * fc2_norms

    elif mlp_type == 'swiglu_fused':
        # w12 has shape (2*hidden_dim, embed_dim): first half = gate, second half = up
        hidden_dim = mlp.w12.out_features // 2
        w12_weight = mlp.w12.weight.data
        gate_norms = w12_weight[:hidden_dim].norm(dim=1)  # (hidden_dim,)
        up_norms = w12_weight[hidden_dim:].norm(dim=1)    # (hidden_dim,)
        w3_norms = mlp.w3.weight.data.norm(dim=0)         # (hidden_dim,)
        return (gate_norms + up_norms) * w3_norms

    elif mlp_type == 'swiglu_split':
        gate_norms = mlp.gate_proj.weight.data.norm(dim=1)  # (hidden_dim,)
        up_norms = mlp.up_proj.weight.data.norm(dim=1)      # (hidden_dim,)
        down_norms = mlp.down_proj.weight.data.norm(dim=0)  # (hidden_dim,)
        return (gate_norms + up_norms) * down_norms

    else:
        raise ValueError(f"Unknown MLP type: {mlp_type}")


def compute_attention_head_weight_importance(attn: nn.Module) -> torch.Tensor:
    """Compute importance for attention heads based on weight magnitude.

    For each head h:
        importance[h] = (||Q_h||_F + ||K_h||_F + ||V_h||_F) * ||proj_h||_F

    Supports both fused QKV (DeiT/DINOv2 with attn.qkv + attn.proj) and
    separate projections (OPT with attn.q_proj, k_proj, v_proj, out_proj).

    Args:
        attn: Attention module

    Returns:
        Importance tensor of shape (num_heads,)
    """
    num_heads = attn.num_heads
    head_dim = attn.head_dim

    # Detect fused vs separate QKV
    has_fused_qkv = hasattr(attn, 'qkv')
    has_separate_qkv = hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj')

    if has_fused_qkv:
        qkv_weight = attn.qkv.weight.data
        proj_weight = attn.proj.weight.data
        device = qkv_weight.device

        importance = torch.zeros(num_heads, device=device)
        for h in range(num_heads):
            q_start = h * head_dim
            k_start = num_heads * head_dim + h * head_dim
            v_start = 2 * num_heads * head_dim + h * head_dim

            q_norm = qkv_weight[q_start:q_start + head_dim, :].norm()
            k_norm = qkv_weight[k_start:k_start + head_dim, :].norm()
            v_norm = qkv_weight[v_start:v_start + head_dim, :].norm()

            proj_norm = proj_weight[:, h * head_dim:(h + 1) * head_dim].norm()
            importance[h] = (q_norm + k_norm + v_norm) * proj_norm

    elif has_separate_qkv:
        q_weight = attn.q_proj.weight.data
        k_weight = attn.k_proj.weight.data
        v_weight = attn.v_proj.weight.data
        out_weight = attn.out_proj.weight.data
        device = q_weight.device

        importance = torch.zeros(num_heads, device=device)
        for h in range(num_heads):
            start = h * head_dim
            end = start + head_dim

            q_norm = q_weight[start:end, :].norm()
            k_norm = k_weight[start:end, :].norm()
            v_norm = v_weight[start:end, :].norm()
            out_norm = out_weight[:, start:end].norm()

            importance[h] = (q_norm + k_norm + v_norm) * out_norm
    else:
        raise ValueError("Attention module has neither fused qkv nor separate q_proj/k_proj/v_proj")

    return importance


class StructureRanker:
    """Ranks structures (channels/heads) for pruning based on importance."""

    def __init__(
        self,
        policy: RankingPolicy = RankingPolicy.ACTIVE_ENERGY,
        min_channels: int = 64,
        keep_topk_outliers: int = 0,
        kurtosis_threshold: float = 3.0,
        max_to_rms_threshold: float = 10.0,
        outlier_safe: bool = False,
    ):
        """Initialize the ranker.

        Args:
            policy: Ranking policy to use
            min_channels: Minimum number of channels to keep
            keep_topk_outliers: Number of high-outlier channels to protect
            kurtosis_threshold: Kurtosis above this is considered outlier-prone
            max_to_rms_threshold: Max/RMS ratio above this is considered outlier
            outlier_safe: If True, boost importance of outlier-prone channels
        """
        self.policy = policy
        self.min_channels = min_channels
        self.keep_topk_outliers = keep_topk_outliers
        self.kurtosis_threshold = kurtosis_threshold
        self.max_to_rms_threshold = max_to_rms_threshold
        self.outlier_safe = outlier_safe

    @classmethod
    def from_config(cls, ranker_type: RankerType, **kwargs) -> 'StructureRanker':
        """Create ranker from config enum.

        Args:
            ranker_type: RankerType enum value
            **kwargs: Additional arguments

        Returns:
            StructureRanker instance
        """
        policy = RankingPolicy(ranker_type.value)
        outlier_safe = kwargs.pop('outlier_safe', False)
        return cls(policy=policy, outlier_safe=outlier_safe, **kwargs)

    def compute_importance(self, report: RedundancyReport) -> torch.Tensor:
        """Compute importance scores for all features.

        Higher importance = less likely to be pruned.

        Args:
            report: Redundancy report for the layer

        Returns:
            Importance scores, shape (feature_dim,)
        """
        if self.policy == RankingPolicy.ENERGY:
            importance = report.energy

        elif self.policy == RankingPolicy.ACTIVE:
            importance = report.active_rate

        elif self.policy == RankingPolicy.ACTIVE_ENERGY:
            importance = report.active_energy_score

        else:
            raise ValueError(f"Unknown policy: {self.policy}")

        # Apply outlier-safe boosting as a post-step when enabled
        if self.outlier_safe:
            importance = importance.clone()
            if report.kurtosis is not None:
                outlier_mask = report.kurtosis > self.kurtosis_threshold
                importance[outlier_mask] *= 1e6
            if report.max_to_rms_ratio is not None:
                outlier_mask = report.max_to_rms_ratio > self.max_to_rms_threshold
                importance[outlier_mask] *= 1e6

        return importance

    def rank(self, report: RedundancyReport) -> torch.Tensor:
        """Rank features by prune priority (lowest importance first).

        Args:
            report: Redundancy report

        Returns:
            Indices sorted by prune priority (first = most prunable)
        """
        importance = self.compute_importance(report)
        # Sort ascending: lowest importance = highest prune priority
        sorted_indices = torch.argsort(importance)
        return sorted_indices

    def select_prune_indices(
        self,
        report: RedundancyReport,
        target_prune_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select indices to prune and keep.

        Args:
            report: Redundancy report
            target_prune_count: Number of features to prune

        Returns:
            Tuple of (prune_indices, survivor_indices)
        """
        feature_dim = report.feature_dim

        # Enforce minimum channels
        max_prune = feature_dim - self.min_channels
        if max_prune < 0:
            logger.warning(
                f"Layer {report.layer_name} has {feature_dim} features, "
                f"less than min_channels={self.min_channels}. Skipping pruning."
            )
            return torch.tensor([], dtype=torch.long), torch.arange(feature_dim)

        actual_prune_count = min(target_prune_count, max_prune)

        if actual_prune_count <= 0:
            return torch.tensor([], dtype=torch.long), torch.arange(feature_dim)

        # Get ranked indices
        ranked = self.rank(report)

        # Handle outlier protection when outlier_safe is enabled
        if self.outlier_safe and self.keep_topk_outliers > 0:
            # Find highest outlier channels
            protected_indices = self._get_outlier_indices(report)

            # Remove protected from prune candidates
            prune_mask = torch.ones(feature_dim, dtype=torch.bool)
            prune_mask[protected_indices] = False

            # Filter ranked indices
            ranked = ranked[prune_mask[ranked]]

            # Adjust prune count
            actual_prune_count = min(actual_prune_count, len(ranked))

        prune_indices = ranked[:actual_prune_count]
        survivor_indices = self._compute_survivors(feature_dim, prune_indices)

        logger.debug(
            f"Layer {report.layer_name}: pruning {len(prune_indices)}/{feature_dim} "
            f"({100*len(prune_indices)/feature_dim:.1f}%)"
        )

        return prune_indices, survivor_indices

    def _get_outlier_indices(self, report: RedundancyReport) -> torch.Tensor:
        """Get indices of outlier-prone channels to protect.

        Args:
            report: Redundancy report

        Returns:
            Indices to protect
        """
        protected = set()

        if report.kurtosis is not None:
            # Top K by kurtosis
            topk = min(self.keep_topk_outliers, len(report.kurtosis))
            top_kurtosis = torch.topk(report.kurtosis, topk).indices
            protected.update(top_kurtosis.tolist())

        if report.max_to_rms_ratio is not None:
            # Top K by max/rms ratio
            topk = min(self.keep_topk_outliers, len(report.max_to_rms_ratio))
            top_ratio = torch.topk(report.max_to_rms_ratio, topk).indices
            protected.update(top_ratio.tolist())

        return torch.tensor(list(protected), dtype=torch.long)

    def _compute_survivors(
        self,
        feature_dim: int,
        prune_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute survivor indices as complement of prune indices.

        Args:
            feature_dim: Total number of features
            prune_indices: Indices being pruned

        Returns:
            Survivor indices (sorted)
        """
        all_indices = set(range(feature_dim))
        prune_set = set(prune_indices.tolist())
        survivors = sorted(all_indices - prune_set)
        return torch.tensor(survivors, dtype=torch.long)

    def compute_importance_from_weights(
        self,
        module: nn.Module,
        module_type: str = 'mlp'
    ) -> torch.Tensor:
        """Compute importance from weight magnitudes.

        Args:
            module: MLP or Attention module
            module_type: 'mlp' or 'attn'

        Returns:
            Importance tensor
        """
        if module_type == 'mlp':
            return compute_mlp_weight_importance(module)
        elif module_type == 'attn':
            return compute_attention_head_weight_importance(module)
        else:
            raise ValueError(f"Unknown module_type: {module_type}")

    def select_for_sparsity(
        self,
        report: RedundancyReport,
        sparsity: float,
        module: Optional[nn.Module] = None,
        module_type: str = 'mlp',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select indices to achieve target sparsity.

        If policy is WEIGHT_MAGNITUDE, uses module weights for importance.
        Otherwise uses activation-based importance from report.

        Args:
            report: Redundancy report
            sparsity: Target sparsity (0.0 to 1.0)
            module: Optional module for weight-based ranking
            module_type: 'mlp' or 'attn' (used only for weight-based ranking)

        Returns:
            Tuple of (prune_indices, survivor_indices)
        """
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"Sparsity must be in [0, 1], got {sparsity}")

        feature_dim = report.feature_dim

        # Policies that require module weights
        if self.policy in (RankingPolicy.WEIGHT_MAGNITUDE, RankingPolicy.ENERGY_WEIGHTMAGNITUDE,
                           RankingPolicy.ACTIVE_WEIGHTMAGNITUDE, RankingPolicy.ENSEMBLE):
            if module is None:
                raise ValueError(f"module required for {self.policy.value} policy")
            weight_importance = self.compute_importance_from_weights(module, module_type)

            if self.policy == RankingPolicy.ENERGY_WEIGHTMAGNITUDE:
                importance = report.energy * weight_importance.to(report.energy.device)
            elif self.policy == RankingPolicy.ACTIVE_WEIGHTMAGNITUDE:
                importance = report.active_rate * weight_importance.to(report.active_rate.device)
            elif self.policy == RankingPolicy.ENSEMBLE:
                importance = report.active_rate * report.energy * weight_importance.to(report.energy.device)
            else:
                importance = weight_importance

            # Apply outlier-safe boosting when enabled
            if self.outlier_safe:
                importance = importance.clone()
                if report.kurtosis is not None:
                    outlier_mask = report.kurtosis > self.kurtosis_threshold
                    importance[outlier_mask] *= 1e6
                if report.max_to_rms_ratio is not None:
                    outlier_mask = report.max_to_rms_ratio > self.max_to_rms_threshold
                    importance[outlier_mask] *= 1e6

            # Compute target prune count
            target_prune_count = int(feature_dim * sparsity)

            # Enforce minimum channels (skip for attention heads where feature_dim is small)
            if module_type == 'attn':
                max_prune = feature_dim - 1  # keep at least 1 head
            else:
                max_prune = feature_dim - self.min_channels
            if max_prune < 0:
                logger.warning(
                    f"Layer {report.layer_name} has {feature_dim} features, "
                    f"less than min_channels={self.min_channels}. Skipping pruning."
                )
                return torch.tensor([], dtype=torch.long), torch.arange(feature_dim)

            actual_prune_count = min(target_prune_count, max_prune)

            if actual_prune_count <= 0:
                return torch.tensor([], dtype=torch.long), torch.arange(feature_dim)

            # Rank by importance (ascending = lowest importance first = most prunable)
            ranked = torch.argsort(importance)
            prune_indices = ranked[:actual_prune_count]
            survivor_indices = self._compute_survivors(feature_dim, prune_indices)

            logger.debug(
                f"Layer {report.layer_name}: pruning {len(prune_indices)}/{feature_dim} "
                f"({100*len(prune_indices)/feature_dim:.1f}%) [{self.policy.value}]"
            )

            return prune_indices, survivor_indices

        # For other policies, use the existing activation-based logic
        target_prune_count = int(report.feature_dim * sparsity)
        return self.select_prune_indices(report, target_prune_count)


class QKDimRanker:
    """Ranks Q/K dimensions within each attention head for pruning."""

    def __init__(self, min_qk_dim: int = 8):
        """Initialize the Q/K dimension ranker.

        Args:
            min_qk_dim: Minimum number of Q/K dimensions to keep per head
        """
        self.min_qk_dim = min_qk_dim

    def compute_importance(self, report: QKDimReport) -> torch.Tensor:
        """Compute importance scores for Q/K dimensions.

        Prefers exact score = E[q_j^2 * k_j^2] when available,
        otherwise falls back to proxy = sqrt(E[q_j^2] * E[k_j^2]).
        Higher importance = less likely to be pruned.

        Args:
            report: QKDimReport for the head

        Returns:
            Importance scores, shape (head_dim,)
        """
        if report.qk_energy is not None:
            # Use exact joint energy score
            return report.qk_energy
        # Fall back to proxy score
        return report.joint_score

    def rank(self, report: QKDimReport) -> torch.Tensor:
        """Rank dimensions by prune priority (lowest importance first).

        Args:
            report: QKDimReport for the head

        Returns:
            Indices sorted by prune priority (first = most prunable)
        """
        importance = self.compute_importance(report)
        # Sort ascending: lowest importance = highest prune priority
        sorted_indices = torch.argsort(importance)
        return sorted_indices

    def select_for_sparsity(
        self,
        report: QKDimReport,
        sparsity: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select dimensions to achieve target sparsity.

        Args:
            report: QKDimReport for the head
            sparsity: Target sparsity (0.0 to 1.0)

        Returns:
            Tuple of (prune_indices, survivor_indices)
        """
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"Sparsity must be in [0, 1], got {sparsity}")

        head_dim = report.head_dim
        target_prune_count = int(head_dim * sparsity)

        # Enforce minimum dimensions
        max_prune = head_dim - self.min_qk_dim
        if max_prune < 0:
            logger.warning(
                f"Head {report.head_idx} in {report.layer_name} has {head_dim} dims, "
                f"less than min_qk_dim={self.min_qk_dim}. Skipping pruning."
            )
            return torch.tensor([], dtype=torch.long), torch.arange(head_dim)

        actual_prune_count = min(target_prune_count, max_prune)

        if actual_prune_count <= 0:
            return torch.tensor([], dtype=torch.long), torch.arange(head_dim)

        # Get ranked indices
        ranked = self.rank(report)

        prune_indices = ranked[:actual_prune_count]
        survivor_indices = self._compute_survivors(head_dim, prune_indices)

        logger.debug(
            f"{report.layer_name} head {report.head_idx}: "
            f"pruning {len(prune_indices)}/{head_dim} dims "
            f"({100*len(prune_indices)/head_dim:.1f}%)"
        )

        return prune_indices, survivor_indices

    def _compute_survivors(
        self,
        feature_dim: int,
        prune_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute survivor indices as complement of prune indices.

        Args:
            feature_dim: Total number of features
            prune_indices: Indices being pruned

        Returns:
            Survivor indices (sorted)
        """
        all_indices = set(range(feature_dim))
        prune_set = set(prune_indices.tolist())
        survivors = sorted(all_indices - prune_set)
        return torch.tensor(survivors, dtype=torch.long)
