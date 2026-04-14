"""
Pruning Schedules.

Defines pruning schedules: layerwise and global (with optional multi-round).
"""

import math
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Dict, Optional
import logging

from config.schemas import ScheduleType, PruningConfig, PruneTarget, AttentionPruneMode

logger = logging.getLogger(__name__)


def _is_attention_layer(layer_name: str, target: PruneTarget) -> bool:
    """Check if a layer is an attention layer based on name and target.

    Args:
        layer_name: Name of the layer
        target: Pruning target type

    Returns:
        True if this is an attention layer
    """
    # Use value comparison to handle different enum imports
    target_value = target.value if hasattr(target, 'value') else target

    if target_value == 'attn':
        return True
    elif target_value == 'mlp':
        return False
    else:  # 'both'
        return layer_name.endswith('.attn')


def _get_min_keep(
    layer_name: str,
    target: PruneTarget,
    min_channels: int,
    min_heads: int,
    min_qk_dim: int = 8,
    attn_prune_mode: AttentionPruneMode = AttentionPruneMode.HEAD,
) -> int:
    """Get minimum number of features to keep for a layer.

    Args:
        layer_name: Name of the layer
        target: Pruning target type
        min_channels: Minimum MLP channels to keep
        min_heads: Minimum attention heads to keep (for head pruning)
        min_qk_dim: Minimum Q/K dimensions to keep (for dim pruning)
        attn_prune_mode: Mode for attention pruning

    Returns:
        Minimum number of features to keep
    """
    if _is_attention_layer(layer_name, target):
        # Use value comparison to handle different enum imports
        mode_value = attn_prune_mode.value if hasattr(attn_prune_mode, 'value') else attn_prune_mode
        if mode_value == 'dim-logit':
            return min_qk_dim
        else:
            return min_heads
    else:
        return min_channels


@dataclass
class PruneStep:
    """A single pruning step.

    Attributes:
        layer_name: Name of layer to prune
        prune_indices: Indices to prune
        sparsity: Target sparsity for this step
        round_num: Round number (for iterative)
        step_num: Step number within round
    """
    layer_name: str
    prune_indices: torch.Tensor
    sparsity: float
    round_num: int = 0
    step_num: int = 0


class PruneSchedule(ABC):
    """Abstract base class for pruning schedules."""

    @abstractmethod
    def iterate(
        self,
        layer_dims: Dict[str, int],
        total_sparsity: float,
    ) -> Iterator[PruneStep]:
        """Iterate over pruning steps.

        Args:
            layer_dims: Dict of layer_name -> feature dimension
            total_sparsity: Target sparsity

        Yields:
            PruneStep for each pruning operation
        """
        pass

    @abstractmethod
    def requires_recalibration(self) -> bool:
        """Whether this schedule requires recalibration between steps."""
        pass


class LayerwiseSchedule(PruneSchedule):
    """Layerwise pruning schedule.

    Prunes one layer at a time, allowing recalibration between layers.
    """

    def __init__(
        self,
        layer_order: Optional[List[str]] = None,
        min_channels: int = 64,
        min_heads: int = 1,
        min_qk_dim: int = 8,
        target: PruneTarget = PruneTarget.MLP,
        attn_prune_mode: AttentionPruneMode = AttentionPruneMode.HEAD,
    ):
        """Initialize layerwise schedule.

        Args:
            layer_order: Order to prune layers (None = default order)
            min_channels: Minimum channels to keep per layer (for MLP)
            min_heads: Minimum heads to keep per layer (for attention head pruning)
            min_qk_dim: Minimum Q/K dims to keep per head (for attention dim pruning)
            target: Pruning target type
            attn_prune_mode: Mode for attention pruning
        """
        self.layer_order = layer_order
        self.min_channels = min_channels
        self.min_heads = min_heads
        self.min_qk_dim = min_qk_dim
        self.target = target
        self.attn_prune_mode = attn_prune_mode

    def iterate(
        self,
        layer_dims: Dict[str, int],
        total_sparsity: float,
    ) -> Iterator[PruneStep]:
        """Iterate over layers one at a time.

        Args:
            layer_dims: Dict of layer_name -> feature dimension
            total_sparsity: Target sparsity per layer

        Yields:
            PruneStep for each layer
        """
        order = self.layer_order or list(layer_dims.keys())

        for step_num, layer_name in enumerate(order):
            if layer_name not in layer_dims:
                logger.warning(f"Layer {layer_name} not found, skipping")
                continue

            dim = layer_dims[layer_name]
            # Use appropriate min_keep based on layer type and attention mode
            min_keep = _get_min_keep(
                layer_name, self.target,
                self.min_channels, self.min_heads, self.min_qk_dim,
                self.attn_prune_mode
            )
            n_prune = int(dim * total_sparsity)
            n_prune = min(n_prune, dim - min_keep)
            n_prune = max(0, n_prune)

            if n_prune > 0:
                # Placeholder indices - actual selection done by ranker
                prune_indices = torch.arange(n_prune)

                # Each layer gets its own round_num to trigger recalibration
                # after each layer is pruned (activations change for subsequent layers)
                yield PruneStep(
                    layer_name=layer_name,
                    prune_indices=prune_indices,
                    sparsity=n_prune / dim,
                    round_num=step_num,  # Different round for each layer
                    step_num=0,  # Reset step within each "round"
                )

    def requires_recalibration(self) -> bool:
        return True


class GlobalSchedule(PruneSchedule):
    """Global pruning schedule.

    Prunes all layers simultaneously to target sparsity.
    Supports multi-round pruning with recalibration between rounds.
    Uses fixed delta per round, automatically calculating rounds needed.
    If delta_per_round is None, prunes in a single round (one-shot).
    """

    def __init__(
        self,
        min_channels: int = 64,
        min_heads: int = 1,
        min_qk_dim: int = 8,
        delta_per_round: Optional[float] = None,
        target: PruneTarget = PruneTarget.MLP,
        attn_prune_mode: AttentionPruneMode = AttentionPruneMode.HEAD,
    ):
        """Initialize global schedule.

        Args:
            min_channels: Minimum channels per layer (for MLP)
            min_heads: Minimum heads per layer (for attention head pruning)
            min_qk_dim: Minimum Q/K dims per head (for attention dim pruning)
            delta_per_round: Fixed sparsity increment per round (e.g., 0.1 = 10% per round).
                If None, prunes in a single round to reach target sparsity.
            target: Pruning target type
            attn_prune_mode: Mode for attention pruning
        """
        self.min_channels = min_channels
        self.min_heads = min_heads
        self.min_qk_dim = min_qk_dim
        self.delta_per_round = delta_per_round
        self.target = target
        self.attn_prune_mode = attn_prune_mode

    def iterate(
        self,
        layer_dims: Dict[str, int],
        total_sparsity: float,
    ) -> Iterator[PruneStep]:
        """Iterate over global pruning steps.

        Args:
            layer_dims: Dict of layer_name -> feature dimension
            total_sparsity: Target sparsity

        Yields:
            PruneStep for each layer (all in same round)
        """
        # Store original dimensions (layer_dims gets updated by runner)
        original_dims = dict(layer_dims)

        # Calculate number of rounds needed
        # If delta is None, single round (one-shot)
        if self.delta_per_round is None or self.delta_per_round >= total_sparsity:
            num_rounds = 1
        else:
            num_rounds = math.ceil(total_sparsity / self.delta_per_round)

        cumulative_sparsity = 0.0

        for round_num in range(num_rounds):
            # Last round (or single round): prune remainder to reach exact target
            if round_num == num_rounds - 1 or self.delta_per_round is None:
                round_delta = total_sparsity - cumulative_sparsity
            else:
                round_delta = self.delta_per_round

            cumulative_sparsity += round_delta

            for step_num, layer_name in enumerate(original_dims.keys()):
                orig_dim = original_dims[layer_name]
                current_dim = layer_dims[layer_name]

                # Use appropriate min_keep based on layer type and attention mode
                min_keep = _get_min_keep(
                    layer_name, self.target,
                    self.min_channels, self.min_heads, self.min_qk_dim,
                    self.attn_prune_mode
                )

                # Calculate target pruned count from ORIGINAL dimension
                target_pruned = int(orig_dim * cumulative_sparsity)
                target_pruned = min(target_pruned, orig_dim - min_keep)

                # How many already pruned = original - current
                already_pruned = orig_dim - current_dim

                # How many more to prune this round
                n_prune = max(0, target_pruned - already_pruned)

                if n_prune > 0:
                    prune_indices = torch.arange(n_prune)

                    yield PruneStep(
                        layer_name=layer_name,
                        prune_indices=prune_indices,
                        sparsity=n_prune / current_dim,
                        round_num=round_num,
                        step_num=step_num,
                    )

    def requires_recalibration(self) -> bool:
        # Multi-round requires recalibration; single-round (delta is None) does not
        return self.delta_per_round is not None


def create_schedule(config: PruningConfig) -> PruneSchedule:
    """Create a pruning schedule from config.

    Args:
        config: Pruning configuration

    Returns:
        Appropriate PruneSchedule instance
    """
    # Compare by value to handle different enum imports
    schedule_value = config.schedule.value if hasattr(config.schedule, 'value') else config.schedule

    if schedule_value == 'layerwise':
        return LayerwiseSchedule(
            layer_order=None,
            min_channels=config.min_channels,
            min_heads=config.min_heads,
            min_qk_dim=config.min_qk_dim,
            target=config.target,
            attn_prune_mode=config.attn_prune_mode,
        )
    elif schedule_value == 'global':
        return GlobalSchedule(
            min_channels=config.min_channels,
            min_heads=config.min_heads,
            min_qk_dim=config.min_qk_dim,
            delta_per_round=config.delta_per_round,
            target=config.target,
            attn_prune_mode=config.attn_prune_mode,
        )
    else:
        raise ValueError(f"Unknown schedule type: {config.schedule}")
