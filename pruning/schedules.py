"""
Pruning Schedules.

Layerwise (one layer at a time, recalibrate between) and global (one-shot, all
layers in a single pass).
"""

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Dict, Optional
import logging

from config.schemas import ScheduleType, PruningConfig, PruneTarget

logger = logging.getLogger(__name__)


def _is_attention_layer(layer_name: str, target: PruneTarget) -> bool:
    target_value = target.value if hasattr(target, 'value') else target
    if target_value == 'attn':
        return True
    elif target_value == 'mlp':
        return False
    else:
        return layer_name.endswith('.attn')


def _get_min_keep(
    layer_name: str,
    target: PruneTarget,
    min_channels: int,
    min_qk_dim: int = 8,
) -> int:
    if _is_attention_layer(layer_name, target):
        return min_qk_dim
    return min_channels


@dataclass
class PruneStep:
    layer_name: str
    prune_indices: torch.Tensor
    sparsity: float
    round_num: int = 0
    step_num: int = 0


class PruneSchedule(ABC):
    @abstractmethod
    def iterate(
        self,
        layer_dims: Dict[str, int],
        total_sparsity: float,
    ) -> Iterator[PruneStep]:
        pass

    @abstractmethod
    def requires_recalibration(self) -> bool:
        pass


class LayerwiseSchedule(PruneSchedule):
    """Prune one layer at a time, recalibrating between layers."""

    def __init__(
        self,
        layer_order: Optional[List[str]] = None,
        min_channels: int = 64,
        min_qk_dim: int = 8,
        target: PruneTarget = PruneTarget.MLP,
    ):
        self.layer_order = layer_order
        self.min_channels = min_channels
        self.min_qk_dim = min_qk_dim
        self.target = target

    def iterate(
        self,
        layer_dims: Dict[str, int],
        total_sparsity: float,
    ) -> Iterator[PruneStep]:
        order = self.layer_order or list(layer_dims.keys())

        for step_num, layer_name in enumerate(order):
            if layer_name not in layer_dims:
                logger.warning(f"Layer {layer_name} not found, skipping")
                continue

            dim = layer_dims[layer_name]
            min_keep = _get_min_keep(
                layer_name, self.target,
                self.min_channels, self.min_qk_dim,
            )
            n_prune = int(dim * total_sparsity)
            n_prune = min(n_prune, dim - min_keep)
            n_prune = max(0, n_prune)

            if n_prune > 0:
                prune_indices = torch.arange(n_prune)
                yield PruneStep(
                    layer_name=layer_name,
                    prune_indices=prune_indices,
                    sparsity=n_prune / dim,
                    round_num=step_num,
                    step_num=0,
                )

    def requires_recalibration(self) -> bool:
        return True


class GlobalSchedule(PruneSchedule):
    """One-shot global pruning: every layer pruned once to target sparsity."""

    def __init__(
        self,
        min_channels: int = 64,
        min_qk_dim: int = 8,
        target: PruneTarget = PruneTarget.MLP,
    ):
        self.min_channels = min_channels
        self.min_qk_dim = min_qk_dim
        self.target = target

    def iterate(
        self,
        layer_dims: Dict[str, int],
        total_sparsity: float,
    ) -> Iterator[PruneStep]:
        for step_num, (layer_name, dim) in enumerate(layer_dims.items()):
            min_keep = _get_min_keep(
                layer_name, self.target,
                self.min_channels, self.min_qk_dim,
            )
            n_prune = int(dim * total_sparsity)
            n_prune = min(n_prune, dim - min_keep)
            n_prune = max(0, n_prune)

            if n_prune > 0:
                yield PruneStep(
                    layer_name=layer_name,
                    prune_indices=torch.arange(n_prune),
                    sparsity=n_prune / dim,
                    round_num=0,
                    step_num=step_num,
                )

    def requires_recalibration(self) -> bool:
        return False


def create_schedule(config: PruningConfig) -> PruneSchedule:
    schedule_value = config.schedule.value if hasattr(config.schedule, 'value') else config.schedule

    if schedule_value == 'layerwise':
        return LayerwiseSchedule(
            layer_order=None,
            min_channels=config.min_channels,
            min_qk_dim=config.min_qk_dim,
            target=config.target,
        )
    elif schedule_value == 'global':
        return GlobalSchedule(
            min_channels=config.min_channels,
            min_qk_dim=config.min_qk_dim,
            target=config.target,
        )
    else:
        raise ValueError(f"Unknown schedule type: {config.schedule}")
