"""Enums and dataclasses for the pruning pipeline.

These types are consumed by `pruning/` (the algorithm core) and `model_prune/`
(the orchestration layer). Keep the shape stable — changing an attribute here
touches every call site.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class PruneTarget(Enum):
    MLP = "mlp"
    ATTN = "attn"
    BOTH = "both"


class ScheduleType(Enum):
    LAYERWISE = "layerwise"
    GLOBAL = "global"


class RankerType(Enum):
    ENERGY = "energy"
    ACTIVE = "active"
    ACTIVE_ENERGY = "active_energy"
    WEIGHT_MAGNITUDE = "weight_magnitude"
    ENERGY_WEIGHTMAGNITUDE = "energy_weightmagnitude"
    ACTIVE_WEIGHTMAGNITUDE = "active_weightmagnitude"
    ENSEMBLE = "ensemble"


class CovarianceMode(Enum):
    EXACT = "exact"
    SKETCH = "sketch"


@dataclass
class CollectorConfig:
    target: PruneTarget = PruneTarget.MLP
    subsample_tokens: Optional[int] = None
    covariance_mode: CovarianceMode = CovarianceMode.EXACT
    store_raw: bool = False
    keep_cls_token: bool = True
    sketch_dim: int = 256
    batch_size: int = 32


@dataclass
class PruningConfig:
    target: PruneTarget = PruneTarget.MLP
    schedule: ScheduleType = ScheduleType.LAYERWISE
    sparsity: float = 0.3
    ranker: RankerType = RankerType.ACTIVE_ENERGY
    lambda_reg: float = 1e-3
    auto_shrinkage: bool = True
    min_channels: int = 64
    min_heads: int = 1
    min_qk_dim: int = 8
    qk_sparsity: float = 0.3
    keep_topk_outliers: int = 0


@dataclass
class RunnerConfig:
    device: str = "cuda"
    output_dir: Path = field(default_factory=lambda: Path("logs"))
    save_pruned_path: Optional[Path] = None
    results_log: Optional[Path] = None
    calib_samples: int = 1024
    dtype: str = "float32"
    seed: int = 42


@dataclass
class FullConfig:
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
