"""
Activation Collection for Structured Pruning.

Provides forward hooks to capture activations from ViT/DeiT models,
with streaming statistics computation using Welford's algorithm.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator, Callable
from contextlib import contextmanager
import logging

from config.schemas import CollectorConfig, CovarianceMode


def _covariance_mode_value(mode):
    """Get the value of a CovarianceMode enum, handling different imports."""
    return mode.value if hasattr(mode, 'value') else mode

logger = logging.getLogger(__name__)


def detect_mlp_type(mlp) -> str:
    """Detect the type of MLP module.

    Returns:
        'standard' for fc1/fc2 MLPs,
        'swiglu_fused' for DINOv2 SwiGLUFFNFused (w12/w3),
        'swiglu_split' for LLaMA LlamaMLP (gate_proj/up_proj/down_proj),
        'unknown' otherwise.
    """
    if hasattr(mlp, 'fc1') and hasattr(mlp, 'fc2'):
        return 'standard'
    if hasattr(mlp, 'w12') and hasattr(mlp, 'w3'):
        return 'swiglu_fused'
    if hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj') and hasattr(mlp, 'down_proj'):
        return 'swiglu_split'
    return 'unknown'


def get_mlp_intermediate_dim(mlp) -> int:
    """Get the intermediate (hidden) dimension of an MLP.

    For SwiGLU variants, this is the dimension after gating (not the raw layer output).
    """
    mlp_type = detect_mlp_type(mlp)
    if mlp_type == 'standard':
        return mlp.fc1.out_features
    elif mlp_type == 'swiglu_fused':
        # w12 output is 2*hidden_dim (gate + up concatenated)
        return mlp.w12.out_features // 2
    elif mlp_type == 'swiglu_split':
        return mlp.gate_proj.out_features
    else:
        raise ValueError(f"Cannot determine intermediate dim for MLP type: {mlp_type}")


@dataclass
class LayerActivationStats:
    """Statistics for activations at a single layer.

    Attributes:
        layer_name: Name of the layer
        feature_dim: Dimensionality of features
        n_samples: Number of samples seen
        sum_x: Sum of activations (feature_dim,)
        sum_x2: Sum of squared activations (feature_dim,)
        covariance: Covariance matrix (feature_dim, feature_dim) or sketch
        active_count: Count of active elements per feature (for active rate)
        raw_activations: Raw activations if store_raw=True
        sum_q2k2: Sum of q^2 * k^2 per dimension for exact joint energy (feature_dim,)
    """
    layer_name: str
    feature_dim: int
    n_samples: int = 0
    sum_x: Optional[torch.Tensor] = None
    sum_x2: Optional[torch.Tensor] = None
    covariance: Optional[torch.Tensor] = None
    active_count: Optional[torch.Tensor] = None
    raw_activations: Optional[List[torch.Tensor]] = None
    sum_q2k2: Optional[torch.Tensor] = None  # For Q/K dimension exact joint energy
    _cov_accum: Optional[torch.Tensor] = None  # X.T @ X accumulator

    def __post_init__(self):
        if self.sum_x is None:
            self.sum_x = torch.zeros(self.feature_dim)
        if self.sum_x2 is None:
            self.sum_x2 = torch.zeros(self.feature_dim)
        if self.active_count is None:
            self.active_count = torch.zeros(self.feature_dim)

    @property
    def mean(self) -> torch.Tensor:
        """Compute mean from running sums."""
        if self.n_samples == 0:
            return torch.zeros(self.feature_dim)
        return self.sum_x / self.n_samples

    @property
    def variance(self) -> torch.Tensor:
        """Compute variance from running sums."""
        if self.n_samples == 0:
            return torch.zeros(self.feature_dim)
        mean = self.mean
        return self.sum_x2 / self.n_samples - mean ** 2

    @property
    def std(self) -> torch.Tensor:
        """Compute standard deviation."""
        return torch.sqrt(torch.clamp(self.variance, min=1e-10))

    @property
    def active_rate(self) -> torch.Tensor:
        """Compute active rate per feature."""
        if self.n_samples == 0:
            return torch.zeros(self.feature_dim)
        return self.active_count / self.n_samples


class ActivationCollector:
    """Collects activations from model layers using forward hooks.

    Supports streaming statistics computation for memory efficiency,
    with optional covariance computation (exact or sketched).
    """

    def __init__(
        self,
        model: nn.Module,
        config: CollectorConfig,
        device: str = "cuda",
    ):
        """Initialize the collector.

        Args:
            model: The model to collect activations from
            config: Collection configuration
            device: Device to store statistics on
        """
        self.model = model
        self.config = config
        self.device = device

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._stats: Dict[str, LayerActivationStats] = {}
        self._active_threshold: Dict[str, float] = {}
        self._sketch_matrices: Dict[str, torch.Tensor] = {}
        self._collecting = False

    def register_hooks(
        self,
        layer_names: List[str],
        model_structure: Optional[Dict] = None,
    ) -> None:
        """Register forward hooks on specified layers."""
        self.clear_hooks()

        if model_structure is None:
            model_structure = detect_model_structure(self.model)

        num_heads = model_structure.get('num_heads', 12)
        embed_dim = model_structure.get('embed_dim', 768)
        head_dim = model_structure.get('head_dim', embed_dim // num_heads if num_heads > 0 else 64)
        qk_dim = model_structure.get('qk_dim', head_dim)

        logger.debug(f"register_hooks: head_dim={head_dim}, qk_dim={qk_dim}, model_structure={model_structure}")

        for name in layer_names:
            module = self._get_module(name)
            if module is None:
                logger.warning(f"Module {name} not found, skipping")
                continue

            if name.endswith('.mlp'):
                hook = self._create_input_hook(name)
            elif name.endswith('.mlp.w3') or name.endswith('.mlp.down_proj'):
                hook = self._create_input_hook(name)
            elif name.endswith('.attn.qkv'):
                attn_name = name[:-4]
                hook = self._create_qk_dim_hook(attn_name, num_heads, qk_dim, head_dim)
            else:
                hook = self._create_output_hook(name)

            handle = module.register_forward_hook(hook)
            self._hooks.append(handle)
            logger.debug(f"Registered hook on {name}")

    def _get_module(self, name: str) -> Optional[nn.Module]:
        """Get a module by its name path."""
        parts = name.split('.')
        module = self.model

        # Handle compiled models
        if hasattr(module, '_orig_mod'):
            module = module._orig_mod

        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif hasattr(module, '__getitem__'):
                try:
                    module = module[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return module

    def _create_input_hook(self, layer_name: str) -> Callable:
        """Create a hook that captures module input."""
        def hook(module: nn.Module, inputs, output):
            if not self._collecting:
                return
            # inputs is a tuple, get the first element
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            self._process_activation(layer_name, x)
        return hook

    def _create_output_hook(self, layer_name: str) -> Callable:
        """Create a hook that captures module output."""
        def hook(module: nn.Module, inputs, output):
            if not self._collecting:
                return
            self._process_activation(layer_name, output)
        return hook

    def _create_qk_dim_hook(self, layer_name: str, num_heads: int, qk_dim: int, v_dim: int) -> Callable:
        """Create a hook that captures per-head per-dimension Q/K statistics.

        For Q/K dimension pruning, we need to collect statistics for each dimension
        within each head separately. This hooks the QKV output and splits Q/K.

        Handles both unpruned and pruned models:
        - Unpruned: QKV = [Q, K, V] where all have same dim (qk_dim == v_dim)
        - Pruned: QKV = [Q, K, V] where Q/K have qk_dim, V has v_dim

        Args:
            layer_name: Name of the attention layer (e.g., 'blocks.0.attn')
            num_heads: Number of attention heads
            qk_dim: Dimension per head for Q/K (may be reduced after pruning)
            v_dim: Dimension per head for V (original head_dim)
        """
        logger.debug(f"Creating QK dim hook for {layer_name}: num_heads={num_heads}, qk_dim={qk_dim}, v_dim={v_dim}")

        def hook(module: nn.Module, inputs, output):
            if not self._collecting:
                return

            # Output of qkv is (batch, seq_len, qkv_total)
            x = output.detach()
            batch_size, seq_len, qkv_total = x.shape

            # Calculate expected dimension based on whether model is pruned
            if qk_dim == v_dim:
                # Unpruned model: 3 * num_heads * head_dim
                expected_dim = 3 * num_heads * qk_dim
            else:
                # Pruned model: 2 * qk_dim * num_heads + v_dim * num_heads
                expected_dim = 2 * qk_dim * num_heads + v_dim * num_heads

            assert qkv_total == expected_dim, \
                f"Expected qkv_dim={expected_dim}, got {qkv_total}"

            # Split Q, K, V based on layout
            qk_total = qk_dim * num_heads

            if qk_dim == v_dim:
                # Unpruned: reshape to (batch, seq_len, 3, num_heads, head_dim)
                x = x.view(batch_size, seq_len, 3, num_heads, qk_dim)
                q = x[:, :, 0, :, :]  # (batch, seq_len, num_heads, qk_dim)
                k = x[:, :, 1, :, :]
            else:
                # Pruned: Q and K have qk_dim, V has v_dim
                q_flat = x[:, :, :qk_total]  # (batch, seq_len, qk_total)
                k_flat = x[:, :, qk_total:2*qk_total]
                # V is not needed for Q/K statistics

                q = q_flat.reshape(batch_size, seq_len, num_heads, qk_dim)
                k = k_flat.reshape(batch_size, seq_len, num_heads, qk_dim)

            # Token subsampling
            if self.config.subsample_tokens is not None:
                n_keep = min(self.config.subsample_tokens, seq_len)
                if self.config.keep_cls_token and n_keep > 1:
                    q_cls = q[:, 0:1, :, :]
                    k_cls = k[:, 0:1, :, :]
                    q_other = q[:, 1:, :, :]
                    k_other = k[:, 1:, :, :]
                    n_sample_others = n_keep - 1
                    if n_sample_others > 0 and q_other.shape[1] > 0:
                        indices = torch.randperm(q_other.shape[1])[:n_sample_others]
                        q = torch.cat([q_cls, q_other[:, indices, :, :]], dim=1)
                        k = torch.cat([k_cls, k_other[:, indices, :, :]], dim=1)
                    else:
                        q = q_cls
                        k = k_cls
                else:
                    indices = torch.randperm(seq_len)[:n_keep]
                    q = q[:, indices, :, :]
                    k = k[:, indices, :, :]

            # Process each head's Q and K dimensions separately
            for h in range(num_heads):
                # Q dims for this head: (batch * seq_len, qk_dim)
                q_h = q[:, :, h, :].reshape(-1, qk_dim)
                k_h = k[:, :, h, :].reshape(-1, qk_dim)

                # Store Q stats
                q_key = f"{layer_name}.q.head_{h}"
                self._process_activation_raw(q_key, q_h, is_head_stat=False)

                # Store K stats
                k_key = f"{layer_name}.k.head_{h}"
                self._process_activation_raw(k_key, k_h, is_head_stat=False)

                # Compute and accumulate q^2 * k^2 for exact joint energy score
                # This captures E[q_j^2 * k_j^2] per dimension
                qk_key = f"{layer_name}.qk.head_{h}"
                q2k2 = (q_h ** 2) * (k_h ** 2)  # (n_tokens, qk_dim)
                self._accumulate_q2k2(qk_key, q2k2, qk_dim)

        return hook

    def _accumulate_q2k2(
        self,
        key: str,
        q2k2: torch.Tensor,
        feature_dim: int,
    ) -> None:
        """Accumulate q^2 * k^2 statistics for exact joint energy score.

        Args:
            key: Stats key (e.g., 'blocks.0.attn.qk.head_0')
            q2k2: Tensor of q^2 * k^2 values, shape (n_tokens, feature_dim)
            feature_dim: Dimension of features (head_dim)
        """
        # Move to CPU float64 for stable accumulation
        q2k2_cpu = q2k2.double().cpu()
        n_new = q2k2_cpu.shape[0]

        # Initialize stats if needed
        if key not in self._stats:
            self._stats[key] = LayerActivationStats(
                layer_name=key,
                feature_dim=feature_dim,
            )
            self._stats[key].sum_q2k2 = torch.zeros(feature_dim, dtype=torch.float64)

        stats = self._stats[key]

        # Initialize sum_q2k2 if not present
        if stats.sum_q2k2 is None:
            stats.sum_q2k2 = torch.zeros(feature_dim, dtype=torch.float64)

        # Accumulate
        stats.sum_q2k2 += q2k2_cpu.sum(dim=0)
        stats.n_samples += n_new

    def _process_activation_raw(
        self,
        layer_name: str,
        x: torch.Tensor,
        is_head_stat: bool = False,
    ) -> None:
        """Process raw activation tensor for statistics.

        Args:
            layer_name: Name of the layer
            x: Activation tensor of shape (n_samples, feature_dim)
            is_head_stat: Whether this is a per-head statistic (for attention)
        """
        feature_dim = x.shape[-1]
        x_cpu = x.float().cpu()

        # Initialize stats if needed
        if layer_name not in self._stats:
            self._stats[layer_name] = LayerActivationStats(
                layer_name=layer_name,
                feature_dim=feature_dim,
            )
            if self.config.store_raw:
                self._stats[layer_name].raw_activations = []

            # Initialize covariance accumulator
            if _covariance_mode_value(self.config.covariance_mode) == 'exact':
                self._stats[layer_name]._cov_accum = torch.zeros(
                    feature_dim, feature_dim, device=self.device, dtype=torch.float64
                )

        stats = self._stats[layer_name]
        assert x_cpu.shape[-1] == stats.feature_dim, \
            f"Feature dim mismatch: {x_cpu.shape[-1]} vs {stats.feature_dim}"

        n_new = x_cpu.shape[0]

        # Update running sums
        stats.sum_x = stats.sum_x.to(x_cpu.device)
        stats.sum_x2 = stats.sum_x2.to(x_cpu.device)
        stats.active_count = stats.active_count.to(x_cpu.device)

        stats.sum_x += x_cpu.sum(dim=0)
        stats.sum_x2 += (x_cpu ** 2).sum(dim=0)

        # Compute RMS for active threshold
        rms = (x_cpu ** 2).mean().sqrt().item()
        threshold = 0.01 * rms
        stats.active_count += (x_cpu.abs() > threshold).float().sum(dim=0)

        stats.n_samples += n_new

        # Update covariance
        if _covariance_mode_value(self.config.covariance_mode) == 'exact':
            x_gpu = x.double().to(self.device)
            stats._cov_accum += x_gpu.T @ x_gpu

        # Store raw if requested
        if self.config.store_raw:
            stats.raw_activations.append(x_cpu)

    def _process_activation(self, layer_name: str, activation: torch.Tensor) -> None:
        """Process and accumulate activation statistics.

        Args:
            layer_name: Name of the layer
            activation: Activation tensor, shape (batch, seq_len, dim) or (batch, dim)
        """
        # Detach and move to CPU for accumulation
        x = activation.detach()

        # Handle different shapes
        if x.ndim == 3:
            batch_size, seq_len, feature_dim = x.shape
            # Token subsampling
            x = self._subsample_tokens(x)
            # Flatten to (n_tokens, feature_dim)
            x = x.reshape(-1, feature_dim)
        elif x.ndim == 2:
            feature_dim = x.shape[-1]
        else:
            raise ValueError(f"Unexpected activation shape: {x.shape}")

        # Initialize stats if needed
        if layer_name not in self._stats:
            self._stats[layer_name] = LayerActivationStats(
                layer_name=layer_name,
                feature_dim=feature_dim,
            )
            if self.config.store_raw:
                self._stats[layer_name].raw_activations = []

            # Initialize covariance accumulator if exact mode (on GPU for speed)
            if _covariance_mode_value(self.config.covariance_mode) == 'exact':
                self._stats[layer_name]._cov_accum = torch.zeros(
                    feature_dim, feature_dim, device=self.device, dtype=torch.float64
                )

            # Initialize sketch matrix if sketch mode
            if _covariance_mode_value(self.config.covariance_mode) == 'sketch':
                sketch_dim = self.config.sketch_dim
                self._sketch_matrices[layer_name] = torch.randn(
                    sketch_dim, feature_dim, device='cpu', dtype=torch.float32
                ) / (sketch_dim ** 0.5)

        stats = self._stats[layer_name]
        assert x.shape[-1] == stats.feature_dim, \
            f"Feature dim mismatch: {x.shape[-1]} vs {stats.feature_dim}"

        n_new = x.shape[0]
        x_cpu = x.float().cpu()

        # Update running sums (Welford's algorithm components)
        stats.sum_x = stats.sum_x.to(x_cpu.device)
        stats.sum_x2 = stats.sum_x2.to(x_cpu.device)
        stats.active_count = stats.active_count.to(x_cpu.device)

        stats.sum_x += x_cpu.sum(dim=0)
        stats.sum_x2 += (x_cpu ** 2).sum(dim=0)

        # Compute RMS for active threshold
        rms = (x_cpu ** 2).mean().sqrt().item()
        threshold = 0.01 * rms
        stats.active_count += (x_cpu.abs() > threshold).float().sum(dim=0)

        stats.n_samples += n_new

        # Update covariance (on GPU for speed)
        if _covariance_mode_value(self.config.covariance_mode) == 'exact':
            x_gpu = x.double()  # Keep on GPU
            stats._cov_accum += x_gpu.T @ x_gpu

        elif _covariance_mode_value(self.config.covariance_mode) == 'sketch':
            R = self._sketch_matrices[layer_name]
            # Sketch: R @ X.T gives (sketch_dim, n_new)
            sketch = R @ x_cpu.T  # (sketch_dim, n_new)
            if stats.covariance is None:
                stats.covariance = torch.zeros(
                    self.config.sketch_dim, self.config.sketch_dim,
                    device='cpu', dtype=torch.float64
                )
            stats.covariance += (sketch.double() @ sketch.T.double())

        # Store raw if requested
        if self.config.store_raw:
            stats.raw_activations.append(x_cpu)

    def _subsample_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample tokens from activation tensor.

        Args:
            x: Activation tensor of shape (batch, seq_len, dim)

        Returns:
            Subsampled tensor
        """
        if self.config.subsample_tokens is None:
            return x

        batch_size, seq_len, dim = x.shape
        n_keep = min(self.config.subsample_tokens, seq_len)

        if self.config.keep_cls_token and n_keep > 1:
            # Keep CLS token (index 0) and random sample others
            cls_token = x[:, 0:1, :]  # (batch, 1, dim)
            other_tokens = x[:, 1:, :]  # (batch, seq_len-1, dim)

            n_sample_others = n_keep - 1
            if n_sample_others > 0 and other_tokens.shape[1] > 0:
                indices = torch.randperm(other_tokens.shape[1])[:n_sample_others]
                sampled_others = other_tokens[:, indices, :]
                return torch.cat([cls_token, sampled_others], dim=1)
            else:
                return cls_token
        else:
            # Random sample all tokens
            indices = torch.randperm(seq_len)[:n_keep]
            return x[:, indices, :]

    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        logger.debug("Cleared all hooks")

    @contextmanager
    def collect(self) -> Iterator['ActivationCollector']:
        """Context manager for activation collection.

        Yields:
            Self, with collection enabled
        """
        self._collecting = True
        try:
            yield self
        finally:
            self._collecting = False
            # Finalize covariance computation
            self._finalize_covariance()

    def _finalize_covariance(self) -> None:
        """Finalize covariance computation after collection."""
        for name, stats in self._stats.items():
            if stats.n_samples == 0:
                continue

            if _covariance_mode_value(self.config.covariance_mode) == 'exact':
                if stats._cov_accum is not None:
                    # Compute full covariance: E[X.T @ X] - mu @ mu.T
                    # Keep computation on GPU, then move result to CPU
                    mean = stats.mean.double().to(stats._cov_accum.device)
                    cov = stats._cov_accum / stats.n_samples
                    cov -= torch.outer(mean, mean)
                    stats.covariance = cov.float().cpu()  # Move to CPU for storage
                    # Free the accumulator to reclaim GPU memory
                    del stats._cov_accum
                    stats._cov_accum = None

            elif _covariance_mode_value(self.config.covariance_mode) == 'sketch':
                if stats.covariance is not None:
                    stats.covariance = (stats.covariance / stats.n_samples).float()

    def get_layer_stats(self, layer_name: str) -> LayerActivationStats:
        """Get statistics for a specific layer.

        Args:
            layer_name: Name of the layer

        Returns:
            LayerActivationStats for the layer

        Raises:
            KeyError: If layer was not collected
        """
        if layer_name not in self._stats:
            raise KeyError(f"No stats collected for layer {layer_name}")
        return self._stats[layer_name]

    def get_all_stats(self) -> Dict[str, LayerActivationStats]:
        """Get statistics for all collected layers."""
        return self._stats.copy()

    def reset(self) -> None:
        """Reset all collected statistics."""
        self._stats.clear()
        self._active_threshold.clear()
        self._sketch_matrices.clear()
        logger.debug("Reset all statistics")

    def get_raw_activations(self, layer_name: str) -> torch.Tensor:
        """Get concatenated raw activations for a layer.

        Args:
            layer_name: Name of the layer

        Returns:
            Tensor of shape (n_samples, feature_dim)

        Raises:
            ValueError: If store_raw was False or layer not found
        """
        if not self.config.store_raw:
            raise ValueError("Raw activations not stored (set store_raw=True)")

        stats = self.get_layer_stats(layer_name)
        if stats.raw_activations is None or len(stats.raw_activations) == 0:
            raise ValueError(f"No raw activations for layer {layer_name}")

        return torch.cat(stats.raw_activations, dim=0)


def detect_model_structure(model: nn.Module) -> Dict:
    """Detect ViT/DeiT model structure.

    Args:
        model: The model to analyze

    Returns:
        Dictionary with model structure info:
        - num_blocks: Number of transformer blocks
        - embed_dim: Embedding dimension
        - mlp_hidden_dim: MLP hidden dimension
        - num_heads: Number of attention heads
        - head_dim: Dimension per attention head (for V, original dimension)
        - qk_dim: Q/K dimension per head (may differ from head_dim if pruned)
        - block_names: List of block names
    """
    # Handle compiled models
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod

    info = {
        'num_blocks': 0,
        'embed_dim': 0,
        'mlp_hidden_dim': 0,
        'num_heads': 0,
        'head_dim': 0,
        'qk_dim': 0,  # Q/K dimension (may differ from head_dim after pruning)
        'block_names': [],
    }

    # Try to find blocks
    if hasattr(model, 'blocks'):
        blocks = model.blocks
        info['num_blocks'] = len(blocks)

        if len(blocks) > 0:
            block = blocks[0]

            # Get MLP dimensions
            if hasattr(block, 'mlp'):
                mlp = block.mlp
                mlp_type = detect_mlp_type(mlp)
                info['mlp_type'] = mlp_type
                if mlp_type == 'standard':
                    info['embed_dim'] = mlp.fc1.in_features
                    info['mlp_hidden_dim'] = mlp.fc1.out_features
                elif mlp_type == 'swiglu_fused':
                    info['embed_dim'] = mlp.w12.in_features
                    info['mlp_hidden_dim'] = mlp.w12.out_features // 2
                elif mlp_type == 'swiglu_split':
                    info['embed_dim'] = mlp.gate_proj.in_features
                    info['mlp_hidden_dim'] = mlp.gate_proj.out_features

            # Get attention info
            if hasattr(block, 'attn'):
                attn = block.attn
                if hasattr(attn, 'num_heads'):
                    info['num_heads'] = attn.num_heads

                # Detect head_dim and qk_dim from actual layer dimensions
                # This is more robust than using attn.head_dim which may not be
                # restored when loading from checkpoint
                if hasattr(attn, 'qkv') and hasattr(attn, 'proj'):
                    qkv_out = attn.qkv.out_features
                    num_heads = info['num_heads']

                    # V dimension from proj layer (unchanged by Q/K pruning)
                    v_total = attn.proj.in_features
                    v_dim = v_total // num_heads

                    # Check if this is a pruned model
                    # Original layout: 3 * v_dim * num_heads
                    # Pruned layout: 2 * qk_dim * num_heads + v_dim * num_heads
                    expected_original = 3 * v_dim * num_heads

                    if qkv_out == expected_original:
                        # Unpruned model: Q/K/V all have same dimension
                        info['head_dim'] = v_dim
                        info['qk_dim'] = v_dim
                    else:
                        # Pruned model: Q/K have different dimension than V
                        # qkv_out = 2 * qk_dim * num_heads + v_dim * num_heads
                        # Solve for qk_dim:
                        qk_total = qkv_out - v_total
                        qk_dim = qk_total // (2 * num_heads)
                        info['head_dim'] = v_dim  # V dimension (original)
                        info['qk_dim'] = qk_dim  # Q/K dimension (pruned)

                elif hasattr(attn, 'qkv'):
                    # No proj layer, assume unpruned
                    qkv_out = attn.qkv.out_features
                    num_heads = info['num_heads']
                    info['head_dim'] = qkv_out // (3 * num_heads)
                    info['qk_dim'] = info['head_dim']
                elif info['num_heads'] > 0 and info['embed_dim'] > 0:
                    info['head_dim'] = info['embed_dim'] // info['num_heads']
                    info['qk_dim'] = info['head_dim']

        # Build block names
        info['block_names'] = [f'blocks.{i}' for i in range(info['num_blocks'])]

    return info


def get_hook_points(model: nn.Module, target: str = 'mlp') -> List[str]:
    """Get hook point names for the model.

    Args:
        model: The model to analyze
        target: 'mlp', 'attn', or 'both'
    """
    info = detect_model_structure(model)
    points = []

    mlp_type = info.get('mlp_type', 'standard')

    for i in range(info['num_blocks']):
        if target in ('mlp', 'both'):
            points.append(f'blocks.{i}.mlp')
            if mlp_type == 'standard':
                points.append(f'blocks.{i}.mlp.act')
            elif mlp_type == 'swiglu_fused':
                points.append(f'blocks.{i}.mlp.w3')
            elif mlp_type == 'swiglu_split':
                points.append(f'blocks.{i}.mlp.down_proj')

        if target in ('attn', 'both'):
            points.append(f'blocks.{i}.attn.qkv')

    return points
