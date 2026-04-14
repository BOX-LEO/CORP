"""
Structured Pruning Application.

Applies structural pruning to model layers with shape validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, TYPE_CHECKING
import logging
import copy
import types

from .compensate import CompensationResult, QKDimCompensationResult, QKDimCompensator
from .collect import detect_mlp_type

logger = logging.getLogger(__name__)


def _make_qk_dim_pruned_attention_forward(attn, qk_dim: int, v_dim: int, num_heads: int):
    """Create a patched forward method for Q/K dimension pruned attention.

    For dim-logit pruning, Q/K have reduced dimensions while V keeps full dimensions.
    QKV layout: [Q_h0(qk_dim), ..., K_h0(qk_dim), ..., V_h0(v_dim), ...]

    Args:
        attn: The attention module
        qk_dim: Dimension per head for Q/K (reduced)
        v_dim: Dimension per head for V (original head_dim)
        num_heads: Number of attention heads
    """
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, N, C = x.shape

        # Get QKV output
        qkv_out = self.qkv(x)  # (B, N, 2 * qk_dim * num_heads + v_dim * num_heads)

        # Split into Q, K, V sections
        qk_total = qk_dim * num_heads
        v_total = v_dim * num_heads

        q_flat = qkv_out[:, :, :qk_total]
        k_flat = qkv_out[:, :, qk_total:2*qk_total]
        v_flat = qkv_out[:, :, 2*qk_total:]

        # Reshape to (B, N, num_heads, dim) then (B, num_heads, N, dim)
        q = q_flat.reshape(B, N, num_heads, qk_dim).permute(0, 2, 1, 3)
        k = k_flat.reshape(B, N, num_heads, qk_dim).permute(0, 2, 1, 3)
        v = v_flat.reshape(B, N, num_heads, v_dim).permute(0, 2, 1, 3)

        # Apply normalization if present
        if hasattr(self, 'q_norm') and self.q_norm is not None:
            q = self.q_norm(q)
        if hasattr(self, 'k_norm') and self.k_norm is not None:
            k = self.k_norm(k)

        # Attention with scale based on reduced Q/K dimension
        scale = qk_dim ** -0.5

        if hasattr(self, 'fused_attn') and self.fused_attn:
            # Note: scaled_dot_product_attention expects same dim for Q/K/V
            # We need to use manual attention for different dims
            pass

        # Manual attention computation (required for different Q/K vs V dims)
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        attn_weights = attn_weights.softmax(dim=-1)
        if hasattr(self, 'attn_drop') and callable(self.attn_drop):
            attn_weights = self.attn_drop(attn_weights)

        # attn_weights: (B, num_heads, N, N)
        # v: (B, num_heads, N, v_dim)
        x = attn_weights @ v  # (B, num_heads, N, v_dim)

        # Reshape back
        x = x.transpose(1, 2).reshape(B, N, num_heads * v_dim)
        x = self.proj(x)
        if hasattr(self, 'proj_drop'):
            x = self.proj_drop(x)
        return x

    # Store dimensions for reference
    attn._qk_dim = qk_dim
    attn._v_dim = v_dim

    return types.MethodType(forward, attn)


class MaskApplier:
    """Applies structured pruning masks to model layers.

    Handles the coupled shapes between layers (e.g., fc1.out == fc2.in)
    and creates new layers with reduced dimensions.
    """

    def __init__(self, validate_shapes: bool = True):
        """Initialize the mask applier.

        Args:
            validate_shapes: Whether to validate shape coupling
        """
        self.validate_shapes = validate_shapes

    @staticmethod
    def validate_shape_coupling(fc1: nn.Linear, fc2: nn.Linear) -> None:
        """Validate that fc1 output matches fc2 input.

        Args:
            fc1: First linear layer
            fc2: Second linear layer

        Raises:
            AssertionError: If shapes don't match
        """
        assert fc1.out_features == fc2.in_features, (
            f"Shape mismatch: fc1.out_features={fc1.out_features} != "
            f"fc2.in_features={fc2.in_features}"
        )

    def prune_ffn_intermediate(
        self,
        mlp: nn.Module,
        prune_indices: torch.Tensor,
        compensation: Optional[CompensationResult] = None,
    ) -> nn.Module:
        """Prune FFN intermediate dimension.

        Dispatches to the appropriate helper based on MLP type.

        Args:
            mlp: MLP module
            prune_indices: Indices of intermediate features to prune
            compensation: Optional compensation to fold into down projection

        Returns:
            Modified MLP module
        """
        mlp_type = detect_mlp_type(mlp)

        if mlp_type == 'standard':
            return self._prune_standard_mlp(mlp, prune_indices, compensation)
        elif mlp_type == 'swiglu_fused':
            return self._prune_swiglu_fused_mlp(mlp, prune_indices, compensation)
        elif mlp_type == 'swiglu_split':
            return self._prune_swiglu_split_mlp(mlp, prune_indices, compensation)
        else:
            raise ValueError(f"Unknown MLP type for pruning: {mlp_type}")

    def _prune_standard_mlp(
        self,
        mlp: nn.Module,
        prune_indices: torch.Tensor,
        compensation: Optional[CompensationResult] = None,
    ) -> nn.Module:
        """Prune standard fc1 -> act -> fc2 MLP."""
        fc1 = mlp.fc1
        fc2 = mlp.fc2

        if self.validate_shapes:
            self.validate_shape_coupling(fc1, fc2)

        n_prune = len(prune_indices)
        if n_prune == 0:
            logger.debug("No pruning needed, returning original MLP")
            return mlp

        all_indices = set(range(fc1.out_features))
        prune_set = set(prune_indices.tolist())
        survivor_indices = torch.tensor(
            sorted(all_indices - prune_set), dtype=torch.long
        )

        n_survivors = len(survivor_indices)
        logger.debug(
            f"Pruning FFN: {fc1.out_features} -> {n_survivors} "
            f"({n_prune} pruned, {100*n_prune/fc1.out_features:.1f}%)"
        )

        new_fc1 = nn.Linear(
            fc1.in_features, n_survivors,
            bias=fc1.bias is not None,
            device=fc1.weight.device, dtype=fc1.weight.dtype,
        )
        with torch.no_grad():
            new_fc1.weight.copy_(fc1.weight[survivor_indices, :])
            if fc1.bias is not None:
                new_fc1.bias.copy_(fc1.bias[survivor_indices])

        new_fc2 = nn.Linear(
            n_survivors, fc2.out_features,
            bias=fc2.bias is not None,
            device=fc2.weight.device, dtype=fc2.weight.dtype,
        )

        if compensation is not None:
            W_tilde, b_tilde = self._fold_compensation_fc2(
                fc2, compensation, survivor_indices
            )
            with torch.no_grad():
                new_fc2.weight.copy_(W_tilde)
                if new_fc2.bias is not None:
                    new_fc2.bias.copy_(b_tilde)
        else:
            with torch.no_grad():
                new_fc2.weight.copy_(fc2.weight[:, survivor_indices])
                if fc2.bias is not None:
                    new_fc2.bias.copy_(fc2.bias)

        mlp.fc1 = new_fc1
        mlp.fc2 = new_fc2

        if self.validate_shapes:
            self.validate_shape_coupling(mlp.fc1, mlp.fc2)

        return mlp

    def _prune_swiglu_fused_mlp(
        self,
        mlp: nn.Module,
        prune_indices: torch.Tensor,
        compensation: Optional[CompensationResult] = None,
    ) -> nn.Module:
        """Prune SwiGLU fused MLP (DINOv2 SwiGLUFFNFused with w12/w3).

        w12 has shape (2*hidden_dim, embed_dim) with layout [gate_rows, up_rows].
        w3 has shape (embed_dim, hidden_dim).
        Forward: x1, x2 = w12(x).chunk(2, -1); hidden = silu(x1) * x2; out = w3(hidden)
        """
        hidden_dim = mlp.w12.out_features // 2

        n_prune = len(prune_indices)
        if n_prune == 0:
            logger.debug("No pruning needed, returning original MLP")
            return mlp

        all_indices = set(range(hidden_dim))
        prune_set = set(prune_indices.tolist())
        survivor_indices = torch.tensor(
            sorted(all_indices - prune_set), dtype=torch.long
        )
        n_survivors = len(survivor_indices)

        logger.debug(
            f"Pruning SwiGLU fused FFN: {hidden_dim} -> {n_survivors} "
            f"({n_prune} pruned, {100*n_prune/hidden_dim:.1f}%)"
        )

        # w12: select survivor rows from gate half and up half, then concatenate
        # Gate rows are [0..hidden_dim), up rows are [hidden_dim..2*hidden_dim)
        gate_survivor_rows = survivor_indices
        up_survivor_rows = survivor_indices + hidden_dim
        w12_survivor_rows = torch.cat([gate_survivor_rows, up_survivor_rows])

        w12 = mlp.w12
        new_w12 = nn.Linear(
            w12.in_features, 2 * n_survivors,
            bias=w12.bias is not None,
            device=w12.weight.device, dtype=w12.weight.dtype,
        )
        with torch.no_grad():
            new_w12.weight.copy_(w12.weight[w12_survivor_rows, :])
            if w12.bias is not None:
                new_w12.bias.copy_(w12.bias[w12_survivor_rows])

        # w3 (down projection): keep survivor columns + fold compensation
        w3 = mlp.w3
        new_w3 = nn.Linear(
            n_survivors, w3.out_features,
            bias=w3.bias is not None,
            device=w3.weight.device, dtype=w3.weight.dtype,
        )

        if compensation is not None:
            W_tilde, b_tilde = self._fold_compensation_fc2(
                w3, compensation, survivor_indices
            )
            with torch.no_grad():
                new_w3.weight.copy_(W_tilde)
                if new_w3.bias is not None:
                    new_w3.bias.copy_(b_tilde)
        else:
            with torch.no_grad():
                new_w3.weight.copy_(w3.weight[:, survivor_indices])
                if w3.bias is not None:
                    new_w3.bias.copy_(w3.bias)

        mlp.w12 = new_w12
        mlp.w3 = new_w3

        return mlp

    def _prune_swiglu_split_mlp(
        self,
        mlp: nn.Module,
        prune_indices: torch.Tensor,
        compensation: Optional[CompensationResult] = None,
    ) -> nn.Module:
        """Prune SwiGLU split MLP (LLaMA with gate_proj/up_proj/down_proj).

        Forward: hidden = silu(gate_proj(x)) * up_proj(x); out = down_proj(hidden)
        """
        hidden_dim = mlp.gate_proj.out_features

        n_prune = len(prune_indices)
        if n_prune == 0:
            logger.debug("No pruning needed, returning original MLP")
            return mlp

        all_indices = set(range(hidden_dim))
        prune_set = set(prune_indices.tolist())
        survivor_indices = torch.tensor(
            sorted(all_indices - prune_set), dtype=torch.long
        )
        n_survivors = len(survivor_indices)

        logger.debug(
            f"Pruning SwiGLU split FFN: {hidden_dim} -> {n_survivors} "
            f"({n_prune} pruned, {100*n_prune/hidden_dim:.1f}%)"
        )

        # gate_proj: keep survivor rows
        gate = mlp.gate_proj
        new_gate = nn.Linear(
            gate.in_features, n_survivors,
            bias=gate.bias is not None,
            device=gate.weight.device, dtype=gate.weight.dtype,
        )
        with torch.no_grad():
            new_gate.weight.copy_(gate.weight[survivor_indices, :])
            if gate.bias is not None:
                new_gate.bias.copy_(gate.bias[survivor_indices])

        # up_proj: keep survivor rows
        up = mlp.up_proj
        new_up = nn.Linear(
            up.in_features, n_survivors,
            bias=up.bias is not None,
            device=up.weight.device, dtype=up.weight.dtype,
        )
        with torch.no_grad():
            new_up.weight.copy_(up.weight[survivor_indices, :])
            if up.bias is not None:
                new_up.bias.copy_(up.bias[survivor_indices])

        # down_proj: keep survivor columns + fold compensation
        down = mlp.down_proj
        new_down = nn.Linear(
            n_survivors, down.out_features,
            bias=down.bias is not None,
            device=down.weight.device, dtype=down.weight.dtype,
        )

        if compensation is not None:
            W_tilde, b_tilde = self._fold_compensation_fc2(
                down, compensation, survivor_indices
            )
            with torch.no_grad():
                new_down.weight.copy_(W_tilde)
                if new_down.bias is not None:
                    new_down.bias.copy_(b_tilde)
        else:
            with torch.no_grad():
                new_down.weight.copy_(down.weight[:, survivor_indices])
                if down.bias is not None:
                    new_down.bias.copy_(down.bias)

        mlp.gate_proj = new_gate
        mlp.up_proj = new_up
        mlp.down_proj = new_down

        return mlp

    def _fold_compensation_fc2(
        self,
        fc2: nn.Linear,
        compensation: CompensationResult,
        survivor_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fold compensation into fc2 weights.

        Args:
            fc2: Original fc2 layer
            compensation: Compensation result with B, c
            survivor_indices: Indices of survivors

        Returns:
            Tuple of (new_weight, new_bias)
        """
        surv_idx = survivor_indices.tolist()
        prune_idx = compensation.prune_indices.tolist()

        W = fc2.weight  # (out_features, in_features)
        W_S = W[:, surv_idx]
        W_P = W[:, prune_idx]

        # W_tilde = W_S + W_P @ B
        B = compensation.B.to(W.device, W.dtype)
        W_tilde = W_S + W_P @ B

        # b_tilde = b + W_P @ c
        c = compensation.c.to(W.device, W.dtype)
        if fc2.bias is not None:
            b_tilde = fc2.bias + W_P @ c
        else:
            b_tilde = W_P @ c

        return W_tilde, b_tilde

    def prune_attention_qk_dims(
        self,
        attn: nn.Module,
        compensation_results: List[QKDimCompensationResult],
        qk_compensator: Optional[QKDimCompensator] = None,
    ) -> nn.Module:
        """Prune Q/K dimensions from attention layer using dim-logit mode.

        Args:
            attn: Attention module with qkv and proj attributes
            compensation_results: List of compensation results, one per head
            qk_compensator: QKDimCompensator instance for folding weights (None to skip compensation)

        Returns:
            Modified attention module
        """
        if not hasattr(attn, 'qkv') or not hasattr(attn, 'proj'):
            raise ValueError("Attention must have qkv and proj attributes")

        # Note: qk_compensator=None means skip compensation (just keep survivors)

        num_heads = attn.num_heads
        in_features = attn.qkv.in_features
        qkv_out_features = attn.qkv.out_features

        # Detect current Q/K and V dimensions from actual layer dimensions
        # This is robust to checkpoint loading where attn.head_dim may not be restored
        # V dimension from proj layer (unchanged by Q/K pruning)
        if hasattr(attn, 'proj'):
            v_total = attn.proj.in_features
            v_dim_per_head = v_total // num_heads
        else:
            # Fallback: assume unpruned
            v_dim_per_head = qkv_out_features // (3 * num_heads)
            v_total = v_dim_per_head * num_heads

        # Check if model is pruned by comparing QKV dimensions
        expected_original = 3 * v_dim_per_head * num_heads
        if qkv_out_features == expected_original:
            # Unpruned model: Q/K/V all have same dimension
            current_qk_dim = v_dim_per_head
        else:
            # Pruned model: Q/K have different dimension than V
            # qkv_out = 2 * qk_dim * num_heads + v_dim * num_heads
            qk_total = qkv_out_features - v_total
            current_qk_dim = qk_total // (2 * num_heads)

        # QKV weight shape depends on whether model was already pruned
        # Layout: [Q_h0, Q_h1, ..., K_h0, K_h1, ..., V_h0, V_h1, ...]
        qkv_weight = attn.qkv.weight.data
        qkv_bias = attn.qkv.bias.data if attn.qkv.bias is not None else None

        # Extract Q, K, V sections using current dimensions
        q_start = 0
        k_start = num_heads * current_qk_dim
        v_start = 2 * num_heads * current_qk_dim

        # Q/K dims are reduced, V unchanged
        # All heads should have same survivor count
        if len(compensation_results) == 0:
            return attn

        n_surv = len(compensation_results[0].survivor_indices)
        new_qk_dim = n_surv * num_heads

        # Create new QKV projection
        # New shape: (2 * new_qk_dim + v_dim, in_features)
        v_dim = num_heads * v_dim_per_head  # V dimension stays unchanged
        new_total_dim = 2 * new_qk_dim + v_dim

        new_qkv_weight = torch.zeros(
            new_total_dim, in_features,
            device=qkv_weight.device, dtype=qkv_weight.dtype
        )
        new_qkv_bias = torch.zeros(
            new_total_dim,
            device=qkv_weight.device, dtype=qkv_weight.dtype
        ) if qkv_bias is not None else None

        # Check if compensation should be applied
        # qk_compensator=None means skip compensation (just keep survivors)
        apply_compensation = qk_compensator is not None

        # Process each head
        for result in compensation_results:
            h = result.head_idx
            surv_idx = result.survivor_indices

            # Original offsets (using current Q/K dimension)
            q_offset_old = q_start + h * current_qk_dim
            k_offset_old = k_start + h * current_qk_dim

            # New offsets
            q_offset_new = h * n_surv
            k_offset_new = new_qk_dim + h * n_surv

            # Extract per-head weights
            W_Q = qkv_weight[q_offset_old:q_offset_old + current_qk_dim, :]
            W_K = qkv_weight[k_offset_old:k_offset_old + current_qk_dim, :]
            b_Q = qkv_bias[q_offset_old:q_offset_old + current_qk_dim] if qkv_bias is not None else None
            b_K = qkv_bias[k_offset_old:k_offset_old + current_qk_dim] if qkv_bias is not None else None

            if apply_compensation:
                # Fold compensation into weights (U/V transforms)
                W_Q_new, b_Q_new, W_K_new, b_K_new = qk_compensator.fold_dim_logit_weights(
                    result, W_Q, b_Q, W_K, b_K
                )
            else:
                # No compensation: just extract survivor weights
                W_Q_new = W_Q[surv_idx]
                W_K_new = W_K[surv_idx]
                b_Q_new = b_Q[surv_idx] if b_Q is not None else None
                b_K_new = b_K[surv_idx] if b_K is not None else None

            # Store transformed weights
            new_qkv_weight[q_offset_new:q_offset_new + n_surv, :] = W_Q_new
            new_qkv_weight[k_offset_new:k_offset_new + n_surv, :] = W_K_new

            if qkv_bias is not None and b_Q_new is not None:
                new_qkv_bias[q_offset_new:q_offset_new + n_surv] = b_Q_new
                new_qkv_bias[k_offset_new:k_offset_new + n_surv] = b_K_new

        # Copy V weights unchanged
        v_offset_new = 2 * new_qk_dim
        new_qkv_weight[v_offset_new:, :] = qkv_weight[v_start:, :]
        if qkv_bias is not None:
            new_qkv_bias[v_offset_new:] = qkv_bias[v_start:]

        # Create new QKV layer
        new_qkv = nn.Linear(
            in_features,
            new_total_dim,
            bias=qkv_bias is not None,
            device=qkv_weight.device,
            dtype=qkv_weight.dtype,
        )

        with torch.no_grad():
            new_qkv.weight.copy_(new_qkv_weight)
            if qkv_bias is not None:
                new_qkv.bias.copy_(new_qkv_bias)

        attn.qkv = new_qkv

        # Update attention parameters for reduced Q/K dims
        attn.head_dim = n_surv  # New reduced head_dim for Q/K
        attn.scale = n_surv ** -0.5  # Update scale for new Q/K dim

        # Patch forward to handle different Q/K vs V dimensions
        attn.forward = _make_qk_dim_pruned_attention_forward(
            attn, n_surv, v_dim_per_head, num_heads
        )

        return attn

def get_model_parameter_count(model: nn.Module) -> int:
    """Get total parameter count of a model."""
    return sum(p.numel() for p in model.parameters())



def compare_model_sizes(
    original: nn.Module,
    pruned: nn.Module,
) -> dict:
    """Compare sizes of original and pruned models.

    Args:
        original: Original model
        pruned: Pruned model

    Returns:
        Dictionary with comparison metrics
    """
    orig_params = get_model_parameter_count(original)
    pruned_params = get_model_parameter_count(pruned)

    return {
        'original_params': orig_params,
        'pruned_params': pruned_params,
        'params_removed': orig_params - pruned_params,
        'compression_ratio': orig_params / max(pruned_params, 1),
        'sparsity': 1 - pruned_params / max(orig_params, 1),
    }
