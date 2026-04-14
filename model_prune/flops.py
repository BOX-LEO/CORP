"""FLOP and parameter counting utilities.

Two code paths, ported from `ombs/activation_prune/`:

- Vision (timm ViT/DeiT, DINOv2): `fvcore.nn.FlopCountAnalysis` with a custom
  SDPA handler, since fvcore doesn't count `aten::scaled_dot_product_attention`
  by default. Input is a dummy `(1, 3, H, W)` on the model's device.

- OPT: analytical layer-wise summation (no fvcore), parameterized by seq_len.
  Inspects live `q_proj/k_proj/v_proj/out_proj/fc1/fc2/lm_head` dimensions so it
  works identically for original and pruned models.

All results are FLOPs (multiply-accumulate counted as 2 ops), matching the
convention used in the ombs results caches.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _sdpa_flop_jit(inputs, outputs):
    """FLOPs for aten::scaled_dot_product_attention.

    2 * B * H * N * N * D_qk   (Q @ K^T)
    + 2 * B * H * N * N * D_v   (attn @ V)
    """
    q = inputs[0]
    v = inputs[2]
    q_shape = q.type().sizes()
    v_shape = v.type().sizes()
    B, H, N, D_qk = q_shape
    D_v = v_shape[-1]
    return 2 * B * H * N * N * D_qk + 2 * B * H * N * N * D_v


def compute_vision_flops(model: nn.Module, input_shape: Tuple[int, int, int, int]) -> int:
    """Run fvcore with the SDPA handler. `input_shape` is (B, C, H, W)."""
    from fvcore.nn import FlopCountAnalysis

    device = next(model.parameters()).device
    dummy = torch.zeros(*input_shape, device=device)
    fa = FlopCountAnalysis(model, dummy)
    fa.set_op_handle("aten::scaled_dot_product_attention", _sdpa_flop_jit)
    # Suppress noisy "unsupported op" logs from fvcore; we handle SDPA manually.
    fa.unsupported_ops_warnings(False)
    fa.uncalled_modules_warnings(False)
    total = int(fa.total())
    del fa, dummy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return total


def compute_opt_flops(model: nn.Module, seq_len: int = 2048) -> int:
    """Analytical FLOPs for HuggingFace OPT (works on pruned models too)."""
    N = seq_len
    total = 0

    decoder = model.model.decoder
    for layer in decoder.layers:
        attn = layer.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            proj = getattr(attn, proj_name)
            total += 2 * N * proj.in_features * proj.out_features

        num_heads = attn.num_heads
        head_dim = attn.head_dim
        total += 2 * num_heads * N * N * head_dim  # Q @ K^T
        total += 2 * num_heads * N * N * head_dim  # attn @ V

        total += 2 * N * attn.out_proj.in_features * attn.out_proj.out_features
        total += 2 * N * layer.fc1.in_features * layer.fc1.out_features
        total += 2 * N * layer.fc2.in_features * layer.fc2.out_features

    if hasattr(model, "lm_head"):
        total += 2 * N * model.lm_head.in_features * model.lm_head.out_features

    return int(total)


def _vision_input_shape_from_loader(loader) -> Optional[Tuple[int, int, int, int]]:
    """Peek one batch, return (1, C, H, W). None if the loader yields tensors
    we don't recognize as images."""
    try:
        batch = next(iter(loader))
    except Exception as e:
        logger.warning(f"Could not peek calib loader for FLOP input shape: {e}")
        return None
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    if not isinstance(x, torch.Tensor) or x.dim() != 4:
        return None
    _, C, H, W = x.shape
    return (1, C, H, W)


def flops_for_model(
    model: nn.Module,
    meta: dict,
    task_cfg,
    calib_loader,
) -> Optional[int]:
    """Dispatcher. Returns FLOPs or None on failure (logged)."""
    source = (meta or {}).get("source")
    try:
        if source == "hf_opt":
            seq_len = int(task_cfg.dataset.get("seq_len", 2048))
            return compute_opt_flops(model, seq_len=seq_len)

        input_shape = _vision_input_shape_from_loader(calib_loader)
        if input_shape is None:
            logger.warning("FLOP counting skipped: could not infer input shape")
            return None
        was_training = model.training
        model.eval()
        try:
            return compute_vision_flops(model, input_shape)
        finally:
            if was_training:
                model.train()
    except Exception as e:
        logger.warning(f"FLOP counting failed ({source}): {e}")
        return None
