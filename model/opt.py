"""OPT loader via HuggingFace transformers.

Exposes two post-pruning hooks used by orchestration:
- `prune_attention_heads`: physically remove whole attention heads
  from OPT's split q/k/v/out_proj layers (not fused like ViT).
- `prune_attention_qk_dims`: dim-logit pruning — shrink per-head Q/K dim
  while V and out_proj stay intact, with optional Sylvester-based
  compensation folded into Q/K weights. Patches `self_attn.forward` since
  HF OPT's upstream forward assumes Q/K/V share the same per-head dim.
"""

import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


OPT_CONFIGS = {
    "125m": {"layers": 12, "embed_dim": 768, "mlp_hidden": 3072, "heads": 12, "head_dim": 64},
    "350m": {"layers": 24, "embed_dim": 1024, "mlp_hidden": 4096, "heads": 16, "head_dim": 64},
    "1.3b": {"layers": 24, "embed_dim": 2048, "mlp_hidden": 8192, "heads": 32, "head_dim": 64},
    "2.7b": {"layers": 32, "embed_dim": 2560, "mlp_hidden": 10240, "heads": 32, "head_dim": 80},
    "6.7b": {"layers": 32, "embed_dim": 4096, "mlp_hidden": 16384, "heads": 32, "head_dim": 128},
}


def load(
    name: str = "125m",
    dtype: str = "float32",
    device: Optional[str] = None,
    **_,
) -> Tuple[nn.Module, Dict]:
    """Load an OPT causal LM.

    Args:
        name: OPT size suffix ('125m', '350m', '1.3b', ...) — mapped to ``facebook/opt-{name}``.
        dtype: torch dtype name.
        device: if given, model is moved to this device.

    Returns:
        (model, meta) where meta['tokenizer'] carries the matching AutoTokenizer.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = f"facebook/opt-{name}"
    torch_dtype = getattr(torch, dtype)
    # Force eager attention: the dim-logit patched forward in
    # `prune_attention_qk_dims` is a manual eager implementation, so the
    # unpruned baseline must also use eager to give an apples-to-apples
    # comparison (sdpa and eager produce slightly different numerics that
    # compound across layers).
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, attn_implementation="eager",
    )
    model.eval()
    if device:
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    cfg = OPT_CONFIGS.get(name, {})
    meta = {
        "source": "hf_opt",
        "name": name,
        "model_id": model_id,
        "tokenizer": tokenizer,
        **cfg,
    }
    return model, meta


def prune_attention_heads(self_attn: nn.Module, prune_head_indices: torch.Tensor) -> None:
    """Physically remove attention heads from an OPT layer.

    OPT uses separate q_proj, k_proj, v_proj, out_proj (unlike fused QKV in
    ViT/DINOv2). The generic MaskApplier doesn't handle this shape — this
    helper is the OPT-specific escape hatch.
    """
    n_heads = self_attn.num_heads
    head_dim = self_attn.head_dim

    n_prune = len(prune_head_indices)
    if n_prune == 0:
        return

    all_heads = set(range(n_heads))
    prune_set = set(prune_head_indices.tolist())
    survivor_heads = torch.tensor(sorted(all_heads - prune_set), dtype=torch.long)
    n_survivors = len(survivor_heads)

    survivor_rows = []
    for h in survivor_heads:
        start = h.item() * head_dim
        survivor_rows.extend(range(start, start + head_dim))
    survivor_rows = torch.tensor(survivor_rows, dtype=torch.long)

    for proj_name in ("q_proj", "k_proj", "v_proj"):
        proj = getattr(self_attn, proj_name)
        new_proj = nn.Linear(
            proj.in_features, n_survivors * head_dim,
            bias=proj.bias is not None,
            device=proj.weight.device, dtype=proj.weight.dtype,
        )
        with torch.no_grad():
            new_proj.weight.copy_(proj.weight[survivor_rows, :])
            if proj.bias is not None:
                new_proj.bias.copy_(proj.bias[survivor_rows])
        setattr(self_attn, proj_name, new_proj)

    out_proj = self_attn.out_proj
    new_out_proj = nn.Linear(
        n_survivors * head_dim, out_proj.out_features,
        bias=out_proj.bias is not None,
        device=out_proj.weight.device, dtype=out_proj.weight.dtype,
    )
    with torch.no_grad():
        new_out_proj.weight.copy_(out_proj.weight[:, survivor_rows])
        if out_proj.bias is not None:
            new_out_proj.bias.copy_(out_proj.bias)
    self_attn.out_proj = new_out_proj

    self_attn.num_heads = n_survivors
    self_attn.embed_dim = n_survivors * head_dim


def _make_opt_qk_dim_pruned_forward(self_attn: nn.Module):
    """Patched forward for OPT attention with Q/K dim < V dim.

    Mirrors HF `OPTAttention.forward` but reshapes Q/K with `_qk_dim` and V with
    `head_dim`, runs manual eager attention (matmul + scale + mask + softmax +
    matmul), and calls `past_key_values.update(...)` so KV-cache decoding still
    works (cache stores K and V independently per layer, so dim mismatch is fine).
    Returns `(attn_output, attn_weights)` to match the decoder layer's
    `hidden_states, _ = self.self_attn(...)` unpacking.
    """
    num_heads = self_attn.num_heads
    qk_dim = self_attn._qk_dim
    v_dim = self_attn.head_dim

    def forward(self, hidden_states, past_key_values=None, attention_mask=None,
                output_attentions=False, **kwargs):
        B, T, _ = hidden_states.shape
        q = (self.q_proj(hidden_states) * self.scaling).view(B, T, num_heads, qk_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, num_heads, qk_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, num_heads, v_dim).transpose(1, 2)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        attn_weights = q @ k.transpose(-2, -1)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : attn_weights.shape[-1]]
        attn_weights = attn_weights.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        out = (attn_weights @ v).transpose(1, 2).reshape(B, T, num_heads * v_dim)
        out = self.out_proj(out)
        return out, (attn_weights if output_attentions else None)

    return types.MethodType(forward, self_attn)


def prune_attention_qk_dims(
    self_attn: nn.Module,
    compensation_results: List,
    qk_compensator=None,
    keep_original_scale: bool = False,
) -> None:
    """Per-head Q/K dim pruning for OPT (dim-logit mode).

    Shrinks `q_proj` and `k_proj` from `(num_heads * head_dim, embed_dim)` to
    `(num_heads * n_surv, embed_dim)`. Optionally folds U/V Sylvester transforms
    via `qk_compensator.fold_dim_logit_weights`. V and out_proj are untouched.
    `num_heads`, `head_dim`, `embed_dim` are unchanged; only the per-head Q/K
    dimension shrinks. Updates `scaling` to `n_surv ** -0.5` (or `head_dim ** -0.5`
    when `keep_original_scale=True`) and patches `self_attn.forward` to handle
    the asymmetric Q/K vs V dim.

    Args:
        self_attn: OPT attention module
        compensation_results: list of QKDimCompensationResult, one per head
        qk_compensator: QKDimCompensator instance (None to skip compensation)
        keep_original_scale: if True, keep the original `1/sqrt(head_dim)`
            softmax scale instead of switching to `1/sqrt(n_surv)`. Useful for
            dim-logit pruning where the surviving dims carry most of the
            dot-product magnitude, so the textbook variance argument doesn't
            apply and the rescale just sharpens softmax vs baseline.
    """
    if not compensation_results:
        return

    num_heads = self_attn.num_heads
    head_dim = self_attn.head_dim
    embed_dim = self_attn.embed_dim

    n_surv = len(compensation_results[0].survivor_indices)
    apply_comp = qk_compensator is not None

    q_proj = self_attn.q_proj
    k_proj = self_attn.k_proj
    has_q_bias = q_proj.bias is not None
    has_k_bias = k_proj.bias is not None
    device = q_proj.weight.device
    dtype = q_proj.weight.dtype

    new_q = nn.Linear(embed_dim, num_heads * n_surv, bias=has_q_bias, device=device, dtype=dtype)
    new_k = nn.Linear(embed_dim, num_heads * n_surv, bias=has_k_bias, device=device, dtype=dtype)

    with torch.no_grad():
        for result in compensation_results:
            h = result.head_idx
            row_start = h * head_dim
            row_end = row_start + head_dim

            W_Q_h = q_proj.weight.data[row_start:row_end, :]
            W_K_h = k_proj.weight.data[row_start:row_end, :]
            b_Q_h = q_proj.bias.data[row_start:row_end] if has_q_bias else None
            b_K_h = k_proj.bias.data[row_start:row_end] if has_k_bias else None

            if apply_comp:
                W_Q_new, b_Q_new, W_K_new, b_K_new = qk_compensator.fold_dim_logit_weights(
                    result, W_Q_h, b_Q_h, W_K_h, b_K_h,
                )
            else:
                surv = result.survivor_indices
                W_Q_new = W_Q_h[surv]
                W_K_new = W_K_h[surv]
                b_Q_new = b_Q_h[surv] if b_Q_h is not None else None
                b_K_new = b_K_h[surv] if b_K_h is not None else None

            new_q.weight.data[h * n_surv:(h + 1) * n_surv] = W_Q_new.to(device=device, dtype=dtype)
            new_k.weight.data[h * n_surv:(h + 1) * n_surv] = W_K_new.to(device=device, dtype=dtype)
            if has_q_bias and b_Q_new is not None:
                new_q.bias.data[h * n_surv:(h + 1) * n_surv] = b_Q_new.to(device=device, dtype=dtype)
            if has_k_bias and b_K_new is not None:
                new_k.bias.data[h * n_surv:(h + 1) * n_surv] = b_K_new.to(device=device, dtype=dtype)

    self_attn.q_proj = new_q
    self_attn.k_proj = new_k
    self_attn._qk_dim = n_surv
    self_attn.scaling = (head_dim ** -0.5) if keep_original_scale else (n_surv ** -0.5)
    self_attn.forward = _make_opt_qk_dim_pruned_forward(self_attn)
