"""OPT loader via HuggingFace transformers.

Exposes a post-pruning hook (`prune_attention_heads`) used by orchestration
when attention heads need to be physically removed from OPT's split
q/k/v/out_proj layers (not fused like ViT).
"""

from typing import Dict, Tuple, Optional

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
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
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
