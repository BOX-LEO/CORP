"""WikiText-2 perplexity evaluation loader."""

import torch
from torch.utils.data import DataLoader, TensorDataset


def build_eval(tokenizer, cfg: dict) -> DataLoader:
    """Load WikiText-2 test split, chunk into fixed-length segments.

    Args:
        tokenizer: HuggingFace tokenizer matching the model under evaluation.
        cfg: dataset block from YAML. Uses seq_len (default 2048), batch_size (4).

    Returns:
        DataLoader yielding ``(input_ids,)`` tuples.
    """
    from datasets import load_dataset

    seq_len = int(cfg.get("seq_len", 2048))
    batch_size = int(cfg.get("batch_size", 4))

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    tokens = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].squeeze(0)

    n_segments = tokens.shape[0] // seq_len
    tokens = tokens[: n_segments * seq_len].reshape(n_segments, seq_len)

    return DataLoader(TensorDataset(tokens), batch_size=batch_size, shuffle=False)
