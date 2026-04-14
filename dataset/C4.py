"""C4 streaming calibration loader for language-model pruning."""

import torch
from torch.utils.data import DataLoader, TensorDataset


def build_calibration(tokenizer, cfg: dict) -> DataLoader:
    """Stream tokens from C4, chunk into fixed-length segments, return a DataLoader.

    Args:
        tokenizer: HuggingFace tokenizer matching the model under evaluation.
        cfg: dataset block from YAML. Uses
            n_calib_segments (default 128), seq_len (2048), batch_size (4), seed (42).

    Returns:
        DataLoader yielding ``(input_ids,)`` tuples shaped ``(batch, seq_len)``.
    """
    from datasets import load_dataset

    n_segments = int(cfg.get("n_calib_segments", 128))
    seq_len = int(cfg.get("seq_len", 2048))
    batch_size = int(cfg.get("batch_size", 4))
    seed = int(cfg.get("seed", 42))

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    total_needed = n_segments * seq_len
    all_tokens = []
    count = 0
    for example in ds:
        toks = tokenizer(
            example["text"], return_tensors="pt", add_special_tokens=False
        )["input_ids"].squeeze(0)
        all_tokens.append(toks)
        count += len(toks)
        if count >= total_needed * 2:
            break

    all_tokens = torch.cat(all_tokens, dim=0)
    n_available = all_tokens.shape[0] // seq_len
    all_tokens = all_tokens[: n_available * seq_len].reshape(n_available, seq_len)

    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_available, generator=gen)[:n_segments]
    segments = all_tokens[indices]

    return DataLoader(TensorDataset(segments), batch_size=batch_size, shuffle=False)
