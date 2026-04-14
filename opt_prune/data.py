"""
Data loading for OPT pruning: C4 calibration and WikiText-2 evaluation.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def create_c4_calibration_loader(tokenizer, n_segments=128, seq_len=2048, batch_size=4, seed=42):
    """Create calibration DataLoader from C4 dataset.

    Loads streaming C4 data, tokenizes, concatenates tokens, and chunks
    into fixed-length segments for calibration.

    Args:
        tokenizer: HuggingFace tokenizer
        n_segments: Number of token segments to collect
        seq_len: Length of each segment in tokens
        batch_size: Batch size for DataLoader
        seed: Random seed for segment selection

    Returns:
        DataLoader yielding input_ids tensors of shape (batch, seq_len)
    """
    from datasets import load_dataset

    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

    # Collect enough tokens
    total_tokens_needed = n_segments * seq_len
    all_tokens = []
    current_count = 0

    for example in dataset:
        tokens = tokenizer(example["text"], return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        all_tokens.append(tokens)
        current_count += len(tokens)
        if current_count >= total_tokens_needed * 2:
            # Collect 2x to have room for random selection
            break

    # Concatenate all tokens
    all_tokens = torch.cat(all_tokens, dim=0)

    # Chunk into seq_len segments
    n_available = all_tokens.shape[0] // seq_len
    all_tokens = all_tokens[:n_available * seq_len].reshape(n_available, seq_len)

    # Randomly select n_segments
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_available, generator=generator)[:n_segments]
    segments = all_tokens[indices]

    dataset = TensorDataset(segments)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_wikitext2_eval_loader(tokenizer, seq_len=2048, batch_size=4):
    """Create evaluation DataLoader from WikiText-2.

    Loads WikiText-2 test split, concatenates all text, tokenizes,
    and chunks into fixed-length segments.

    Args:
        tokenizer: HuggingFace tokenizer
        seq_len: Length of each segment in tokens
        batch_size: Batch size for DataLoader

    Returns:
        DataLoader yielding input_ids tensors of shape (batch, seq_len)
    """
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    # Tokenize
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)

    # Chunk into seq_len segments
    n_segments = tokens.shape[0] // seq_len
    tokens = tokens[:n_segments * seq_len].reshape(n_segments, seq_len)

    dataset = TensorDataset(tokens)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
