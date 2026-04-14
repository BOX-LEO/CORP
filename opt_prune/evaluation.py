"""
Perplexity evaluation for OPT models.
"""

import torch
import math
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_perplexity(model, eval_loader, device):
    """Evaluate perplexity on a DataLoader of input_ids.

    Uses the standard non-overlapping chunk approach (as in SparseGPT, Wanda).
    The model predicts tokens 1..N from tokens 0..N-1 (causal shift), so each
    sequence of length N contributes N-1 predicted tokens to the NLL.

    Args:
        model: OPT CausalLM model
        eval_loader: DataLoader yielding (input_ids,) tuples
        device: Device to run evaluation on

    Returns:
        Perplexity (float)
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for batch in eval_loader:
        input_ids = batch[0].to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        # outputs.loss is mean cross-entropy over (seq_len - 1) shifted tokens
        # per sequence, averaged across the batch
        batch_size, seq_len = input_ids.shape
        n_predicted = batch_size * (seq_len - 1)
        total_nll += outputs.loss.item() * n_predicted
        total_tokens += n_predicted

    mean_nll = total_nll / total_tokens
    perplexity = math.exp(mean_nll)
    logger.info(f"Perplexity: {perplexity:.2f} (mean NLL: {mean_nll:.4f}, tokens: {total_tokens})")
    return perplexity
