"""Causal-LM perplexity (WikiText-2 style non-overlapping chunks)."""

import math
import logging

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model, eval_loader, device: str) -> float:
    """Return perplexity on ``eval_loader`` (yields ``(input_ids,)`` tuples)."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in eval_loader:
        input_ids = batch[0].to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        batch_size, seq_len = input_ids.shape
        n_predicted = batch_size * (seq_len - 1)
        total_nll += outputs.loss.item() * n_predicted
        total_tokens += n_predicted
    mean_nll = total_nll / total_tokens
    ppl = math.exp(mean_nll)
    logger.info(f"Perplexity: {ppl:.2f} (mean NLL: {mean_nll:.4f}, tokens: {total_tokens})")
    return ppl
