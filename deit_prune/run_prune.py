#!/usr/bin/env python
"""
CLI Entry Point for Activation-Based Structured Pruning.

Usage:
    # MLP pruning:
    python deit_prune/run_prune.py --model deit_tiny_patch16_224 --target mlp --sparsity 0.3

    # Attention head pruning:
    python deit_prune/run_prune.py --model deit_tiny_patch16_224 --target attn --sparsity 0.25

    # Q/K dimension pruning with Sylvester-based logit-matching compensation:
    python deit_prune/run_prune.py --model deit_small_patch16_224 --target attn \
        --attn_mode dim-logit --qk_sparsity 0.3 --covariance_mode exact
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.schemas import (
    CollectorConfig, PruningConfig, RunnerConfig, FullConfig,
    PruneTarget, ScheduleType, RankerType, CovarianceMode, AttentionPruneMode,
)
from pruning.runner import PruneRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Activation-based structured pruning for ViT/DeiT models'
    )

    # Model arguments
    parser.add_argument(
        '--model', type=str, default='deit_base_patch16_224',
        help='Model name (from timm) or path to checkpoint'
    )
    parser.add_argument(
        '--model_path', type=str, default=None,
        help='Path to model checkpoint (if not using timm)'
    )

    # Data arguments
    parser.add_argument(
        '--calib_path', type=str, default='/home/boxiang/work/dao2/eff_vit/imagenet-mini/val',
        help='Path to calibration data (ImageNet format or tensor file)'
    )
    parser.add_argument(
        '--val_path', type=str, default='/home/boxiang/work/ombs/imagenet-mini/imagenet-val',
        help='Path to validation data (optional)'
    )
    parser.add_argument(
        '--calib_samples', type=int, default=3923,
        help='Number of calibration samples to use'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for calibration'
    )

    # Pruning arguments
    parser.add_argument(
        '--target', type=str, default='mlp', choices=['mlp', 'attn', 'both'],
        help='Target structures to prune'
    )
    parser.add_argument(
        '--schedule', type=str, default='layerwise',
        choices=['layerwise', 'global'],
        help='Pruning schedule (global supports multi-round with --rounds)'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0.3,
        help='Target sparsity (0.0 to 1.0)'
    )
    parser.add_argument(
        '--ranker', type=str, default='active_energy',
        choices=['energy', 'active', 'active_energy', 'weight_magnitude', 'energy_weightmagnitude', 'active_weightmagnitude', 'ensemble'],
        help='Ranking policy for pruning'
    )
    parser.add_argument(
        '--outlier_safe', action='store_true', default=False,
        help='Enable outlier-safe mode (boost importance of high-kurtosis channels)'
    )
    parser.add_argument(
        '--lambda_reg', type=float, default=1e-3,
        help='Ridge regularization strength'
    )
    parser.add_argument(
        '--min_channels', type=int, default=64,
        help='Minimum MLP hidden channels to keep per layer'
    )
    parser.add_argument(
        '--min_heads', type=int, default=1,
        help='Minimum attention heads to keep per layer'
    )
    parser.add_argument(
        '--delta', type=float, default=None,
        help='Sparsity increment per round for global schedule. If not set, prunes in single round.'
    )

    # Attention dimension pruning arguments
    parser.add_argument(
        '--attn_mode', type=str, default='head',
        choices=['head', 'dim-logit'],
        help='Attention pruning mode: '
             'head (prune entire heads), '
             'dim-logit (prune Q/K dims with Sylvester-based logit-matching compensation - '
             'applies different U/V transforms to Q/K, requires covariance_mode=exact)'
    )
    parser.add_argument(
        '--qk_sparsity', type=float, default=0.3,
        help='Fraction of Q/K dimensions to prune per head (for dim-ac/dim-logit modes). '
             'Uses E[q^2 * k^2] joint energy score for ranking.'
    )
    parser.add_argument(
        '--min_qk_dim', type=int, default=8,
        help='Minimum Q/K dimensions to keep per head (for dim-ac/dim-logit modes)'
    )

    # Collection arguments
    parser.add_argument(
        '--subsample_tokens', type=int, default=None,
        help='Number of tokens to subsample per image'
    )
    parser.add_argument(
        '--covariance_mode', type=str, default='exact',
        choices=['exact', 'sketch'],
        help='Covariance computation mode'
    )
    parser.add_argument(
        '--sketch_dim', type=int, default=256,
        help='Sketch dimension for covariance approximation'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir', type=str, default='./logs',
        help='Output directory for logs and artifacts'
    )
    parser.add_argument(
        '--save_pruned_path', type=str, default=None,
        help='Path to save pruned model'
    )

    # Runtime arguments
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to run on'
    )
    parser.add_argument(
        '--dtype', type=str, default='float32',
        choices=['float32', 'float16', 'bfloat16'],
        help='Data type for computation'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    return parser.parse_args()


def load_model(args):
    """Load the model."""
    try:
        import timm
        model = timm.create_model(args.model, pretrained=True)
        logger.info(f"Loaded model {args.model} from timm")
    except ImportError:
        logger.warning("timm not installed, trying torch.hub")
        model = torch.hub.load(
            'facebookresearch/deit:main',
            args.model,
            pretrained=True
        )

    if args.model_path:
        state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {args.model_path}")

    return model


# Path to baseline accuracy cache file
BASELINE_CACHE_PATH = Path(__file__).parent / "baseline_accuracy_cache.json"


def load_baseline_cache() -> dict:
    """Load baseline accuracy cache from JSON file."""
    if BASELINE_CACHE_PATH.exists():
        with open(BASELINE_CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_baseline_cache(cache: dict) -> None:
    """Save baseline accuracy cache to JSON file."""
    with open(BASELINE_CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)
    logger.info(f"Saved baseline accuracy cache to {BASELINE_CACHE_PATH}")


def get_baseline_accuracy(model_name: str, val_path: str) -> Optional[dict]:
    """Get cached baseline accuracy and params for a model+dataset combination.

    Returns:
        Dict with 'top1', 'top5', and optionally 'params' keys, or None if not cached.
    """
    cache = load_baseline_cache()
    # Key includes both model name and val_path to handle different datasets
    cache_key = f"{model_name}|{val_path}"
    return cache.get(cache_key)


def save_baseline_accuracy(model_name: str, val_path: str, accuracy: dict) -> None:
    """Save baseline accuracy and params for a model+dataset combination.

    Args:
        model_name: Name of the model
        val_path: Path to validation dataset
        accuracy: Dict with 'top1', 'top5', and optionally 'params' keys
    """
    cache = load_baseline_cache()
    cache_key = f"{model_name}|{val_path}"
    cache[cache_key] = accuracy
    save_baseline_cache(cache)


def load_validation_data(args) -> DataLoader:
    """Load validation data for accuracy evaluation."""
    val_path = Path(args.val_path)

    if not val_path.exists():
        raise ValueError(f"Validation path does not exist: {val_path}")

    if not val_path.is_dir():
        raise ValueError(f"Validation path must be a directory: {val_path}")

    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = datasets.ImageFolder(val_path, transform=transform)
    logger.info(f"Loaded {len(dataset)} validation samples from {val_path}")

    eval_bs = getattr(args, 'eval_batch_size', None) or args.batch_size
    num_workers = getattr(args, 'num_workers', 4)
    loader = DataLoader(
        dataset,
        batch_size=eval_bs,
        shuffle=False,  # No shuffle for evaluation
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    desc: str = "Evaluating"
) -> Tuple[float, float]:
    """
    Evaluate model accuracy on a dataset.

    Returns:
        Tuple of (top1_accuracy, top5_accuracy) as percentages.
    """
    model.eval()
    model.to(device)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for batch in tqdm(dataloader, desc=desc):
        if isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1]
        else:
            images, labels = batch, None

        if labels is None:
            logger.warning("No labels found in batch, skipping accuracy evaluation")
            return 0.0, 0.0

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Top-1 accuracy
        _, pred_top1 = outputs.topk(1, dim=1)
        correct_top1 += (pred_top1.squeeze() == labels).sum().item()

        # Top-5 accuracy
        _, pred_top5 = outputs.topk(5, dim=1)
        correct_top5 += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.size(0)

    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total

    return top1_acc, top5_acc


def load_calibration_data(args):
    """Load calibration data."""
    calib_path = Path(args.calib_path)

    if calib_path.suffix in ['.pt', '.pth']:
        # Load tensor file
        data = torch.load(calib_path)
        if isinstance(data, dict):
            images = data.get('images', data.get('data'))
        else:
            images = data
        dataset = TensorDataset(images)
        logger.info(f"Loaded {len(dataset)} samples from tensor file")

    elif calib_path.is_dir():
        # Assume ImageNet-style directory
        try:
            from torchvision import datasets, transforms

            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            dataset = datasets.ImageFolder(calib_path, transform=transform)
            logger.info(f"Loaded {len(dataset)} samples from ImageFolder")

        except ImportError:
            logger.error("torchvision required for ImageFolder loading")
            raise

    else:
        raise ValueError(f"Unknown calibration data format: {calib_path}")

    num_workers = getattr(args, 'num_workers', 4)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


def create_config(args) -> FullConfig:
    """Create configuration from arguments."""
    collector = CollectorConfig(
        target=PruneTarget(args.target),
        subsample_tokens=args.subsample_tokens,
        covariance_mode=CovarianceMode(args.covariance_mode),
        sketch_dim=args.sketch_dim,
        store_raw=False,  # Compensation uses covariance, not raw activations
        batch_size=args.batch_size,
    )

    pruning = PruningConfig(
        target=PruneTarget(args.target),
        schedule=ScheduleType(args.schedule),
        sparsity=args.sparsity,
        ranker=RankerType(args.ranker),
        lambda_reg=args.lambda_reg,
        min_channels=args.min_channels,
        min_heads=args.min_heads,
        delta_per_round=args.delta,
        attn_prune_mode=AttentionPruneMode(args.attn_mode),
        qk_sparsity=args.qk_sparsity,
        min_qk_dim=args.min_qk_dim,
    )

    runner = RunnerConfig(
        device=args.device,
        dtype=args.dtype,
        output_dir=Path(args.output_dir),
        save_pruned_path=Path(args.save_pruned_path) if args.save_pruned_path else None,
        calib_samples=args.calib_samples,
        seed=args.seed,
    )

    return FullConfig(collector=collector, pruning=pruning, runner=runner)


def main():
    """Main entry point."""
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    logger.info("=" * 60)
    logger.info("Activation-Based Structured Pruning")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Schedule: {args.schedule}")
    logger.info(f"Sparsity: {args.sparsity}")
    logger.info(f"Ranker: {args.ranker}")
    logger.info("=" * 60)

    # Load model
    model = load_model(args)

    # Load validation data for accuracy evaluation
    val_loader = load_validation_data(args)

    # Get or compute baseline accuracy
    logger.info("=" * 60)
    logger.info("Baseline Accuracy Evaluation")
    logger.info("=" * 60)

    cached_baseline = get_baseline_accuracy(args.model, args.val_path)
    if cached_baseline is not None:
        baseline_top1 = cached_baseline["top1"]
        baseline_top5 = cached_baseline["top5"]
        logger.info(f"Using cached baseline accuracy for {args.model}")
        logger.info(f"Baseline Top-1: {baseline_top1:.2f}%")
        logger.info(f"Baseline Top-5: {baseline_top5:.2f}%")
    else:
        logger.info(f"Computing baseline accuracy for {args.model}...")
        baseline_top1, baseline_top5 = evaluate_accuracy(
            model, val_loader, device=args.device, desc="Baseline evaluation"
        )
        logger.info(f"Baseline Top-1: {baseline_top1:.2f}%")
        logger.info(f"Baseline Top-5: {baseline_top5:.2f}%")

        # Cache the baseline accuracy
        save_baseline_accuracy(args.model, args.val_path, {
            "top1": baseline_top1,
            "top5": baseline_top5,
        })

    # Load calibration data
    calib_loader = load_calibration_data(args)

    # Create config
    config = create_config(args)

    # Create runner
    runner = PruneRunner(config)

    # Run pruning WITHOUT compensation first
    logger.info("=" * 60)
    logger.info("Running Pruning (without compensation)")
    logger.info("=" * 60)
    result_no_comp = runner.run(model, calib_loader, skip_compensation=True)

    # Evaluate pruned model (no compensation)
    logger.info("=" * 60)
    logger.info("Pruned Model Accuracy (No Compensation)")
    logger.info("=" * 60)

    pruned_no_comp_top1, pruned_no_comp_top5 = evaluate_accuracy(
        result_no_comp.pruned_model, val_loader, device=args.device,
        desc="Pruned (no compensation) evaluation"
    )
    logger.info(f"Pruned (no comp) Top-1: {pruned_no_comp_top1:.2f}%")
    logger.info(f"Pruned (no comp) Top-5: {pruned_no_comp_top5:.2f}%")

    # Run pruning WITH compensation
    logger.info("=" * 60)
    logger.info("Running Pruning (with compensation)")
    logger.info("=" * 60)
    result = runner.run(model, calib_loader, skip_compensation=False)

    # Evaluate pruned model (with compensation)
    logger.info("=" * 60)
    logger.info("Pruned Model Accuracy (With Compensation)")
    logger.info("=" * 60)

    pruned_top1, pruned_top5 = evaluate_accuracy(
        result.pruned_model, val_loader, device=args.device,
        desc="Pruned (with compensation) evaluation"
    )
    logger.info(f"Pruned (compensated) Top-1: {pruned_top1:.2f}%")
    logger.info(f"Pruned (compensated) Top-5: {pruned_top5:.2f}%")

    # Compute accuracy drops
    top1_drop_no_comp = baseline_top1 - pruned_no_comp_top1
    top5_drop_no_comp = baseline_top5 - pruned_no_comp_top5
    top1_drop = baseline_top1 - pruned_top1
    top5_drop = baseline_top5 - pruned_top5
    top1_recovery = pruned_top1 - pruned_no_comp_top1
    top5_recovery = pruned_top5 - pruned_no_comp_top5

    # Report results
    logger.info("=" * 60)
    logger.info("Final Summary")
    logger.info("=" * 60)
    logger.info(f"Success: {result.success}")
    logger.info(f"Original params: {result.original_params:,}")
    logger.info(f"Pruned params: {result.pruned_params:,}")
    logger.info(f"Compression ratio: {result.compression_ratio:.2f}x")
    logger.info("-" * 40)
    logger.info("Accuracy Comparison:")
    logger.info(f"  Baseline:              Top-1: {baseline_top1:.2f}%  |  Top-5: {baseline_top5:.2f}%")
    logger.info(f"  Pruned (no comp):      Top-1: {pruned_no_comp_top1:.2f}%  |  Top-5: {pruned_no_comp_top5:.2f}%  |  Drop: {top1_drop_no_comp:.2f}%/{top5_drop_no_comp:.2f}%")
    logger.info(f"  Pruned (compensated):  Top-1: {pruned_top1:.2f}%  |  Top-5: {pruned_top5:.2f}%  |  Drop: {top1_drop:.2f}%/{top5_drop:.2f}%")
    logger.info(f"  Compensation recovery: Top-1: +{top1_recovery:.2f}%  |  Top-5: +{top5_recovery:.2f}%")
    logger.info("-" * 40)
    logger.info(f"Output directory: {result.output_dir}")

    if result.error_message:
        logger.error(f"Error: {result.error_message}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
