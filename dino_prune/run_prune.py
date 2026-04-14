#!/usr/bin/env python
"""
Layer-by-Layer Activation Pruning for DINOv2 Models.

This script performs activation-based structured pruning on DINOv2 models,
using NYU Depth V2 and ADE20k datasets for calibration and evaluation.

Usage:
    python dino_prune/run_prune.py --model base --sparsity 0.3 \
        --nyu_path /path/to/nyu --use_ade

    # MLP-only pruning with weight magnitude ranking
    python dino_prune/run_prune.py --model base --target mlp --sparsity 0.3 \
        --ranker weight_magnitude --nyu_path /path/to/nyu

    # Attention + MLP simultaneous pruning with dim-logit mode
    python dino_prune/run_prune.py --model base --target both --sparsity 0.3 \
        --attn_mode dim-logit --nyu_path /path/to/nyu
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.schemas import (
    CollectorConfig, PruningConfig, RunnerConfig, FullConfig,
    PruneTarget, ScheduleType, RankerType, CovarianceMode,
    AttentionPruneMode,
)
from pruning.runner import PruneRunner

from dino_prune.models import load_model, count_parameters
from dino_prune._datasets import create_calibration_loader, create_validation_loaders
from dino_prune.evaluation import FeatureEvaluator, format_feature_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Layer-by-layer activation pruning for DINOv2 models'
    )

    # Model arguments
    parser.add_argument(
        '--model', type=str, default='base',
        choices=['small', 'base', 'large', 'huge'],
        help='DINOv2 model size (default: base)'
    )

    # Data arguments
    parser.add_argument(
        '--nyu_path', type=str, default=None,
        help='Path to NYU Depth V2 dataset'
    )
    parser.add_argument(
        '--use_ade', action='store_true',
        help='Include ADE20k dataset (loaded from HuggingFace)'
    )
    parser.add_argument(
        '--nyu_samples', type=int, default=3000,
        help='Number of NYU samples for calibration (default: 3000)'
    )
    parser.add_argument(
        '--ade_samples', type=int, default=3000,
        help='Number of ADE20k samples for calibration (default: 3000)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--image_size', type=int, default=518,
        help='Image size (default: 518 for DINOv2/v3)'
    )

    # Pruning arguments
    parser.add_argument(
        '--target', type=str, default='both',
        choices=['mlp', 'attn', 'both'],
        help='Pruning target (default: both)'
    )
    parser.add_argument(
        '--schedule', type=str, default='layerwise',
        choices=['layerwise', 'global'],
        help='Pruning schedule (default: layerwise)'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0.3,
        help='Target sparsity (0.0 to 1.0, default: 0.3)'
    )
    parser.add_argument(
        '--ranker', type=str, default='weight_magnitude',
        choices=['energy', 'active', 'active_energy', 'weight_magnitude', 'energy_weightmagnitude', 'active_weightmagnitude', 'ensemble'],
        help='Ranking policy for pruning (default: weight_magnitude)'
    )
    parser.add_argument(
        '--outlier_safe', action='store_true', default=False,
        help='Enable outlier-safe mode (boost importance of high-kurtosis channels)'
    )
    parser.add_argument(
        '--lambda_reg', type=float, default=1e-3,
        help='Ridge regularization strength (default: 1e-3)'
    )
    parser.add_argument(
        '--min_channels', type=int, default=64,
        help='Minimum channels to keep per layer (default: 64)'
    )
    parser.add_argument(
        '--min_heads', type=int, default=1,
        help='Minimum attention heads to keep per layer (default: 1)'
    )
    parser.add_argument(
        '--skip_compensation', action='store_true',
        help='Skip compensation (prune without weight adjustment)'
    )

    # Attention pruning arguments
    parser.add_argument(
        '--attn_mode', type=str, default='dim-logit',
        choices=['head', 'dim-logit'],
        help='Attention pruning mode (default: dim-logit)'
    )
    parser.add_argument(
        '--qk_sparsity', type=float, default=0.3,
        help='Fraction of Q/K dims to prune per head (default: 0.3)'
    )
    parser.add_argument(
        '--min_qk_dim', type=int, default=8,
        help='Minimum Q/K dimensions to keep per head (default: 8)'
    )

    # Schedule arguments
    parser.add_argument(
        '--delta', type=float, default=None,
        help='Global schedule sparsity increment per round (default: None = single round)'
    )

    # Collection arguments
    parser.add_argument(
        '--calib_samples', type=int, default=6000,
        help='Number of samples for activation collection (default: 6000)'
    )
    parser.add_argument(
        '--subsample_tokens', type=int, default=100,
        help='Number of tokens to subsample per image (default: 100)'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir', type=str, default='./dino_logs',
        help='Output directory for logs and checkpoints'
    )
    parser.add_argument(
        '--save_model', action='store_true',
        help='Save pruned model checkpoint'
    )
    parser.add_argument(
        '--save_pruned_path', type=str, default=None,
        help='Path to save pruned model state dict'
    )

    # Runtime arguments
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to run on (default: cuda)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers (default: 4)'
    )

    return parser.parse_args()


def create_config(args) -> FullConfig:
    """Create FullConfig from CLI arguments."""
    collector = CollectorConfig(
        target=PruneTarget(args.target),
        subsample_tokens=args.subsample_tokens,
        covariance_mode=CovarianceMode.EXACT,
        store_raw=False,
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
    logger.info("DINOv2 Layer-by-Layer Activation Pruning")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Sparsity: {args.sparsity}")
    logger.info(f"Ranker: {args.ranker}")
    logger.info(f"Attention mode: {args.attn_mode}")
    logger.info(f"Calib samples: {args.calib_samples}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    # Check data paths
    if args.nyu_path is None and not args.use_ade:
        logger.error("At least one of --nyu_path or --use_ade must be provided")
        return 1

    # Load model
    logger.info("Loading DINOv2 model...")
    model, model_config = load_model(model_name=args.model)
    model = model.to(args.device)
    model.eval()

    original_params = count_parameters(model)
    logger.info(f"Model loaded: {model_config.get('timm_name', args.model)}")
    logger.info(f"Parameters: {original_params:,}")

    # Create calibration loader
    logger.info("Creating calibration data loader...")
    calib_loader = create_calibration_loader(
        nyu_root=args.nyu_path,
        use_ade=args.use_ade,
        nyu_samples=args.nyu_samples,
        ade_samples=args.ade_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    logger.info(f"Calibration samples: {len(calib_loader.dataset)}")

    # Create validation loaders
    logger.info("Creating validation data loaders...")
    nyu_val_loader, ade_val_loader = create_validation_loaders(
        nyu_root=args.nyu_path,
        use_ade=args.use_ade,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # Create config and runner
    config = create_config(args)
    runner = PruneRunner(config)

    # Run pruning
    logger.info("=" * 60)
    logger.info(f"Running Pruning (compensation={'disabled' if args.skip_compensation else 'enabled'})")
    logger.info("=" * 60)

    result = runner.run(model, calib_loader, skip_compensation=args.skip_compensation)

    if not result.success:
        logger.error(f"Pruning failed: {result.error_message}")
        return 1

    # Report results
    logger.info("=" * 60)
    logger.info("Pruning Complete")
    logger.info("=" * 60)
    logger.info(f"Original params: {result.original_params:,}")
    logger.info(f"Pruned params: {result.pruned_params:,}")
    logger.info(f"Compression ratio: {result.compression_ratio:.2f}x")
    logger.info(f"Output directory: {result.output_dir}")

    # Compare features between original and pruned
    if result.pruned_model is not None:
        logger.info("=" * 60)
        logger.info("Feature Comparison (Original vs Pruned)")
        logger.info("=" * 60)

        feature_evaluator = FeatureEvaluator()

        # Use validation loader for comparison
        eval_loader = nyu_val_loader or ade_val_loader or calib_loader

        # Reload original model for comparison
        original_model, _ = load_model(model_name=args.model)
        original_model = original_model.to(args.device)
        original_model.eval()

        metrics = feature_evaluator.compare_models(
            original_model, result.pruned_model,
            eval_loader, device=args.device,
            max_samples=500,
        )
        logger.info(format_feature_metrics(metrics))

        # Clean up
        del original_model
        torch.cuda.empty_cache()

    # Save pruned model if requested
    if args.save_model and result.pruned_model is not None:
        save_path = result.output_dir / "pruned_model.pt"
        torch.save(result.pruned_model.state_dict(), save_path)
        logger.info(f"Saved pruned model to {save_path}")

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
