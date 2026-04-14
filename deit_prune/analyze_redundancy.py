#!/usr/bin/env python
"""
Analyze activation signatures that imply redundancy for each layer.

Logs detailed redundancy metrics to a file for analysis.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.schemas import (
    CollectorConfig, PruneTarget, CovarianceMode,
)
from pruning.collect import (
    ActivationCollector, detect_model_structure, get_hook_points,
)
from pruning.stats import RedundancyAnalyzer, RedundancyReport

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze activation redundancy')
    parser.add_argument('--model', type=str, default='deit3_huge_patch14_224')
    parser.add_argument('--calib_path', type=str,
                        default='/home/boxiang/work/dao2/eff_vit/imagenet-mini/val')
    parser.add_argument('--calib_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--subsample_tokens', type=int, default=50)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--store_raw', action='store_true',
                        help='Store raw activations for additional metrics (correlation, kurtosis, '
                             'effective_rank). WARNING: ~1 GB CPU RAM per layer for large models')
    return parser.parse_args()


def load_model(model_name: str):
    """Load model from timm."""
    import timm
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    return model


def load_calibration_data(calib_path: str, batch_size: int):
    """Load calibration data."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(calib_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def compute_additional_metrics(stats) -> dict:
    """Compute additional redundancy metrics from raw activations."""
    metrics = {}

    if stats.raw_activations is None or len(stats.raw_activations) == 0:
        return metrics

    X = torch.cat(stats.raw_activations, dim=0)

    # Per-channel statistics
    mean = X.mean(dim=0)
    std = X.std(dim=0)

    # Coefficient of variation (std / |mean|)
    cv = std / (mean.abs() + 1e-10)
    metrics['coeff_variation_mean'] = cv.mean().item()
    metrics['coeff_variation_max'] = cv.max().item()

    # Sparsity: fraction of near-zero activations
    threshold = 0.01 * X.abs().mean()
    sparsity = (X.abs() < threshold).float().mean().item()
    metrics['activation_sparsity'] = sparsity

    # Per-channel sparsity
    channel_sparsity = (X.abs() < threshold).float().mean(dim=0)
    metrics['channels_very_sparse'] = (channel_sparsity > 0.9).sum().item()
    metrics['channels_mostly_zero'] = (channel_sparsity > 0.99).sum().item()

    # Correlation analysis (sample for efficiency)
    if X.shape[0] > 500:
        idx = torch.randperm(X.shape[0])[:500]
        X_sample = X[idx]
    else:
        X_sample = X

    # Compute correlation matrix
    X_centered = X_sample - X_sample.mean(dim=0)
    norms = X_centered.norm(dim=0, keepdim=True)
    norms = norms.clamp(min=1e-10)
    X_normed = X_centered / norms
    corr = X_normed.T @ X_normed / X_sample.shape[0]

    # Find highly correlated pairs (excluding diagonal)
    corr_no_diag = corr.clone()
    corr_no_diag.fill_diagonal_(0)
    high_corr_threshold = 0.9
    high_corr_pairs = (corr_no_diag.abs() > high_corr_threshold).sum().item() // 2
    metrics['highly_correlated_pairs'] = high_corr_pairs
    metrics['max_correlation'] = corr_no_diag.abs().max().item()

    # Effective rank estimation
    try:
        cov = X_centered.T @ X_centered / X_sample.shape[0]
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = eigvals.clamp(min=0)
        eigvals = eigvals.flip(0)  # Descending

        # Effective rank (entropy-based)
        eigvals_norm = eigvals / eigvals.sum()
        eigvals_norm = eigvals_norm[eigvals_norm > 1e-10]
        entropy = -(eigvals_norm * eigvals_norm.log()).sum()
        effective_rank = torch.exp(entropy).item()
        metrics['effective_rank'] = effective_rank
        metrics['effective_rank_ratio'] = effective_rank / X.shape[1]

        # Variance explained by top-k
        cumsum = eigvals.cumsum(0) / eigvals.sum()
        for pct in [0.5, 0.9, 0.95, 0.99]:
            k = (cumsum < pct).sum().item() + 1
            metrics[f'k_for_{int(pct*100)}pct_var'] = k

    except Exception as e:
        logger.warning(f"Eigenvalue computation failed: {e}")

    # Kurtosis (outlier indicator)
    var = X.var(dim=0)
    var_safe = var.clamp(min=1e-10)
    fourth_moment = ((X - mean) ** 4).mean(dim=0)
    kurtosis = fourth_moment / (var_safe ** 2) - 3
    metrics['mean_kurtosis'] = kurtosis.mean().item()
    metrics['median_kurtosis'] = kurtosis.median().item()
    metrics['max_kurtosis'] = kurtosis.max().item()
    metrics['high_kurtosis_channels'] = (kurtosis > 5).sum().item()

    # Max/RMS ratio per channel
    max_abs = X.abs().max(dim=0)[0]
    rms = torch.sqrt((X ** 2).mean(dim=0))
    rms_safe = rms.clamp(min=1e-10)
    max_to_rms = max_abs / rms_safe
    metrics['mean_max_to_rms'] = max_to_rms.mean().item()
    metrics['max_max_to_rms'] = max_to_rms.max().item()

    return metrics


def analyze_layer(stats, analyzer: RedundancyAnalyzer, has_raw: bool) -> dict:
    """Analyze a single layer's redundancy.

    Args:
        stats: LayerActivationStats for the layer
        analyzer: RedundancyAnalyzer instance
        has_raw: Whether raw activations are available
    """
    report = analyzer.generate_report(stats)

    result = {
        'layer_name': report.layer_name,
        'feature_dim': report.feature_dim,
        'n_samples': report.n_samples,

        # Energy statistics
        'energy_mean': report.energy.mean().item(),
        'energy_std': report.energy.std().item(),
        'energy_median': report.energy.median().item(),
        'energy_min': report.energy.min().item(),
        'energy_max': report.energy.max().item(),
        'low_energy_channels': (report.energy < report.energy.mean() * 0.1).sum().item(),

        # Active rate statistics
        'active_rate_mean': report.active_rate.mean().item(),
        'active_rate_median': report.active_rate.median().item(),
        'active_rate_std': report.active_rate.std().item(),
        'active_rate_min': report.active_rate.min().item(),
        'active_rate_max': report.active_rate.max().item(),
        'low_active_channels': (report.active_rate < 0.1).sum().item(),

        # Active energy score
        'active_energy_score_mean': report.active_energy_score.mean().item(),
        'active_energy_score_std': report.active_energy_score.std().item(),
        'active_energy_score_min': report.active_energy_score.min().item(),

        # Pruning candidates at different thresholds
        'prune_candidates_10pct': (report.active_energy_score < report.active_energy_score.quantile(0.1)).sum().item(),
        'prune_candidates_20pct': (report.active_energy_score < report.active_energy_score.quantile(0.2)).sum().item(),
        'prune_candidates_30pct': (report.active_energy_score < report.active_energy_score.quantile(0.3)).sum().item(),
    }

    # Covariance spectrum
    if report.eigenvalues is not None:
        result['k95'] = report.k95
        result['k95_ratio'] = report.k95 / report.feature_dim
        result['condition_number'] = report.condition_number

        # Top eigenvalue concentration
        top_eig = report.eigenvalues[:10].sum() / report.eigenvalues.sum()
        result['top10_eigenvalue_fraction'] = top_eig.item()

    # Additional metrics from raw activations
    if has_raw:
        additional = compute_additional_metrics(stats)
        result.update(additional)

    return result


def main():
    args = parse_args()

    # Setup output file
    output_dir = Path(__file__).parent / "redundancy_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"redundancy_analysis_{args.model}_{timestamp}.json"
    else:
        output_file = Path(args.output_file)

    logger.info(f"Model: {args.model}")
    logger.info(f"Calibration path: {args.calib_path}")
    logger.info(f"Output file: {output_file}")

    # Load model
    model = load_model(args.model)
    model = model.to(args.device)
    logger.info(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Detect structure
    structure = detect_model_structure(model)
    logger.info(f"Model structure: {structure['num_blocks']} blocks, "
                f"embed_dim={structure['embed_dim']}, mlp_hidden={structure['mlp_hidden_dim']}")

    # Load calibration data
    loader = load_calibration_data(args.calib_path, args.batch_size)
    logger.info(f"Loaded calibration data")

    # Setup collector - collect from MLP hidden activations
    config = CollectorConfig(
        target=PruneTarget.MLP,
        covariance_mode=CovarianceMode.EXACT,
        store_raw=args.store_raw,
        subsample_tokens=args.subsample_tokens,
    )

    collector = ActivationCollector(model, config, device=args.device)

    # Get hook points for MLP activations (after GELU)
    hook_points = []
    for i in range(structure['num_blocks']):
        hook_points.append(f'blocks.{i}.mlp.act')

    collector.register_hooks(hook_points)
    logger.info(f"Registered hooks on {len(hook_points)} layers")

    # Collect activations
    n_samples = 0
    with collector.collect():
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(loader):
                images = images.to(args.device)
                model(images)
                n_samples += images.shape[0]

                if batch_idx % 10 == 0:
                    logger.info(f"Processed {n_samples} samples")

                if n_samples >= args.calib_samples:
                    break

    collector.clear_hooks()
    logger.info(f"Collected activations from {n_samples} samples")

    all_stats = collector.get_all_stats()
    has_raw = args.store_raw

    # Analyze each layer
    analyzer = RedundancyAnalyzer()

    results = {
        'model': args.model,
        'calib_samples': n_samples,
        'timestamp': datetime.now().isoformat(),
        'model_structure': structure,
        'layers': {},
    }

    logger.info("\n" + "=" * 80)
    logger.info("REDUNDANCY ANALYSIS RESULTS")
    logger.info("=" * 80)

    for layer_name, stats in sorted(all_stats.items()):
        layer_result = analyze_layer(stats, analyzer, has_raw)
        results['layers'][layer_name] = layer_result

        # Print summary for each layer
        logger.info(f"\n{layer_name}:")
        logger.info(f"  Feature dim: {layer_result['feature_dim']}")
        logger.info(f"  Energy - mean: {layer_result['energy_mean']:.4f}, "
                   f"low channels: {layer_result['low_energy_channels']}")
        logger.info(f"  Active rate - mean: {layer_result['active_rate_mean']:.4f}, "
                   f"median: {layer_result['active_rate_median']:.4f}, "
                   f"low channels: {layer_result['low_active_channels']}")
        if 'effective_rank' in layer_result:
            logger.info(f"  Effective rank: {layer_result['effective_rank']:.1f} "
                       f"({layer_result['effective_rank_ratio']*100:.1f}% of dim)")
        if 'k95' in layer_result:
            logger.info(f"  k95: {layer_result['k95']} ({layer_result['k95_ratio']*100:.1f}% of dim)")
        if 'highly_correlated_pairs' in layer_result:
            logger.info(f"  Highly correlated pairs: {layer_result['highly_correlated_pairs']}")
        logger.info(f"  Prune candidates @30%: {layer_result['prune_candidates_30pct']}")

    # Compute summary statistics
    layers = results['layers'].values()
    summary = {
        'total_channels': sum(r['feature_dim'] for r in layers),
        'total_low_energy': sum(r['low_energy_channels'] for r in layers),
        'total_low_active': sum(r['low_active_channels'] for r in layers),
    }

    if has_raw:
        summary['mean_effective_rank_ratio'] = sum(
            r.get('effective_rank_ratio', 0) for r in layers
        ) / len(results['layers'])

    summary['mean_k95_ratio'] = sum(
        r.get('k95_ratio', 0) for r in layers
    ) / len(results['layers'])

    results['summary'] = summary

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total channels: {summary['total_channels']}")
    logger.info(f"Total low-energy channels: {summary['total_low_energy']}")
    logger.info(f"Total low-active channels: {summary['total_low_active']}")
    if 'mean_effective_rank_ratio' in summary:
        logger.info(f"Mean effective rank ratio: {summary['mean_effective_rank_ratio']*100:.1f}%")
    logger.info(f"Mean k95 ratio: {summary['mean_k95_ratio']*100:.1f}%")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
