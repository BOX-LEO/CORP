"""
Redundancy Statistics for Structured Pruning.

Computes energy, active rate, covariance spectrum, and other metrics
for analyzing activation redundancy.
"""

import torch
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
import logging

from .collect import LayerActivationStats

logger = logging.getLogger(__name__)


@dataclass
class QKDimReport:
    """Report containing Q/K dimension metrics for attention pruning.

    Attributes:
        layer_name: Name of the attention layer
        head_idx: Index of the attention head
        head_dim: Dimension per head
        n_samples: Number of samples used for computation
        q_energy: E[q_j^2] per dimension - shape (head_dim,)
        k_energy: E[k_j^2] per dimension - shape (head_dim,)
        joint_score: sqrt(E[q_j^2] * E[k_j^2]) - shape (head_dim,) (proxy score)
        qk_energy: E[q_j^2 * k_j^2] per dimension - shape (head_dim,) (exact score)
        q_stats: LayerActivationStats for Q
        k_stats: LayerActivationStats for K
        qk_stats: Optional LayerActivationStats for Q/K joint (contains sum_q2k2)
    """
    layer_name: str
    head_idx: int
    head_dim: int
    n_samples: int
    q_energy: torch.Tensor
    k_energy: torch.Tensor
    joint_score: torch.Tensor
    q_stats: 'LayerActivationStats'
    k_stats: 'LayerActivationStats'
    qk_energy: Optional[torch.Tensor] = None
    qk_stats: Optional['LayerActivationStats'] = None


@dataclass
class RedundancyReport:
    """Report containing redundancy metrics for a layer.

    Attributes:
        layer_name: Name of the layer
        feature_dim: Number of features/channels
        n_samples: Number of samples used for computation

        energy: Energy per feature E[x^2] - shape (feature_dim,)
        active_rate: Active rate per feature P(|x| > eps) - shape (feature_dim,)
        active_energy_score: energy * active_rate - shape (feature_dim,)

        eigenvalues: Eigenvalues of covariance matrix (if computed)
        k95: Number of components for 95% variance
        condition_number: Condition number of covariance

        kurtosis: Excess kurtosis per feature (for outlier detection)
        max_to_rms_ratio: max(|x|) / RMS(x) per feature

        residual_ratio: ||output|| / ||input|| for layer
    """
    layer_name: str
    feature_dim: int
    n_samples: int

    energy: torch.Tensor
    active_rate: torch.Tensor
    active_energy_score: torch.Tensor

    eigenvalues: Optional[torch.Tensor] = None
    k95: Optional[int] = None
    condition_number: Optional[float] = None

    kurtosis: Optional[torch.Tensor] = None
    max_to_rms_ratio: Optional[torch.Tensor] = None

    residual_ratio: Optional[float] = None


class RedundancyAnalyzer:
    """Analyzes activation statistics to compute redundancy metrics."""

    def __init__(
        self,
        eps_active_ratio: float = 0.01,
        variance_threshold: float = 0.95,
    ):
        """Initialize the analyzer.

        Args:
            eps_active_ratio: Ratio of RMS to use as activation threshold
            variance_threshold: Fraction of variance for k95 computation
        """
        self.eps_active_ratio = eps_active_ratio
        self.variance_threshold = variance_threshold

    def compute_energy(self, stats: LayerActivationStats) -> torch.Tensor:
        """Compute energy (mean squared activation) per feature.

        Energy = E[x^2] = sum_x2 / n

        Args:
            stats: Activation statistics

        Returns:
            Energy tensor of shape (feature_dim,)
        """
        if stats.n_samples == 0:
            return torch.zeros(stats.feature_dim)

        energy = stats.sum_x2 / stats.n_samples
        return energy

    def compute_active_rate(self, stats: LayerActivationStats) -> torch.Tensor:
        """Get pre-computed active rate from stats.

        Active rate = P(|x| > eps) where eps = 0.01 * RMS(layer)

        Args:
            stats: Activation statistics

        Returns:
            Active rate tensor of shape (feature_dim,)
        """
        return stats.active_rate

    def compute_active_energy_score(
        self,
        energy: torch.Tensor,
        active_rate: torch.Tensor,
    ) -> torch.Tensor:
        """Compute active energy importance score.

        active_energy = energy * active_rate

        Features with low energy AND low active rate are most prunable.

        Args:
            energy: Energy per feature
            active_rate: Active rate per feature

        Returns:
            Active energy score tensor
        """
        return energy * active_rate

    def compute_covariance_spectrum(
        self,
        stats: LayerActivationStats,
    ) -> Tuple[Optional[torch.Tensor], Optional[int], Optional[float]]:
        """Compute eigenvalue spectrum of covariance matrix.

        Args:
            stats: Activation statistics with covariance

        Returns:
            Tuple of (eigenvalues, k95, condition_number)
            k95 is the number of components explaining variance_threshold
        """
        if stats.covariance is None:
            return None, None, None

        try:
            # Compute eigenvalues (covariance is symmetric PSD)
            cov = stats.covariance.double()

            # Symmetrize to handle numerical errors
            cov = (cov + cov.T) / 2

            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues.flip(0)  # Descending order
            eigenvalues = eigenvalues.clamp(min=0)  # Handle numerical negatives

            # Compute k95 (number of components for 95% variance)
            total_var = eigenvalues.sum()
            if total_var > 0:
                cumsum = eigenvalues.cumsum(0)
                k95 = (cumsum < self.variance_threshold * total_var).sum().item() + 1
                k95 = min(k95, len(eigenvalues))
            else:
                k95 = len(eigenvalues)

            # Condition number
            max_eig = eigenvalues[0].item()
            min_eig_pos = eigenvalues[eigenvalues > 1e-10]
            if len(min_eig_pos) > 0:
                condition_number = max_eig / min_eig_pos[-1].item()
            else:
                condition_number = float('inf')

            return eigenvalues.float(), k95, condition_number

        except RuntimeError as e:
            logger.warning(f"Eigenvalue computation failed: {e}")
            return None, None, None

    def compute_kurtosis(self, stats: LayerActivationStats) -> Optional[torch.Tensor]:
        """Compute excess kurtosis per feature.

        Kurtosis = E[(x - mu)^4] / var^2 - 3

        High kurtosis indicates outlier-prone features.

        Note: Requires raw activations (store_raw=True in collector).

        Args:
            stats: Activation statistics

        Returns:
            Kurtosis tensor or None if raw activations unavailable
        """
        if stats.raw_activations is None or len(stats.raw_activations) == 0:
            return None

        x = torch.cat(stats.raw_activations, dim=0)
        mean = x.mean(dim=0)
        var = x.var(dim=0)

        # Avoid division by zero
        var_safe = var.clamp(min=1e-10)
        centered = x - mean
        fourth_moment = (centered ** 4).mean(dim=0)
        kurtosis = fourth_moment / (var_safe ** 2) - 3

        return kurtosis

    def compute_max_to_rms_ratio(
        self,
        stats: LayerActivationStats,
    ) -> Optional[torch.Tensor]:
        """Compute max-to-RMS ratio per feature.

        Ratio = max(|x|) / RMS(x)

        High ratio indicates potential outlier channels.

        Note: Requires tracking max values (not currently in basic stats).

        Args:
            stats: Activation statistics

        Returns:
            Ratio tensor or None
        """
        if stats.raw_activations is None or len(stats.raw_activations) == 0:
            return None

        x = torch.cat(stats.raw_activations, dim=0)
        max_abs = x.abs().max(dim=0)[0]
        rms = torch.sqrt((x ** 2).mean(dim=0))

        rms_safe = rms.clamp(min=1e-10)
        ratio = max_abs / rms_safe

        return ratio

    def compute_residual_ratio(
        self,
        input_stats: LayerActivationStats,
        output_stats: LayerActivationStats,
    ) -> float:
        """Compute residual ratio ||f(x)|| / ||x||.

        Args:
            input_stats: Statistics for layer input
            output_stats: Statistics for layer output

        Returns:
            Ratio of output to input RMS
        """
        input_rms = torch.sqrt(self.compute_energy(input_stats).mean()).item()
        output_rms = torch.sqrt(self.compute_energy(output_stats).mean()).item()

        if input_rms < 1e-10:
            return float('inf')

        return output_rms / input_rms

    def generate_report(
        self,
        stats: LayerActivationStats,
        input_stats: Optional[LayerActivationStats] = None,
    ) -> RedundancyReport:
        """Generate a complete redundancy report for a layer.

        Args:
            stats: Activation statistics for the layer
            input_stats: Optional input statistics for residual ratio

        Returns:
            RedundancyReport with all computed metrics
        """
        energy = self.compute_energy(stats)
        active_rate = self.compute_active_rate(stats)
        active_energy_score = self.compute_active_energy_score(energy, active_rate)

        eigenvalues, k95, condition_number = self.compute_covariance_spectrum(stats)
        kurtosis = self.compute_kurtosis(stats)
        max_to_rms = self.compute_max_to_rms_ratio(stats)

        residual_ratio = None
        if input_stats is not None:
            residual_ratio = self.compute_residual_ratio(input_stats, stats)

        return RedundancyReport(
            layer_name=stats.layer_name,
            feature_dim=stats.feature_dim,
            n_samples=stats.n_samples,
            energy=energy,
            active_rate=active_rate,
            active_energy_score=active_energy_score,
            eigenvalues=eigenvalues,
            k95=k95,
            condition_number=condition_number,
            kurtosis=kurtosis,
            max_to_rms_ratio=max_to_rms,
            residual_ratio=residual_ratio,
        )

    def report_from_stats(self, stats: LayerActivationStats) -> RedundancyReport:
        """Create a lightweight report from stats for ranking (no eigvalsh).

        Computes only the cheap metrics needed by rankers: energy, active_rate,
        active_energy_score, kurtosis, max_to_rms_ratio.  Skips the expensive
        eigenvalue decomposition which is only useful for diagnostics.
        """
        energy = self.compute_energy(stats)
        active_rate = self.compute_active_rate(stats)
        active_energy_score = self.compute_active_energy_score(energy, active_rate)
        kurtosis = self.compute_kurtosis(stats)
        max_to_rms = self.compute_max_to_rms_ratio(stats)

        return RedundancyReport(
            layer_name=stats.layer_name,
            feature_dim=stats.feature_dim,
            n_samples=stats.n_samples,
            energy=energy,
            active_rate=active_rate,
            active_energy_score=active_energy_score,
            kurtosis=kurtosis,
            max_to_rms_ratio=max_to_rms,
        )

    def reports_from_all_stats(
        self,
        all_stats: Dict[str, LayerActivationStats],
    ) -> Dict[str, RedundancyReport]:
        """Create lightweight reports for all layers (no eigvalsh).

        Use generate_all_reports() instead if you need eigenvalue diagnostics.
        """
        return {name: self.report_from_stats(s) for name, s in all_stats.items()}

    def generate_all_reports(
        self,
        all_stats: Dict[str, LayerActivationStats],
    ) -> Dict[str, RedundancyReport]:
        """Generate full diagnostic reports for all layers (includes eigvalsh).

        This is expensive for large layers. Use reports_from_all_stats() for
        the pruning pipeline where eigenvalue diagnostics are not needed.

        Args:
            all_stats: Dictionary of layer name to stats

        Returns:
            Dictionary of layer name to report
        """
        reports = {}
        for name, stats in all_stats.items():
            reports[name] = self.generate_report(stats)
        return reports

    def compute_qk_joint_score(
        self,
        q_stats: LayerActivationStats,
        k_stats: LayerActivationStats,
    ) -> torch.Tensor:
        """Compute joint importance score for Q/K dimensions (proxy).

        Score = sqrt(E[q_j^2] * E[k_j^2])

        This captures the contribution of each dimension to the attention logits,
        as logit_contribution ~ q_j * k_j.

        Args:
            q_stats: Statistics for Q activations
            k_stats: Statistics for K activations

        Returns:
            Joint score tensor of shape (head_dim,)
        """
        q_energy = self.compute_energy(q_stats)
        k_energy = self.compute_energy(k_stats)
        joint_score = torch.sqrt(q_energy * k_energy + 1e-10)
        return joint_score

    def compute_qk_energy_score(
        self,
        qk_stats: LayerActivationStats,
    ) -> Optional[torch.Tensor]:
        """Compute exact joint energy score E[q_j^2 * k_j^2] for Q/K dimensions.

        This is the exact joint energy score computed from accumulated q^2 * k^2.

        Args:
            qk_stats: Statistics containing sum_q2k2 from Q/K joint collection

        Returns:
            Exact qk_energy tensor of shape (head_dim,), or None if not available
        """
        if qk_stats is None or qk_stats.sum_q2k2 is None:
            return None
        if qk_stats.n_samples == 0:
            return None
        return (qk_stats.sum_q2k2 / qk_stats.n_samples).float()

    def compute_gram_matrices(
        self,
        q_stats: LayerActivationStats,
        k_stats: LayerActivationStats,
        prune_idx: torch.Tensor,
        surv_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute Gram matrices for Sylvester equation solver.

        Computes:
            G_Q = n * (Σ_SS_Q + μ_S_Q @ μ_S_Q^T)   # (d_s, d_s)
            G_K = n * (Σ_SS_K + μ_S_K @ μ_S_K^T)   # (d_s, d_s)
            C_Q = n * (Σ_SP_Q + μ_S_Q @ μ_P_Q^T)   # (d_s, d_p)
            C_K = n * (Σ_PS_K + μ_P_K @ μ_S_K^T)   # (d_p, d_s)

        These are used to solve the Sylvester equation for dim-logit compensation.

        Args:
            q_stats: Statistics for Q activations (with covariance)
            k_stats: Statistics for K activations (with covariance)
            prune_idx: Indices of dimensions to prune
            surv_idx: Indices of surviving dimensions

        Returns:
            Dictionary with G_Q, G_K, C_Q, C_K tensors
        """
        if q_stats.covariance is None or k_stats.covariance is None:
            raise ValueError("Covariance required for Gram matrix computation")

        n = q_stats.n_samples
        p_idx = prune_idx.tolist()
        s_idx = surv_idx.tolist()

        # Get covariance submatrices for Q
        q_cov = q_stats.covariance.double()
        q_mean = q_stats.mean.double()
        q_Sigma_SS = q_cov[s_idx][:, s_idx]  # (d_s, d_s)
        q_Sigma_SP = q_cov[s_idx][:, p_idx]  # (d_s, d_p)
        q_mu_S = q_mean[s_idx]  # (d_s,)
        q_mu_P = q_mean[p_idx]  # (d_p,)

        # Get covariance submatrices for K
        k_cov = k_stats.covariance.double()
        k_mean = k_stats.mean.double()
        k_Sigma_SS = k_cov[s_idx][:, s_idx]  # (d_s, d_s)
        k_Sigma_PS = k_cov[p_idx][:, s_idx]  # (d_p, d_s)
        k_mu_S = k_mean[s_idx]  # (d_s,)
        k_mu_P = k_mean[p_idx]  # (d_p,)

        # Compute Gram matrices
        # G_Q = n * (Σ_SS_Q + μ_S_Q @ μ_S_Q^T) = E[q_S q_S^T] * n
        G_Q = n * (q_Sigma_SS + torch.outer(q_mu_S, q_mu_S))

        # G_K = n * (Σ_SS_K + μ_S_K @ μ_S_K^T) = E[k_S k_S^T] * n
        G_K = n * (k_Sigma_SS + torch.outer(k_mu_S, k_mu_S))

        # C_Q = n * (Σ_SP_Q + μ_S_Q @ μ_P_Q^T)
        C_Q = n * (q_Sigma_SP + torch.outer(q_mu_S, q_mu_P))

        # C_K = n * (Σ_PS_K + μ_P_K @ μ_S_K^T)
        C_K = n * (k_Sigma_PS + torch.outer(k_mu_P, k_mu_S))

        return {
            'G_Q': G_Q,
            'G_K': G_K,
            'C_Q': C_Q,
            'C_K': C_K,
            'n_samples': n,
        }

    def generate_qk_dim_report(
        self,
        q_stats: LayerActivationStats,
        k_stats: LayerActivationStats,
        layer_name: str,
        head_idx: int,
        qk_stats: Optional[LayerActivationStats] = None,
    ) -> QKDimReport:
        """Generate a Q/K dimension report for a single attention head.

        Args:
            q_stats: Statistics for Q activations of this head
            k_stats: Statistics for K activations of this head
            layer_name: Name of the attention layer
            head_idx: Index of the attention head
            qk_stats: Optional statistics containing sum_q2k2 for exact joint energy

        Returns:
            QKDimReport with dimension-level metrics
        """
        q_energy = self.compute_energy(q_stats)
        k_energy = self.compute_energy(k_stats)
        joint_score = self.compute_qk_joint_score(q_stats, k_stats)

        # Compute exact qk_energy if qk_stats available
        qk_energy = self.compute_qk_energy_score(qk_stats) if qk_stats is not None else None

        return QKDimReport(
            layer_name=layer_name,
            head_idx=head_idx,
            head_dim=q_stats.feature_dim,
            n_samples=q_stats.n_samples,
            q_energy=q_energy,
            k_energy=k_energy,
            joint_score=joint_score,
            q_stats=q_stats,
            k_stats=k_stats,
            qk_energy=qk_energy,
            qk_stats=qk_stats,
        )

    def generate_all_qk_dim_reports(
        self,
        all_stats: Dict[str, LayerActivationStats],
    ) -> Dict[str, QKDimReport]:
        """Generate Q/K dimension reports for all attention heads.

        Looks for stats keys like '{layer}.q.head_{h}', '{layer}.k.head_{h}',
        and '{layer}.qk.head_{h}' and generates reports for each head.

        Args:
            all_stats: Dictionary of layer name to stats

        Returns:
            Dictionary of '{layer}.attn.head_{h}' to QKDimReport
        """
        reports = {}

        # Group stats by layer and head
        # Keys are like 'blocks.0.attn.q.head_0', 'blocks.0.attn.k.head_0', 'blocks.0.attn.qk.head_0'
        q_stats_by_head = {}
        k_stats_by_head = {}
        qk_stats_by_head = {}

        for key, stats in all_stats.items():
            if '.q.head_' in key:
                # Extract layer name and head index
                # e.g., 'blocks.0.attn.q.head_0' -> layer='blocks.0.attn', head=0
                parts = key.rsplit('.q.head_', 1)
                layer_name = parts[0]
                head_idx = int(parts[1])
                q_stats_by_head[(layer_name, head_idx)] = stats
            elif '.k.head_' in key:
                parts = key.rsplit('.k.head_', 1)
                layer_name = parts[0]
                head_idx = int(parts[1])
                k_stats_by_head[(layer_name, head_idx)] = stats
            elif '.qk.head_' in key:
                # Joint Q/K stats for exact energy score
                parts = key.rsplit('.qk.head_', 1)
                layer_name = parts[0]
                head_idx = int(parts[1])
                qk_stats_by_head[(layer_name, head_idx)] = stats

        # Generate reports for matching Q/K pairs
        for (layer_name, head_idx), q_stats in q_stats_by_head.items():
            k_stats = k_stats_by_head.get((layer_name, head_idx))
            if k_stats is None:
                logger.warning(f"Missing K stats for {layer_name} head {head_idx}")
                continue

            # Get optional qk_stats for exact energy score
            qk_stats = qk_stats_by_head.get((layer_name, head_idx))

            report = self.generate_qk_dim_report(
                q_stats, k_stats, layer_name, head_idx, qk_stats
            )
            report_key = f"{layer_name}.attn.head_{head_idx}"
            reports[report_key] = report

        return reports
