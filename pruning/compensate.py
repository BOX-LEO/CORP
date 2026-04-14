"""
Affine Compensation for Structured Pruning.

Implements ridge regression compensation: x_P ≈ B @ x_S + c
to recover pruned channel outputs through surviving channels.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

from .collect import LayerActivationStats

logger = logging.getLogger(__name__)


@dataclass
class QKDimCompensationResult:
    """Result of Q/K dimension compensation fitting (dim-logit mode).

    Attributes:
        mode: Compensation mode ('dim-logit')
        layer_name: Name of the attention layer
        head_idx: Index of the attention head
        prune_indices: Indices of pruned dimensions
        survivor_indices: Indices of surviving dimensions
        U: Transform matrix for Q weights (n_surv, n_surv) - Sylvester method
        V: Transform matrix for K weights (n_surv, n_surv) - Sylvester method
        mu_q_S: Mean of Q survivors (n_surv,)
        mu_k_S: Mean of K survivors (n_surv,)
        lambda_used: Regularization strength used
    """
    mode: str
    layer_name: str
    head_idx: int
    prune_indices: torch.Tensor
    survivor_indices: torch.Tensor
    U: Optional[torch.Tensor] = None  # Transform for Q
    V: Optional[torch.Tensor] = None  # Transform for K
    mu_q_S: Optional[torch.Tensor] = None
    mu_k_S: Optional[torch.Tensor] = None
    lambda_used: float = 0.0


@dataclass
class CompensationResult:
    """Result of affine compensation fitting.

    Attributes:
        B: Linear compensation matrix (n_pruned, n_survivors)
        c: Bias compensation vector (n_pruned,)
        prune_indices: Indices of pruned features
        survivor_indices: Indices of surviving features
        lambda_used: Regularization strength used
        condition_number: Condition number of (Sigma_SS + lambda*I)
        reconstruction_error: MSE of reconstruction on calibration data
    """
    B: torch.Tensor
    c: torch.Tensor
    prune_indices: torch.Tensor
    survivor_indices: torch.Tensor
    lambda_used: float
    condition_number: float
    reconstruction_error: Optional[float] = None


class AffineCompensator:
    """Fits affine compensation to recover pruned channel outputs.

    Given activations X with survivors X_S and pruned X_P, fits:
        X_P ≈ X_S @ B.T + c

    The compensation can then be folded into output weights:
        W_out @ X = W_out_S @ X_S + W_out_P @ X_P
                  ≈ W_out_S @ X_S + W_out_P @ (X_S @ B.T + c)
                  = (W_out_S + W_out_P @ B) @ X_S + W_out_P @ c
    """

    def __init__(
        self,
        lambda_reg: float = 1e-3,
        auto_shrinkage: bool = True,
        solve_dtype: torch.dtype = torch.float64,
    ):
        """Initialize the compensator.

        Args:
            lambda_reg: Ridge regularization strength
            auto_shrinkage: Use Oracle Approximating Shrinkage for lambda
            solve_dtype: Data type for ridge solve (recommend float64)
        """
        self.lambda_reg = lambda_reg
        self.auto_shrinkage = auto_shrinkage
        self.solve_dtype = solve_dtype

    def fit(
        self,
        stats: LayerActivationStats,
        prune_indices: torch.Tensor,
        survivor_indices: torch.Tensor,
    ) -> CompensationResult:
        """Fit affine compensation from activation statistics.

        Uses covariance matrices to compute compensation without storing
        all raw activations.

        Args:
            stats: Layer activation statistics with covariance
            prune_indices: Indices of features to prune
            survivor_indices: Indices of surviving features

        Returns:
            CompensationResult with B, c matrices
        """
        if stats.covariance is None:
            raise ValueError("Covariance matrix required for compensation fitting")

        n_prune = len(prune_indices)
        n_surv = len(survivor_indices)

        if n_prune == 0:
            return CompensationResult(
                B=torch.zeros(0, n_surv),
                c=torch.zeros(0),
                prune_indices=prune_indices,
                survivor_indices=survivor_indices,
                lambda_used=self.lambda_reg,
                condition_number=1.0,
            )

        # Extract submatrices from full covariance
        cov = stats.covariance.to(self.solve_dtype)
        mean = stats.mean.to(self.solve_dtype)

        # Get indices as lists for indexing
        p_idx = prune_indices.tolist()
        s_idx = survivor_indices.tolist()

        # Extract covariance submatrices
        # Sigma_SS = Cov(X_S, X_S) - shape (n_surv, n_surv)
        Sigma_SS = cov[s_idx][:, s_idx]

        # Sigma_PS = Cov(X_P, X_S) - shape (n_prune, n_surv)
        Sigma_PS = cov[p_idx][:, s_idx]

        # Extract means
        mu_S = mean[s_idx]  # (n_surv,)
        mu_P = mean[p_idx]  # (n_prune,)

        # Determine lambda
        lambda_val = self._compute_lambda(Sigma_SS, stats.n_samples)

        # Ridge solve: B = Sigma_PS @ (Sigma_SS + lambda*I)^-1
        # Using Cholesky for numerical stability
        n_s = Sigma_SS.shape[0]
        regularized = Sigma_SS + lambda_val * torch.eye(n_s, dtype=self.solve_dtype)

        try:
            L = torch.linalg.cholesky(regularized)
            # Solve L @ L.T @ B.T = Sigma_PS.T
            # First: L @ y = Sigma_PS.T
            y = torch.linalg.solve_triangular(L, Sigma_PS.T, upper=False)
            # Then: L.T @ B.T = y
            B_T = torch.linalg.solve_triangular(L.T, y, upper=True)
            B = B_T.T  # (n_prune, n_surv)

        except RuntimeError as e:
            logger.warning(f"Cholesky failed, falling back to lstsq: {e}")
            # Fallback to least squares
            B = torch.linalg.lstsq(regularized, Sigma_PS.T).solution.T

        # Compute bias: c = mu_P - B @ mu_S
        c = mu_P - B @ mu_S

        # Compute condition number
        try:
            eigvals = torch.linalg.eigvalsh(regularized)
            cond = (eigvals.max() / eigvals.min()).item()
        except RuntimeError:
            cond = float('inf')

        # Convert back to float32
        B_out = B.float()
        c_out = c.float()

        logger.debug(
            f"Compensation fit: B shape {B_out.shape}, "
            f"lambda={lambda_val:.2e}, cond={cond:.2e}"
        )

        return CompensationResult(
            B=B_out,
            c=c_out,
            prune_indices=prune_indices,
            survivor_indices=survivor_indices,
            lambda_used=lambda_val,
            condition_number=cond,
        )

    def fit_from_raw(
        self,
        X: torch.Tensor,
        prune_indices: torch.Tensor,
        survivor_indices: torch.Tensor,
    ) -> CompensationResult:
        """Fit affine compensation from raw activations.

        Direct computation from raw data, useful for validation.

        Args:
            X: Raw activations, shape (n_samples, feature_dim)
            prune_indices: Indices of features to prune
            survivor_indices: Indices of surviving features

        Returns:
            CompensationResult with B, c matrices
        """
        n_samples = X.shape[0]
        n_prune = len(prune_indices)
        n_surv = len(survivor_indices)

        if n_prune == 0:
            return CompensationResult(
                B=torch.zeros(0, n_surv),
                c=torch.zeros(0),
                prune_indices=prune_indices,
                survivor_indices=survivor_indices,
                lambda_used=self.lambda_reg,
                condition_number=1.0,
            )

        # Extract subsets
        X_S = X[:, survivor_indices].to(self.solve_dtype)  # (n, n_surv)
        X_P = X[:, prune_indices].to(self.solve_dtype)     # (n, n_prune)

        # Compute means
        mu_S = X_S.mean(dim=0)  # (n_surv,)
        mu_P = X_P.mean(dim=0)  # (n_prune,)

        # Center
        X_S_c = X_S - mu_S
        X_P_c = X_P - mu_P

        # Compute covariances
        Sigma_SS = (X_S_c.T @ X_S_c) / n_samples  # (n_surv, n_surv)
        Sigma_PS = (X_P_c.T @ X_S_c) / n_samples  # (n_prune, n_surv)

        # Determine lambda
        lambda_val = self._compute_lambda(Sigma_SS, n_samples)

        # Ridge solve
        n_s = Sigma_SS.shape[0]
        regularized = Sigma_SS + lambda_val * torch.eye(n_s, dtype=self.solve_dtype)

        try:
            L = torch.linalg.cholesky(regularized)
            y = torch.linalg.solve_triangular(L, Sigma_PS.T, upper=False)
            B_T = torch.linalg.solve_triangular(L.T, y, upper=True)
            B = B_T.T
        except RuntimeError as e:
            logger.warning(f"Cholesky failed: {e}")
            B = torch.linalg.lstsq(regularized, Sigma_PS.T).solution.T

        # Bias
        c = mu_P - B @ mu_S

        # Condition number
        try:
            eigvals = torch.linalg.eigvalsh(regularized)
            cond = (eigvals.max() / eigvals.min()).item()
        except RuntimeError:
            cond = float('inf')

        # Reconstruction error
        X_P_pred = X_S @ B.T + c
        mse = ((X_P - X_P_pred) ** 2).mean().item()

        return CompensationResult(
            B=B.float(),
            c=c.float(),
            prune_indices=prune_indices,
            survivor_indices=survivor_indices,
            lambda_used=lambda_val,
            condition_number=cond,
            reconstruction_error=mse,
        )

    def _compute_lambda(
        self,
        Sigma_SS: torch.Tensor,
        n_samples: int,
    ) -> float:
        """Compute regularization strength.

        Args:
            Sigma_SS: Survivor covariance matrix
            n_samples: Number of samples

        Returns:
            Regularization strength
        """
        if not self.auto_shrinkage:
            return self.lambda_reg

        # Oracle Approximating Shrinkage (OAS) estimator
        # See Chen et al. "Shrinkage Algorithms for MMSE Covariance Estimation"
        n = n_samples
        p = Sigma_SS.shape[0]

        if n <= 1 or p == 0:
            return self.lambda_reg

        trace_S = torch.trace(Sigma_SS)
        trace_S2 = torch.trace(Sigma_SS @ Sigma_SS)

        # Ledoit-Wolf shrinkage intensity
        try:
            rho = min(
                1.0,
                ((1 - 2/p) * trace_S2 + trace_S**2) /
                ((n + 1 - 2/p) * (trace_S2 - trace_S**2 / p))
            )
        except (ZeroDivisionError, RuntimeError):
            rho = 0.5

        # Scale lambda by trace to be scale-invariant
        lambda_scaled = rho * trace_S / p

        # Combine with base lambda
        lambda_final = max(self.lambda_reg, lambda_scaled.item())

        logger.debug(f"Auto-shrinkage: rho={rho:.4f}, lambda={lambda_final:.2e}")

        return lambda_final

    def fold_into_weights(
        self,
        W_out: torch.Tensor,
        b_out: Optional[torch.Tensor],
        result: CompensationResult,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fold compensation into output weights.

        Given output layer: y = W_out @ x + b_out
        After pruning: y = W_out_tilde @ x_S + b_out_tilde

        Where:
            W_out_tilde = W_out[:, surv_idx] + W_out[:, prune_idx] @ B
            b_out_tilde = b_out + W_out[:, prune_idx] @ c

        Args:
            W_out: Output weight matrix, shape (out_features, in_features)
            b_out: Output bias, shape (out_features,) or None
            result: Compensation result

        Returns:
            Tuple of (W_out_tilde, b_out_tilde)
        """
        prune_idx = result.prune_indices.tolist()
        surv_idx = result.survivor_indices.tolist()

        # Get weight submatrices
        W_S = W_out[:, surv_idx]   # (out, n_surv)
        W_P = W_out[:, prune_idx]  # (out, n_prune)

        # Fold: W_tilde = W_S + W_P @ B
        W_tilde = W_S + W_P @ result.B  # (out, n_surv)

        # Fold bias
        if b_out is None:
            b_out = torch.zeros(W_out.shape[0], device=W_out.device, dtype=W_out.dtype)
        b_tilde = b_out + W_P @ result.c  # (out,)

        return W_tilde, b_tilde


def validate_compensation(
    X: torch.Tensor,
    W_out: torch.Tensor,
    b_out: Optional[torch.Tensor],
    result: CompensationResult,
    W_out_tilde: torch.Tensor,
    b_out_tilde: torch.Tensor,
) -> dict:
    """Validate compensation by comparing outputs before and after.

    Args:
        X: Raw activations, shape (n_samples, feature_dim)
        W_out: Original output weights
        b_out: Original output bias
        result: Compensation result
        W_out_tilde: Folded output weights
        b_out_tilde: Folded output bias

    Returns:
        Dictionary with validation metrics
    """
    surv_idx = result.survivor_indices.tolist()

    # Original output
    if b_out is None:
        b_out = torch.zeros(W_out.shape[0], device=W_out.device, dtype=W_out.dtype)
    y_orig = X @ W_out.T + b_out

    # Pruned output (only survivors)
    X_S = X[:, surv_idx]
    y_pruned = X_S @ W_out_tilde.T + b_out_tilde

    # Metrics
    mse = ((y_orig - y_pruned) ** 2).mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        y_orig.flatten(), y_pruned.flatten(), dim=0
    ).item()
    rel_error = mse / (y_orig ** 2).mean().item()

    return {
        'mse': mse,
        'cosine_similarity': cosine,
        'relative_error': rel_error,
    }


def _solve_sylvester(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    lambda_val: float,
    max_kronecker_dim: int = 64,
) -> torch.Tensor:
    """Solve the Sylvester equation: A @ X @ B + λX = C

    Tries methods in order:
    1. torch.linalg.solve_sylvester (PyTorch 2.0+, not yet available)
    2. Vectorized Kronecker approach: (B^T ⊗ A + λI) @ vec(X) = vec(C)
    3. scipy.linalg.solve_sylvester fallback

    Args:
        A: Left matrix (d_s, d_s)
        B: Right matrix (d_s, d_s)
        C: RHS matrix (d_s, d_s)
        lambda_val: Regularization parameter
        max_kronecker_dim: Max dimension for Kronecker approach (memory safety)

    Returns:
        Solution X (d_s, d_s)
    """
    d_s = A.shape[0]
    device = A.device
    dtype = A.dtype

    # Method 1: Try PyTorch native (not yet available in stable PyTorch)
    if hasattr(torch.linalg, 'solve_sylvester'):
        try:
            # Note: PyTorch solve_sylvester solves A @ X + X @ B = C
            # We have A @ X @ B + λX = C, which is different
            # Need to transform the equation
            pass  # Not directly usable, skip to other methods
        except Exception:
            pass

    # Method 2: Vectorized Kronecker approach (only for small d_s)
    # Equation: A @ X @ B + λX = C
    # Vec form (column-major): (B^T ⊗ A + λI) @ vec_col(X) = vec_col(C)
    # where vec_col(X) = X.T.reshape(-1) (column-major vectorization)
    if d_s <= max_kronecker_dim:
        try:
            # Compute Kronecker product: B^T ⊗ A
            # kron(B^T, A) has shape (d_s^2, d_s^2)
            B_T = B.T.contiguous()
            A_contig = A.contiguous()
            kron = torch.kron(B_T, A_contig)

            # Add regularization: kron + λI
            kron_reg = kron + lambda_val * torch.eye(d_s * d_s, device=device, dtype=dtype)

            # Solve linear system using column-major vectorization
            # vec_col(C) = C.T.contiguous().reshape(-1)
            vec_C = C.T.contiguous().reshape(-1)
            vec_X = torch.linalg.solve(kron_reg, vec_C)

            # Convert back from column-major: X = vec_X.reshape(d_s, d_s).T
            X = vec_X.reshape(d_s, d_s).T
            return X
        except RuntimeError as e:
            logger.warning(f"Kronecker Sylvester solve failed: {e}")

    # Method 3: scipy/numpy fallback with Kronecker approach
    try:
        import numpy as np

        A_np = A.cpu().numpy()
        B_np = B.cpu().numpy()
        C_np = C.cpu().numpy()

        # Build Kronecker system using column-major vectorization
        kron_np = np.kron(B_np.T, A_np) + lambda_val * np.eye(d_s * d_s)
        # Column-major vectorization: vec_col(C) = C.T.reshape(-1)
        vec_C_np = C_np.T.reshape(-1)
        vec_X_np = np.linalg.solve(kron_np, vec_C_np)
        # Convert back from column-major
        X_np = vec_X_np.reshape(d_s, d_s).T

        X = torch.from_numpy(X_np).to(device=device, dtype=dtype)
        return X
    except Exception as e:
        logger.warning(f"numpy Sylvester solve failed: {e}")

    # Fallback: return zero matrix (no compensation)
    logger.warning("All Sylvester solvers failed, returning zero matrix")
    return torch.zeros_like(C)


class QKDimCompensator:
    """Handles Q/K dimension pruning with Sylvester-based compensation.

    Uses dim-logit mode: Logit matching with Sylvester equation solver.
    Applies different U, V transforms to Q/K weights to preserve attention logits.
    """

    def __init__(
        self,
        lambda_reg: float = 1e-3,
        auto_shrinkage: bool = True,
        solve_dtype: torch.dtype = torch.float64,
    ):
        """Initialize the Q/K dimension compensator.

        Args:
            lambda_reg: Ridge regularization strength
            auto_shrinkage: Use Oracle Approximating Shrinkage for lambda
            solve_dtype: Data type for ridge solve (recommend float64)
        """
        self.lambda_reg = lambda_reg
        self.auto_shrinkage = auto_shrinkage
        self.solve_dtype = solve_dtype
        self.affine_compensator = AffineCompensator(
            lambda_reg=lambda_reg,
            auto_shrinkage=auto_shrinkage,
            solve_dtype=solve_dtype,
        )

    def fit_dim_logit(
        self,
        q_stats: LayerActivationStats,
        k_stats: LayerActivationStats,
        prune_indices: torch.Tensor,
        survivor_indices: torch.Tensor,
        layer_name: str,
        head_idx: int,
    ) -> QKDimCompensationResult:
        """Fit logit-matching compensation for Q/K dimensions using Sylvester equation.

        Optimizes for attention logit reconstruction via Sylvester equation:
        1. Compute Gram matrices G_Q, G_K, C_Q, C_K from covariance
        2. Solve Sylvester equation: G_Q @ M @ G_K + λM = C_Q @ C_K
        3. SVD of I + M to get U, V transforms
        4. Apply different U, V transforms to Q/K weights

        Args:
            q_stats: Statistics for Q activations
            k_stats: Statistics for K activations
            prune_indices: Indices of dimensions to prune
            survivor_indices: Indices of surviving dimensions
            layer_name: Name of the attention layer
            head_idx: Index of the attention head

        Returns:
            QKDimCompensationResult with U, V (and L for backward compatibility)
        """
        n_prune = len(prune_indices)
        n_surv = len(survivor_indices)

        if n_prune == 0:
            # No pruning: identity transforms
            I = torch.eye(n_surv)
            return QKDimCompensationResult(
                mode='dim-logit',
                layer_name=layer_name,
                head_idx=head_idx,
                prune_indices=prune_indices,
                survivor_indices=survivor_indices,
                L=I,  # Backward compatibility
                U=I,
                V=I,
                mu_q_S=torch.zeros(n_surv),
                mu_k_S=torch.zeros(n_surv),
                lambda_used=self.lambda_reg,
            )

        if q_stats.covariance is None or k_stats.covariance is None:
            raise ValueError("Covariance matrix required for dim-logit compensation")

        # Get indices
        p_idx = prune_indices.tolist()
        s_idx = survivor_indices.tolist()
        n = q_stats.n_samples

        # Extract covariance submatrices and means for Q
        q_cov = q_stats.covariance.to(self.solve_dtype)
        q_mean = q_stats.mean.to(self.solve_dtype)
        q_Sigma_SS = q_cov[s_idx][:, s_idx]  # (d_s, d_s)
        q_Sigma_SP = q_cov[s_idx][:, p_idx]  # (d_s, d_p)
        q_mu_S = q_mean[s_idx]  # (d_s,)
        q_mu_P = q_mean[p_idx]  # (d_p,)

        # Extract covariance submatrices and means for K
        k_cov = k_stats.covariance.to(self.solve_dtype)
        k_mean = k_stats.mean.to(self.solve_dtype)
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

        # Compute RHS of Sylvester equation: RHS = C_Q @ C_K
        # This is (d_s, d_p) @ (d_p, d_s) = (d_s, d_s)
        RHS = C_Q @ C_K

        # Compute lambda
        lambda_val = self._compute_lambda(q_Sigma_SS, n)

        # Solve Sylvester equation: G_Q @ M @ G_K + λM = RHS
        M = _solve_sylvester(G_Q, G_K, RHS, lambda_val)

        # Check for NaNs
        if torch.isnan(M).any():
            logger.warning(f"NaNs in Sylvester solution M for {layer_name} head {head_idx}, using identity")
            I = torch.eye(n_surv, dtype=self.solve_dtype)
            return QKDimCompensationResult(
                mode='dim-logit',
                layer_name=layer_name,
                head_idx=head_idx,
                prune_indices=prune_indices,
                survivor_indices=survivor_indices,
                L=I.float(),
                U=I.float(),
                V=I.float(),
                mu_q_S=q_mu_S.float(),
                mu_k_S=k_mu_S.float(),
                lambda_used=lambda_val,
            )

        # Compute I + M
        I = torch.eye(n_surv, dtype=self.solve_dtype)
        I_plus_M = I + M

        # Make symmetric for stability
        I_plus_M = (I_plus_M + I_plus_M.T) / 2

        # SVD: I + M = R @ Σ @ S^T
        try:
            R, Sigma, S_T = torch.linalg.svd(I_plus_M)
            S = S_T.T

            # Clamp singular values to be positive for stability
            Sigma = torch.clamp(Sigma, min=1e-6)

            # Compute U = R @ Σ^{1/2} and V = S @ Σ^{1/2}
            Sigma_sqrt = torch.sqrt(Sigma)
            U = R @ torch.diag(Sigma_sqrt)
            V = S @ torch.diag(Sigma_sqrt)

            # For backward compatibility, also compute L (Cholesky-like)
            # L = U (or V) when U = V for symmetric case
            L = U  # Use U as the "Cholesky-like" factor

        except RuntimeError as e:
            logger.warning(f"SVD failed for I+M in dim-logit: {e}, using eigendecomposition")
            # Fallback: eigendecomposition (for symmetric matrices)
            eigenvalues, eigenvectors = torch.linalg.eigh(I_plus_M)
            eigenvalues = torch.clamp(eigenvalues, min=1e-6)
            sqrt_eigenvalues = torch.sqrt(eigenvalues)

            # For symmetric case, U = V
            U = eigenvectors @ torch.diag(sqrt_eigenvalues)
            V = U.clone()
            L = U

        logger.debug(
            f"dim-logit Sylvester: {layer_name} head {head_idx}, "
            f"lambda={lambda_val:.2e}, M norm={M.norm().item():.2e}"
        )

        return QKDimCompensationResult(
            mode='dim-logit',
            layer_name=layer_name,
            head_idx=head_idx,
            prune_indices=prune_indices,
            survivor_indices=survivor_indices,
            U=U.float(),
            V=V.float(),
            mu_q_S=q_mu_S.float(),
            mu_k_S=k_mu_S.float(),
            lambda_used=lambda_val,
        )

    def _compute_lambda(
        self,
        Sigma_SS: torch.Tensor,
        n_samples: int,
    ) -> float:
        """Compute regularization strength (delegates to AffineCompensator)."""
        return self.affine_compensator._compute_lambda(Sigma_SS, n_samples)

    def fold_dim_logit_weights(
        self,
        result: QKDimCompensationResult,
        W_Q: torch.Tensor,
        b_Q: Optional[torch.Tensor],
        W_K: torch.Tensor,
        b_K: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fold dim-logit compensation into Q/K weights.

        The goal is to preserve attention logits: Q @ K.T ≈ Q_new @ K_new.T
        Using the Sylvester solution with different U, V transforms:
        - Q_new = Q_S @ U
        - K_new = K_S @ V

        Since Q_S = input @ W_Q_S.T + b_Q_S, we have:
        Q_new = Q_S @ U = (input @ W_Q_S.T + b_Q_S) @ U = input @ W_Q_S.T @ U + b_Q_S @ U

        So the transformed weights are:
        - W_Q_new = U.T @ W_Q[surv]  (gives input @ W_Q_S.T @ U)
        - W_K_new = V.T @ W_K[surv]  (gives input @ W_K_S.T @ V)
        - b_Q_new = b_Q[surv] @ U    (gives b_Q_S @ U)
        - b_K_new = b_K[surv] @ V    (gives b_K_S @ V)

        Args:
            result: QKDimCompensationResult from fit_dim_logit
            W_Q: Original Q weight for this head (head_dim, in_features)
            b_Q: Original Q bias for this head (head_dim,) or None
            W_K: Original K weight for this head (head_dim, in_features)
            b_K: Original K bias for this head (head_dim,) or None

        Returns:
            Tuple of (W_Q_new, b_Q_new, W_K_new, b_K_new) with n_surv dimensions
        """
        surv_idx = result.survivor_indices.tolist()

        if result.U is None or result.V is None:
            raise ValueError("QKDimCompensationResult must have U and V transforms")
        U = result.U.to(W_Q.device, W_Q.dtype)
        V = result.V.to(W_Q.device, W_Q.dtype)

        # Extract survivor weights
        W_Q_S = W_Q[surv_idx]  # (n_surv, in_features)
        W_K_S = W_K[surv_idx]  # (n_surv, in_features)

        # Transform weights: W_Q_new = U.T @ W_Q_S, W_K_new = V.T @ W_K_S
        # This gives: input @ W_Q_new.T = input @ W_Q_S.T @ U
        W_Q_new = U.T @ W_Q_S
        W_K_new = V.T @ W_K_S

        # Transform biases: b_Q_new = b_Q_S @ U, b_K_new = b_K_S @ V
        # This gives: Q_new = input @ W_Q_S.T @ U + b_Q_S @ U = (input @ W_Q_S.T + b_Q_S) @ U = Q_S @ U
        if b_Q is not None:
            b_Q_S = b_Q[surv_idx]
            b_Q_new = b_Q_S @ U  # (n_surv,) @ (n_surv, n_surv) = (n_surv,)
        else:
            b_Q_new = None

        if b_K is not None:
            b_K_S = b_K[surv_idx]
            b_K_new = b_K_S @ V
        else:
            b_K_new = None

        return W_Q_new, b_Q_new, W_K_new, b_K_new
