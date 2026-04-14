"""
Diagnostics for Structured Pruning.

Pre/post-prune failure detection and quality metrics.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """Result of diagnostic check on a layer.

    Attributes:
        layer_name: Name of the layer
        mse: Mean squared error between baseline and pruned outputs
        cosine_similarity: Cosine similarity between outputs
        outlier_rate: Fraction of outliers (|x| > k*sigma)
        has_nan: Whether NaN values were detected
        has_inf: Whether Inf values were detected
        residual_rms_ratio: RMS(output) / RMS(input) drift
        max_abs_error: Maximum absolute error
        relative_error: MSE / baseline_variance
    """
    layer_name: str
    mse: float
    cosine_similarity: float
    outlier_rate: float
    has_nan: bool
    has_inf: bool
    residual_rms_ratio: Optional[float] = None
    max_abs_error: Optional[float] = None
    relative_error: Optional[float] = None
    passed: bool = True
    failure_reason: Optional[str] = None


class DiagnosticsChecker:
    """Checks for pruning failures and quality degradation."""

    def __init__(
        self,
        outlier_k: float = 6.0,
        mse_threshold: float = 0.1,
        cosine_threshold: float = 0.95,
        relative_error_threshold: float = 0.5,
        residual_drift_threshold: float = 2.0,
    ):
        """Initialize the diagnostics checker.

        Args:
            outlier_k: Number of sigmas for outlier detection
            mse_threshold: Maximum acceptable MSE
            cosine_threshold: Minimum acceptable cosine similarity
            relative_error_threshold: Maximum acceptable relative error
            residual_drift_threshold: Maximum acceptable residual ratio drift
        """
        self.outlier_k = outlier_k
        self.mse_threshold = mse_threshold
        self.cosine_threshold = cosine_threshold
        self.relative_error_threshold = relative_error_threshold
        self.residual_drift_threshold = residual_drift_threshold

    def check_layer(
        self,
        baseline_output: torch.Tensor,
        pruned_output: torch.Tensor,
        layer_name: str,
        baseline_input: Optional[torch.Tensor] = None,
        pruned_input: Optional[torch.Tensor] = None,
    ) -> DiagnosticResult:
        """Check a single layer for pruning quality.

        Args:
            baseline_output: Output from unpruned model
            pruned_output: Output from pruned model
            layer_name: Name of the layer
            baseline_input: Optional input for residual ratio
            pruned_input: Optional input for residual ratio

        Returns:
            DiagnosticResult with all metrics
        """
        baseline_output = baseline_output.detach().float()
        pruned_output = pruned_output.detach().float()

        # Check for NaN/Inf
        has_nan = torch.isnan(pruned_output).any().item()
        has_inf = torch.isinf(pruned_output).any().item()

        # Flatten for comparison
        baseline_flat = baseline_output.reshape(-1)
        pruned_flat = pruned_output.reshape(-1)

        # Ensure same shape
        assert baseline_flat.shape == pruned_flat.shape, (
            f"Shape mismatch: {baseline_flat.shape} vs {pruned_flat.shape}"
        )

        # MSE
        mse = ((baseline_flat - pruned_flat) ** 2).mean().item()

        # Cosine similarity
        if baseline_flat.norm() > 1e-10 and pruned_flat.norm() > 1e-10:
            cosine = torch.nn.functional.cosine_similarity(
                baseline_flat.unsqueeze(0),
                pruned_flat.unsqueeze(0),
            ).item()
        else:
            cosine = 0.0

        # Outlier rate
        sigma = baseline_flat.std().item()
        if sigma > 1e-10:
            error = (baseline_flat - pruned_flat).abs()
            outlier_rate = (error > self.outlier_k * sigma).float().mean().item()
        else:
            outlier_rate = 0.0

        # Max absolute error
        max_abs_error = (baseline_flat - pruned_flat).abs().max().item()

        # Relative error
        baseline_var = baseline_flat.var().item()
        if baseline_var > 1e-10:
            relative_error = mse / baseline_var
        else:
            relative_error = 0.0 if mse < 1e-10 else float('inf')

        # Residual RMS ratio drift
        residual_rms_ratio = None
        if baseline_input is not None and pruned_input is not None:
            baseline_input = baseline_input.detach().float()
            pruned_input = pruned_input.detach().float()

            baseline_in_rms = torch.sqrt((baseline_input ** 2).mean()).item()
            baseline_out_rms = torch.sqrt((baseline_output ** 2).mean()).item()
            pruned_in_rms = torch.sqrt((pruned_input ** 2).mean()).item()
            pruned_out_rms = torch.sqrt((pruned_output ** 2).mean()).item()

            if baseline_in_rms > 1e-10 and pruned_in_rms > 1e-10:
                baseline_ratio = baseline_out_rms / baseline_in_rms
                pruned_ratio = pruned_out_rms / pruned_in_rms
                if baseline_ratio > 1e-10:
                    residual_rms_ratio = pruned_ratio / baseline_ratio

        result = DiagnosticResult(
            layer_name=layer_name,
            mse=mse,
            cosine_similarity=cosine,
            outlier_rate=outlier_rate,
            has_nan=has_nan,
            has_inf=has_inf,
            residual_rms_ratio=residual_rms_ratio,
            max_abs_error=max_abs_error,
            relative_error=relative_error,
        )

        # Detect failures
        passed, reason = self.detect_failure(result)
        result.passed = passed
        result.failure_reason = reason

        return result

    def detect_failure(self, result: DiagnosticResult) -> Tuple[bool, Optional[str]]:
        """Detect if the diagnostic result indicates a failure.

        Args:
            result: Diagnostic result to check

        Returns:
            Tuple of (passed, failure_reason)
        """
        # Check for NaN/Inf
        if result.has_nan:
            return False, "NaN values detected in output"

        if result.has_inf:
            return False, "Inf values detected in output"

        # Check cosine similarity
        if result.cosine_similarity < self.cosine_threshold:
            return False, f"Low cosine similarity: {result.cosine_similarity:.4f} < {self.cosine_threshold}"

        # Check relative error
        if result.relative_error is not None and result.relative_error > self.relative_error_threshold:
            return False, f"High relative error: {result.relative_error:.4f} > {self.relative_error_threshold}"

        # Check residual drift
        if result.residual_rms_ratio is not None:
            if result.residual_rms_ratio > self.residual_drift_threshold:
                return False, f"Residual imbalance (ratio increased): {result.residual_rms_ratio:.4f}"
            if result.residual_rms_ratio < 1.0 / self.residual_drift_threshold:
                return False, f"Residual imbalance (ratio decreased): {result.residual_rms_ratio:.4f}"

        # Check outlier rate
        if result.outlier_rate > 0.01:  # More than 1% outliers
            logger.warning(
                f"Layer {result.layer_name}: High outlier rate {result.outlier_rate:.4f}"
            )
            # Don't fail, just warn

        return True, None

    def diagnose_failure_type(self, result: DiagnosticResult) -> str:
        """Diagnose the type of failure from diagnostic result.

        Args:
            result: Failed diagnostic result

        Returns:
            Diagnosis string describing the likely cause
        """
        if result.has_nan or result.has_inf:
            return "numerical-instability"

        if result.cosine_similarity < 0.5:
            return "shape-coupling-mismatch"

        if result.relative_error is not None and result.relative_error > 1.0:
            return "compensation-overfit"

        if result.residual_rms_ratio is not None:
            if result.residual_rms_ratio > 2.0 or result.residual_rms_ratio < 0.5:
                return "residual-imbalance"

        if result.mse > self.mse_threshold * 10:
            return "calibration-shift"

        return "unknown"


@dataclass
class ModelDiagnostics:
    """Diagnostics for the full model."""
    layer_results: Dict[str, DiagnosticResult] = field(default_factory=dict)
    total_mse: float = 0.0
    mean_cosine: float = 0.0
    worst_layer: Optional[str] = None
    all_passed: bool = True
    failure_summary: List[str] = field(default_factory=list)


class ModelDiagnosticsChecker:
    """Runs diagnostics across the full model."""

    def __init__(
        self,
        layer_checker: Optional[DiagnosticsChecker] = None,
    ):
        """Initialize the model diagnostics checker.

        Args:
            layer_checker: Layer-level diagnostics checker
        """
        self.layer_checker = layer_checker or DiagnosticsChecker()

    def check_model_outputs(
        self,
        baseline_outputs: Dict[str, torch.Tensor],
        pruned_outputs: Dict[str, torch.Tensor],
        baseline_inputs: Optional[Dict[str, torch.Tensor]] = None,
        pruned_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> ModelDiagnostics:
        """Check all layer outputs for pruning quality.

        Args:
            baseline_outputs: Dict of layer_name -> baseline output
            pruned_outputs: Dict of layer_name -> pruned output
            baseline_inputs: Optional dict of layer inputs
            pruned_inputs: Optional dict of layer inputs

        Returns:
            ModelDiagnostics with all results
        """
        diagnostics = ModelDiagnostics()

        total_mse = 0.0
        cosine_sum = 0.0
        worst_cosine = 1.0
        worst_layer = None

        for layer_name in baseline_outputs:
            if layer_name not in pruned_outputs:
                logger.warning(f"Missing pruned output for {layer_name}")
                continue

            baseline_in = baseline_inputs.get(layer_name) if baseline_inputs else None
            pruned_in = pruned_inputs.get(layer_name) if pruned_inputs else None

            result = self.layer_checker.check_layer(
                baseline_outputs[layer_name],
                pruned_outputs[layer_name],
                layer_name,
                baseline_in,
                pruned_in,
            )

            diagnostics.layer_results[layer_name] = result

            total_mse += result.mse
            cosine_sum += result.cosine_similarity

            if result.cosine_similarity < worst_cosine:
                worst_cosine = result.cosine_similarity
                worst_layer = layer_name

            if not result.passed:
                diagnostics.all_passed = False
                diagnostics.failure_summary.append(
                    f"{layer_name}: {result.failure_reason}"
                )

        n_layers = len(diagnostics.layer_results)
        if n_layers > 0:
            diagnostics.total_mse = total_mse
            diagnostics.mean_cosine = cosine_sum / n_layers
            diagnostics.worst_layer = worst_layer

        return diagnostics

    def check_final_output(
        self,
        baseline_logits: torch.Tensor,
        pruned_logits: torch.Tensor,
    ) -> DiagnosticResult:
        """Check final model output quality.

        Args:
            baseline_logits: Logits from unpruned model
            pruned_logits: Logits from pruned model

        Returns:
            DiagnosticResult for final output
        """
        return self.layer_checker.check_layer(
            baseline_logits,
            pruned_logits,
            "final_output",
        )


def run_quick_validation(
    model: nn.Module,
    pruned_model: nn.Module,
    sample_input: torch.Tensor,
    device: str = "cuda",
) -> Tuple[bool, str]:
    """Run a quick validation of pruned model.

    Args:
        model: Original model
        pruned_model: Pruned model
        sample_input: Sample input tensor
        device: Device to run on

    Returns:
        Tuple of (passed, message)
    """
    model = model.to(device).eval()
    pruned_model = pruned_model.to(device).eval()
    sample_input = sample_input.to(device)

    with torch.no_grad():
        try:
            baseline_out = model(sample_input)
            pruned_out = pruned_model(sample_input)
        except RuntimeError as e:
            return False, f"Forward pass failed: {e}"

    checker = DiagnosticsChecker()
    result = checker.check_layer(baseline_out, pruned_out, "model_output")

    if result.passed:
        return True, f"Validation passed (cosine={result.cosine_similarity:.4f})"
    else:
        return False, f"Validation failed: {result.failure_reason}"
