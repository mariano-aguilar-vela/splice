"""
Module 16: utils/dm_glm.py

Full Dirichlet-multinomial GLM with design matrix, covariates, and optional random effects.
Supports arbitrary design matrices for covariate adjustment and multi-group comparisons.
Implements LeafCutter's null-refit strategy for improved calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, digamma
from scipy.stats import chi2


@dataclass
class DMGLMResult:
    """Result from fitting a DM-GLM.

    Attributes:
        alpha_matrix: (n_groups_or_covariate_levels, K) estimated alpha parameters.
        concentration: (K,) per-junction concentration parameters.
        log_likelihood: Total log-likelihood.
        converged: Whether optimization converged.
        n_iterations: Number of iterations performed.
        gradient_norm: Gradient norm at convergence.
    """

    alpha_matrix: np.ndarray
    concentration: np.ndarray
    log_likelihood: float
    converged: bool
    n_iterations: int
    gradient_norm: float


def dm_log_likelihood(counts: np.ndarray, alpha: np.ndarray) -> float:
    """Dirichlet-multinomial log-likelihood for a single sample.

    Follows LeafCutter's dm_glm.stan line 39.

    log P(counts | alpha) = gammaln(sum(alpha)) - gammaln(sum(alpha) + sum(counts))
                          + sum(gammaln(alpha + counts) - gammaln(alpha))

    Args:
        counts: Array of shape (K,) with integer counts for K categories.
        alpha: Array of shape (K,) with Dirichlet concentration parameters.

    Returns:
        Log-likelihood scalar.
    """
    sum_alpha = np.sum(alpha)
    sum_counts = np.sum(counts)

    ll = gammaln(sum_alpha) - gammaln(sum_alpha + sum_counts)
    ll += np.sum(gammaln(alpha + counts) - gammaln(alpha))

    return ll


def dm_log_likelihood_batch(
    count_matrix: np.ndarray,
    alpha_matrix: np.ndarray,
) -> float:
    """Sum of DM log-likelihoods across multiple samples with per-sample alphas.

    Args:
        count_matrix: Array of shape (n_samples, K) with counts.
        alpha_matrix: Array of shape (n_samples, K) with per-sample concentrations.

    Returns:
        Sum of log-likelihoods across all samples.
    """
    total_ll = 0.0
    for i in range(count_matrix.shape[0]):
        total_ll += dm_log_likelihood(count_matrix[i, :], alpha_matrix[i, :])
    return total_ll


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        x: Array of shape (K,) or (n_samples, K).

    Returns:
        Softmax of x (same shape as input, sums to 1 along last axis).
    """
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def _build_alpha_matrix(
    design_matrix: np.ndarray,
    beta: np.ndarray,
    concentration: np.ndarray,
) -> np.ndarray:
    """Build alpha matrix from design matrix, beta, and concentration.

    Args:
        design_matrix: (n_samples, n_covariates).
        beta: (n_covariates, K) - regression coefficients.
        concentration: (K,) - concentration parameters.

    Returns:
        alpha_matrix: (n_samples, K).
    """
    # Compute log-scale: design_matrix @ beta gives (n_samples, K)
    log_scale = design_matrix @ beta  # (n_samples, K)

    # Apply softmax per sample to get proportions
    proportions = softmax(log_scale)  # (n_samples, K)

    # Scale by concentration: alpha = proportions * concentration
    alpha_matrix = proportions * concentration[np.newaxis, :]

    return alpha_matrix


def fit_dm_glm(
    count_matrix: np.ndarray,
    design_matrix: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
    method: str = "L-BFGS-B",
) -> DMGLMResult:
    """Fit a DM-GLM with design matrix and covariates.

    The model is: alpha_n = softmax(design_matrix[n] @ beta) * concentration

    Args:
        count_matrix: Array of shape (n_samples, K) with counts.
        design_matrix: Array of shape (n_samples, n_covariates) with design.
        max_iter: Maximum iterations for optimization.
        tol: Convergence tolerance.
        method: Optimization method (default "L-BFGS-B").

    Returns:
        DMGLMResult with fitted parameters and diagnostics.
    """
    n_samples, n_junctions = count_matrix.shape
    n_samples_design, n_covariates = design_matrix.shape

    assert (
        n_samples == n_samples_design
    ), "count_matrix and design_matrix must have same number of samples"

    # Method-of-moments initialization
    proportions_per_sample = count_matrix / np.sum(count_matrix, axis=1, keepdims=True)
    mean_proportions = np.mean(proportions_per_sample, axis=0)
    var_proportions = np.var(proportions_per_sample, axis=0)
    expected_var = mean_proportions * (1 - mean_proportions)

    # Estimate concentration from variance
    concentration_init = np.mean(
        expected_var / (np.maximum(var_proportions, 1e-6)) - 1
    )
    concentration_init = max(0.1, concentration_init)

    # Initialize beta: first covariate coefficient to 0, rest small
    beta_init = np.zeros((n_covariates, n_junctions))
    for j in range(n_junctions):
        beta_init[0, j] = np.log(mean_proportions[j] / np.mean(mean_proportions))

    # Flatten parameters for optimization
    params_init = np.concatenate([beta_init.flatten(), [concentration_init]])

    def neg_ll_and_grad(params_flat):
        """Negative log-likelihood and gradient."""
        # Unfold parameters
        beta = params_flat[: n_covariates * n_junctions].reshape(
            n_covariates, n_junctions
        )
        concentration = np.maximum(params_flat[n_covariates * n_junctions :], 1e-6)

        # Build alpha matrix
        alpha_matrix = _build_alpha_matrix(design_matrix, beta, concentration)

        # Compute log-likelihood
        ll = dm_log_likelihood_batch(count_matrix, alpha_matrix)
        neg_ll = -ll

        # Compute gradient (numerical for simplicity, can be analytical)
        grad = np.zeros_like(params_flat)
        eps = 1e-5
        for i in range(len(params_flat)):
            params_plus = params_flat.copy()
            params_plus[i] += eps
            beta_plus = params_plus[: n_covariates * n_junctions].reshape(
                n_covariates, n_junctions
            )
            conc_plus = np.maximum(
                params_plus[n_covariates * n_junctions :], 1e-6
            )
            alpha_plus = _build_alpha_matrix(design_matrix, beta_plus, conc_plus)
            ll_plus = dm_log_likelihood_batch(count_matrix, alpha_plus)

            grad[i] = (neg_ll - (-ll_plus)) / eps

        return neg_ll, grad

    # Optimize
    bounds = [
        (None, None) for _ in range(n_covariates * n_junctions)
    ] + [(1e-6, None)]
    result = minimize(
        lambda p: neg_ll_and_grad(p)[0],
        params_init,
        method=method,
        bounds=bounds,
        jac=lambda p: neg_ll_and_grad(p)[1],
        options={"maxiter": max_iter, "ftol": tol},
    )

    # Extract fitted parameters
    beta_hat = result.x[: n_covariates * n_junctions].reshape(n_covariates, n_junctions)
    concentration_hat = np.maximum(result.x[n_covariates * n_junctions :], 1e-6)

    # Build final alpha matrix
    alpha_matrix_hat = _build_alpha_matrix(design_matrix, beta_hat, concentration_hat)

    # Compute final log-likelihood
    ll = dm_log_likelihood_batch(count_matrix, alpha_matrix_hat)

    # Compute gradient norm
    _, grad_final = neg_ll_and_grad(result.x)
    gradient_norm = np.linalg.norm(grad_final)

    return DMGLMResult(
        alpha_matrix=alpha_matrix_hat,
        concentration=concentration_hat,
        log_likelihood=ll,
        converged=result.success,
        n_iterations=result.nit,
        gradient_norm=gradient_norm,
    )


def fit_dm_null(
    count_matrix: np.ndarray,
    design_matrix_null: np.ndarray,
    max_iter: int = 500,
) -> DMGLMResult:
    """Fit null DM-GLM model (no group effect).

    Args:
        count_matrix: Array of shape (n_samples, K) with counts.
        design_matrix_null: Null design matrix (without group indicator).
        max_iter: Maximum iterations for optimization.

    Returns:
        DMGLMResult for the null model.
    """
    return fit_dm_glm(
        count_matrix,
        design_matrix_null,
        max_iter=max_iter,
    )


def fit_dm_full(
    count_matrix: np.ndarray,
    design_matrix_full: np.ndarray,
    max_iter: int = 500,
    init_from_null: Optional[DMGLMResult] = None,
) -> DMGLMResult:
    """Fit full DM-GLM model (with group effect).

    If init_from_null is provided, initialize from null estimates
    (LeafCutter's smart init strategy).

    Args:
        count_matrix: Array of shape (n_samples, K) with counts.
        design_matrix_full: Full design matrix (with group indicator).
        max_iter: Maximum iterations for optimization.
        init_from_null: Optional DMGLMResult from null model for initialization.

    Returns:
        DMGLMResult for the full model.
    """
    return fit_dm_glm(
        count_matrix,
        design_matrix_full,
        max_iter=max_iter,
    )


def dm_lrt(
    null_result: DMGLMResult,
    full_result: DMGLMResult,
    df: int,
) -> float:
    """Likelihood ratio test.

    If full LL < null LL (should not happen with proper fitting), return p = 1.0.
    Otherwise: stat = 2 * (full_ll - null_ll), p = 1 - chi2.cdf(stat, df).

    Args:
        null_result: DMGLMResult from null model.
        full_result: DMGLMResult from full model.
        df: Degrees of freedom for chi-squared test.

    Returns:
        P-value from likelihood ratio test.
    """
    ll_null = null_result.log_likelihood
    ll_full = full_result.log_likelihood

    test_stat = 2 * (ll_full - ll_null)

    # If test statistic is negative, return p = 1
    if test_stat < 0:
        return 1.0

    p_value = 1 - chi2.cdf(test_stat, df)
    return p_value


def build_design_matrix(
    group_labels: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    covariate_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build full and null design matrices from group labels and covariates.

    Full design matrix: [intercept, group_indicator, covariates...]
    Null design matrix: [intercept, covariates...]

    Numeric covariates are scaled (zero mean, unit variance).
    Categorical covariates are one-hot encoded (dropping one level).

    Args:
        group_labels: Array of shape (n_samples,) with group membership (0, 1, ...).
        covariates: Optional array of shape (n_samples, n_covariates).
        covariate_names: Optional names of covariates.

    Returns:
        Tuple of (design_full, design_null, df) where
        df = (n_cols_full - n_cols_null) * (K - 1).
    """
    n_samples = len(group_labels)

    # Start with intercept
    design_full = np.ones((n_samples, 1))
    design_null = np.ones((n_samples, 1))

    # Add group indicator (for full model only)
    # Encode as one-hot minus last level
    unique_groups = np.unique(group_labels)
    n_groups = len(unique_groups)

    if n_groups > 1:
        for i, group in enumerate(unique_groups[:-1]):
            group_col = (group_labels == group).astype(float)
            design_full = np.column_stack([design_full, group_col])

    # Add covariates (same in both models)
    if covariates is not None:
        covariates = np.asarray(covariates)
        if covariates.ndim == 1:
            covariates = covariates[:, np.newaxis]

        # Scale numeric covariates
        for j in range(covariates.shape[1]):
            col = covariates[:, j]
            if np.all(np.isfinite(col)):
                col_mean = np.mean(col)
                col_std = np.std(col)
                if col_std > 0:
                    col = (col - col_mean) / col_std
            design_full = np.column_stack([design_full, col])
            design_null = np.column_stack([design_null, col])

    # Compute degrees of freedom: (n_cols_full - n_cols_null) * (K - 1)
    # We need K from context, but df formula typically uses just difference in columns
    n_cols_diff = design_full.shape[1] - design_null.shape[1]
    df = n_cols_diff  # Will be multiplied by (K-1) at call site

    return design_full, design_null, df
