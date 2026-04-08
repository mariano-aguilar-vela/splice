"""
Module 14: utils/stats.py

Statistical utility functions for differential splicing testing.
Dirichlet-multinomial log-likelihood, Benjamini-Hochberg FDR correction,
Beta posterior helpers, and likelihood ratio testing.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import gammaln, digamma
from scipy.stats import chi2
from scipy.optimize import minimize


def dm_log_likelihood(
    counts: np.ndarray,
    alpha: np.ndarray,
) -> float:
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
    alpha: np.ndarray,
) -> float:
    """Sum of DM log-likelihoods across multiple samples.

    Args:
        count_matrix: Array of shape (n_samples, K) with counts.
        alpha: Array of shape (K,) with concentrations (shared across samples).

    Returns:
        Sum of log-likelihoods across all samples.
    """
    total_ll = 0.0
    for i in range(count_matrix.shape[0]):
        total_ll += dm_log_likelihood(count_matrix[i, :], alpha)
    return total_ll


def fit_dm_null(
    count_matrix: np.ndarray,
    max_iter: int = 200,
) -> Tuple[np.ndarray, float, bool]:
    """Fit a Dirichlet-multinomial null model (no group effect).

    Estimates alpha by maximum likelihood using scipy.optimize.minimize.
    Initializes via method-of-moments: proportions from mean row-normalized counts,
    then estimates concentration from variance of proportions.

    Args:
        count_matrix: Array of shape (n_samples, K) with counts.
        max_iter: Maximum iterations for optimization.

    Returns:
        Tuple of (alpha_hat, log_likelihood, converged).
    """
    n_samples, n_junctions = count_matrix.shape

    # Method-of-moments initialization
    # Normalize each row and compute mean proportions
    proportions_per_sample = count_matrix / np.sum(count_matrix, axis=1, keepdims=True)
    mean_proportions = np.mean(proportions_per_sample, axis=0)

    # Estimate concentration from variance of proportions
    # Var(p) ≈ p(1-p) / (concentration + 1), so concentration ≈ p(1-p) / Var(p) - 1
    var_proportions = np.var(proportions_per_sample, axis=0)
    expected_var = mean_proportions * (1 - mean_proportions)

    # Avoid division by zero; use a minimum concentration of 0.1
    concentration_init = np.mean(expected_var / (np.maximum(var_proportions, 1e-6)) - 1)
    concentration_init = max(0.1, concentration_init)

    # Initialize alpha
    alpha_init = mean_proportions * concentration_init

    # Define negative log-likelihood objective
    def neg_ll(alpha_flat):
        alpha = np.maximum(alpha_flat, 1e-6)  # Ensure positivity
        return -dm_log_likelihood_batch(count_matrix, alpha)

    # Optimize
    result = minimize(
        neg_ll,
        alpha_init,
        method="L-BFGS-B",
        bounds=[(1e-6, None) for _ in range(n_junctions)],
        options={"maxiter": max_iter},
    )

    alpha_hat = np.maximum(result.x, 1e-6)
    ll = -result.fun
    converged = result.success

    return alpha_hat, ll, converged


def fit_dm_full(
    count_matrix: np.ndarray,
    group_labels: np.ndarray,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """Fit a DM model with separate alpha for each group.

    Estimates separate alpha vectors for group 0 and group 1.
    Initializes from null model estimates, splits by group.

    Args:
        count_matrix: Array of shape (n_samples, K) with counts.
        group_labels: Array of shape (n_samples,) with values 0 or 1.
        max_iter: Maximum iterations for optimization.

    Returns:
        Tuple of (alpha_0, alpha_1, log_likelihood, converged).
    """
    n_samples, n_junctions = count_matrix.shape

    # Fit null model to get initial alpha
    alpha_null, _, _ = fit_dm_null(count_matrix, max_iter)

    # Split data by group
    group_0_mask = group_labels == 0
    group_1_mask = group_labels == 1

    counts_0 = count_matrix[group_0_mask, :]
    counts_1 = count_matrix[group_1_mask, :]

    # Initialize with null model alphas
    alpha_0_init = alpha_null.copy()
    alpha_1_init = alpha_null.copy()

    # Define negative log-likelihood for full model
    def neg_ll_full(alpha_flat):
        alpha_0 = np.maximum(alpha_flat[:n_junctions], 1e-6)
        alpha_1 = np.maximum(alpha_flat[n_junctions:], 1e-6)

        ll_0 = dm_log_likelihood_batch(counts_0, alpha_0) if counts_0.shape[0] > 0 else 0
        ll_1 = dm_log_likelihood_batch(counts_1, alpha_1) if counts_1.shape[0] > 0 else 0

        return -(ll_0 + ll_1)

    # Optimize
    alpha_init_flat = np.concatenate([alpha_0_init, alpha_1_init])
    result = minimize(
        neg_ll_full,
        alpha_init_flat,
        method="L-BFGS-B",
        bounds=[(1e-6, None) for _ in range(2 * n_junctions)],
        options={"maxiter": max_iter},
    )

    alpha_0 = np.maximum(result.x[:n_junctions], 1e-6)
    alpha_1 = np.maximum(result.x[n_junctions:], 1e-6)
    ll = -result.fun
    converged = result.success

    return alpha_0, alpha_1, ll, converged


def likelihood_ratio_test(
    ll_null: float,
    ll_full: float,
    df: int,
) -> float:
    """Likelihood ratio test p-value.

    test_stat = 2 * (ll_full - ll_null)
    p_value = 1 - chi2.cdf(test_stat, df)

    Args:
        ll_null: Log-likelihood under null model.
        ll_full: Log-likelihood under full model.
        df: Degrees of freedom (typically n_junctions - 1).

    Returns:
        P-value from chi-squared test.
    """
    test_stat = 2 * (ll_full - ll_null)

    # If test statistic is negative, return p-value of 1
    if test_stat < 0:
        return 1.0

    p_value = 1 - chi2.cdf(test_stat, df)
    return p_value


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of p-values.

    Returns:
        Adjusted p-values (Benjamini-Hochberg).
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Compute adjusted p-values
    adjusted_p = np.zeros(n)
    for i in range(n):
        # Rank is i+1 (1-indexed)
        rank = i + 1
        adjusted_p[sorted_idx[i]] = sorted_p[i] * n / rank

    # Ensure monotonicity: adjusted_p[i] <= adjusted_p[i+1]
    for i in range(n - 2, -1, -1):
        adjusted_p[sorted_idx[i]] = min(adjusted_p[sorted_idx[i]], adjusted_p[sorted_idx[i + 1]])

    # Clamp to [0, 1]
    adjusted_p = np.minimum(adjusted_p, 1.0)

    return adjusted_p


def beta_posterior_psi(
    counts: np.ndarray,
    total: float,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Beta posterior parameters for PSI of each junction.

    For junction k with count c_k and total N:
      alpha_k = c_k + prior_alpha
      beta_k = (N - c_k) + prior_beta
      psi_k = alpha_k / (alpha_k + beta_k)

    Args:
        counts: Array of shape (K,) with junction counts.
        total: Total counts across all junctions.
        prior_alpha: Prior alpha for Beta distribution (default 0.5).
        prior_beta: Prior beta for Beta distribution (default 0.5).

    Returns:
        Tuple of (psi_array, credible_interval_width_array) where
        credible_interval_width is the 95% CI width.
    """
    alpha_posterior = counts + prior_alpha
    beta_posterior = (total - counts) + prior_beta

    # PSI point estimate
    psi = alpha_posterior / (alpha_posterior + beta_posterior)

    # Credible interval width (simplified: using variance approximation)
    # Var(X) = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
    var_psi = (
        alpha_posterior * beta_posterior
        / ((alpha_posterior + beta_posterior) ** 2 * (alpha_posterior + beta_posterior + 1))
    )

    # 95% CI width (approximately 4 * std = 4 * sqrt(var))
    ci_width = 4 * np.sqrt(var_psi)

    return psi, ci_width
