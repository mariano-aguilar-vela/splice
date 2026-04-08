"""
Module 13: core/bootstrap.py

Bootstrap resampling for uncertainty estimation.
Generates bootstrap replicates of junction counts and PSI values.
Follows MAJIQ's approach for robust confidence interval estimation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from splicekit.core.effective_length import length_normalize_counts


def bootstrap_junction_counts(
    count_matrix: np.ndarray,
    n_bootstraps: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """Generate bootstrap resamples of junction counts.

    For each sample, the total reads are redistributed across junctions
    via multinomial resampling with probabilities proportional to observed counts.

    Args:
        count_matrix: Array of shape (n_junctions, n_samples) with raw counts.
        n_bootstraps: Number of bootstrap replicates (default 30).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Array of shape (n_bootstraps, n_junctions, n_samples) with resampled counts.
    """
    rng = np.random.RandomState(seed)

    n_junctions, n_samples = count_matrix.shape
    bootstrap_counts = np.zeros((n_bootstraps, n_junctions, n_samples), dtype=int)

    for sample_idx in range(n_samples):
        # Get counts for this sample
        sample_counts = count_matrix[:, sample_idx]
        total_count = np.sum(sample_counts)

        # Compute probabilities
        if total_count > 0:
            probs = sample_counts / total_count
        else:
            probs = np.ones(n_junctions) / n_junctions

        # Generate bootstrap resamples for this sample
        for boot_idx in range(n_bootstraps):
            bootstrap_counts[boot_idx, :, sample_idx] = rng.multinomial(
                total_count, probs
            )

    return bootstrap_counts


def bootstrap_psi(
    bootstrap_counts: np.ndarray,
    effective_lengths: np.ndarray,
    prior_alpha: float = 0.5,
) -> np.ndarray:
    """Compute PSI for each bootstrap replicate.

    Length-normalizes counts before computing PSI.

    Args:
        bootstrap_counts: Array of shape (n_bootstraps, n_junctions, n_samples).
        effective_lengths: Array of shape (n_junctions,).
        prior_alpha: Dirichlet prior (currently unused, reserved for future use).

    Returns:
        Array of shape (n_bootstraps, n_junctions, n_samples) with PSI values in [0, 1].
    """
    n_bootstraps, n_junctions, n_samples = bootstrap_counts.shape

    psi_matrix = np.zeros((n_bootstraps, n_junctions, n_samples), dtype=float)

    for boot_idx in range(n_bootstraps):
        # Get this bootstrap replicate
        counts = bootstrap_counts[boot_idx, :, :].astype(float)

        # Length-normalize: divide each row by its effective length
        normalized = counts / effective_lengths[:, np.newaxis]

        # Compute PSI: normalized_junc / sum_of_normalized_per_sample
        # Sum over junctions (rows) to get per-sample totals
        col_sums = np.sum(normalized, axis=0)

        # Compute PSI for each sample with nonzero reads
        for sample_idx in range(n_samples):
            if col_sums[sample_idx] > 0:
                psi_matrix[boot_idx, :, sample_idx] = (
                    normalized[:, sample_idx] / col_sums[sample_idx]
                )

    return psi_matrix


def bootstrap_confidence_intervals(
    bootstrap_psi: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute percentile confidence intervals from bootstrap PSI.

    Args:
        bootstrap_psi: Array of shape (n_bootstraps, n_junctions, n_samples).
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (ci_low, ci_high) each of shape (n_junctions, n_samples).
    """
    n_bootstraps, n_junctions, n_samples = bootstrap_psi.shape

    ci_low = np.zeros((n_junctions, n_samples), dtype=float)
    ci_high = np.zeros((n_junctions, n_samples), dtype=float)

    # Percentiles for confidence intervals
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    for j in range(n_junctions):
        for s in range(n_samples):
            ci_low[j, s] = np.percentile(bootstrap_psi[:, j, s], lower_percentile)
            ci_high[j, s] = np.percentile(bootstrap_psi[:, j, s], upper_percentile)

    return ci_low, ci_high


def bootstrap_mean_psi(bootstrap_psi: np.ndarray) -> np.ndarray:
    """Compute mean PSI from bootstrap replicates.

    Args:
        bootstrap_psi: Array of shape (n_bootstraps, n_junctions, n_samples).

    Returns:
        Array of shape (n_junctions, n_samples) with mean PSI.
    """
    return np.mean(bootstrap_psi, axis=0)


def bootstrap_std_psi(bootstrap_psi: np.ndarray) -> np.ndarray:
    """Compute standard deviation of PSI from bootstrap replicates.

    Args:
        bootstrap_psi: Array of shape (n_bootstraps, n_junctions, n_samples).

    Returns:
        Array of shape (n_junctions, n_samples) with PSI standard deviation.
    """
    return np.std(bootstrap_psi, axis=0)
