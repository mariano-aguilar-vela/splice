"""
Module 13: core/bootstrap.py

Bootstrap resampling for uncertainty estimation.
Generates bootstrap replicates of junction counts and PSI values.
Follows MAJIQ's approach for robust confidence interval estimation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from splice.core.effective_length import length_normalize_counts


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
        sample_counts = count_matrix[:, sample_idx]
        total_count = int(np.sum(sample_counts))
        if total_count > 0:
            probs = sample_counts / total_count
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
    """Compute PSI for each bootstrap replicate. Fully vectorized.

    Args:
        bootstrap_counts: Array of shape (n_bootstraps, n_junctions, n_samples).
        effective_lengths: Array of shape (n_junctions,).
        prior_alpha: Dirichlet prior (currently unused, reserved for future use).

    Returns:
        Array of shape (n_bootstraps, n_junctions, n_samples) with PSI values in [0, 1].
    """
    # Length-normalize: divide each junction by its effective length
    # effective_lengths shape: (n_junctions,) -> broadcast to (1, n_junctions, 1)
    eff_len = effective_lengths.reshape(1, -1, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.where(eff_len > 0, bootstrap_counts.astype(float) / eff_len, 0.0)

    # PSI = normalized / sum(normalized) per sample per bootstrap
    col_sums = np.sum(normalized, axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        psi_matrix = np.where(col_sums > 0, normalized / col_sums, 0.0)

    return psi_matrix


def bootstrap_confidence_intervals(
    bootstrap_psi: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute percentile confidence intervals from bootstrap PSI. Fully vectorized.

    Args:
        bootstrap_psi: Array of shape (n_bootstraps, n_junctions, n_samples).
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (ci_low, ci_high) each of shape (n_junctions, n_samples).
    """
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100
    ci_low = np.percentile(bootstrap_psi, lower_pct, axis=0)
    ci_high = np.percentile(bootstrap_psi, upper_pct, axis=0)
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
