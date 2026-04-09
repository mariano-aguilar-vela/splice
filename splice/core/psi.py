"""
Module 15: core/psi.py

Per-sample PSI quantification with Beta posteriors and bootstrap confidence intervals.
Integrates length normalization, bootstrap resampling, and uncertainty estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from splice.core.bootstrap import (
    bootstrap_confidence_intervals,
    bootstrap_junction_counts,
    bootstrap_psi,
)
from splice.core.evidence import ModuleEvidence


@dataclass(frozen=True, slots=True)
class ModulePSI:
    """Per-sample PSI for a splicing module.

    Attributes:
        module_id: Module identifier.
        psi_matrix: (n_junctions, n_samples) point estimates of PSI.
        ci_low_matrix: (n_junctions, n_samples) lower bound of 95% bootstrap CI.
        ci_high_matrix: (n_junctions, n_samples) upper bound of 95% bootstrap CI.
        bootstrap_psi: (n_bootstraps, n_junctions, n_samples) all bootstrap PSI replicates.
        total_counts: (n_samples,) total junction counts per sample.
        effective_n: (n_samples,) effective sample size (total weighted counts).
    """

    module_id: str
    psi_matrix: np.ndarray
    ci_low_matrix: np.ndarray
    ci_high_matrix: np.ndarray
    bootstrap_psi: np.ndarray
    total_counts: np.ndarray
    effective_n: np.ndarray


def quantify_psi(
    module_evidence_list: List[ModuleEvidence],
    n_bootstraps: int = 30,
    prior_alpha: float = 0.5,
    seed: int = 42,
) -> List[ModulePSI]:
    """Compute per-sample PSI with bootstrap confidence intervals.

    For each module:
    1. Length-normalize junction counts using effective_lengths.
    2. Compute PSI as normalized_count_k / sum(normalized_counts).
    3. Generate bootstrap resamples and compute bootstrap PSI.
    4. Compute 95% bootstrap percentile confidence intervals.
    5. Effective_n = total_weighted_counts (reflects MAPQ weighting).

    Args:
        module_evidence_list: List of ModuleEvidence objects.
        n_bootstraps: Number of bootstrap replicates (default 30).
        prior_alpha: Prior alpha for Beta distribution (unused, reserved for future).
        seed: Random seed for reproducibility (default 42).

    Returns:
        List of ModulePSI objects, one per module.
    """
    psi_list: List[ModulePSI] = []

    for evidence in module_evidence_list:
        module_id = evidence.module.module_id
        n_junctions, n_samples = evidence.junction_count_matrix.shape

        # Step 1: Length-normalize junction counts
        # Divide each row by its effective length (with zero handling)
        normalized_counts = np.zeros_like(
            evidence.junction_count_matrix, dtype=float
        )
        for j in range(n_junctions):
            if evidence.junction_effective_lengths[j] > 0:
                normalized_counts[j, :] = (
                    evidence.junction_count_matrix[j, :]
                    / evidence.junction_effective_lengths[j]
                )

        # Step 2: Compute point estimate PSI
        # PSI = normalized_count_k / sum(normalized_counts per sample)
        col_sums = np.sum(normalized_counts, axis=0)
        psi_matrix = np.zeros((n_junctions, n_samples), dtype=float)
        for sample_idx in range(n_samples):
            if col_sums[sample_idx] > 0:
                psi_matrix[:, sample_idx] = (
                    normalized_counts[:, sample_idx] / col_sums[sample_idx]
                )

        # Step 3: Bootstrap resampling
        # Generate bootstrap resamples of junction counts
        bootstrap_counts = bootstrap_junction_counts(
            evidence.junction_count_matrix,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )

        # Compute PSI for each bootstrap replicate
        bootstrap_psi_matrix = bootstrap_psi(
            bootstrap_counts,
            evidence.junction_effective_lengths,
            prior_alpha=prior_alpha,
        )

        # Step 4: Compute 95% bootstrap percentile confidence intervals
        ci_low, ci_high = bootstrap_confidence_intervals(
            bootstrap_psi_matrix,
            alpha=0.05,
        )

        # Step 5: Effective_n = total_weighted_counts
        effective_n = evidence.total_weighted

        # Create ModulePSI object
        module_psi = ModulePSI(
            module_id=module_id,
            psi_matrix=psi_matrix,
            ci_low_matrix=ci_low,
            ci_high_matrix=ci_high,
            bootstrap_psi=bootstrap_psi_matrix,
            total_counts=evidence.total_counts,
            effective_n=effective_n,
        )

        psi_list.append(module_psi)

    return psi_list
