"""
Module 11: core/effective_length.py

Compute effective lengths for inclusion and skipping isoforms.
Follows rMATS's length normalization to eliminate length bias in PSI estimation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from splicekit.utils.genomic import GenomicInterval, Junction


def compute_se_effective_lengths(
    target_exon: GenomicInterval,
    upstream_exon: GenomicInterval,
    downstream_exon: GenomicInterval,
    read_length: int,
    anchor_length: int = 1,
) -> Tuple[float, float]:
    """Compute effective lengths for skipped exon (SE) events.

    Follows rMATS sm_inclen() (rmatspipeline.pyx lines 1355-1366).
    Effective length is the number of unique read positions that can
    generate a junction-spanning read.

    Args:
        target_exon: The exon that may be skipped (GenomicInterval).
        upstream_exon: Exon before target (GenomicInterval).
        downstream_exon: Exon after target (GenomicInterval).
        read_length: Sequencing read length.
        anchor_length: Minimum anchor required per side (default 1).

    Returns:
        Tuple of (inc_effective_length, skip_effective_length).
    """
    # JC (Junction Counts) mode
    jc_inc_len = (read_length - 2 * anchor_length + 1) + min(
        target_exon.length, read_length - 2 * anchor_length + 1
    )
    jc_skip_len = read_length - 2 * anchor_length + 1

    # JCEC (Junction Counts + Exon Body) mode: add exon body contribution
    exon_body_contribution = max(0, target_exon.length - read_length + 1)
    jcec_inc_len = jc_inc_len + exon_body_contribution
    jcec_skip_len = jc_skip_len

    return jcec_inc_len, jcec_skip_len


def compute_effective_lengths_for_module(
    module_junctions: List[Junction],
    gene_exons: List[GenomicInterval],
    read_length: int,
    anchor_length: int = 1,
) -> np.ndarray:
    """Compute effective length for each junction in a module.

    For junction counting evidence, effective length is the number of
    unique read starting positions that can span that junction.

    effective_length = read_length - 2 * anchor_length + 1

    For exon body evidence, effective_length = max(0, exon_length - read_length + 1).

    Args:
        module_junctions: List of Junction objects in module.
        gene_exons: List of GenomicInterval objects (exons) in gene.
        read_length: Sequencing read length.
        anchor_length: Minimum anchor per side (default 1).

    Returns:
        Array of shape (n_junctions,) with effective lengths.
    """
    n = len(module_junctions)
    effective_lengths = np.zeros(n, dtype=float)

    # For junction evidence: simple formula based on read length and anchor
    for i in range(n):
        effective_lengths[i] = read_length - 2 * anchor_length + 1

    return effective_lengths


def compute_exon_body_effective_lengths(
    gene_exons: List[GenomicInterval],
    read_length: int,
) -> np.ndarray:
    """Compute effective lengths for exon body reads.

    For exon body evidence, the effective length is the number of
    unique read positions that can start within the exon.

    effective_length = max(0, exon_length - read_length + 1)

    Args:
        gene_exons: List of GenomicInterval objects (exons).
        read_length: Sequencing read length.

    Returns:
        Array of shape (n_exons,) with exon body effective lengths.
    """
    n = len(gene_exons)
    effective_lengths = np.zeros(n, dtype=float)

    for i, exon in enumerate(gene_exons):
        effective_lengths[i] = max(0, exon.length - read_length + 1)

    return effective_lengths


def length_normalize_counts(
    counts: np.ndarray, effective_lengths: np.ndarray
) -> np.ndarray:
    """Normalize counts by effective length.

    Handles zero effective lengths by returning 0 for those positions.

    Args:
        counts: Count array (any shape).
        effective_lengths: Effective length array (same shape as counts).

    Returns:
        Normalized array (same shape as counts).
    """
    # Avoid division by zero
    normalized = np.zeros_like(counts, dtype=float)
    nonzero_mask = effective_lengths > 0

    normalized[nonzero_mask] = counts[nonzero_mask] / effective_lengths[nonzero_mask]

    return normalized


def compute_library_size_factors(
    count_matrix: np.ndarray,
) -> np.ndarray:
    """Compute library size factors using simple ratio of column sums.

    Adjusts for differences in sequencing depth across samples.
    Uses a simple depth-based approach for robustness.

    Args:
        count_matrix: Array of shape (n_features, n_samples) with counts.

    Returns:
        Array of shape (n_samples,) with size factors.
    """
    # Simple approach: size factor = total counts in sample / median total
    col_sums = np.sum(count_matrix, axis=0).astype(float)
    median_sum = np.median(col_sums)

    # Avoid division by zero
    size_factors = np.where(col_sums > 0, col_sums / median_sum, 1.0)

    # Normalize so geometric mean = 1
    nonzero_factors = size_factors[size_factors > 0]
    if len(nonzero_factors) > 0:
        geo_mean = np.exp(np.mean(np.log(nonzero_factors)))
        size_factors = size_factors / geo_mean

    return size_factors
