"""
Module 12: core/evidence.py

Build multi-tier evidence matrices for splicing modules.
Combines junction counts, exon body reads, effective length normalization,
and confidence scores into matrices ready for PSI and differential testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from splicekit.core.effective_length import (
    compute_effective_lengths_for_module,
    length_normalize_counts,
)
from splicekit.core.junction_extractor import JunctionEvidence
from splicekit.core.splicegraph import SplicingModule
from splicekit.utils.genomic import Junction


@dataclass
class ModuleEvidence:
    """Evidence matrices for a splicing module.

    Attributes:
        module: SplicingModule object.
        junction_count_matrix: (n_junctions, n_samples) raw integer counts.
        junction_weighted_matrix: (n_junctions, n_samples) MAPQ-weighted counts.
        junction_mapq_matrix: (n_junctions, n_samples) mean MAPQ per junction per sample.
        exon_body_count_matrix: (n_exons, n_samples) exon body reads (optional).
        exon_body_weighted_matrix: (n_exons, n_samples) weighted exon body reads (optional).
        junction_effective_lengths: (n_junctions,) effective length per junction.
        normalized_count_matrix: (n_junctions, n_samples) length-normalized counts.
        total_counts: (n_samples,) total junction reads per sample.
        total_weighted: (n_samples,) total weighted junction reads per sample.
        junction_confidence: (n_junctions,) composite confidence scores.
        is_annotated: (n_junctions,) bool, whether junction is annotated.
    """

    module: SplicingModule
    junction_count_matrix: np.ndarray
    junction_weighted_matrix: np.ndarray
    junction_mapq_matrix: np.ndarray
    exon_body_count_matrix: Optional[np.ndarray]
    exon_body_weighted_matrix: Optional[np.ndarray]
    junction_effective_lengths: np.ndarray
    normalized_count_matrix: np.ndarray
    total_counts: np.ndarray
    total_weighted: np.ndarray
    junction_confidence: np.ndarray
    is_annotated: np.ndarray


def build_evidence_matrices(
    modules: List[SplicingModule],
    junction_evidence: Dict[Junction, JunctionEvidence],
    junction_confidence: Optional[Dict] = None,
    read_length: int = 101,
    n_samples: int = 0,
) -> List[ModuleEvidence]:
    """Build evidence matrices for all modules.

    Assembles junction counts, effective lengths, and normalization.

    Args:
        modules: List of SplicingModule objects.
        junction_evidence: Dict mapping Junction -> JunctionEvidence.
        junction_confidence: Optional dict mapping Junction -> confidence scores.
        read_length: Sequencing read length for effective length computation.
        n_samples: Number of samples (inferred from evidence if 0).

    Returns:
        List of ModuleEvidence objects, one per module.
    """
    if not modules:
        return []

    # Infer n_samples from first junction evidence
    if n_samples == 0 and junction_evidence:
        first_evidence = next(iter(junction_evidence.values()))
        n_samples = len(first_evidence.sample_counts)

    module_evidence_list: List[ModuleEvidence] = []

    for module in modules:
        # Extract junction evidence for this module
        count_matrix = np.zeros((module.n_connections, n_samples), dtype=int)
        weighted_matrix = np.zeros((module.n_connections, n_samples), dtype=float)
        mapq_matrix = np.zeros((module.n_connections, n_samples), dtype=float)
        confidence_scores = np.zeros(module.n_connections, dtype=float)
        annotated_flags = np.zeros(module.n_connections, dtype=bool)

        for i, junction in enumerate(module.junctions):
            if junction in junction_evidence:
                evidence = junction_evidence[junction]
                count_matrix[i, :] = evidence.sample_counts
                weighted_matrix[i, :] = evidence.sample_weighted_counts
                mapq_matrix[i, :] = evidence.sample_mapq_mean
                annotated_flags[i] = evidence.is_annotated

                # Get confidence score if available
                if junction_confidence and junction in junction_confidence:
                    conf = junction_confidence[junction]
                    # Handle both dict and dataclass
                    if hasattr(conf, "composite_score"):
                        confidence_scores[i] = conf.composite_score
                    elif isinstance(conf, dict):
                        confidence_scores[i] = conf.get("composite_score", 0.5)

        # Compute effective lengths
        gene_exons = []  # Empty for now; could be populated from gene model
        effective_lengths = compute_effective_lengths_for_module(
            module.junctions, gene_exons, read_length
        )

        # Length-normalize counts
        normalized_matrix = length_normalize_counts(count_matrix, effective_lengths)

        # Compute totals
        total_counts = np.sum(count_matrix, axis=0)
        total_weighted = np.sum(weighted_matrix, axis=0)

        # Create ModuleEvidence
        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=count_matrix,
            junction_weighted_matrix=weighted_matrix,
            junction_mapq_matrix=mapq_matrix,
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=effective_lengths,
            normalized_count_matrix=normalized_matrix,
            total_counts=total_counts,
            total_weighted=total_weighted,
            junction_confidence=confidence_scores,
            is_annotated=annotated_flags,
        )
        module_evidence_list.append(evidence)

    return module_evidence_list


def filter_evidence_by_depth(
    module_evidence_list: List[ModuleEvidence],
    min_total_reads: int = 20,
    min_samples_with_reads: int = 3,
) -> List[ModuleEvidence]:
    """Filter modules by minimum read depth.

    A module passes if:
    - At least min_samples_with_reads samples have total_counts >= 1
    - The sum of total_counts across all samples >= min_total_reads

    Args:
        module_evidence_list: List of ModuleEvidence objects.
        min_total_reads: Minimum total reads across all samples.
        min_samples_with_reads: Minimum samples with reads.

    Returns:
        Filtered list of ModuleEvidence objects.
    """
    filtered = []

    for evidence in module_evidence_list:
        # Check number of samples with reads
        samples_with_reads = np.sum(evidence.total_counts >= 1)
        if samples_with_reads < min_samples_with_reads:
            continue

        # Check total reads
        total_reads = np.sum(evidence.total_counts)
        if total_reads < min_total_reads:
            continue

        filtered.append(evidence)

    return filtered


def filter_evidence_by_size(
    module_evidence_list: List[ModuleEvidence],
    min_junctions: int = 2,
) -> List[ModuleEvidence]:
    """Filter modules by minimum number of junctions.

    Args:
        module_evidence_list: List of ModuleEvidence objects.
        min_junctions: Minimum number of junctions.

    Returns:
        Filtered list of ModuleEvidence objects.
    """
    return [m for m in module_evidence_list if m.module.n_connections >= min_junctions]


def get_module_psi_matrix(evidence: ModuleEvidence) -> np.ndarray:
    """Compute PSI (Percent Spliced In) matrix from evidence.

    PSI for each junction = normalized_count / total_normalized_counts

    Args:
        evidence: ModuleEvidence object.

    Returns:
        Array of shape (n_junctions, n_samples) with PSI values in [0, 1].
    """
    # Sum normalized counts across junctions per sample
    total_normalized = np.sum(evidence.normalized_count_matrix, axis=0)

    # Avoid division by zero
    psi = np.zeros_like(evidence.normalized_count_matrix)
    nonzero_mask = total_normalized > 0

    psi[:, nonzero_mask] = (
        evidence.normalized_count_matrix[:, nonzero_mask]
        / total_normalized[nonzero_mask]
    )

    return psi
