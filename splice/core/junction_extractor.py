"""
Module 5: core/junction_extractor.py

Junction extraction and aggregation across multiple BAM files.
Produces per-sample junction counts, MAPQ statistics, motif classification,
and junction co-occurrence evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from splice.io.bam_utils import extract_evidence_from_bam, extract_junction_stats_streaming
from splice.utils.genomic import Junction, JunctionPair
from splice.utils.motif import classify_motif, extract_motif_from_genome, score_motif


@dataclass
class JunctionEvidence:
    """Aggregated evidence for a junction across multiple samples.

    Attributes:
        junction: The junction object (chrom, start, end, strand).
        sample_counts: (n_samples,) array of raw junction counts.
        sample_weighted_counts: (n_samples,) array of 1/NH-weighted counts.
        sample_mapq_mean: (n_samples,) mean MAPQ for junction in each sample.
        sample_mapq_median: (n_samples,) median MAPQ for junction in each sample.
        sample_nh_distribution: (n_samples,) mean NH tag value for junction reads.
        is_annotated: Whether junction is in the known annotation.
        motif: Splice site motif class (e.g., "GT/AG", "non-canonical", "").
        motif_score: Confidence score for motif (0.0 to 1.0).
        max_anchor: Maximum anchor length observed across all samples.
        n_samples_detected: Number of samples where junction appears (count > 0).
        cross_sample_recurrence: Fraction of samples with this junction.
    """

    junction: Junction
    sample_counts: np.ndarray
    sample_weighted_counts: np.ndarray
    sample_mapq_mean: np.ndarray
    sample_mapq_median: np.ndarray
    sample_nh_distribution: np.ndarray
    is_annotated: bool
    motif: str
    motif_score: float
    max_anchor: int
    n_samples_detected: int
    cross_sample_recurrence: float


@dataclass
class CooccurrenceEvidence:
    """Evidence for two junctions appearing in the same read.

    Attributes:
        pair: The pair of junctions.
        sample_counts: (n_samples,) array of co-occurrence counts per sample.
    """

    pair: JunctionPair
    sample_counts: np.ndarray


def extract_all_junctions(
    bam_paths: List[str],
    sample_names: List[str],
    known_junctions: Set[Junction],
    genome_fasta_path: Optional[str] = None,
    min_anchor: int = 6,
    min_mapq: int = 0,
    mapq_weight_threshold: int = 20,
) -> Tuple[Dict[Junction, JunctionEvidence], Dict[JunctionPair, CooccurrenceEvidence]]:
    """Extract junctions and co-occurrences from all BAM files.

    Uses streaming mode: each BAM is processed read-by-read, aggregating
    junction counts directly into shared dicts. Never stores per-read objects.
    Memory usage is proportional to the number of unique junctions (~100K-500K),
    not the number of reads (~100M+).

    Args:
        bam_paths: List of paths to BAM files (must be indexed).
        sample_names: List of sample names (one per BAM).
        known_junctions: Set of annotated junctions.
        genome_fasta_path: Optional path to genome FASTA for motif classification.
        min_anchor: Minimum anchor length for valid junctions.
        min_mapq: Minimum mapping quality to count a junction.
        mapq_weight_threshold: MAPQ threshold for weighting (currently unused).

    Returns:
        Tuple of (junction_dict, cooccurrence_dict) where:
          - junction_dict: Maps Junction -> JunctionEvidence
          - cooccurrence_dict: Maps JunctionPair -> CooccurrenceEvidence
    """
    import sys

    n_samples = len(bam_paths)

    # Shared aggregation structures -- populated by streaming BAM reader
    junction_stats: Dict[Junction, Dict] = {}
    cooccurrence_counts: Dict[JunctionPair, np.ndarray] = {}

    # Process each BAM file in streaming mode
    for sample_idx, (bam_path, sample_name) in enumerate(zip(bam_paths, sample_names)):
        print(f"    [{sample_idx+1}/{n_samples}] {sample_name}...", end=" ", flush=True)

        bam_stats = extract_junction_stats_streaming(
            bam_path=bam_path,
            sample_idx=sample_idx,
            junction_stats=junction_stats,
            cooccurrence_counts=cooccurrence_counts,
            n_samples=n_samples,
            min_anchor=min_anchor,
            min_mapq=min_mapq,
        )

        print(
            f"{bam_stats['mapped_reads']:,} mapped, "
            f"{bam_stats['junction_reads']:,} junction reads, "
            f"{len(junction_stats):,} unique junctions so far",
            flush=True,
        )

    # Build JunctionEvidence objects from aggregated stats
    junction_evidence: Dict[Junction, JunctionEvidence] = {}

    for junc, samples_stats in junction_stats.items():
        sample_counts = np.zeros(n_samples, dtype=int)
        sample_weighted_counts = np.zeros(n_samples, dtype=float)
        sample_mapq_mean = np.zeros(n_samples, dtype=float)
        sample_mapq_median = np.zeros(n_samples, dtype=float)
        sample_nh_distribution = np.zeros(n_samples, dtype=float)
        max_anchor_global = 0

        for sample_idx, stats in samples_stats.items():
            n = stats["n"]
            sample_counts[sample_idx] = stats["counts"]

            # Weighted count: sum of 1/NH (approximate from mean NH)
            mean_nh = stats["nh_sum"] / n if n > 0 else 1.0
            sample_weighted_counts[sample_idx] = stats["counts"] / mean_nh

            # Mean MAPQ from running sum
            sample_mapq_mean[sample_idx] = stats["mapq_sum"] / n if n > 0 else 0.0
            # Median not available in streaming mode; use mean as proxy
            sample_mapq_median[sample_idx] = sample_mapq_mean[sample_idx]

            # Mean NH
            sample_nh_distribution[sample_idx] = mean_nh

            max_anchor_global = max(max_anchor_global, stats["max_anchor"])

        # Check if annotated
        is_annotated = junc in known_junctions

        # Classify motif if genome provided
        motif_str = ""
        motif_score_val = 0.0
        if genome_fasta_path:
            try:
                donor_dinuc, acceptor_dinuc, motif_str = extract_motif_from_genome(
                    genome_fasta_path,
                    junc.chrom,
                    junc.start,
                    junc.end,
                    junc.strand,
                )
                motif_score_val = score_motif(motif_str)
            except Exception:
                motif_str = ""
                motif_score_val = 0.0

        # Compute cross-sample recurrence
        n_samples_detected = int(np.sum(sample_counts > 0))
        cross_sample_recurrence = n_samples_detected / n_samples if n_samples > 0 else 0.0

        junction_evidence[junc] = JunctionEvidence(
            junction=junc,
            sample_counts=sample_counts,
            sample_weighted_counts=sample_weighted_counts,
            sample_mapq_mean=sample_mapq_mean,
            sample_mapq_median=sample_mapq_median,
            sample_nh_distribution=sample_nh_distribution,
            is_annotated=is_annotated,
            motif=motif_str,
            motif_score=motif_score_val,
            max_anchor=max_anchor_global,
            n_samples_detected=n_samples_detected,
            cross_sample_recurrence=cross_sample_recurrence,
        )

    # Build CooccurrenceEvidence objects
    cooccurrence_evidence: Dict[JunctionPair, CooccurrenceEvidence] = {
        pair: CooccurrenceEvidence(pair=pair, sample_counts=counts)
        for pair, counts in cooccurrence_counts.items()
    }

    return junction_evidence, cooccurrence_evidence
