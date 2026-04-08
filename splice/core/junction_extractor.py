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

from splicekit.io.bam_utils import extract_evidence_from_bam
from splicekit.utils.genomic import Junction, JunctionPair
from splicekit.utils.motif import classify_motif, extract_motif_from_genome, score_motif


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

    Processes each BAM file to extract junction evidence, then aggregates across
    samples to produce per-junction statistics and co-occurrence counts.

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
    n_samples = len(bam_paths)

    # Initialize aggregation structures
    # junction -> {sample_idx -> [counts, mapq_values, nh_values]}
    junction_stats: Dict[Junction, Dict[int, List | np.ndarray]] = {}
    # pair -> [counts per sample]
    cooccurrence_counts: Dict[JunctionPair, np.ndarray] = {}

    # Process each BAM file
    for sample_idx, (bam_path, sample_name) in enumerate(zip(bam_paths, sample_names)):
        evidence_list, stats = extract_evidence_from_bam(
            bam_path, min_anchor=min_anchor, min_mapq=min_mapq
        )

        # Aggregate junction evidence from this sample
        for evidence in evidence_list:
            # For each read, count each junction and track statistics
            for i, junc in enumerate(evidence.junctions):
                # Initialize if not present
                if junc not in junction_stats:
                    junction_stats[junc] = {}
                if sample_idx not in junction_stats[junc]:
                    junction_stats[junc][sample_idx] = {
                        "counts": 0,
                        "mapq_values": [],
                        "nh_values": [],
                        "max_anchor": 0,
                    }

                # Raw count: increment by 1
                junction_stats[junc][sample_idx]["counts"] += 1

                # Collect MAPQ and NH values
                junction_stats[junc][sample_idx]["mapq_values"].append(evidence.mapq)
                junction_stats[junc][sample_idx]["nh_values"].append(evidence.nh)

                # Track maximum anchor (using exon blocks flanking the junction)
                if i < len(evidence.exon_blocks):
                    anchor = evidence.exon_blocks[i].length
                    junction_stats[junc][sample_idx]["max_anchor"] = max(
                        junction_stats[junc][sample_idx]["max_anchor"], anchor
                    )
                if i + 1 < len(evidence.exon_blocks):
                    anchor = evidence.exon_blocks[i + 1].length
                    junction_stats[junc][sample_idx]["max_anchor"] = max(
                        junction_stats[junc][sample_idx]["max_anchor"], anchor
                    )

            # Aggregate co-occurrence evidence
            for pair in evidence.junction_pairs:
                if pair not in cooccurrence_counts:
                    cooccurrence_counts[pair] = np.zeros(n_samples, dtype=int)
                cooccurrence_counts[pair][sample_idx] += 1

    # Build JunctionEvidence objects
    junction_evidence: Dict[Junction, JunctionEvidence] = {}

    for junc, samples_stats in junction_stats.items():
        # Initialize arrays for all samples
        sample_counts = np.zeros(n_samples, dtype=int)
        sample_weighted_counts = np.zeros(n_samples, dtype=float)
        sample_mapq_mean = np.zeros(n_samples, dtype=float)
        sample_mapq_median = np.zeros(n_samples, dtype=float)
        sample_nh_distribution = np.zeros(n_samples, dtype=float)
        max_anchor_global = 0

        # Fill in statistics for samples that have this junction
        for sample_idx, stats in samples_stats.items():
            sample_counts[sample_idx] = stats["counts"]

            # Weighted count: sum of 1/NH for each read
            mapq_vals = np.array(stats["mapq_values"], dtype=float)
            nh_vals = np.array(stats["nh_values"], dtype=float)

            sample_weighted_counts[sample_idx] = np.sum(1.0 / nh_vals)
            sample_mapq_mean[sample_idx] = np.mean(mapq_vals) if len(mapq_vals) > 0 else 0.0
            sample_mapq_median[sample_idx] = (
                np.median(mapq_vals) if len(mapq_vals) > 0 else 0.0
            )
            sample_nh_distribution[sample_idx] = np.mean(nh_vals) if len(nh_vals) > 0 else 0.0

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
                # If motif extraction fails, leave empty
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
