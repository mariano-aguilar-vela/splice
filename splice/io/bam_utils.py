"""
Module 4: io/bam_utils.py

BAM file reading, CIGAR parsing, and evidence extraction (junctions, exon blocks, co-occurrences).
Filters reads by mapping quality, secondary alignment status, duplicates, and QC flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from splice.utils.genomic import GenomicInterval, Junction, JunctionPair

if TYPE_CHECKING:
    import pysam
else:
    try:
        import pysam
    except ImportError:
        pysam = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ReadEvidence:
    """Evidence extracted from a single read.

    Attributes:
        junctions: List of junctions (N operations in CIGAR).
        junction_pairs: All pairwise combinations of junctions (co-occurrence).
        exon_blocks: Contiguous aligned segments (M/=/X operations).
        mapq: Mapping quality score.
        nh: Number of alignments (NH tag, default 1).
        is_proper_pair: Whether read is part of a properly paired pair.
        alignment_score: Alignment score from AS tag (if present).
    """

    junctions: List[Junction]
    junction_pairs: List[JunctionPair]
    exon_blocks: List[GenomicInterval]
    mapq: int
    nh: int
    is_proper_pair: bool
    alignment_score: Optional[int]


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------


def _extract_junctions_from_cigar(
    read: pysam.AlignedSegment,
    min_anchor: int = 6
) -> List[Junction]:
    """Extract junctions from a read's aligned blocks.

    Uses pysam.AlignedSegment.get_blocks() to get exon blocks, then infers
    introns from gaps between consecutive blocks. Each intron becomes a junction.
    Junctions are only included if both flanking blocks have min_anchor bases.

    Args:
        read: pysam AlignedSegment object.
        min_anchor: Minimum aligned bases required on each side of intron.

    Returns:
        List of Junction objects.
    """
    junctions = []
    blocks = read.get_blocks()

    for i in range(len(blocks) - 1):
        intron_start = blocks[i][1]  # End of previous block
        intron_end = blocks[i + 1][0]  # Start of next block

        # Check minimum anchor length on both sides
        left_anchor = blocks[i][1] - blocks[i][0]
        right_anchor = blocks[i + 1][1] - blocks[i + 1][0]

        if left_anchor >= min_anchor and right_anchor >= min_anchor:
            junction = Junction(
                chrom=read.reference_name,
                start=intron_start,
                end=intron_end,
                strand="+" if not read.is_reverse else "-"
            )
            junctions.append(junction)

    return junctions


def _extract_exon_blocks(
    read: pysam.AlignedSegment
) -> List[GenomicInterval]:
    """Extract exon blocks (aligned regions) from a read.

    Uses pysam.AlignedSegment.get_blocks() to retrieve all aligned blocks.

    Args:
        read: pysam AlignedSegment object.

    Returns:
        List of GenomicInterval objects representing exon blocks.
    """
    blocks = []
    for start, end in read.get_blocks():
        block = GenomicInterval(
            chrom=read.reference_name,
            start=start,
            end=end,
            strand="+" if not read.is_reverse else "-"
        )
        blocks.append(block)
    return blocks


def _extract_junction_pairs(junctions: List[Junction]) -> List[JunctionPair]:
    """Generate all pairwise combinations of junctions.

    For a read with junctions [J1, J2, J3], generates [(J1,J2), (J1,J3), (J2,J3)].

    Args:
        junctions: List of Junction objects.

    Returns:
        List of JunctionPair objects (all 2-combinations).
    """
    return [JunctionPair(j1, j2) for j1, j2 in combinations(junctions, 2)]


def _passes_quality_filters(read: pysam.AlignedSegment) -> bool:
    """Check if a read passes basic quality filters.

    Excludes: unmapped, secondary alignments, duplicates, QC failures.

    Args:
        read: pysam AlignedSegment object.

    Returns:
        True if read passes all filters, False otherwise.
    """
    return not (
        read.is_unmapped
        or read.is_secondary
        or read.is_duplicate
        or read.is_qcfail
    )


def _get_nh_tag(read: pysam.AlignedSegment) -> int:
    """Get the number of alignments (NH tag) for a read.

    Args:
        read: pysam AlignedSegment object.

    Returns:
        NH tag value, or 1 if not present (single unique alignment).
    """
    try:
        return read.get_tag("NH")
    except KeyError:
        return 1


def _get_as_tag(read: pysam.AlignedSegment) -> Optional[int]:
    """Get the alignment score (AS tag) for a read.

    Args:
        read: pysam AlignedSegment object.

    Returns:
        AS tag value if present, otherwise None.
    """
    try:
        return read.get_tag("AS")
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Main API functions
# ---------------------------------------------------------------------------


def extract_evidence_from_read(
    read: pysam.AlignedSegment,
    min_anchor: int = 6
) -> Optional[ReadEvidence]:
    """Extract all evidence from a single read.

    Parses the read's CIGAR string to extract junctions, exon blocks, and
    junction co-occurrence information. Filters reads that fail quality checks.

    Args:
        read: pysam AlignedSegment object.
        min_anchor: Minimum aligned bases required on each side of a junction.

    Returns:
        ReadEvidence object if read passes filters, None otherwise.
    """
    if not _passes_quality_filters(read):
        return None

    junctions = _extract_junctions_from_cigar(read, min_anchor)
    junction_pairs = _extract_junction_pairs(junctions)
    exon_blocks = _extract_exon_blocks(read)

    nh = _get_nh_tag(read)
    as_tag = _get_as_tag(read)

    return ReadEvidence(
        junctions=junctions,
        junction_pairs=junction_pairs,
        exon_blocks=exon_blocks,
        mapq=read.mapping_quality,
        nh=nh,
        is_proper_pair=read.is_proper_pair,
        alignment_score=as_tag
    )


def extract_evidence_from_bam(
    bam_path: str,
    region: Optional[str] = None,
    min_anchor: int = 6,
    min_mapq: int = 0
) -> Tuple[List[ReadEvidence], Dict]:
    """Extract evidence from all reads in a BAM file or region.

    WARNING: This function loads ALL read evidence into memory. For large BAM
    files (>10M reads), this will consume tens of GB of RAM. Use
    extract_junction_stats_streaming() instead for production workloads.
    This function is intended for small BAMs, testing, or debugging only.

    Iterates through reads, applies quality filters, and extracts evidence
    for each passing read. Collects statistics across all reads.

    Args:
        bam_path: Path to BAM file (must be indexed).
        region: Optional region string (e.g., "chr1:1000-2000").
        min_anchor: Minimum aligned bases on each side of junction.
        min_mapq: Minimum mapping quality to include read.

    Returns:
        Tuple of (evidence_list, stats_dict) where stats_dict contains:
          - total_reads: Total reads in BAM
          - mapped_reads: Non-unmapped reads
          - junction_reads: Reads with at least one junction
          - multi_mapped_reads: Reads with NH > 1
          - mean_mapq: Mean mapping quality across mapped reads
          - median_mapq: Median mapping quality across mapped reads
    """
    import warnings
    warnings.warn(
        "extract_evidence_from_bam loads all reads into memory. "
        "For large BAMs, use extract_junction_stats_streaming() instead.",
        ResourceWarning,
        stacklevel=2,
    )
    evidence_list = []
    mapq_values = []
    total_reads = 0
    mapped_reads = 0
    junction_reads = 0
    multi_mapped_reads = 0

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(region=region):
            total_reads += 1

            if read.is_unmapped:
                continue

            mapped_reads += 1
            mapq_values.append(read.mapping_quality)

            if read.mapping_quality < min_mapq:
                continue

            evidence = extract_evidence_from_read(read, min_anchor)
            if evidence is None:
                continue

            evidence_list.append(evidence)

            if evidence.junctions:
                junction_reads += 1

            if evidence.nh > 1:
                multi_mapped_reads += 1

    # Compute statistics
    stats = {
        "total_reads": total_reads,
        "mapped_reads": mapped_reads,
        "junction_reads": junction_reads,
        "multi_mapped_reads": multi_mapped_reads,
        "mean_mapq": float(np.mean(mapq_values)) if mapq_values else 0.0,
        "median_mapq": float(np.median(mapq_values)) if mapq_values else 0.0,
    }

    return evidence_list, stats


def count_exon_body_reads(
    bam_path: str,
    exon: GenomicInterval,
    min_mapq: int = 0
) -> Tuple[int, float]:
    """Count reads falling entirely within an exon body.

    A read counts if all its aligned blocks fall completely within the exon
    boundaries. Multi-mapped reads (NH > 1) contribute fractionally (1/NH).

    Args:
        bam_path: Path to BAM file (must be indexed).
        exon: GenomicInterval defining the exon boundary.
        min_mapq: Minimum mapping quality to include read.

    Returns:
        Tuple of (raw_count, mapq_weighted_count) where:
          - raw_count: Number of reads entirely within exon
          - mapq_weighted_count: Sum of 1/NH for all counted reads
    """
    raw_count = 0
    weighted_count = 0.0

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(
            contig=exon.chrom,
            start=exon.start,
            stop=exon.end
        ):
            if not _passes_quality_filters(read):
                continue

            if read.mapping_quality < min_mapq:
                continue

            # Check if all aligned blocks are within exon boundaries
            blocks = read.get_blocks()
            if not blocks:
                continue

            all_within = all(
                block[0] >= exon.start and block[1] <= exon.end
                for block in blocks
            )
            if not all_within:
                continue

            raw_count += 1

            # Add weighted count (1 / NH for multi-mapped reads)
            nh = _get_nh_tag(read)
            weighted_count += 1.0 / nh

    return raw_count, weighted_count


def extract_junction_stats_streaming(
    bam_path: str,
    sample_idx: int,
    junction_stats: Dict,
    cooccurrence_counts: Dict,
    n_samples: int,
    min_anchor: int = 6,
    min_mapq: int = 0,
    region: Optional[str] = None,
) -> Dict:
    """Extract junction statistics from a BAM file in streaming mode.

    Reads the BAM file one read at a time, extracts junctions from each read,
    and immediately aggregates into the provided junction_stats and
    cooccurrence_counts dicts. Never stores per-read objects in memory.

    Memory usage is proportional to the number of unique junctions (~100K-500K),
    not the number of reads (~100M+).

    Args:
        bam_path: Path to BAM file (must be indexed).
        sample_idx: Index of this sample in the sample array.
        junction_stats: Shared dict to accumulate into.
            Maps Junction -> {sample_idx -> {counts, mapq_sum, mapq_sq_sum, nh_sum, n, max_anchor}}.
        cooccurrence_counts: Shared dict to accumulate into.
            Maps JunctionPair -> np.ndarray of shape (n_samples,).
        n_samples: Total number of samples (for initializing arrays).
        min_anchor: Minimum anchor length for valid junctions.
        min_mapq: Minimum mapping quality to include read.
        region: Optional region string (e.g., "chr1") for chromosome-level fetch.
            When None, reads the entire BAM file.

    Returns:
        Dict with BAM-level statistics (total_reads, mapped_reads, etc.).
    """
    total_reads = 0
    mapped_reads = 0
    junction_reads = 0
    multi_mapped_reads = 0
    mapq_sum = 0.0
    mapq_count = 0

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(region=region):
            total_reads += 1

            if read.is_unmapped:
                continue

            mapped_reads += 1
            mapq_count += 1
            mapq_sum += read.mapping_quality

            if not _passes_quality_filters(read):
                continue

            if read.mapping_quality < min_mapq:
                continue

            # Extract junctions directly from CIGAR - no ReadEvidence object
            junctions = _extract_junctions_from_cigar(read, min_anchor)
            if not junctions:
                continue

            junction_reads += 1

            nh = _get_nh_tag(read)
            mapq = read.mapping_quality
            if nh > 1:
                multi_mapped_reads += 1

            # Get exon blocks for anchor calculation
            blocks = read.get_blocks()

            # Aggregate per-junction stats directly
            for i, junc in enumerate(junctions):
                if junc not in junction_stats:
                    junction_stats[junc] = {}
                if sample_idx not in junction_stats[junc]:
                    junction_stats[junc][sample_idx] = {
                        "counts": 0,
                        "mapq_sum": 0.0,
                        "mapq_sq_sum": 0.0,
                        "nh_sum": 0.0,
                        "n": 0,
                        "max_anchor": 0,
                    }

                s = junction_stats[junc][sample_idx]
                s["counts"] += 1
                s["mapq_sum"] += mapq
                s["mapq_sq_sum"] += mapq * mapq
                s["nh_sum"] += nh
                s["n"] += 1

                # Track max anchor from flanking blocks
                if i < len(blocks):
                    anchor = blocks[i][1] - blocks[i][0]
                    s["max_anchor"] = max(s["max_anchor"], anchor)
                if i + 1 < len(blocks):
                    anchor = blocks[i + 1][1] - blocks[i + 1][0]
                    s["max_anchor"] = max(s["max_anchor"], anchor)

            # Aggregate co-occurrence for multi-junction reads
            if len(junctions) >= 2:
                for j1, j2 in combinations(junctions, 2):
                    pair = JunctionPair(j1, j2)
                    if pair not in cooccurrence_counts:
                        cooccurrence_counts[pair] = np.zeros(n_samples, dtype=int)
                    cooccurrence_counts[pair][sample_idx] += 1

    stats = {
        "total_reads": total_reads,
        "mapped_reads": mapped_reads,
        "junction_reads": junction_reads,
        "multi_mapped_reads": multi_mapped_reads,
        "mean_mapq": mapq_sum / mapq_count if mapq_count > 0 else 0.0,
    }

    return stats
