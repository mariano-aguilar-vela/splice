"""
Tests for core/junction_extractor.py

Covers: JunctionEvidence, CooccurrenceEvidence, and extract_all_junctions.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from splice.core.junction_extractor import (
    CooccurrenceEvidence,
    JunctionEvidence,
    extract_all_junctions,
)
from splice.io.bam_utils import ReadEvidence
from splice.utils.genomic import GenomicInterval, Junction, JunctionPair


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def create_read_evidence(
    junctions=None,
    junction_pairs=None,
    exon_blocks=None,
    mapq=60,
    nh=1,
    is_proper_pair=True,
    alignment_score=None,
):
    """Create a mock ReadEvidence."""
    return ReadEvidence(
        junctions=junctions or [],
        junction_pairs=junction_pairs or [],
        exon_blocks=exon_blocks or [],
        mapq=mapq,
        nh=nh,
        is_proper_pair=is_proper_pair,
        alignment_score=alignment_score,
    )


# ---------------------------------------------------------------------------
# JunctionEvidence tests
# ---------------------------------------------------------------------------


class TestJunctionEvidence:
    def test_construction(self):
        """Test basic JunctionEvidence construction."""
        junc = Junction("chr1", 100, 200, "+")
        counts = np.array([5, 3, 2])
        weighted = np.array([4.5, 3.0, 1.5])
        mapq_mean = np.array([55.0, 50.0, 45.0])
        mapq_median = np.array([56.0, 51.0, 46.0])
        nh_dist = np.array([1.1, 1.2, 1.3])

        evidence = JunctionEvidence(
            junction=junc,
            sample_counts=counts,
            sample_weighted_counts=weighted,
            sample_mapq_mean=mapq_mean,
            sample_mapq_median=mapq_median,
            sample_nh_distribution=nh_dist,
            is_annotated=True,
            motif="GT/AG",
            motif_score=1.0,
            max_anchor=10,
            n_samples_detected=3,
            cross_sample_recurrence=1.0,
        )

        assert evidence.junction == junc
        assert evidence.is_annotated is True
        assert evidence.motif == "GT/AG"
        assert evidence.motif_score == 1.0
        assert evidence.max_anchor == 10
        assert evidence.n_samples_detected == 3

    def test_array_shapes(self):
        """Test that arrays have correct shapes."""
        junc = Junction("chr1", 100, 200, "+")
        n_samples = 5

        evidence = JunctionEvidence(
            junction=junc,
            sample_counts=np.zeros(n_samples, dtype=int),
            sample_weighted_counts=np.zeros(n_samples, dtype=float),
            sample_mapq_mean=np.zeros(n_samples, dtype=float),
            sample_mapq_median=np.zeros(n_samples, dtype=float),
            sample_nh_distribution=np.zeros(n_samples, dtype=float),
            is_annotated=False,
            motif="",
            motif_score=0.0,
            max_anchor=0,
            n_samples_detected=0,
            cross_sample_recurrence=0.0,
        )

        assert evidence.sample_counts.shape == (n_samples,)
        assert evidence.sample_weighted_counts.shape == (n_samples,)
        assert evidence.sample_mapq_mean.shape == (n_samples,)


# ---------------------------------------------------------------------------
# CooccurrenceEvidence tests
# ---------------------------------------------------------------------------


class TestCooccurrenceEvidence:
    def test_construction(self):
        """Test basic CooccurrenceEvidence construction."""
        j1 = Junction("chr1", 100, 200, "+")
        j2 = Junction("chr1", 300, 400, "+")
        pair = JunctionPair(j1, j2)
        counts = np.array([5, 3, 2])

        evidence = CooccurrenceEvidence(pair=pair, sample_counts=counts)

        assert evidence.pair == pair
        assert len(evidence.sample_counts) == 3
        assert evidence.sample_counts[0] == 5

    def test_array_shape(self):
        """Test that sample_counts has correct shape."""
        j1 = Junction("chr1", 100, 200, "+")
        j2 = Junction("chr1", 300, 400, "+")
        pair = JunctionPair(j1, j2)
        n_samples = 4

        evidence = CooccurrenceEvidence(
            pair=pair, sample_counts=np.zeros(n_samples, dtype=int)
        )

        assert evidence.sample_counts.shape == (n_samples,)


# ---------------------------------------------------------------------------
# extract_all_junctions tests
# ---------------------------------------------------------------------------


def _make_streaming_mock(reads_by_bam):
    """Create a mock for extract_junction_stats_streaming.

    Simulates BAM reading by populating junction_stats and cooccurrence_counts
    dicts from ReadEvidence objects, matching extract_junction_stats_streaming's
    in-place aggregation behavior.

    Args:
        reads_by_bam: dict mapping bam_path -> (list_of_reads, stats_dict)
                      or a single (list_of_reads, stats_dict) for all BAMs.
    """
    from itertools import combinations as _combinations

    def streaming_side_effect(bam_path, sample_idx, junction_stats,
                              cooccurrence_counts, n_samples,
                              min_anchor=6, min_mapq=0):
        if callable(reads_by_bam):
            read_list, bam_stats = reads_by_bam(bam_path)
        elif isinstance(reads_by_bam, dict):
            read_list, bam_stats = reads_by_bam.get(
                bam_path, ([], {"total_reads": 0})
            )
        else:
            read_list, bam_stats = reads_by_bam

        mapped = 0
        junction_read_count = 0

        for read in read_list:
            mapped += 1
            if not read.junctions:
                continue
            junction_read_count += 1

            for junc_idx, junc in enumerate(read.junctions):
                if junc not in junction_stats:
                    junction_stats[junc] = {}
                if sample_idx not in junction_stats[junc]:
                    junction_stats[junc][sample_idx] = {
                        "counts": 0, "mapq_sum": 0.0, "mapq_sq_sum": 0.0,
                        "nh_sum": 0.0, "n": 0, "max_anchor": 0,
                    }
                stats = junction_stats[junc][sample_idx]
                stats["counts"] += 1
                stats["mapq_sum"] += read.mapq
                stats["mapq_sq_sum"] += read.mapq ** 2
                stats["nh_sum"] += read.nh
                stats["n"] += 1

                # Compute anchor from exon blocks
                if read.exon_blocks and junc_idx < len(read.exon_blocks):
                    for block in read.exon_blocks:
                        if block.end == junc.start:
                            stats["max_anchor"] = max(
                                stats["max_anchor"], block.length
                            )
                        if block.start == junc.end:
                            stats["max_anchor"] = max(
                                stats["max_anchor"], block.length
                            )

            # Co-occurrence pairs
            for pair in read.junction_pairs:
                if pair not in cooccurrence_counts:
                    cooccurrence_counts[pair] = np.zeros(n_samples, dtype=int)
                cooccurrence_counts[pair][sample_idx] += 1

        return {
            "total_reads": bam_stats.get("total_reads", len(read_list)),
            "mapped_reads": mapped,
            "junction_reads": junction_read_count,
            "multi_mapped_reads": 0,
            "mean_mapq": 0.0,
            "median_mapq": 0.0,
        }

    return streaming_side_effect


class TestExtractAllJunctions:
    def test_single_sample_single_junction(self):
        """Test extraction from a single BAM with one junction."""
        j1 = Junction("chr1", 100, 200, "+")
        block1 = GenomicInterval("chr1", 50, 100, "+")
        block2 = GenomicInterval("chr1", 200, 250, "+")

        read_evidence = create_read_evidence(
            junctions=[j1],
            exon_blocks=[block1, block2],
            mapq=60,
            nh=1,
        )

        mock_fn = _make_streaming_mock(([read_evidence], {"total_reads": 1}))
        with patch(
            "splice.core.junction_extractor.extract_junction_stats_streaming",
            side_effect=mock_fn,
        ):
            junc_dict, cooc_dict = extract_all_junctions(
                ["mock.bam"],
                ["sample1"],
                set(),
            )

        assert len(junc_dict) == 1
        assert j1 in junc_dict
        assert junc_dict[j1].sample_counts[0] == 1
        assert junc_dict[j1].n_samples_detected == 1
        assert junc_dict[j1].cross_sample_recurrence == 1.0

    def test_multiple_samples_same_junction(self):
        """Test extraction from multiple BAMs with same junction."""
        j1 = Junction("chr1", 100, 200, "+")
        block1 = GenomicInterval("chr1", 50, 100, "+")
        block2 = GenomicInterval("chr1", 200, 250, "+")

        read1 = create_read_evidence(
            junctions=[j1], exon_blocks=[block1, block2], mapq=60, nh=1
        )
        read2 = create_read_evidence(
            junctions=[j1], exon_blocks=[block1, block2], mapq=50, nh=2
        )

        def reads_by_bam(bam_path):
            if "sample1.bam" in bam_path:
                return ([read1], {"total_reads": 1})
            else:
                return ([read2], {"total_reads": 1})

        mock_fn = _make_streaming_mock(reads_by_bam)
        with patch(
            "splice.core.junction_extractor.extract_junction_stats_streaming",
            side_effect=mock_fn,
        ):
            junc_dict, cooc_dict = extract_all_junctions(
                ["sample1.bam", "sample2.bam"],
                ["sample1", "sample2"],
                set(),
            )

        assert j1 in junc_dict
        evidence = junc_dict[j1]
        assert evidence.sample_counts[0] == 1
        assert evidence.sample_counts[1] == 1
        assert evidence.n_samples_detected == 2
        assert evidence.cross_sample_recurrence == 1.0

    def test_mapq_statistics(self):
        """Test MAPQ mean and median calculation."""
        j1 = Junction("chr1", 100, 200, "+")
        block1 = GenomicInterval("chr1", 50, 100, "+")
        block2 = GenomicInterval("chr1", 200, 250, "+")

        reads = [
            create_read_evidence(
                junctions=[j1], exon_blocks=[block1, block2], mapq=60, nh=1
            ),
            create_read_evidence(
                junctions=[j1], exon_blocks=[block1, block2], mapq=50, nh=1
            ),
            create_read_evidence(
                junctions=[j1], exon_blocks=[block1, block2], mapq=40, nh=1
            ),
        ]

        mock_fn = _make_streaming_mock((reads, {"total_reads": 3}))
        with patch(
            "splice.core.junction_extractor.extract_junction_stats_streaming",
            side_effect=mock_fn,
        ):
            junc_dict, _ = extract_all_junctions(
                ["mock.bam"], ["sample1"], set()
            )

        evidence = junc_dict[j1]
        assert evidence.sample_mapq_mean[0] == 50.0
        # In streaming mode, median is approximated by mean
        assert evidence.sample_mapq_median[0] == 50.0

    def test_nh_weighting(self):
        """Test 1/NH weighting in weighted counts."""
        j1 = Junction("chr1", 100, 200, "+")
        block1 = GenomicInterval("chr1", 50, 100, "+")
        block2 = GenomicInterval("chr1", 200, 250, "+")

        reads = [
            create_read_evidence(
                junctions=[j1], exon_blocks=[block1, block2], nh=1
            ),
            create_read_evidence(
                junctions=[j1], exon_blocks=[block1, block2], nh=2
            ),
        ]

        mock_fn = _make_streaming_mock((reads, {"total_reads": 2}))
        with patch(
            "splice.core.junction_extractor.extract_junction_stats_streaming",
            side_effect=mock_fn,
        ):
            junc_dict, _ = extract_all_junctions(
                ["mock.bam"], ["sample1"], set()
            )

        evidence = junc_dict[j1]
        # Streaming mode: weighted_count = counts / mean_nh = 2 / 1.5 = 1.333...
        # (not 1/1 + 1/2 = 1.5 from per-read mode)
        mean_nh = (1 + 2) / 2.0
        expected_weighted = 2 / mean_nh
        assert evidence.sample_weighted_counts[0] == pytest.approx(expected_weighted)

    def test_annotation_marking(self):
        """Test marking of annotated junctions."""
        j1 = Junction("chr1", 100, 200, "+")
        j2 = Junction("chr1", 300, 400, "+")
        block1 = GenomicInterval("chr1", 50, 100, "+")
        block2 = GenomicInterval("chr1", 200, 250, "+")

        read1 = create_read_evidence(
            junctions=[j1], exon_blocks=[block1, block2]
        )
        read2 = create_read_evidence(
            junctions=[j2], exon_blocks=[block1, block2]
        )

        known_junctions = {j1}

        mock_fn = _make_streaming_mock(([read1, read2], {"total_reads": 2}))
        with patch(
            "splice.core.junction_extractor.extract_junction_stats_streaming",
            side_effect=mock_fn,
        ):
            junc_dict, _ = extract_all_junctions(
                ["mock.bam"],
                ["sample1"],
                known_junctions,
            )

        assert junc_dict[j1].is_annotated is True
        assert junc_dict[j2].is_annotated is False

    def test_co_occurrence_tracking(self):
        """Test co-occurrence tracking for junction pairs."""
        j1 = Junction("chr1", 100, 200, "+")
        j2 = Junction("chr1", 300, 400, "+")
        pair = JunctionPair(j1, j2)
        block1 = GenomicInterval("chr1", 50, 100, "+")
        block2 = GenomicInterval("chr1", 200, 250, "+")
        block3 = GenomicInterval("chr1", 400, 450, "+")

        read = create_read_evidence(
            junctions=[j1, j2],
            junction_pairs=[pair],
            exon_blocks=[block1, block2, block3],
        )

        mock_fn = _make_streaming_mock(([read], {"total_reads": 1}))
        with patch(
            "splice.core.junction_extractor.extract_junction_stats_streaming",
            side_effect=mock_fn,
        ):
            _, cooc_dict = extract_all_junctions(
                ["mock.bam"],
                ["sample1"],
                set(),
            )

        assert pair in cooc_dict
        assert cooc_dict[pair].sample_counts[0] == 1

    def test_cross_sample_recurrence(self):
        """Test cross-sample recurrence calculation."""
        j1 = Junction("chr1", 100, 200, "+")
        j2 = Junction("chr1", 300, 400, "+")
        block1 = GenomicInterval("chr1", 50, 100, "+")
        block2 = GenomicInterval("chr1", 200, 250, "+")

        read1 = create_read_evidence(
            junctions=[j1], exon_blocks=[block1, block2]
        )
        read2 = create_read_evidence(
            junctions=[j2], exon_blocks=[block1, block2]
        )

        def reads_by_bam(bam_path):
            if "sample1.bam" in bam_path:
                return ([read1], {"total_reads": 1})
            elif "sample2.bam" in bam_path:
                return ([read1, read2], {"total_reads": 2})
            else:
                return ([], {"total_reads": 0})

        mock_fn = _make_streaming_mock(reads_by_bam)
        with patch(
            "splice.core.junction_extractor.extract_junction_stats_streaming",
            side_effect=mock_fn,
        ):
            junc_dict, _ = extract_all_junctions(
                ["sample1.bam", "sample2.bam", "sample3.bam"],
                ["sample1", "sample2", "sample3"],
                set(),
            )

        # j1 appears in 2/3 samples
        assert junc_dict[j1].n_samples_detected == 2
        assert junc_dict[j1].cross_sample_recurrence == pytest.approx(2.0 / 3.0)

        # j2 appears in 1/3 samples
        assert junc_dict[j2].n_samples_detected == 1
        assert junc_dict[j2].cross_sample_recurrence == pytest.approx(1.0 / 3.0)

    def test_max_anchor_tracking(self):
        """Test tracking of maximum anchor length."""
        j1 = Junction("chr1", 100, 200, "+")
        block1 = GenomicInterval("chr1", 50, 120, "+")  # length 70, ends at junc.start
        block2 = GenomicInterval("chr1", 200, 250, "+")  # length 50, starts at junc.end

        read = create_read_evidence(
            junctions=[j1], exon_blocks=[block1, block2]
        )

        mock_fn = _make_streaming_mock(([read], {"total_reads": 1}))
        with patch(
            "splice.core.junction_extractor.extract_junction_stats_streaming",
            side_effect=mock_fn,
        ):
            junc_dict, _ = extract_all_junctions(
                ["mock.bam"], ["sample1"], set()
            )

        # Block1 ends at 120 != junc.start(100), block2 starts at 200 == junc.end(200)
        # So only block2 (length 50) matches. But block1 end(120) != junc.start(100).
        # In the real streaming reader, anchors come from CIGAR blocks adjacent to introns.
        # Our mock matches blocks by coordinate: block.end==junc.start or block.start==junc.end
        assert junc_dict[j1].max_anchor == 50

    def test_motif_classification(self):
        """Test motif classification when genome is provided."""
        j1 = Junction("chr1", 100, 200, "+")
        block1 = GenomicInterval("chr1", 50, 100, "+")
        block2 = GenomicInterval("chr1", 200, 250, "+")

        read = create_read_evidence(
            junctions=[j1], exon_blocks=[block1, block2]
        )

        mock_fn = _make_streaming_mock(([read], {"total_reads": 1}))
        with patch(
            "splice.core.junction_extractor.extract_junction_stats_streaming",
            side_effect=mock_fn,
        ):
            with patch(
                "splice.core.junction_extractor.extract_motif_from_genome"
            ) as mock_motif:
                mock_motif.return_value = ("GT", "AG", "GT/AG")
                junc_dict, _ = extract_all_junctions(
                    ["mock.bam"],
                    ["sample1"],
                    set(),
                    genome_fasta_path="genome.fasta",
                )

        assert junc_dict[j1].motif == "GT/AG"
        assert junc_dict[j1].motif_score == 1.0

    def test_motif_classification_failure(self):
        """Test handling of motif classification failure."""
        j1 = Junction("chr1", 100, 200, "+")
        block1 = GenomicInterval("chr1", 50, 100, "+")
        block2 = GenomicInterval("chr1", 200, 250, "+")

        read = create_read_evidence(
            junctions=[j1], exon_blocks=[block1, block2]
        )

        mock_fn = _make_streaming_mock(([read], {"total_reads": 1}))
        with patch(
            "splice.core.junction_extractor.extract_junction_stats_streaming",
            side_effect=mock_fn,
        ):
            with patch(
                "splice.core.junction_extractor.extract_motif_from_genome",
                side_effect=Exception("File not found"),
            ):
                junc_dict, _ = extract_all_junctions(
                    ["mock.bam"],
                    ["sample1"],
                    set(),
                    genome_fasta_path="genome.fasta",
                )

        assert junc_dict[j1].motif == ""
        assert junc_dict[j1].motif_score == 0.0

    def test_empty_bam(self):
        """Test handling of empty BAM file."""
        mock_fn = _make_streaming_mock(([], {"total_reads": 0}))
        with patch(
            "splice.core.junction_extractor.extract_junction_stats_streaming",
            side_effect=mock_fn,
        ):
            junc_dict, cooc_dict = extract_all_junctions(
                ["mock.bam"],
                ["sample1"],
                set(),
            )

        assert len(junc_dict) == 0
        assert len(cooc_dict) == 0

    def test_multiple_junctions_per_read(self):
        """Test handling of reads with multiple junctions."""
        j1 = Junction("chr1", 100, 200, "+")
        j2 = Junction("chr1", 300, 400, "+")
        j3 = Junction("chr1", 500, 600, "+")
        pair1 = JunctionPair(j1, j2)
        pair2 = JunctionPair(j1, j3)
        pair3 = JunctionPair(j2, j3)

        block1 = GenomicInterval("chr1", 50, 100, "+")
        block2 = GenomicInterval("chr1", 200, 250, "+")
        block3 = GenomicInterval("chr1", 300, 350, "+")
        block4 = GenomicInterval("chr1", 400, 450, "+")

        read = create_read_evidence(
            junctions=[j1, j2, j3],
            junction_pairs=[pair1, pair2, pair3],
            exon_blocks=[block1, block2, block3, block4],
        )

        mock_fn = _make_streaming_mock(([read], {"total_reads": 1}))
        with patch(
            "splice.core.junction_extractor.extract_junction_stats_streaming",
            side_effect=mock_fn,
        ):
            junc_dict, cooc_dict = extract_all_junctions(
                ["mock.bam"],
                ["sample1"],
                set(),
            )

        assert len(junc_dict) == 3
        assert all(junc in junc_dict for junc in [j1, j2, j3])

        assert len(cooc_dict) == 3
        assert all(pair in cooc_dict for pair in [pair1, pair2, pair3])

    def test_nh_mean_calculation(self):
        """Test mean NH calculation."""
        j1 = Junction("chr1", 100, 200, "+")
        block1 = GenomicInterval("chr1", 50, 100, "+")
        block2 = GenomicInterval("chr1", 200, 250, "+")

        reads = [
            create_read_evidence(
                junctions=[j1], exon_blocks=[block1, block2], nh=1
            ),
            create_read_evidence(
                junctions=[j1], exon_blocks=[block1, block2], nh=2
            ),
            create_read_evidence(
                junctions=[j1], exon_blocks=[block1, block2], nh=3
            ),
        ]

        mock_fn = _make_streaming_mock((reads, {"total_reads": 3}))
        with patch(
            "splice.core.junction_extractor.extract_junction_stats_streaming",
            side_effect=mock_fn,
        ):
            junc_dict, _ = extract_all_junctions(
                ["mock.bam"],
                ["sample1"],
                set(),
            )

        # Mean NH = (1 + 2 + 3) / 3 = 2.0
        assert junc_dict[j1].sample_nh_distribution[0] == 2.0
