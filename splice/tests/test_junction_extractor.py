"""
Tests for core/junction_extractor.py

Covers: JunctionEvidence, CooccurrenceEvidence, and extract_all_junctions.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from splicekit.core.junction_extractor import (
    CooccurrenceEvidence,
    JunctionEvidence,
    extract_all_junctions,
)
from splicekit.io.bam_utils import ReadEvidence
from splicekit.utils.genomic import GenomicInterval, Junction, JunctionPair


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

        with patch("splicekit.core.junction_extractor.extract_evidence_from_bam") as mock:
            mock.return_value = ([read_evidence], {"total_reads": 1})
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

        def mock_extract(bam_path, **kwargs):
            if "sample1.bam" in bam_path:
                return ([read1], {"total_reads": 1})
            else:
                return ([read2], {"total_reads": 1})

        with patch(
            "splicekit.core.junction_extractor.extract_evidence_from_bam", side_effect=mock_extract
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

        with patch("splicekit.core.junction_extractor.extract_evidence_from_bam") as mock:
            mock.return_value = (reads, {"total_reads": 3})
            junc_dict, _ = extract_all_junctions(
                ["mock.bam"], ["sample1"], set()
            )

        evidence = junc_dict[j1]
        assert evidence.sample_mapq_mean[0] == 50.0
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

        with patch("splicekit.core.junction_extractor.extract_evidence_from_bam") as mock:
            mock.return_value = (reads, {"total_reads": 2})
            junc_dict, _ = extract_all_junctions(
                ["mock.bam"], ["sample1"], set()
            )

        evidence = junc_dict[j1]
        # 1/1 + 1/2 = 1.5
        assert evidence.sample_weighted_counts[0] == 1.5

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

        known_junctions = {j1}  # j1 is annotated, j2 is not

        with patch("splicekit.core.junction_extractor.extract_evidence_from_bam") as mock:
            mock.return_value = ([read1, read2], {"total_reads": 2})
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

        with patch("splicekit.core.junction_extractor.extract_evidence_from_bam") as mock:
            mock.return_value = ([read], {"total_reads": 1})
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

        def mock_extract(bam_path, **kwargs):
            if "sample1.bam" in bam_path:
                return ([read1], {"total_reads": 1})
            elif "sample2.bam" in bam_path:
                return ([read1, read2], {"total_reads": 2})
            else:
                return ([], {"total_reads": 0})

        with patch(
            "splicekit.core.junction_extractor.extract_evidence_from_bam", side_effect=mock_extract
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
        block1 = GenomicInterval("chr1", 50, 120, "+")  # length 70
        block2 = GenomicInterval("chr1", 200, 250, "+")  # length 50

        read = create_read_evidence(
            junctions=[j1], exon_blocks=[block1, block2]
        )

        with patch("splicekit.core.junction_extractor.extract_evidence_from_bam") as mock:
            mock.return_value = ([read], {"total_reads": 1})
            junc_dict, _ = extract_all_junctions(
                ["mock.bam"], ["sample1"], set()
            )

        # max_anchor should be max(70, 50) = 70
        assert junc_dict[j1].max_anchor == 70

    def test_motif_classification(self):
        """Test motif classification when genome is provided."""
        j1 = Junction("chr1", 100, 200, "+")
        block1 = GenomicInterval("chr1", 50, 100, "+")
        block2 = GenomicInterval("chr1", 200, 250, "+")

        read = create_read_evidence(
            junctions=[j1], exon_blocks=[block1, block2]
        )

        with patch("splicekit.core.junction_extractor.extract_evidence_from_bam") as mock:
            mock.return_value = ([read], {"total_reads": 1})

            with patch(
                "splicekit.core.junction_extractor.extract_motif_from_genome"
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

        with patch("splicekit.core.junction_extractor.extract_evidence_from_bam") as mock:
            mock.return_value = ([read], {"total_reads": 1})

            with patch(
                "splicekit.core.junction_extractor.extract_motif_from_genome",
                side_effect=Exception("File not found"),
            ):
                junc_dict, _ = extract_all_junctions(
                    ["mock.bam"],
                    ["sample1"],
                    set(),
                    genome_fasta_path="genome.fasta",
                )

        # Should gracefully handle failure
        assert junc_dict[j1].motif == ""
        assert junc_dict[j1].motif_score == 0.0

    def test_empty_bam(self):
        """Test handling of empty BAM file."""
        with patch("splicekit.core.junction_extractor.extract_evidence_from_bam") as mock:
            mock.return_value = ([], {"total_reads": 0})
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

        with patch("splicekit.core.junction_extractor.extract_evidence_from_bam") as mock:
            mock.return_value = ([read], {"total_reads": 1})
            junc_dict, cooc_dict = extract_all_junctions(
                ["mock.bam"],
                ["sample1"],
                set(),
            )

        # All 3 junctions should be extracted
        assert len(junc_dict) == 3
        assert all(junc in junc_dict for junc in [j1, j2, j3])

        # All 3 pairs should be tracked
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

        with patch("splicekit.core.junction_extractor.extract_evidence_from_bam") as mock:
            mock.return_value = (reads, {"total_reads": 3})
            junc_dict, _ = extract_all_junctions(
                ["mock.bam"],
                ["sample1"],
                set(),
            )

        # Mean NH = (1 + 2 + 3) / 3 = 2.0
        assert junc_dict[j1].sample_nh_distribution[0] == 2.0
