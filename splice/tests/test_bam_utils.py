"""
Tests for io/bam_utils.py

Covers: ReadEvidence, extract_evidence_from_read, extract_evidence_from_bam,
and count_exon_body_reads, using mocked pysam.
"""

from unittest.mock import MagicMock, patch

import pytest

from splice.io.bam_utils import (
    ReadEvidence,
    count_exon_body_reads,
    extract_evidence_from_bam,
    extract_evidence_from_read,
)
from splice.utils.genomic import GenomicInterval, Junction


# ---------------------------------------------------------------------------
# Test fixtures: Mock pysam AlignedSegment objects
# ---------------------------------------------------------------------------


def create_mock_read(
    query_name="read1",
    query_sequence="A" * 20,
    flag=0,
    reference_name="chr1",
    reference_start=100,
    mapping_quality=60,
    cigartuples=None,
    is_proper_pair=True,
    is_unmapped=False,
    is_secondary=False,
    is_duplicate=False,
    is_qcfail=False,
    is_reverse=False,
    blocks=None,
    nh_tag=1,
    as_tag=None,
):
    """Create a mock pysam.AlignedSegment with specified properties."""
    read = MagicMock()
    read.query_name = query_name
    read.query_sequence = query_sequence
    read.flag = flag
    read.reference_name = reference_name
    read.reference_start = reference_start
    read.mapping_quality = mapping_quality
    read.cigartuples = cigartuples or []
    read.is_proper_pair = is_proper_pair
    read.is_unmapped = is_unmapped
    read.is_secondary = is_secondary
    read.is_duplicate = is_duplicate
    read.is_qcfail = is_qcfail
    read.is_reverse = is_reverse

    # Default blocks based on CIGAR if not provided
    if blocks is None:
        blocks = _infer_blocks_from_cigar(reference_start, cigartuples)

    read.get_blocks.return_value = blocks

    # Mock get_tag for NH and AS
    def get_tag(tag_name):
        if tag_name == "NH":
            return nh_tag
        elif tag_name == "AS":
            if as_tag is None:
                raise KeyError(f"Tag {tag_name} not found")
            return as_tag
        raise KeyError(f"Tag {tag_name} not found")

    read.get_tag.side_effect = get_tag

    return read


def _infer_blocks_from_cigar(reference_start, cigartuples):
    """Infer aligned blocks from CIGAR string."""
    if not cigartuples:
        return []
    blocks = []
    pos = reference_start
    for op, length in cigartuples:
        if op in [0, 7, 8]:  # M, =, X
            blocks.append((pos, pos + length))
            pos += length
        elif op in [2, 3]:  # D, N
            pos += length
    return blocks


# ---------------------------------------------------------------------------
# ReadEvidence dataclass tests
# ---------------------------------------------------------------------------


class TestReadEvidence:
    def test_construction(self):
        """Test basic ReadEvidence construction."""
        j1 = Junction("chr1", 100, 110, "+")
        j2 = Junction("chr1", 120, 130, "+")
        jp = [(j1, j2)]
        eb = [GenomicInterval("chr1", 100, 110, "+")]

        evidence = ReadEvidence(
            junctions=[j1, j2],
            junction_pairs=jp,
            exon_blocks=eb,
            mapq=60,
            nh=1,
            is_proper_pair=True,
            alignment_score=100
        )

        assert len(evidence.junctions) == 2
        assert len(evidence.junction_pairs) == 1
        assert evidence.mapq == 60
        assert evidence.nh == 1

    def test_frozen_immutability(self):
        """Test that ReadEvidence is immutable."""
        evidence = ReadEvidence(
            junctions=[],
            junction_pairs=[],
            exon_blocks=[],
            mapq=60,
            nh=1,
            is_proper_pair=True,
            alignment_score=None
        )

        with pytest.raises((AttributeError, TypeError)):
            evidence.mapq = 30  # type: ignore[misc]


# ---------------------------------------------------------------------------
# extract_evidence_from_read tests
# ---------------------------------------------------------------------------


class TestExtractEvidenceFromRead:
    def test_read_with_junction(self):
        """Test extracting evidence from a read with a junction."""
        # Read with junction: 10M10N10M (blocks at [100-110], [120-130])
        read = create_mock_read(
            reference_name="chr1",
            reference_start=100,
            mapping_quality=60,
            cigartuples=[(0, 10), (3, 10), (0, 10)],
            blocks=[(100, 110), (120, 130)],
            nh_tag=1,
            as_tag=100
        )

        evidence = extract_evidence_from_read(read, min_anchor=6)

        assert evidence is not None
        assert len(evidence.junctions) == 1
        assert evidence.junctions[0].start == 110
        assert evidence.junctions[0].end == 120
        assert evidence.mapq == 60
        assert evidence.nh == 1
        assert evidence.alignment_score == 100

    def test_read_without_junction(self):
        """Test extracting evidence from a read without junctions."""
        read = create_mock_read(
            mapping_quality=60,
            cigartuples=[(0, 10)],
            blocks=[(50, 60)]
        )

        evidence = extract_evidence_from_read(read, min_anchor=6)

        assert evidence is not None
        assert len(evidence.junctions) == 0
        assert len(evidence.junction_pairs) == 0
        assert evidence.mapq == 60

    def test_unmapped_read_filtered(self):
        """Test that unmapped reads are filtered out."""
        read = create_mock_read(is_unmapped=True)
        evidence = extract_evidence_from_read(read, min_anchor=6)
        assert evidence is None

    def test_secondary_read_filtered(self):
        """Test that secondary reads are filtered out."""
        read = create_mock_read(is_secondary=True)
        evidence = extract_evidence_from_read(read, min_anchor=6)
        assert evidence is None

    def test_duplicate_read_filtered(self):
        """Test that duplicate reads are filtered out."""
        read = create_mock_read(is_duplicate=True)
        evidence = extract_evidence_from_read(read, min_anchor=6)
        assert evidence is None

    def test_qcfail_read_filtered(self):
        """Test that QC-failed reads are filtered out."""
        read = create_mock_read(is_qcfail=True)
        evidence = extract_evidence_from_read(read, min_anchor=6)
        assert evidence is None

    def test_multi_junction_read(self):
        """Test extracting evidence from a read with multiple junctions."""
        # CIGAR: 10M5N10M5N10M5N10M -> 3 junctions
        read = create_mock_read(
            reference_start=100,
            cigartuples=[(0, 10), (3, 5), (0, 10), (3, 5), (0, 10), (3, 5), (0, 10)],
            blocks=[(100, 110), (115, 125), (130, 140), (145, 155)]
        )

        evidence = extract_evidence_from_read(read, min_anchor=6)

        assert evidence is not None
        # 3 junctions (between blocks)
        assert len(evidence.junctions) == 3
        # C(3, 2) = 3 junction pairs
        assert len(evidence.junction_pairs) == 3

    def test_exon_blocks_extracted(self):
        """Test that exon blocks are correctly extracted."""
        read = create_mock_read(
            reference_name="chr1",
            reference_start=100,
            cigartuples=[(0, 10), (3, 10), (0, 10)],
            blocks=[(100, 110), (120, 130)]
        )

        evidence = extract_evidence_from_read(read, min_anchor=6)

        assert evidence is not None
        assert len(evidence.exon_blocks) == 2
        assert evidence.exon_blocks[0].start == 100
        assert evidence.exon_blocks[0].end == 110
        assert evidence.exon_blocks[1].start == 120
        assert evidence.exon_blocks[1].end == 130

    def test_anchor_length_filtering(self):
        """Test that junctions with short anchors are filtered."""
        # CIGAR: 3M10N10M (short left anchor, should be filtered with min_anchor=6)
        read = create_mock_read(
            reference_start=100,
            cigartuples=[(0, 3), (3, 10), (0, 10)],
            blocks=[(100, 103), (113, 123)]
        )

        evidence = extract_evidence_from_read(read, min_anchor=6)

        assert evidence is not None
        assert len(evidence.junctions) == 0

    def test_nh_tag_extraction(self):
        """Test that NH tag is correctly extracted."""
        read = create_mock_read(nh_tag=3)
        evidence = extract_evidence_from_read(read, min_anchor=6)
        assert evidence is not None
        assert evidence.nh == 3

    def test_as_tag_extraction(self):
        """Test that AS tag is correctly extracted."""
        read = create_mock_read(as_tag=100)
        evidence = extract_evidence_from_read(read, min_anchor=6)
        assert evidence is not None
        assert evidence.alignment_score == 100

    def test_missing_as_tag(self):
        """Test that missing AS tag returns None."""
        read = create_mock_read(as_tag=None)
        evidence = extract_evidence_from_read(read, min_anchor=6)
        assert evidence is not None
        assert evidence.alignment_score is None

    def test_reverse_strand_read(self):
        """Test that reverse strand reads are handled correctly."""
        read = create_mock_read(
            is_reverse=True,
            cigartuples=[(0, 10), (3, 10), (0, 10)],
            blocks=[(100, 110), (120, 130)]
        )

        evidence = extract_evidence_from_read(read, min_anchor=6)

        assert evidence is not None
        assert evidence.junctions[0].strand == "-"
        assert evidence.exon_blocks[0].strand == "-"


# ---------------------------------------------------------------------------
# extract_evidence_from_bam tests
# ---------------------------------------------------------------------------


class TestExtractEvidenceFromBAM:
    def test_extract_all_reads(self):
        """Test extracting evidence from all reads in BAM."""
        # Create mock BAM with 3 reads: 2 mapped (MAPQ 60, 10), 1 unmapped
        reads = [
            create_mock_read(mapping_quality=60),
            create_mock_read(mapping_quality=10),
            create_mock_read(is_unmapped=True),
        ]

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = reads
            evidence_list, stats = extract_evidence_from_bam("mock.bam")

        assert len(evidence_list) == 2
        assert stats["total_reads"] == 3
        assert stats["mapped_reads"] == 2

    def test_mapq_filtering(self):
        """Test MAPQ filtering in BAM extraction."""
        reads = [
            create_mock_read(mapping_quality=60),
            create_mock_read(mapping_quality=10),
        ]

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = reads
            evidence_list, stats = extract_evidence_from_bam("mock.bam", min_mapq=30)

        # Only the high-quality read (MAPQ=60) should pass the MAPQ filter
        assert len(evidence_list) == 1
        assert stats["mapped_reads"] == 2

    def test_junction_read_counting(self):
        """Test that junction reads are counted correctly."""
        # Create a read with junctions
        read = create_mock_read(
            cigartuples=[(0, 10), (3, 5), (0, 10)],
            blocks=[(100, 110), (115, 125)]
        )

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = [read]
            evidence_list, stats = extract_evidence_from_bam("mock.bam")

        assert stats["junction_reads"] == 1
        assert len(evidence_list[0].junctions) == 1

    def test_multi_mapped_counting(self):
        """Test that multi-mapped reads are counted correctly."""
        read = create_mock_read(nh_tag=3)

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = [read]
            evidence_list, stats = extract_evidence_from_bam("mock.bam")

        assert stats["multi_mapped_reads"] == 1
        assert evidence_list[0].nh == 3

    def test_mapq_statistics(self):
        """Test MAPQ statistics calculation."""
        reads = [
            create_mock_read(mapping_quality=60),
            create_mock_read(mapping_quality=10),
        ]

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = reads
            evidence_list, stats = extract_evidence_from_bam("mock.bam")

        # Two reads with MAPQ 60 and 10 -> mean=35, median=35
        assert stats["mean_mapq"] == 35.0
        assert stats["median_mapq"] == 35.0

    def test_region_filtering(self):
        """Test extracting evidence from a specific region."""
        read = create_mock_read()

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = [read]
            evidence_list, stats = extract_evidence_from_bam(
                "mock.bam", region="chr1:100-200"
            )

        # Verify fetch was called with region parameter
        mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.assert_called_once_with(
            region="chr1:100-200"
        )
        assert len(evidence_list) == 1

    def test_empty_bam(self):
        """Test extracting from an empty BAM."""
        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = []
            evidence_list, stats = extract_evidence_from_bam("mock.bam")

        assert len(evidence_list) == 0
        assert stats["total_reads"] == 0
        assert stats["mapped_reads"] == 0
        assert stats["mean_mapq"] == 0.0


# ---------------------------------------------------------------------------
# count_exon_body_reads tests
# ---------------------------------------------------------------------------


class TestCountExonBodyReads:
    def test_count_reads_within_exon(self):
        """Test counting reads completely within exon."""
        # Read 1: completely within [100, 150]
        read1 = create_mock_read(
            reference_start=110,
            mapping_quality=60,
            blocks=[(110, 150)],
            nh_tag=1
        )
        # Read 2: completely within [100, 150], multi-mapped
        read2 = create_mock_read(
            reference_start=115,
            mapping_quality=60,
            blocks=[(115, 135)],
            nh_tag=2
        )

        exon = GenomicInterval("chr1", 100, 150, "+")

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = [
                read1, read2
            ]
            raw_count, weighted_count = count_exon_body_reads("mock.bam", exon)

        assert raw_count == 2
        assert weighted_count == 1.5  # 1.0 + 0.5

    def test_exclude_reads_extending_before_exon(self):
        """Test that reads extending before exon start are excluded."""
        # Read extending before exon start
        read = create_mock_read(
            reference_start=90,
            mapping_quality=60,
            blocks=[(90, 130)]  # Extends before exon start [100, 150]
        )

        exon = GenomicInterval("chr1", 100, 150, "+")

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = [read]
            raw_count, weighted_count = count_exon_body_reads("mock.bam", exon)

        assert raw_count == 0

    def test_exclude_reads_extending_after_exon(self):
        """Test that reads extending after exon end are excluded."""
        # Read extending after exon end
        read = create_mock_read(
            reference_start=140,
            mapping_quality=60,
            blocks=[(140, 180)]  # Extends after exon end [100, 150]
        )

        exon = GenomicInterval("chr1", 100, 150, "+")

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = [read]
            raw_count, weighted_count = count_exon_body_reads("mock.bam", exon)

        assert raw_count == 0

    def test_weighted_count_nh_tag(self):
        """Test that multi-mapped reads are weighted by 1/NH."""
        read1 = create_mock_read(blocks=[(110, 150)], nh_tag=1)
        read2 = create_mock_read(blocks=[(115, 135)], nh_tag=2)

        exon = GenomicInterval("chr1", 100, 150, "+")

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = [
                read1, read2
            ]
            raw_count, weighted_count = count_exon_body_reads("mock.bam", exon)

        assert raw_count == 2
        assert weighted_count == 1.5

    def test_mapq_filtering_exon(self):
        """Test MAPQ filtering for exon body counting."""
        read1 = create_mock_read(
            mapping_quality=60,
            blocks=[(110, 150)],
            nh_tag=1
        )
        read2 = create_mock_read(
            mapping_quality=40,
            blocks=[(115, 135)],
            nh_tag=1
        )

        exon = GenomicInterval("chr1", 100, 150, "+")

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = [
                read1, read2
            ]
            # With min_mapq=50, only read1 (MAPQ=60) should pass
            raw_count, weighted_count = count_exon_body_reads(
                "mock.bam", exon, min_mapq=50
            )

        assert raw_count == 1
        assert weighted_count == 1.0

    def test_empty_exon(self):
        """Test counting reads in an exon with no reads."""
        exon = GenomicInterval("chr1", 5000, 5100, "+")

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = []
            raw_count, weighted_count = count_exon_body_reads("mock.bam", exon)

        assert raw_count == 0
        assert weighted_count == 0.0

    def test_filtered_read_not_counted(self):
        """Test that filtered reads are not counted."""
        read_unmapped = create_mock_read(is_unmapped=True)
        read_secondary = create_mock_read(is_secondary=True)
        read_duplicate = create_mock_read(is_duplicate=True)

        exon = GenomicInterval("chr1", 100, 150, "+")

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = [
                read_unmapped, read_secondary, read_duplicate
            ]
            raw_count, weighted_count = count_exon_body_reads("mock.bam", exon)

        assert raw_count == 0
        assert weighted_count == 0.0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_pipeline_junction_detection(self):
        """Test complete pipeline: extract from BAM and detect junctions."""
        # Create a read with a junction
        read = create_mock_read(
            cigartuples=[(0, 10), (3, 10), (0, 10)],
            blocks=[(100, 110), (120, 130)]
        )

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = [read]
            evidence_list, stats = extract_evidence_from_bam("mock.bam")

        assert len(evidence_list) == 1
        assert len(evidence_list[0].junctions) == 1
        assert stats["junction_reads"] == 1

    def test_multi_junction_pairs(self):
        """Test that junction pairs are correctly generated."""
        # Create a read with 3 junctions
        read = create_mock_read(
            cigartuples=[(0, 10), (3, 5), (0, 10), (3, 5), (0, 10), (3, 5), (0, 10)],
            blocks=[(100, 110), (115, 125), (130, 140), (145, 155)]
        )

        with patch("splice.io.bam_utils.pysam") as mock_pysam:
            mock_pysam.AlignmentFile.return_value.__enter__.return_value.fetch.return_value = [read]
            evidence_list, stats = extract_evidence_from_bam("mock.bam")

        assert len(evidence_list) == 1
        evidence = evidence_list[0]
        # 3 junctions should give 3 pairs (C(3,2) = 3)
        assert len(evidence.junction_pairs) == 3
