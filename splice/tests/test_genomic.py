"""
Tests for utils/genomic.py

Covers: GenomicInterval, Junction, JunctionPair, overlaps,
merge_intervals, subtract_interval.
"""

import pytest
from splicekit.utils.genomic import (
    GenomicInterval,
    Junction,
    JunctionPair,
    merge_intervals,
    overlaps,
    subtract_interval,
)


# ---------------------------------------------------------------------------
# GenomicInterval
# ---------------------------------------------------------------------------


class TestGenomicInterval:
    def test_length_basic(self):
        iv = GenomicInterval("chr1", 100, 200, "+")
        assert iv.length == 100

    def test_length_single_base(self):
        iv = GenomicInterval("chr1", 50, 51, "+")
        assert iv.length == 1

    def test_length_zero(self):
        iv = GenomicInterval("chr1", 75, 75, ".")
        assert iv.length == 0

    def test_frozen_immutability(self):
        iv = GenomicInterval("chr1", 100, 200, "+")
        with pytest.raises((AttributeError, TypeError)):
            iv.start = 0  # type: ignore[misc]

    def test_hashable(self):
        iv = GenomicInterval("chr1", 100, 200, "+")
        s = {iv, iv}
        assert len(s) == 1

    def test_equality(self):
        iv1 = GenomicInterval("chr1", 100, 200, "+")
        iv2 = GenomicInterval("chr1", 100, 200, "+")
        assert iv1 == iv2

    def test_inequality_chrom(self):
        iv1 = GenomicInterval("chr1", 100, 200, "+")
        iv2 = GenomicInterval("chr2", 100, 200, "+")
        assert iv1 != iv2

    def test_inequality_strand(self):
        iv1 = GenomicInterval("chr1", 100, 200, "+")
        iv2 = GenomicInterval("chr1", 100, 200, "-")
        assert iv1 != iv2

    def test_ordering_by_chrom(self):
        iv1 = GenomicInterval("chr1", 100, 200, "+")
        iv2 = GenomicInterval("chr2", 50, 150, "+")
        assert iv1 < iv2

    def test_ordering_by_start(self):
        iv1 = GenomicInterval("chr1", 100, 200, "+")
        iv2 = GenomicInterval("chr1", 200, 300, "+")
        assert iv1 < iv2

    def test_usable_in_sorted(self):
        ivs = [
            GenomicInterval("chr1", 300, 400, "+"),
            GenomicInterval("chr1", 100, 200, "+"),
            GenomicInterval("chr1", 200, 300, "+"),
        ]
        result = sorted(ivs)
        assert result[0].start == 100
        assert result[1].start == 200
        assert result[2].start == 300


# ---------------------------------------------------------------------------
# Junction
# ---------------------------------------------------------------------------


class TestJunction:
    def test_donor_plus_strand(self):
        junc = Junction("chr1", 1000, 2000, "+")
        assert junc.donor == 1000

    def test_acceptor_plus_strand(self):
        junc = Junction("chr1", 1000, 2000, "+")
        assert junc.acceptor == 1999

    def test_donor_minus_strand(self):
        junc = Junction("chr1", 1000, 2000, "-")
        assert junc.donor == 1999

    def test_acceptor_minus_strand(self):
        junc = Junction("chr1", 1000, 2000, "-")
        assert junc.acceptor == 1000

    def test_donor_unstranded(self):
        junc = Junction("chr1", 1000, 2000, ".")
        assert junc.donor == 1000

    def test_acceptor_unstranded(self):
        junc = Junction("chr1", 1000, 2000, ".")
        assert junc.acceptor == 1999

    def test_to_string(self):
        junc = Junction("chr1", 1000, 2000, "+")
        assert junc.to_string() == "chr1:1000-2000:+"

    def test_to_string_minus(self):
        junc = Junction("chrX", 5000, 6000, "-")
        assert junc.to_string() == "chrX:5000-6000:-"

    def test_to_string_unstranded(self):
        junc = Junction("chr2", 0, 100, ".")
        assert junc.to_string() == "chr2:0-100:."

    def test_frozen_immutability(self):
        junc = Junction("chr1", 1000, 2000, "+")
        with pytest.raises((AttributeError, TypeError)):
            junc.start = 0  # type: ignore[misc]

    def test_hashable(self):
        junc = Junction("chr1", 1000, 2000, "+")
        s = {junc, junc}
        assert len(s) == 1

    def test_equality(self):
        j1 = Junction("chr1", 1000, 2000, "+")
        j2 = Junction("chr1", 1000, 2000, "+")
        assert j1 == j2

    def test_ordering(self):
        j1 = Junction("chr1", 1000, 2000, "+")
        j2 = Junction("chr1", 1500, 2500, "+")
        assert j1 < j2

    # shares_splice_site

    def test_shares_donor_plus(self):
        j1 = Junction("chr1", 1000, 2000, "+")
        j2 = Junction("chr1", 1000, 3000, "+")  # same donor
        assert j1.shares_splice_site(j2)

    def test_shares_acceptor_plus(self):
        j1 = Junction("chr1", 500, 2001, "+")
        j2 = Junction("chr1", 1000, 2001, "+")  # same acceptor (end-1 = 2000)
        assert j1.shares_splice_site(j2)

    def test_shares_donor_minus(self):
        # On '-', donor = end - 1
        j1 = Junction("chr1", 1000, 2000, "-")  # donor = 1999
        j2 = Junction("chr1", 500, 2000, "-")   # donor = 1999
        assert j1.shares_splice_site(j2)

    def test_shares_acceptor_minus(self):
        # On '-', acceptor = start
        j1 = Junction("chr1", 1000, 2000, "-")  # acceptor = 1000
        j2 = Junction("chr1", 1000, 3000, "-")  # acceptor = 1000
        assert j1.shares_splice_site(j2)

    def test_no_shared_splice_site(self):
        j1 = Junction("chr1", 1000, 2000, "+")
        j2 = Junction("chr1", 3000, 4000, "+")
        assert not j1.shares_splice_site(j2)

    def test_no_shared_splice_site_diff_chrom(self):
        j1 = Junction("chr1", 1000, 2000, "+")
        j2 = Junction("chr2", 1000, 2000, "+")
        assert not j1.shares_splice_site(j2)

    def test_no_shared_splice_site_diff_strand(self):
        j1 = Junction("chr1", 1000, 2000, "+")
        j2 = Junction("chr1", 1000, 2000, "-")
        assert not j1.shares_splice_site(j2)

    def test_shares_splice_site_with_self(self):
        j = Junction("chr1", 1000, 2000, "+")
        assert j.shares_splice_site(j)


# ---------------------------------------------------------------------------
# JunctionPair
# ---------------------------------------------------------------------------


class TestJunctionPair:
    def test_construction(self):
        j1 = Junction("chr1", 1000, 2000, "+")
        j2 = Junction("chr1", 3000, 4000, "+")
        pair = JunctionPair(j1, j2)
        assert pair.junction1 == j1
        assert pair.junction2 == j2

    def test_frozen(self):
        j1 = Junction("chr1", 1000, 2000, "+")
        j2 = Junction("chr1", 3000, 4000, "+")
        pair = JunctionPair(j1, j2)
        with pytest.raises((AttributeError, TypeError)):
            pair.junction1 = j2  # type: ignore[misc]

    def test_hashable(self):
        j1 = Junction("chr1", 1000, 2000, "+")
        j2 = Junction("chr1", 3000, 4000, "+")
        pair = JunctionPair(j1, j2)
        s = {pair}
        assert len(s) == 1

    def test_equality(self):
        j1 = Junction("chr1", 1000, 2000, "+")
        j2 = Junction("chr1", 3000, 4000, "+")
        p1 = JunctionPair(j1, j2)
        p2 = JunctionPair(j1, j2)
        assert p1 == p2


# ---------------------------------------------------------------------------
# overlaps
# ---------------------------------------------------------------------------


class TestOverlaps:
    def test_overlapping_same_strand(self):
        a = GenomicInterval("chr1", 100, 200, "+")
        b = GenomicInterval("chr1", 150, 250, "+")
        assert overlaps(a, b)

    def test_overlapping_symmetric(self):
        a = GenomicInterval("chr1", 100, 200, "+")
        b = GenomicInterval("chr1", 150, 250, "+")
        assert overlaps(b, a)

    def test_adjacent_no_overlap(self):
        # [100, 200) and [200, 300) are adjacent but do not share a base
        a = GenomicInterval("chr1", 100, 200, "+")
        b = GenomicInterval("chr1", 200, 300, "+")
        assert not overlaps(a, b)

    def test_no_overlap_gap(self):
        a = GenomicInterval("chr1", 100, 200, "+")
        b = GenomicInterval("chr1", 300, 400, "+")
        assert not overlaps(a, b)

    def test_containment(self):
        a = GenomicInterval("chr1", 100, 500, "+")
        b = GenomicInterval("chr1", 200, 300, "+")
        assert overlaps(a, b)
        assert overlaps(b, a)

    def test_identical(self):
        a = GenomicInterval("chr1", 100, 200, "+")
        assert overlaps(a, a)

    def test_diff_chrom_no_overlap(self):
        a = GenomicInterval("chr1", 100, 200, "+")
        b = GenomicInterval("chr2", 100, 200, "+")
        assert not overlaps(a, b)

    def test_diff_strand_no_overlap(self):
        a = GenomicInterval("chr1", 100, 200, "+")
        b = GenomicInterval("chr1", 150, 250, "-")
        assert not overlaps(a, b)

    def test_unstranded_overlaps_plus(self):
        a = GenomicInterval("chr1", 100, 200, ".")
        b = GenomicInterval("chr1", 150, 250, "+")
        assert overlaps(a, b)

    def test_unstranded_overlaps_minus(self):
        a = GenomicInterval("chr1", 100, 200, ".")
        b = GenomicInterval("chr1", 150, 250, "-")
        assert overlaps(a, b)

    def test_unstranded_overlaps_unstranded(self):
        a = GenomicInterval("chr1", 100, 200, ".")
        b = GenomicInterval("chr1", 150, 250, ".")
        assert overlaps(a, b)

    def test_plus_does_not_overlap_minus(self):
        a = GenomicInterval("chr1", 100, 200, "+")
        b = GenomicInterval("chr1", 100, 200, "-")
        assert not overlaps(a, b)

    def test_single_base_overlap(self):
        # [100, 101) and [100, 200) share one base (position 100)
        a = GenomicInterval("chr1", 100, 101, "+")
        b = GenomicInterval("chr1", 100, 200, "+")
        assert overlaps(a, b)


# ---------------------------------------------------------------------------
# merge_intervals
# ---------------------------------------------------------------------------


class TestMergeIntervals:
    def test_empty(self):
        assert merge_intervals([]) == []

    def test_single(self):
        iv = GenomicInterval("chr1", 100, 200, "+")
        result = merge_intervals([iv])
        assert result == [iv]

    def test_non_overlapping_same_chrom_strand(self):
        iv1 = GenomicInterval("chr1", 100, 200, "+")
        iv2 = GenomicInterval("chr1", 300, 400, "+")
        result = merge_intervals([iv1, iv2])
        assert len(result) == 2
        assert result[0] == iv1
        assert result[1] == iv2

    def test_overlapping_merged(self):
        iv1 = GenomicInterval("chr1", 100, 200, "+")
        iv2 = GenomicInterval("chr1", 150, 300, "+")
        result = merge_intervals([iv1, iv2])
        assert len(result) == 1
        assert result[0] == GenomicInterval("chr1", 100, 300, "+")

    def test_adjacent_merged(self):
        # [100, 200) and [200, 300) share no base but start==end so they touch
        iv1 = GenomicInterval("chr1", 100, 200, "+")
        iv2 = GenomicInterval("chr1", 200, 300, "+")
        result = merge_intervals([iv1, iv2])
        # start <= current.end (200 <= 200) so they merge
        assert len(result) == 1
        assert result[0] == GenomicInterval("chr1", 100, 300, "+")

    def test_contained_interval_merged(self):
        iv1 = GenomicInterval("chr1", 100, 500, "+")
        iv2 = GenomicInterval("chr1", 200, 300, "+")
        result = merge_intervals([iv1, iv2])
        assert len(result) == 1
        assert result[0] == GenomicInterval("chr1", 100, 500, "+")

    def test_different_strands_not_merged(self):
        iv1 = GenomicInterval("chr1", 100, 200, "+")
        iv2 = GenomicInterval("chr1", 150, 250, "-")
        result = merge_intervals([iv1, iv2])
        assert len(result) == 2

    def test_different_chroms_not_merged(self):
        iv1 = GenomicInterval("chr1", 100, 200, "+")
        iv2 = GenomicInterval("chr2", 100, 200, "+")
        result = merge_intervals([iv1, iv2])
        assert len(result) == 2

    def test_unordered_input(self):
        iv1 = GenomicInterval("chr1", 300, 400, "+")
        iv2 = GenomicInterval("chr1", 100, 200, "+")
        iv3 = GenomicInterval("chr1", 150, 350, "+")
        result = merge_intervals([iv1, iv2, iv3])
        assert len(result) == 1
        assert result[0] == GenomicInterval("chr1", 100, 400, "+")

    def test_multi_chrom_multi_strand(self):
        ivs = [
            GenomicInterval("chr1", 100, 200, "+"),
            GenomicInterval("chr1", 150, 250, "+"),
            GenomicInterval("chr1", 100, 200, "-"),
            GenomicInterval("chr2", 100, 200, "+"),
        ]
        result = merge_intervals(ivs)
        # chr1+: [100,250), chr1-: [100,200), chr2+: [100,200)
        assert len(result) == 3

    def test_three_way_merge(self):
        iv1 = GenomicInterval("chr1", 100, 200, "+")
        iv2 = GenomicInterval("chr1", 180, 300, "+")
        iv3 = GenomicInterval("chr1", 280, 400, "+")
        result = merge_intervals([iv1, iv2, iv3])
        assert len(result) == 1
        assert result[0] == GenomicInterval("chr1", 100, 400, "+")

    def test_unstranded_not_merged_with_stranded(self):
        iv1 = GenomicInterval("chr1", 100, 200, ".")
        iv2 = GenomicInterval("chr1", 150, 250, "+")
        result = merge_intervals([iv1, iv2])
        # Different strand values -- not merged
        assert len(result) == 2


# ---------------------------------------------------------------------------
# subtract_interval
# ---------------------------------------------------------------------------


class TestSubtractInterval:
    def test_no_overlap_before(self):
        a = GenomicInterval("chr1", 200, 400, "+")
        b = GenomicInterval("chr1", 100, 200, "+")
        result = subtract_interval(a, b)
        assert result == [a]

    def test_no_overlap_after(self):
        a = GenomicInterval("chr1", 100, 200, "+")
        b = GenomicInterval("chr1", 200, 300, "+")
        result = subtract_interval(a, b)
        assert result == [a]

    def test_no_overlap_different_chrom(self):
        a = GenomicInterval("chr1", 100, 300, "+")
        b = GenomicInterval("chr2", 150, 250, "+")
        result = subtract_interval(a, b)
        assert result == [a]

    def test_complete_coverage_returns_empty(self):
        a = GenomicInterval("chr1", 100, 200, "+")
        b = GenomicInterval("chr1", 50, 300, "+")
        result = subtract_interval(a, b)
        assert result == []

    def test_exact_match_returns_empty(self):
        a = GenomicInterval("chr1", 100, 200, "+")
        result = subtract_interval(a, a)
        assert result == []

    def test_left_trim(self):
        a = GenomicInterval("chr1", 100, 300, "+")
        b = GenomicInterval("chr1", 50, 200, "+")
        result = subtract_interval(a, b)
        assert result == [GenomicInterval("chr1", 200, 300, "+")]

    def test_right_trim(self):
        a = GenomicInterval("chr1", 100, 300, "+")
        b = GenomicInterval("chr1", 200, 400, "+")
        result = subtract_interval(a, b)
        assert result == [GenomicInterval("chr1", 100, 200, "+")]

    def test_middle_excision(self):
        a = GenomicInterval("chr1", 100, 400, "+")
        b = GenomicInterval("chr1", 200, 300, "+")
        result = subtract_interval(a, b)
        assert len(result) == 2
        assert result[0] == GenomicInterval("chr1", 100, 200, "+")
        assert result[1] == GenomicInterval("chr1", 300, 400, "+")

    def test_strand_preserved_from_a(self):
        a = GenomicInterval("chr1", 100, 400, "-")
        b = GenomicInterval("chr1", 200, 300, "+")
        result = subtract_interval(a, b)
        # b's strand ignored for removal; a's strand preserved in output
        for iv in result:
            assert iv.strand == "-"

    def test_subtract_from_single_base(self):
        a = GenomicInterval("chr1", 100, 101, "+")
        b = GenomicInterval("chr1", 100, 101, "+")
        result = subtract_interval(a, b)
        assert result == []

    def test_b_starts_at_a_start(self):
        a = GenomicInterval("chr1", 100, 400, "+")
        b = GenomicInterval("chr1", 100, 200, "+")
        result = subtract_interval(a, b)
        assert result == [GenomicInterval("chr1", 200, 400, "+")]

    def test_b_ends_at_a_end(self):
        a = GenomicInterval("chr1", 100, 400, "+")
        b = GenomicInterval("chr1", 300, 400, "+")
        result = subtract_interval(a, b)
        assert result == [GenomicInterval("chr1", 100, 300, "+")]

    def test_result_lengths_sum_correctly(self):
        a = GenomicInterval("chr1", 0, 1000, "+")
        b = GenomicInterval("chr1", 400, 600, "+")
        result = subtract_interval(a, b)
        total = sum(iv.length for iv in result)
        assert total == a.length - b.length
