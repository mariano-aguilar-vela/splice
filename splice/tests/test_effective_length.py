"""
Test suite for Module 11: core/effective_length.py

Tests effective length computation following rMATS normalization.
"""

import unittest

import numpy as np

from splice.core.effective_length import (
    compute_effective_lengths_for_module,
    compute_exon_body_effective_lengths,
    compute_library_size_factors,
    compute_se_effective_lengths,
    length_normalize_counts,
)
from splice.utils.genomic import GenomicInterval, Junction


class TestComputeSEEffectiveLengths(unittest.TestCase):
    """Test compute_se_effective_lengths function."""

    def test_standard_read_length(self):
        """Test SE effective length with standard read length."""
        target = GenomicInterval(chrom="chr1", start=1000, end=2000, strand="+")
        upstream = GenomicInterval(chrom="chr1", start=500, end=1000, strand="+")
        downstream = GenomicInterval(chrom="chr1", start=2000, end=2500, strand="+")

        read_length = 101
        anchor = 1

        inc_len, skip_len = compute_se_effective_lengths(
            target, upstream, downstream, read_length, anchor
        )

        # JC inc_len = (101 - 2 + 1) + min(1000, 101 - 2 + 1)
        #            = 100 + min(1000, 100) = 100 + 100 = 200
        # JC skip_len = 100
        # JCEC inc_len = 200 + max(0, 1000 - 101 + 1) = 200 + 900 = 1100
        # JCEC skip_len = 100

        self.assertEqual(inc_len, 1100)
        self.assertEqual(skip_len, 100)

    def test_short_exon(self):
        """Test with exon shorter than read length."""
        target = GenomicInterval(chrom="chr1", start=1000, end=1050, strand="+")
        upstream = GenomicInterval(chrom="chr1", start=500, end=1000, strand="+")
        downstream = GenomicInterval(chrom="chr1", start=1050, end=2000, strand="+")

        read_length = 101
        anchor = 1

        inc_len, skip_len = compute_se_effective_lengths(
            target, upstream, downstream, read_length, anchor
        )

        # JC inc_len = 100 + min(50, 100) = 150
        # JCEC inc_len = 150 + max(0, 50 - 101 + 1) = 150 + 0 = 150

        self.assertEqual(inc_len, 150)
        self.assertEqual(skip_len, 100)

    def test_very_short_read_length(self):
        """Test with short read length."""
        target = GenomicInterval(chrom="chr1", start=1000, end=2000, strand="+")
        upstream = GenomicInterval(chrom="chr1", start=500, end=1000, strand="+")
        downstream = GenomicInterval(chrom="chr1", start=2000, end=2500, strand="+")

        read_length = 50
        anchor = 1

        inc_len, skip_len = compute_se_effective_lengths(
            target, upstream, downstream, read_length, anchor
        )

        # JC inc_len = (50 - 2 + 1) + min(1000, 49) = 49 + 49 = 98
        # JCEC inc_len = 98 + max(0, 1000 - 50 + 1) = 98 + 951 = 1049

        self.assertEqual(inc_len, 1049)
        self.assertEqual(skip_len, 49)


class TestComputeEffectiveLengthsForModule(unittest.TestCase):
    """Test compute_effective_lengths_for_module function."""

    def test_single_junction(self):
        """Test effective length for single junction."""
        j = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        exons = [
            GenomicInterval(chrom="chr1", start=500, end=1000, strand="+"),
            GenomicInterval(chrom="chr1", start=2000, end=2500, strand="+"),
        ]

        read_length = 101
        anchor = 1

        lengths = compute_effective_lengths_for_module([j], exons, read_length, anchor)

        self.assertEqual(len(lengths), 1)
        # read_length - 2*anchor + 1 = 101 - 2 + 1 = 100
        self.assertEqual(lengths[0], 100)

    def test_multiple_junctions(self):
        """Test effective lengths for multiple junctions."""
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            Junction(chrom="chr1", start=1500, end=3000, strand="+"),
        ]
        exons = [
            GenomicInterval(chrom="chr1", start=500, end=1000, strand="+"),
            GenomicInterval(chrom="chr1", start=2000, end=2500, strand="+"),
            GenomicInterval(chrom="chr1", start=3000, end=3500, strand="+"),
        ]

        read_length = 101
        anchor = 1

        lengths = compute_effective_lengths_for_module(
            junctions, exons, read_length, anchor
        )

        self.assertEqual(len(lengths), 3)
        # All should have the same junction-level effective length
        np.testing.assert_array_equal(lengths, [100, 100, 100])

    def test_different_anchor_lengths(self):
        """Test that anchor length affects effective length."""
        j = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        exons = []

        read_length = 101

        # Anchor = 1: 101 - 2 + 1 = 100
        lengths_a1 = compute_effective_lengths_for_module([j], exons, read_length, 1)

        # Anchor = 10: 101 - 20 + 1 = 82
        lengths_a10 = compute_effective_lengths_for_module([j], exons, read_length, 10)

        self.assertEqual(lengths_a1[0], 100)
        self.assertEqual(lengths_a10[0], 82)


class TestComputeExonBodyEffectiveLengths(unittest.TestCase):
    """Test compute_exon_body_effective_lengths function."""

    def test_single_exon(self):
        """Test effective length for single exon."""
        exon = GenomicInterval(chrom="chr1", start=1000, end=2000, strand="+")

        read_length = 101

        lengths = compute_exon_body_effective_lengths([exon], read_length)

        self.assertEqual(len(lengths), 1)
        # max(0, 1000 - 101 + 1) = 900
        self.assertEqual(lengths[0], 900)

    def test_short_exon(self):
        """Test with exon shorter than read length."""
        exon = GenomicInterval(chrom="chr1", start=1000, end=1050, strand="+")

        read_length = 101

        lengths = compute_exon_body_effective_lengths([exon], read_length)

        # max(0, 50 - 101 + 1) = 0
        self.assertEqual(lengths[0], 0)

    def test_multiple_exons(self):
        """Test effective lengths for multiple exons."""
        exons = [
            GenomicInterval(chrom="chr1", start=1000, end=2000, strand="+"),
            GenomicInterval(chrom="chr1", start=3000, end=3500, strand="+"),
            GenomicInterval(chrom="chr1", start=4000, end=4050, strand="+"),
        ]

        read_length = 101

        lengths = compute_exon_body_effective_lengths(exons, read_length)

        self.assertEqual(len(lengths), 3)
        np.testing.assert_array_equal(lengths, [900, 400, 0])


class TestLengthNormalizeCounts(unittest.TestCase):
    """Test length_normalize_counts function."""

    def test_basic_normalization(self):
        """Test basic count normalization."""
        counts = np.array([100, 200, 300])
        effective_lengths = np.array([100, 100, 100])

        normalized = length_normalize_counts(counts, effective_lengths)

        np.testing.assert_array_almost_equal(normalized, [1.0, 2.0, 3.0])

    def test_different_lengths(self):
        """Test normalization with different effective lengths."""
        counts = np.array([100, 200, 300])
        effective_lengths = np.array([100, 50, 150])

        normalized = length_normalize_counts(counts, effective_lengths)

        np.testing.assert_array_almost_equal(normalized, [1.0, 4.0, 2.0])

    def test_zero_effective_length(self):
        """Test handling of zero effective length."""
        counts = np.array([100, 200, 300])
        effective_lengths = np.array([100, 0, 100])

        normalized = length_normalize_counts(counts, effective_lengths)

        np.testing.assert_almost_equal(normalized[0], 1.0)
        np.testing.assert_almost_equal(normalized[1], 0.0)  # Undefined, set to 0
        np.testing.assert_almost_equal(normalized[2], 3.0)

    def test_matrix_normalization(self):
        """Test normalization of 2D count matrix."""
        counts = np.array([[100, 200], [300, 400]])
        effective_lengths = np.array([100, 100])

        normalized = length_normalize_counts(counts, effective_lengths)

        np.testing.assert_array_almost_equal(
            normalized, [[1.0, 2.0], [3.0, 4.0]]
        )


class TestComputeLibrarySizeFactors(unittest.TestCase):
    """Test compute_library_size_factors function."""

    def test_equal_library_sizes(self):
        """Test with equal library sizes."""
        # Two samples with equal counts
        count_matrix = np.array([
            [100, 100],
            [200, 200],
            [150, 150],
        ])

        factors = compute_library_size_factors(count_matrix)

        # Should be normalized to geometric mean of 1
        self.assertEqual(len(factors), 2)
        self.assertAlmostEqual(factors[0] * factors[1], 1.0, places=5)
        # With equal counts, should be nearly equal to each other
        self.assertAlmostEqual(factors[0], factors[1], places=2)

    def test_different_library_sizes(self):
        """Test with different library sizes."""
        # Sample 2 has 2x the counts of sample 1
        count_matrix = np.array([
            [100, 200],
            [200, 400],
            [150, 300],
        ])

        factors = compute_library_size_factors(count_matrix)

        # Sample 2 should have higher factor
        self.assertGreater(factors[1], factors[0])
        # Geometric mean should be 1
        self.assertAlmostEqual(factors[0] * factors[1], 1.0, places=5)

    def test_zero_counts(self):
        """Test handling of samples with zero counts."""
        count_matrix = np.array([
            [100, 0],
            [200, 100],
            [150, 50],
        ])

        factors = compute_library_size_factors(count_matrix)

        # Should not produce NaN or inf
        self.assertFalse(np.isnan(factors).any())
        self.assertFalse(np.isinf(factors).any())
        # Geometric mean should still be 1
        self.assertAlmostEqual(factors[0] * factors[1], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
