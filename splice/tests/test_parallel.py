"""
Test suite for Module 25: utils/parallel.py

Tests chromosome-level parallelism via multiprocessing.
"""

import unittest
from typing import Dict, List

from splicekit.utils.parallel import (
    get_default_chromosomes,
    parallel_by_chromosome,
)


# Module-level functions for pickling (required for multiprocessing tests)
def _identity(chrom, **kwargs):
    """Return the chromosome name."""
    return chrom


def _concat_with_prefix(chrom, prefix="", suffix=""):
    """Concatenate chromosome name with prefix and suffix."""
    return f"{prefix}{chrom}{suffix}"


def _count_characters(chrom, **kwargs):
    """Count number of characters in chromosome name."""
    return len(chrom)


def _create_stats(chrom, **kwargs):
    """Create statistics dictionary for chromosome."""
    return {
        "chrom": chrom,
        "length": len(chrom),
        "events": kwargs.get("n_events", 0),
    }


def _upper_chrom(chrom, **kwargs):
    """Return uppercase chromosome name."""
    return chrom.upper()


class TestParallelByChromosome(unittest.TestCase):
    """Test parallel_by_chromosome function."""

    def test_sequential_execution(self):
        """Test sequential execution with n_workers=1."""
        chromosomes = ["chr1", "chr2", "chr3"]
        results = parallel_by_chromosome(
            _identity, chromosomes, n_workers=1
        )

        self.assertEqual(results, chromosomes)

    def test_parallel_execution(self):
        """Test parallel execution with n_workers>1."""
        chromosomes = ["chr1", "chr2", "chr3"]
        results = parallel_by_chromosome(
            _identity, chromosomes, n_workers=2
        )

        # Results should be in same order as input
        self.assertEqual(results, chromosomes)

    def test_with_kwargs(self):
        """Test with keyword arguments."""
        chromosomes = ["chr1", "chr2"]
        results = parallel_by_chromosome(
            _concat_with_prefix,
            chromosomes,
            n_workers=1,
            prefix="PREFIX_",
            suffix="_SUFFIX",
        )

        expected = ["PREFIX_chr1_SUFFIX", "PREFIX_chr2_SUFFIX"]
        self.assertEqual(results, expected)

    def test_with_complex_function(self):
        """Test with function that performs computation."""
        chromosomes = ["chr1", "chr22", "chrX"]
        results = parallel_by_chromosome(
            _count_characters, chromosomes, n_workers=1
        )

        expected = [4, 5, 4]
        self.assertEqual(results, expected)

    def test_with_dictionary_return(self):
        """Test function that returns dictionary."""
        chromosomes = ["chr1", "chr2"]
        results = parallel_by_chromosome(
            _create_stats,
            chromosomes,
            n_workers=1,
            n_events=100,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chrom"], "chr1")
        self.assertEqual(results[0]["events"], 100)
        self.assertEqual(results[1]["chrom"], "chr2")

    def test_empty_chromosome_list(self):
        """Test with empty chromosome list."""
        results = parallel_by_chromosome(_identity, [], n_workers=1)

        self.assertEqual(results, [])

    def test_single_chromosome(self):
        """Test with single chromosome."""
        results = parallel_by_chromosome(
            _identity, ["chr1"], n_workers=1
        )

        self.assertEqual(results, ["chr1"])

    def test_many_chromosomes_sequential(self):
        """Test with many chromosomes in sequential mode."""
        chromosomes = [f"chr{i}" for i in range(1, 50)]
        results = parallel_by_chromosome(
            _identity, chromosomes, n_workers=1
        )

        self.assertEqual(results, chromosomes)


class TestGetDefaultChromosomes(unittest.TestCase):
    """Test get_default_chromosomes function."""

    def test_include_sex_chromosomes(self):
        """Test with include_sex=True."""
        chromosomes = get_default_chromosomes(include_sex=True)

        # Should have 24 chromosomes (22 autosomes + X + Y)
        self.assertEqual(len(chromosomes), 24)

        # Check autosomes
        for i in range(1, 23):
            self.assertIn(f"chr{i}", chromosomes)

        # Check sex chromosomes
        self.assertIn("chrX", chromosomes)
        self.assertIn("chrY", chromosomes)

        # Sex chromosomes should be last
        self.assertEqual(chromosomes[-2:], ["chrX", "chrY"])

    def test_exclude_sex_chromosomes(self):
        """Test with include_sex=False."""
        chromosomes = get_default_chromosomes(include_sex=False)

        # Should have 22 chromosomes (autosomes only)
        self.assertEqual(len(chromosomes), 22)

        # Check autosomes
        for i in range(1, 23):
            self.assertIn(f"chr{i}", chromosomes)

        # Sex chromosomes should not be present
        self.assertNotIn("chrX", chromosomes)
        self.assertNotIn("chrY", chromosomes)

    def test_default_parameter(self):
        """Test that default parameter includes sex chromosomes."""
        chromosomes_default = get_default_chromosomes()
        chromosomes_explicit = get_default_chromosomes(include_sex=True)

        self.assertEqual(chromosomes_default, chromosomes_explicit)

    def test_chromosome_order(self):
        """Test that chromosomes are in correct order."""
        chromosomes = get_default_chromosomes(include_sex=True)

        # First chromosome should be chr1
        self.assertEqual(chromosomes[0], "chr1")

        # Last two should be chrX and chrY
        self.assertEqual(chromosomes[-2], "chrX")
        self.assertEqual(chromosomes[-1], "chrY")

        # Check sequential autosomes
        for i in range(1, 23):
            self.assertEqual(chromosomes[i - 1], f"chr{i}")

    def test_no_duplicates(self):
        """Test that there are no duplicate chromosomes."""
        chromosomes = get_default_chromosomes(include_sex=True)

        self.assertEqual(len(chromosomes), len(set(chromosomes)))

    def test_all_valid_chromosome_names(self):
        """Test that all chromosome names are valid."""
        chromosomes = get_default_chromosomes(include_sex=True)

        for chrom in chromosomes:
            # Should start with 'chr'
            self.assertTrue(chrom.startswith("chr"))
            # Should have valid format
            rest = chrom[3:]
            self.assertTrue(
                rest.isdigit() or rest in ["X", "Y"],
                f"Invalid chromosome name: {chrom}",
            )


class TestParallelIntegration(unittest.TestCase):
    """Integration tests for parallel processing."""

    def test_sequential_with_default_chromosomes(self):
        """Test sequential processing with default chromosomes."""
        chromosomes = get_default_chromosomes(include_sex=True)
        results = parallel_by_chromosome(
            _count_characters, chromosomes, n_workers=1
        )

        # All results should be positive integers
        self.assertEqual(len(results), 24)
        self.assertTrue(all(isinstance(r, int) for r in results))
        self.assertTrue(all(r > 0 for r in results))

    def test_sequential_processing(self):
        """Test sequential execution with multiple calls."""
        chromosomes = ["chr1", "chr2", "chr3"]

        results = parallel_by_chromosome(
            _concat_with_prefix,
            chromosomes,
            n_workers=1,
            prefix="PREFIX_",
            suffix="_SUFFIX",
        )

        expected = [
            "PREFIX_chr1_SUFFIX",
            "PREFIX_chr2_SUFFIX",
            "PREFIX_chr3_SUFFIX",
        ]
        self.assertEqual(results, expected)

    def test_chromosome_subset(self):
        """Test processing subset of chromosomes."""
        # Process only chr1-5
        chromosomes = ["chr1", "chr2", "chr3", "chr4", "chr5"]
        results = parallel_by_chromosome(
            _upper_chrom, chromosomes, n_workers=1
        )

        expected = ["CHR1", "CHR2", "CHR3", "CHR4", "CHR5"]
        self.assertEqual(results, expected)

    def test_complex_chromosome_names(self):
        """Test with complex chromosome names."""
        # Test with various chromosome naming conventions
        chromosomes = [
            "chr1",
            "chr22",
            "chrX",
            "chrY",
            "chrM",  # Mitochondrial
        ]
        results = parallel_by_chromosome(
            _identity, chromosomes, n_workers=1
        )

        self.assertEqual(results, chromosomes)


if __name__ == "__main__":
    unittest.main()
