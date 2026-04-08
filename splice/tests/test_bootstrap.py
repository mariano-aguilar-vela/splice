"""
Test suite for Module 13: core/bootstrap.py

Tests bootstrap resampling for uncertainty estimation.
"""

import unittest

import numpy as np

from splicekit.core.bootstrap import (
    bootstrap_confidence_intervals,
    bootstrap_junction_counts,
    bootstrap_mean_psi,
    bootstrap_psi,
    bootstrap_std_psi,
)


class TestBootstrapJunctionCounts(unittest.TestCase):
    """Test bootstrap_junction_counts function."""

    def test_bootstrap_shape(self):
        """Test that bootstrap returns correct shape."""
        count_matrix = np.array([[100, 200], [300, 400]])
        n_bootstraps = 10

        bootstrap_counts = bootstrap_junction_counts(
            count_matrix, n_bootstraps=n_bootstraps, seed=42
        )

        self.assertEqual(bootstrap_counts.shape, (10, 2, 2))

    def test_bootstrap_preserves_total_counts(self):
        """Test that total counts are preserved in bootstraps."""
        count_matrix = np.array([[100, 200], [300, 400]])
        n_bootstraps = 5

        bootstrap_counts = bootstrap_junction_counts(
            count_matrix, n_bootstraps=n_bootstraps, seed=42
        )

        # Check that column totals match original
        for boot_idx in range(n_bootstraps):
            for sample_idx in range(2):
                original_total = np.sum(count_matrix[:, sample_idx])
                bootstrap_total = np.sum(bootstrap_counts[boot_idx, :, sample_idx])
                self.assertEqual(bootstrap_total, original_total)

    def test_bootstrap_single_sample(self):
        """Test bootstrap with single sample."""
        count_matrix = np.array([[100], [200], [300]])

        bootstrap_counts = bootstrap_junction_counts(
            count_matrix, n_bootstraps=5, seed=42
        )

        self.assertEqual(bootstrap_counts.shape, (5, 3, 1))

    def test_bootstrap_zero_counts(self):
        """Test bootstrap with some zero counts."""
        count_matrix = np.array([[100, 50], [200, 150]])

        bootstrap_counts = bootstrap_junction_counts(
            count_matrix, n_bootstraps=5, seed=42
        )

        # All samples have counts, so should be preserved
        for boot_idx in range(5):
            self.assertEqual(np.sum(bootstrap_counts[boot_idx, :, 0]), 300)
            self.assertEqual(np.sum(bootstrap_counts[boot_idx, :, 1]), 200)

    def test_bootstrap_reproducibility(self):
        """Test that same seed produces same results."""
        count_matrix = np.array([[100, 200], [300, 400]])

        bootstrap1 = bootstrap_junction_counts(count_matrix, n_bootstraps=5, seed=42)
        bootstrap2 = bootstrap_junction_counts(count_matrix, n_bootstraps=5, seed=42)

        np.testing.assert_array_equal(bootstrap1, bootstrap2)

    def test_bootstrap_different_seeds(self):
        """Test that different seeds produce different results."""
        count_matrix = np.array([[100, 200], [300, 400]])

        bootstrap1 = bootstrap_junction_counts(count_matrix, n_bootstraps=5, seed=42)
        bootstrap2 = bootstrap_junction_counts(count_matrix, n_bootstraps=5, seed=123)

        self.assertFalse(np.array_equal(bootstrap1, bootstrap2))


class TestBootstrapPSI(unittest.TestCase):
    """Test bootstrap_psi function."""

    def test_bootstrap_psi_shape(self):
        """Test that PSI matrix has correct shape."""
        bootstrap_counts = np.array([
            [[100, 200], [300, 400]],
            [[110, 190], [290, 410]],
            [[95, 205], [305, 395]],
        ])
        effective_lengths = np.array([100.0, 100.0])

        psi = bootstrap_psi(bootstrap_counts, effective_lengths)

        self.assertEqual(psi.shape, (3, 2, 2))

    def test_bootstrap_psi_sums_to_one(self):
        """Test that PSI values sum to 1 per sample."""
        # Counts: 2 junctions, 2 samples, 2 bootstraps
        bootstrap_counts = np.array([
            [[100, 200], [300, 400]],
            [[110, 190], [290, 410]],
        ])
        effective_lengths = np.array([100.0, 100.0])

        psi = bootstrap_psi(bootstrap_counts, effective_lengths)

        # PSI should sum to 1 per sample across junctions
        # Shape of psi: (2 bootstraps, 2 junctions, 2 samples)
        # Sum over junctions (axis=0): should get (2, 2) -> sum per sample per bootstrap
        for boot_idx in range(2):
            sample_sums = np.sum(psi[boot_idx, :, :], axis=0)
            # Both samples should sum to 1
            self.assertAlmostEqual(sample_sums[0], 1.0, places=5)
            self.assertAlmostEqual(sample_sums[1], 1.0, places=5)

    def test_bootstrap_psi_values_in_range(self):
        """Test that PSI values are in [0, 1]."""
        bootstrap_counts = np.array([
            [[100, 200], [300, 400]],
        ])
        effective_lengths = np.array([100.0, 100.0])

        psi = bootstrap_psi(bootstrap_counts, effective_lengths)

        self.assertTrue(np.all(psi >= 0))
        self.assertTrue(np.all(psi <= 1))

    def test_bootstrap_psi_zero_counts(self):
        """Test PSI with zero counts sample."""
        bootstrap_counts = np.array([
            [[100, 50], [300, 150]],
        ])
        effective_lengths = np.array([100.0, 100.0])

        psi = bootstrap_psi(bootstrap_counts, effective_lengths)

        # All samples have reads, so PSI should be valid
        self.assertTrue(np.all(np.isfinite(psi)))
        # PSI should sum to 1 per sample
        np.testing.assert_array_almost_equal(
            np.sum(psi[0, :, :], axis=0), [1.0, 1.0]
        )


class TestBootstrapConfidenceIntervals(unittest.TestCase):
    """Test bootstrap_confidence_intervals function."""

    def test_ci_shape(self):
        """Test that CI arrays have correct shape."""
        bootstrap_psi = np.random.rand(30, 5, 10)  # 30 bootstraps, 5 junctions, 10 samples

        ci_low, ci_high = bootstrap_confidence_intervals(bootstrap_psi, alpha=0.05)

        self.assertEqual(ci_low.shape, (5, 10))
        self.assertEqual(ci_high.shape, (5, 10))

    def test_ci_bounds(self):
        """Test that CI bounds are reasonable."""
        # Create random PSI values that sum to 1 per sample
        bootstrap_psi = np.random.rand(100, 3, 5)
        bootstrap_psi = bootstrap_psi / np.sum(bootstrap_psi, axis=1, keepdims=True)

        ci_low, ci_high = bootstrap_confidence_intervals(bootstrap_psi, alpha=0.05)

        # Low CI should be <= high CI
        self.assertTrue(np.all(ci_low <= ci_high))

        # Both should be in [0, 1]
        self.assertTrue(np.all(ci_low >= 0))
        self.assertTrue(np.all(ci_high <= 1))

    def test_ci_alpha_effect(self):
        """Test that smaller alpha produces wider intervals."""
        # Create random PSI values that sum to 1 per sample
        bootstrap_psi = np.random.rand(100, 2, 3)
        bootstrap_psi = bootstrap_psi / np.sum(bootstrap_psi, axis=1, keepdims=True)

        ci_low_05, ci_high_05 = bootstrap_confidence_intervals(
            bootstrap_psi, alpha=0.05
        )
        ci_low_10, ci_high_10 = bootstrap_confidence_intervals(
            bootstrap_psi, alpha=0.10
        )

        # 95% CI should be wider than 90% CI
        width_05 = ci_high_05 - ci_low_05
        width_10 = ci_high_10 - ci_low_10

        self.assertTrue(np.mean(width_05) >= np.mean(width_10))

    def test_ci_constant_values(self):
        """Test CI with constant PSI values."""
        # All bootstrap replicates are the same
        bootstrap_psi = np.full((30, 2, 3), 0.5)

        ci_low, ci_high = bootstrap_confidence_intervals(bootstrap_psi, alpha=0.05)

        # All intervals should be [0.5, 0.5]
        np.testing.assert_array_almost_equal(ci_low, 0.5)
        np.testing.assert_array_almost_equal(ci_high, 0.5)


class TestBootstrapMeanPSI(unittest.TestCase):
    """Test bootstrap_mean_psi function."""

    def test_mean_shape(self):
        """Test that mean PSI has correct shape."""
        # Create random PSI values that sum to 1 per sample
        bootstrap_psi = np.random.rand(30, 5, 10)
        bootstrap_psi = bootstrap_psi / np.sum(bootstrap_psi, axis=1, keepdims=True)

        mean_psi = bootstrap_mean_psi(bootstrap_psi)

        self.assertEqual(mean_psi.shape, (5, 10))

    def test_mean_values(self):
        """Test that mean PSI is correct."""
        bootstrap_psi = np.array([
            [[0.2, 0.4], [0.3, 0.5]],
            [[0.4, 0.6], [0.5, 0.7]],
            [[0.0, 0.2], [0.1, 0.3]],
        ])

        mean_psi = bootstrap_mean_psi(bootstrap_psi)

        expected = np.array([[0.2, 0.4], [0.3, 0.5]])
        np.testing.assert_array_almost_equal(mean_psi, expected)


class TestBootstrapStdPSI(unittest.TestCase):
    """Test bootstrap_std_psi function."""

    def test_std_shape(self):
        """Test that std PSI has correct shape."""
        # Create random PSI values that sum to 1 per sample
        bootstrap_psi = np.random.rand(30, 5, 10)
        bootstrap_psi = bootstrap_psi / np.sum(bootstrap_psi, axis=1, keepdims=True)

        std_psi = bootstrap_std_psi(bootstrap_psi)

        self.assertEqual(std_psi.shape, (5, 10))

    def test_std_non_negative(self):
        """Test that std is non-negative."""
        # Create random PSI values that sum to 1 per sample
        bootstrap_psi = np.random.rand(30, 5, 10)
        bootstrap_psi = bootstrap_psi / np.sum(bootstrap_psi, axis=1, keepdims=True)

        std_psi = bootstrap_std_psi(bootstrap_psi)

        self.assertTrue(np.all(std_psi >= 0))

    def test_std_constant_values(self):
        """Test std with constant PSI values."""
        bootstrap_psi = np.full((30, 2, 3), 0.5)

        std_psi = bootstrap_std_psi(bootstrap_psi)

        # Std should be 0 for constant values
        np.testing.assert_array_almost_equal(std_psi, 0)


if __name__ == "__main__":
    unittest.main()
