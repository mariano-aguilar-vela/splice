"""
Test suite for Module 14: utils/stats.py

Tests statistical utilities including DM likelihood, FDR correction, and Beta posteriors.
"""

import unittest

import numpy as np
from scipy.stats import chi2

from splicekit.utils.stats import (
    benjamini_hochberg,
    beta_posterior_psi,
    dm_log_likelihood,
    dm_log_likelihood_batch,
    fit_dm_full,
    fit_dm_null,
    likelihood_ratio_test,
)


class TestDMLogLikelihood(unittest.TestCase):
    """Test Dirichlet-multinomial log-likelihood."""

    def test_single_sample_simple(self):
        """Test DM likelihood with simple counts."""
        counts = np.array([10, 20, 30])
        alpha = np.array([1.0, 1.0, 1.0])

        ll = dm_log_likelihood(counts, alpha)

        # Should be a finite number
        self.assertTrue(np.isfinite(ll))
        self.assertLess(ll, 0)  # Log-likelihood should be negative

    def test_single_sample_zero_counts(self):
        """Test DM likelihood with some zero counts."""
        counts = np.array([0, 10, 20])
        alpha = np.array([0.5, 0.5, 0.5])

        ll = dm_log_likelihood(counts, alpha)

        self.assertTrue(np.isfinite(ll))

    def test_batch_likelihood(self):
        """Test batch DM likelihood computation."""
        count_matrix = np.array([
            [10, 20, 30],
            [15, 25, 35],
            [20, 30, 40],
        ])
        alpha = np.array([1.0, 1.0, 1.0])

        batch_ll = dm_log_likelihood_batch(count_matrix, alpha)

        # Should equal sum of individual likelihoods
        individual_ll = sum(
            dm_log_likelihood(count_matrix[i, :], alpha) for i in range(3)
        )

        self.assertAlmostEqual(batch_ll, individual_ll, places=10)

    def test_likelihood_increases_with_good_alpha(self):
        """Test that likelihood increases with better-fitting alpha."""
        counts = np.array([100, 200, 300])

        # Alpha matching proportions should give higher likelihood
        alpha_good = np.array([1.0, 2.0, 3.0])
        alpha_bad = np.array([3.0, 2.0, 1.0])

        ll_good = dm_log_likelihood(counts, alpha_good)
        ll_bad = dm_log_likelihood(counts, alpha_bad)

        self.assertGreater(ll_good, ll_bad)


class TestFitDMNull(unittest.TestCase):
    """Test null DM model fitting."""

    def test_fit_null_convergence(self):
        """Test that null model fitting converges."""
        count_matrix = np.array([
            [100, 200, 300],
            [120, 220, 320],
            [110, 210, 310],
        ])

        alpha_hat, ll, converged = fit_dm_null(count_matrix)

        self.assertTrue(converged)
        self.assertTrue(np.all(alpha_hat > 0))
        self.assertTrue(np.isfinite(ll))
        self.assertEqual(len(alpha_hat), 3)

    def test_fit_null_better_than_uniform(self):
        """Test that fitted alpha gives higher likelihood than uniform."""
        count_matrix = np.array([
            [100, 200, 300],
            [150, 250, 350],
            [120, 210, 320],
        ])

        alpha_hat, ll_fit, _ = fit_dm_null(count_matrix)

        # Compare with uniform alpha
        alpha_uniform = np.ones(3)
        ll_uniform = dm_log_likelihood_batch(count_matrix, alpha_uniform)

        self.assertGreater(ll_fit, ll_uniform)


class TestFitDMFull(unittest.TestCase):
    """Test full DM model fitting with group effect."""

    def test_fit_full_convergence(self):
        """Test that full model fitting converges."""
        count_matrix = np.array([
            [100, 200],  # group 0
            [110, 210],  # group 0
            [50, 450],   # group 1
            [60, 440],   # group 1
        ])
        group_labels = np.array([0, 0, 1, 1])

        alpha_0, alpha_1, ll, converged = fit_dm_full(count_matrix, group_labels)

        self.assertTrue(converged)
        self.assertTrue(np.all(alpha_0 > 0))
        self.assertTrue(np.all(alpha_1 > 0))
        self.assertTrue(np.isfinite(ll))
        self.assertEqual(len(alpha_0), 2)
        self.assertEqual(len(alpha_1), 2)

    def test_fit_full_vs_null(self):
        """Test that full model gives better or equal likelihood to null."""
        count_matrix = np.array([
            [100, 200],
            [110, 210],
            [50, 450],
            [60, 440],
        ])
        group_labels = np.array([0, 0, 1, 1])

        _, _, ll_full, _ = fit_dm_full(count_matrix, group_labels)
        _, ll_null, _ = fit_dm_null(count_matrix)

        # Full model should give >= likelihood (and usually > due to extra parameters)
        self.assertGreaterEqual(ll_full, ll_null - 1e-4)  # Small tolerance for numeric precision


class TestLikelihoodRatioTest(unittest.TestCase):
    """Test likelihood ratio testing."""

    def test_lrt_no_difference(self):
        """Test LRT when null and full likelihoods are equal."""
        ll_null = -100.0
        ll_full = -100.0

        p_value = likelihood_ratio_test(ll_null, ll_full, df=1)

        self.assertEqual(p_value, 1.0)

    def test_lrt_strong_difference(self):
        """Test LRT with strong difference in likelihoods."""
        ll_null = -100.0
        ll_full = -50.0  # Much better

        p_value = likelihood_ratio_test(ll_null, ll_full, df=1)

        # Should be very small p-value
        self.assertLess(p_value, 0.001)

    def test_lrt_negative_test_stat(self):
        """Test LRT when full model is worse than null."""
        ll_null = -50.0
        ll_full = -100.0  # Worse fit

        p_value = likelihood_ratio_test(ll_null, ll_full, df=1)

        # Should return 1.0 for negative test statistic
        self.assertEqual(p_value, 1.0)

    def test_lrt_matches_chi2(self):
        """Test that LRT p-value matches chi-squared distribution."""
        ll_null = -100.0
        ll_full = -98.0

        test_stat = 2 * (ll_full - ll_null)
        expected_p = 1 - chi2.cdf(test_stat, 1)

        actual_p = likelihood_ratio_test(ll_null, ll_full, df=1)

        self.assertAlmostEqual(actual_p, expected_p, places=10)


class TestBenjaminiHochberg(unittest.TestCase):
    """Test Benjamini-Hochberg FDR correction."""

    def test_bh_all_significant(self):
        """Test BH correction with all very small p-values."""
        p_values = np.array([0.001, 0.002, 0.003, 0.004, 0.005])

        adjusted = benjamini_hochberg(p_values)

        # All should be adjusted
        self.assertEqual(len(adjusted), 5)
        self.assertTrue(np.all(adjusted >= p_values))

    def test_bh_all_insignificant(self):
        """Test BH correction with large p-values."""
        p_values = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        adjusted = benjamini_hochberg(p_values)

        # Should be clamped to 1.0
        self.assertTrue(np.all(adjusted <= 1.0))
        self.assertTrue(np.all(adjusted >= p_values))

    def test_bh_mixed(self):
        """Test BH correction with mixed p-values."""
        p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])

        adjusted = benjamini_hochberg(p_values)

        # Adjusted should be >= original
        self.assertTrue(np.all(adjusted >= p_values))
        # Should be monotonic
        for i in range(len(adjusted) - 1):
            self.assertLessEqual(adjusted[i], adjusted[i + 1])

    def test_bh_single_p_value(self):
        """Test BH with single p-value."""
        p_values = np.array([0.05])

        adjusted = benjamini_hochberg(p_values)

        self.assertEqual(len(adjusted), 1)
        self.assertEqual(adjusted[0], 0.05)  # Single test should not be adjusted

    def test_bh_bounds(self):
        """Test that adjusted p-values stay in [0, 1]."""
        p_values = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99])

        adjusted = benjamini_hochberg(p_values)

        self.assertTrue(np.all(adjusted >= 0))
        self.assertTrue(np.all(adjusted <= 1.0))


class TestBetaPosteriorPSI(unittest.TestCase):
    """Test Beta posterior PSI computation."""

    def test_beta_posterior_simple(self):
        """Test Beta posterior with simple counts."""
        counts = np.array([10, 20, 30])
        total = 60.0

        psi, ci_width = beta_posterior_psi(counts, total)

        # PSI should be close to counts/total
        expected_psi = counts / total
        np.testing.assert_array_almost_equal(psi, expected_psi, decimal=1)

    def test_beta_posterior_shapes(self):
        """Test that output shapes are correct."""
        counts = np.array([100, 200, 300])
        total = 600.0

        psi, ci_width = beta_posterior_psi(counts, total)

        self.assertEqual(psi.shape, (3,))
        self.assertEqual(ci_width.shape, (3,))

    def test_beta_posterior_sums_to_one(self):
        """Test that PSI values sum to approximately 1."""
        counts = np.array([100, 200, 300])
        total = 600.0

        psi, _ = beta_posterior_psi(counts, total)

        # With large counts and uniform prior, should sum to ~1
        self.assertAlmostEqual(np.sum(psi), 1.0, places=2)

    def test_beta_posterior_ci_width_positive(self):
        """Test that credible interval widths are positive."""
        counts = np.array([10, 20, 30])
        total = 60.0

        psi, ci_width = beta_posterior_psi(counts, total)

        self.assertTrue(np.all(ci_width > 0))

    def test_beta_posterior_low_counts(self):
        """Test Beta posterior with low counts (wider CI expected)."""
        counts_low = np.array([1, 2, 3])
        counts_high = np.array([100, 200, 300])
        total_low = 6.0
        total_high = 600.0

        _, ci_low = beta_posterior_psi(counts_low, total_low)
        _, ci_high = beta_posterior_psi(counts_high, total_high)

        # Low counts should have wider CI
        self.assertTrue(np.mean(ci_low) > np.mean(ci_high))


class TestStatisticsIntegration(unittest.TestCase):
    """Integration tests for statistical functions."""

    def test_full_dm_testing_pipeline(self):
        """Test complete DM testing pipeline: fit null, fit full, LRT."""
        count_matrix = np.array([
            [100, 200],
            [110, 210],
            [50, 450],
            [60, 440],
        ])
        group_labels = np.array([0, 0, 1, 1])

        # Fit models
        alpha_null, ll_null, _ = fit_dm_null(count_matrix)
        alpha_0, alpha_1, ll_full, _ = fit_dm_full(count_matrix, group_labels)

        # Compute LRT p-value
        p_value = likelihood_ratio_test(ll_null, ll_full, df=1)

        # Basic sanity checks
        self.assertTrue(0 <= p_value <= 1)
        self.assertTrue(np.isfinite(ll_null))
        self.assertTrue(np.isfinite(ll_full))

    def test_fdr_correction_pipeline(self):
        """Test FDR correction on multiple tests."""
        p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.042])

        adjusted = benjamini_hochberg(p_values)

        # At FDR=0.05, should retain approximately first 3-4 tests
        sig_count = np.sum(adjusted <= 0.05)
        self.assertGreaterEqual(sig_count, 1)
        self.assertLessEqual(sig_count, 5)


if __name__ == "__main__":
    unittest.main()
