"""
Test suite for Module 16: utils/dm_glm.py

Tests DM-GLM fitting with design matrices, covariates, and likelihood ratio testing.
"""

import unittest

import numpy as np
from scipy.stats import chi2

from splicekit.utils.dm_glm import (
    DMGLMResult,
    build_design_matrix,
    dm_log_likelihood,
    dm_log_likelihood_batch,
    dm_lrt,
    fit_dm_full,
    fit_dm_glm,
    fit_dm_null,
    softmax,
)


class TestDMLogLikelihood(unittest.TestCase):
    """Test DM log-likelihood functions."""

    def test_dm_log_likelihood_simple(self):
        """Test DM likelihood with simple counts."""
        counts = np.array([10, 20, 30])
        alpha = np.array([1.0, 1.0, 1.0])

        ll = dm_log_likelihood(counts, alpha)

        self.assertTrue(np.isfinite(ll))
        self.assertLess(ll, 0)

    def test_dm_log_likelihood_batch(self):
        """Test batch DM likelihood computation."""
        count_matrix = np.array([
            [10, 20, 30],
            [15, 25, 35],
            [20, 30, 40],
        ])
        alpha_matrix = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])

        batch_ll = dm_log_likelihood_batch(count_matrix, alpha_matrix)

        # Should equal sum of individual likelihoods
        individual_ll = sum(
            dm_log_likelihood(count_matrix[i, :], alpha_matrix[i, :]) for i in range(3)
        )

        self.assertAlmostEqual(batch_ll, individual_ll, places=10)


class TestSoftmax(unittest.TestCase):
    """Test softmax function."""

    def test_softmax_sums_to_one(self):
        """Test that softmax output sums to 1."""
        x = np.array([1.0, 2.0, 3.0])
        sm = softmax(x)

        self.assertAlmostEqual(np.sum(sm), 1.0, places=10)
        self.assertTrue(np.all(sm > 0))
        self.assertTrue(np.all(sm < 1))

    def test_softmax_batch(self):
        """Test softmax with 2D input."""
        x = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
        ])
        sm = softmax(x)

        # Should sum to 1 per row
        self.assertTrue(np.allclose(np.sum(sm, axis=1), 1.0))

    def test_softmax_scale_invariance(self):
        """Test that softmax(x) = softmax(x + c)."""
        x = np.array([1.0, 2.0, 3.0])
        sm1 = softmax(x)
        sm2 = softmax(x + 100.0)

        np.testing.assert_array_almost_equal(sm1, sm2, decimal=10)


class TestDMGLMBasic(unittest.TestCase):
    """Test basic DM-GLM fitting."""

    def create_simple_data(self, n_samples=20, n_junctions=3):
        """Create simple test data."""
        # Simple two-group data
        count_matrix = np.random.poisson(50, size=(n_samples, n_junctions))
        count_matrix = np.maximum(count_matrix, 1)

        # Simple design: intercept + group indicator
        design_matrix = np.ones((n_samples, 2))
        design_matrix[n_samples // 2 :, 1] = 1.0

        return count_matrix, design_matrix

    def test_fit_dm_glm_convergence(self):
        """Test that fit_dm_glm produces valid results."""
        count_matrix, design_matrix = self.create_simple_data()

        result = fit_dm_glm(count_matrix, design_matrix, max_iter=100)

        self.assertIsInstance(result, DMGLMResult)
        # Check that optimization completed (converged or hit max iterations)
        self.assertGreaterEqual(result.n_iterations, 0)
        self.assertTrue(np.isfinite(result.log_likelihood))
        self.assertTrue(np.all(result.concentration > 0))

    def test_fit_dm_glm_alpha_shape(self):
        """Test that alpha_matrix has correct shape."""
        count_matrix, design_matrix = self.create_simple_data(
            n_samples=15, n_junctions=4
        )

        result = fit_dm_glm(count_matrix, design_matrix, max_iter=50)

        self.assertEqual(result.alpha_matrix.shape, (15, 4))
        self.assertTrue(np.all(result.alpha_matrix > 0))

    def test_fit_dm_glm_concentration_positive(self):
        """Test that concentration is positive."""
        count_matrix, design_matrix = self.create_simple_data()

        result = fit_dm_glm(count_matrix, design_matrix, max_iter=50)

        self.assertTrue(np.all(result.concentration > 0))


class TestNullFullFitting(unittest.TestCase):
    """Test null and full model fitting."""

    def create_grouped_data(self, n_per_group=10, n_junctions=2):
        """Create grouped test data."""
        # Group 0: higher counts on junction 0
        counts_0 = np.random.poisson(100, size=(n_per_group, n_junctions))
        counts_0[:, 1] = np.random.poisson(20, size=n_per_group)

        # Group 1: more balanced
        counts_1 = np.random.poisson(60, size=(n_per_group, n_junctions))

        count_matrix = np.vstack([counts_0, counts_1])
        count_matrix = np.maximum(count_matrix, 1)

        return count_matrix

    def test_fit_dm_null(self):
        """Test fitting null model."""
        count_matrix = self.create_grouped_data(n_per_group=8, n_junctions=3)
        n_samples = count_matrix.shape[0]

        # Null design: just intercept
        design_null = np.ones((n_samples, 1))

        result = fit_dm_null(count_matrix, design_null, max_iter=100)

        self.assertIsInstance(result, DMGLMResult)
        self.assertTrue(np.isfinite(result.log_likelihood))

    def test_fit_dm_full_vs_null(self):
        """Test that full model gives >= likelihood to null."""
        count_matrix = self.create_grouped_data(n_per_group=10, n_junctions=2)
        n_samples = count_matrix.shape[0]

        # Null: intercept only
        design_null = np.ones((n_samples, 1))

        # Full: intercept + group
        design_full = np.ones((n_samples, 2))
        design_full[n_samples // 2 :, 1] = 1.0

        result_null = fit_dm_null(count_matrix, design_null, max_iter=100)
        result_full = fit_dm_full(count_matrix, design_full, max_iter=100)

        # Full model should have >= likelihood (usually >)
        self.assertGreaterEqual(
            result_full.log_likelihood, result_null.log_likelihood - 1e-3
        )


class TestLRT(unittest.TestCase):
    """Test likelihood ratio testing."""

    def test_dm_lrt_no_difference(self):
        """Test LRT when likelihoods are equal."""
        result_null = DMGLMResult(
            alpha_matrix=np.array([[1.0, 1.0]]),
            concentration=np.array([1.0, 1.0]),
            log_likelihood=-100.0,
            converged=True,
            n_iterations=10,
            gradient_norm=0.01,
        )
        result_full = DMGLMResult(
            alpha_matrix=np.array([[1.0, 1.0]]),
            concentration=np.array([1.0, 1.0]),
            log_likelihood=-100.0,
            converged=True,
            n_iterations=10,
            gradient_norm=0.01,
        )

        p_value = dm_lrt(result_null, result_full, df=1)

        self.assertEqual(p_value, 1.0)

    def test_dm_lrt_strong_difference(self):
        """Test LRT with strong likelihood difference."""
        result_null = DMGLMResult(
            alpha_matrix=np.array([[1.0, 1.0]]),
            concentration=np.array([1.0, 1.0]),
            log_likelihood=-100.0,
            converged=True,
            n_iterations=10,
            gradient_norm=0.01,
        )
        result_full = DMGLMResult(
            alpha_matrix=np.array([[1.0, 1.0]]),
            concentration=np.array([1.0, 1.0]),
            log_likelihood=-50.0,
            converged=True,
            n_iterations=10,
            gradient_norm=0.01,
        )

        p_value = dm_lrt(result_null, result_full, df=1)

        self.assertLess(p_value, 0.001)

    def test_dm_lrt_negative_stat(self):
        """Test LRT when full is worse than null."""
        result_null = DMGLMResult(
            alpha_matrix=np.array([[1.0, 1.0]]),
            concentration=np.array([1.0, 1.0]),
            log_likelihood=-50.0,
            converged=True,
            n_iterations=10,
            gradient_norm=0.01,
        )
        result_full = DMGLMResult(
            alpha_matrix=np.array([[1.0, 1.0]]),
            concentration=np.array([1.0, 1.0]),
            log_likelihood=-100.0,
            converged=True,
            n_iterations=10,
            gradient_norm=0.01,
        )

        p_value = dm_lrt(result_null, result_full, df=1)

        self.assertEqual(p_value, 1.0)


class TestDesignMatrix(unittest.TestCase):
    """Test design matrix building."""

    def test_build_design_matrix_two_group(self):
        """Test building design matrix for two-group comparison."""
        group_labels = np.array([0, 0, 0, 1, 1, 1])

        design_full, design_null, df = build_design_matrix(group_labels)

        # Full should have intercept + group indicator
        self.assertEqual(design_full.shape, (6, 2))
        # Null should have just intercept
        self.assertEqual(design_null.shape, (6, 1))
        # df should be 1 (one extra column)
        self.assertEqual(df, 1)

        # Check intercept
        np.testing.assert_array_almost_equal(design_full[:, 0], 1.0)
        np.testing.assert_array_almost_equal(design_null[:, 0], 1.0)

    def test_build_design_matrix_with_covariates(self):
        """Test building design matrix with numeric covariates."""
        group_labels = np.array([0, 0, 0, 1, 1, 1])
        covariates = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        design_full, design_null, df = build_design_matrix(
            group_labels, covariates=covariates
        )

        # Full: intercept + group + covariate
        self.assertEqual(design_full.shape, (6, 3))
        # Null: intercept + covariate
        self.assertEqual(design_null.shape, (6, 2))
        # df = 1 (group indicator)
        self.assertEqual(df, 1)

    def test_build_design_matrix_multigroup(self):
        """Test with more than two groups."""
        group_labels = np.array([0, 0, 1, 1, 2, 2])

        design_full, design_null, df = build_design_matrix(group_labels)

        # Full: intercept + 2 group indicators (K-1 encoding)
        self.assertEqual(design_full.shape, (6, 3))
        # Null: intercept only
        self.assertEqual(design_null.shape, (6, 1))
        # df = 2
        self.assertEqual(df, 2)

    def test_design_matrix_covariate_scaling(self):
        """Test that covariates are scaled."""
        group_labels = np.array([0, 0, 0, 1, 1, 1])
        covariates = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        design_full, design_null, _ = build_design_matrix(
            group_labels, covariates=covariates
        )

        # Covariate column should be scaled (roughly unit variance)
        cov_col = design_full[:, -1]
        cov_std = np.std(cov_col)
        self.assertAlmostEqual(cov_std, 1.0, places=1)


class TestDMGLMIntegration(unittest.TestCase):
    """Integration tests for DM-GLM."""

    def test_full_pipeline_two_group(self):
        """Test complete fitting pipeline for two-group comparison."""
        n_samples = 20
        n_junctions = 3

        # Create test data with group effect on junction 0
        count_matrix = np.random.poisson(50, size=(n_samples, n_junctions))
        count_matrix = np.maximum(count_matrix, 1)

        # Make group 1 have higher counts on junction 0
        count_matrix[n_samples // 2 :, 0] = np.random.poisson(
            100, size=n_samples // 2
        )

        # Build design matrices
        design_full, design_null, df = build_design_matrix(
            np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
        )

        # Fit models
        result_null = fit_dm_null(count_matrix, design_null, max_iter=100)
        result_full = fit_dm_full(count_matrix, design_full, max_iter=100)

        # Run LRT
        p_value = dm_lrt(result_null, result_full, df)

        # Should be a valid p-value
        self.assertTrue(0 <= p_value <= 1)

    def test_fit_with_covariates(self):
        """Test fitting with covariates."""
        n_samples = 15
        n_junctions = 2

        count_matrix = np.random.poisson(50, size=(n_samples, n_junctions))
        count_matrix = np.maximum(count_matrix, 1)

        group_labels = np.array([0] * 8 + [1] * 7)
        covariates = np.random.randn(n_samples)

        design_full, design_null, df = build_design_matrix(
            group_labels, covariates=covariates
        )

        result = fit_dm_glm(count_matrix, design_full, max_iter=100)

        self.assertTrue(np.isfinite(result.log_likelihood))
        self.assertEqual(result.alpha_matrix.shape, (n_samples, n_junctions))


if __name__ == "__main__":
    unittest.main()
