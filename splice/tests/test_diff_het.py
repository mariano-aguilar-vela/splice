"""
Test suite for Module 18: core/diff_het.py

Tests heterogeneity-aware differential splicing testing.
"""

import unittest

import numpy as np

from splicekit.core.diff_het import HetResult, test_heterogeneous_splicing
from splicekit.core.psi import ModulePSI


class TestHetResultDataclass(unittest.TestCase):
    """Test HetResult dataclass."""

    def test_het_result_creation(self):
        """Test basic HetResult creation."""
        result = HetResult(
            module_id="mod_1",
            gene_id="gene_1",
            gene_name="GENE1",
            event_type="SE",
            n_junctions=2,
            sample_psi=np.array([0.2, 0.3, 0.4, 0.8, 0.85, 0.9]),
            group_labels=np.array([0, 0, 0, 1, 1, 1]),
            ttest_pvalue=0.01,
            mannwhitney_pvalue=0.02,
            within_group_variance=np.array([0.01, 0.005]),
            between_group_variance=0.25,
            heterogeneity_index=0.06,
            bimodal_pvalue=0.1,
            n_outlier_samples=1,
            fdr=0.05,
        )

        self.assertEqual(result.module_id, "mod_1")
        self.assertEqual(result.event_type, "SE")
        self.assertAlmostEqual(result.ttest_pvalue, 0.01)
        self.assertEqual(result.n_junctions, 2)

    def test_het_result_frozen(self):
        """Test that HetResult is frozen."""
        result = HetResult(
            module_id="test",
            gene_id="g1",
            gene_name="G1",
            event_type="SE",
            n_junctions=1,
            sample_psi=np.array([0.5, 0.5, 0.5, 0.5]),
            group_labels=np.array([0, 0, 1, 1]),
            ttest_pvalue=1.0,
            mannwhitney_pvalue=1.0,
            within_group_variance=np.array([0.0, 0.0]),
            between_group_variance=0.0,
            heterogeneity_index=0.0,
            bimodal_pvalue=1.0,
            n_outlier_samples=0,
            fdr=1.0,
        )

        with self.assertRaises(AttributeError):
            result.ttest_pvalue = 0.01


class TestHeterogeneousTestingBasic(unittest.TestCase):
    """Test basic heterogeneous splicing testing."""

    def create_test_module_psi(self, n_samples=10, n_junctions=2):
        """Create test ModulePSI with heterogeneous effect."""
        # Create PSI that shows heterogeneity: subsets within groups
        np.random.seed(42)

        psi_matrix = np.zeros((n_junctions, n_samples))

        # Group 0: mostly low PSI, some intermediate
        psi_matrix[0, :n_samples // 2] = np.random.uniform(0.1, 0.4, n_samples // 2)
        # Group 1: mostly high PSI, some intermediate
        psi_matrix[0, n_samples // 2 :] = np.random.uniform(0.6, 0.9, n_samples // 2)

        # Second junction: more uniform
        psi_matrix[1, :] = np.random.uniform(0.4, 0.6, n_samples)

        psi = ModulePSI(
            module_id="test_module",
            psi_matrix=psi_matrix,
            ci_low_matrix=psi_matrix * 0.8,
            ci_high_matrix=psi_matrix * 1.2,
            bootstrap_psi=np.random.rand(20, n_junctions, n_samples) * 0.5
            + psi_matrix[np.newaxis, :, :] * 0.5,
            total_counts=np.ones(n_samples) * 100,
            effective_n=np.ones(n_samples) * 95,
        )

        return psi

    def test_heterogeneous_testing_two_group(self):
        """Test heterogeneous splicing testing with two groups."""
        psi = self.create_test_module_psi(n_samples=12, n_junctions=2)

        group_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        results = test_heterogeneous_splicing([psi], group_labels)

        self.assertEqual(len(results), 1)
        result = results[0]

        # Check basic properties
        self.assertEqual(result.module_id, "test_module")
        self.assertEqual(result.n_junctions, 2)
        self.assertTrue(0 <= result.ttest_pvalue <= 1)
        self.assertTrue(0 <= result.mannwhitney_pvalue <= 1)
        self.assertTrue(0 <= result.fdr <= 1)

    def test_heterogeneous_psi_shape(self):
        """Test that sample PSI has correct shape."""
        psi = self.create_test_module_psi(n_samples=10, n_junctions=3)

        group_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        results = test_heterogeneous_splicing([psi], group_labels)

        result = results[0]

        self.assertEqual(result.sample_psi.shape, (10,))
        self.assertEqual(result.group_labels.shape, (10,))

    def test_heterogeneous_variance_metrics(self):
        """Test variance metrics are computed correctly."""
        psi = self.create_test_module_psi(n_samples=10, n_junctions=2)

        group_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        results = test_heterogeneous_splicing([psi], group_labels)

        result = results[0]

        # Within-group variance should be positive (or zero if no variance)
        self.assertTrue(np.all(result.within_group_variance >= 0))

        # Between-group variance should be non-negative
        self.assertGreaterEqual(result.between_group_variance, 0)

        # Heterogeneity index should be positive
        self.assertGreater(result.heterogeneity_index, 0)

    def test_heterogeneous_outlier_detection(self):
        """Test outlier detection."""
        psi = self.create_test_module_psi(n_samples=10, n_junctions=2)

        group_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        results = test_heterogeneous_splicing([psi], group_labels)

        result = results[0]

        # Number of outliers should be reasonable
        self.assertGreaterEqual(result.n_outlier_samples, 0)
        self.assertLessEqual(result.n_outlier_samples, 10)

    def test_heterogeneous_test_pvalues(self):
        """Test that p-values are valid."""
        psi = self.create_test_module_psi(n_samples=10, n_junctions=2)

        group_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        results = test_heterogeneous_splicing([psi], group_labels)

        result = results[0]

        # P-values should be in [0, 1]
        self.assertTrue(0 <= result.ttest_pvalue <= 1)
        self.assertTrue(0 <= result.mannwhitney_pvalue <= 1)
        self.assertTrue(0 <= result.bimodal_pvalue <= 1)
        self.assertTrue(0 <= result.fdr <= 1)


class TestHeterogeneousTestingMultiple(unittest.TestCase):
    """Test with multiple modules."""

    def create_psi_data(self, module_id, n_samples=12):
        """Create test PSI data."""
        np.random.seed(hash(module_id) % 2**32)

        psi_matrix = np.random.uniform(0.3, 0.7, size=(2, n_samples))

        # Add some group structure
        psi_matrix[0, :n_samples // 2] = np.random.uniform(0.2, 0.4, n_samples // 2)
        psi_matrix[0, n_samples // 2 :] = np.random.uniform(0.6, 0.8, n_samples // 2)

        return ModulePSI(
            module_id=module_id,
            psi_matrix=psi_matrix,
            ci_low_matrix=psi_matrix * 0.8,
            ci_high_matrix=psi_matrix * 1.2,
            bootstrap_psi=np.random.rand(15, 2, n_samples) * 0.4
            + psi_matrix[np.newaxis, :, :] * 0.6,
            total_counts=np.ones(n_samples) * 100,
            effective_n=np.ones(n_samples) * 95,
        )

    def test_multiple_modules(self):
        """Test with multiple modules."""
        psi_list = [
            self.create_psi_data(f"mod_{i}", n_samples=12) for i in range(5)
        ]

        group_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        results = test_heterogeneous_splicing(psi_list, group_labels)

        # Should process all modules
        self.assertEqual(len(results), 5)

        # All results should have valid p-values
        for result in results:
            self.assertTrue(0 <= result.ttest_pvalue <= 1)
            self.assertTrue(0 <= result.fdr <= 1)


class TestHeterogeneousTestingEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_minimum_group_size(self):
        """Test minimum group size filtering."""
        psi = ModulePSI(
            module_id="test",
            psi_matrix=np.random.uniform(0.3, 0.7, size=(2, 5)),
            ci_low_matrix=np.random.uniform(0.2, 0.6, size=(2, 5)),
            ci_high_matrix=np.random.uniform(0.4, 0.8, size=(2, 5)),
            bootstrap_psi=np.random.rand(10, 2, 5),
            total_counts=np.ones(5) * 100,
            effective_n=np.ones(5) * 95,
        )

        # Only 1 sample in group 1 - should fail min_samples check
        group_labels = np.array([0, 0, 0, 0, 1])

        results = test_heterogeneous_splicing(
            [psi], group_labels, min_samples_per_group=3
        )

        # Should return empty list due to insufficient samples
        self.assertEqual(len(results), 0)

    def test_uniform_psi(self):
        """Test with uniform PSI (no heterogeneity)."""
        psi = ModulePSI(
            module_id="uniform",
            psi_matrix=np.full((2, 10), 0.5),
            ci_low_matrix=np.full((2, 10), 0.4),
            ci_high_matrix=np.full((2, 10), 0.6),
            bootstrap_psi=np.full((15, 2, 10), 0.5),
            total_counts=np.ones(10) * 100,
            effective_n=np.ones(10) * 95,
        )

        group_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        results = test_heterogeneous_splicing([psi], group_labels)

        result = results[0]

        # With no heterogeneity (all constant), p-value will be NaN (undefined)
        # This is expected behavior when variance is zero
        self.assertTrue(np.isnan(result.ttest_pvalue) or result.ttest_pvalue > 0.05)

    def test_high_heterogeneity(self):
        """Test with high heterogeneity (bimodal)."""
        psi_matrix = np.zeros((2, 10))

        # Group 0: bimodal (subset with different PSI)
        psi_matrix[0, :3] = np.random.uniform(0.1, 0.2, 3)
        psi_matrix[0, 3:5] = np.random.uniform(0.8, 0.9, 2)
        psi_matrix[0, 5:] = np.random.uniform(0.4, 0.5, 5)

        # Group 1: different levels
        psi_matrix[1, :] = np.random.uniform(0.4, 0.6, 10)

        psi = ModulePSI(
            module_id="hetero",
            psi_matrix=psi_matrix,
            ci_low_matrix=psi_matrix * 0.8,
            ci_high_matrix=psi_matrix * 1.2,
            bootstrap_psi=np.random.rand(15, 2, 10) * 0.3
            + psi_matrix[np.newaxis, :, :] * 0.7,
            total_counts=np.ones(10) * 100,
            effective_n=np.ones(10) * 95,
        )

        group_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        results = test_heterogeneous_splicing([psi], group_labels)

        result = results[0]

        # High heterogeneity metrics should be computed
        self.assertTrue(0 <= result.ttest_pvalue <= 1)
        self.assertTrue(0 <= result.bimodal_pvalue <= 1)
        self.assertTrue(result.heterogeneity_index > 0)


class TestHeterogeneousFDRCorrection(unittest.TestCase):
    """Test FDR correction."""

    def create_module_psi(self, module_id, n_samples=12):
        """Create test PSI."""
        np.random.seed(hash(module_id) % 2**32)

        psi_matrix = np.random.uniform(0.3, 0.7, size=(2, n_samples))
        psi_matrix[0, :n_samples // 2] = np.random.uniform(0.2, 0.35, n_samples // 2)
        psi_matrix[0, n_samples // 2 :] = np.random.uniform(0.65, 0.8, n_samples // 2)

        return ModulePSI(
            module_id=module_id,
            psi_matrix=psi_matrix,
            ci_low_matrix=psi_matrix * 0.8,
            ci_high_matrix=psi_matrix * 1.2,
            bootstrap_psi=np.random.rand(10, 2, n_samples) * 0.4
            + psi_matrix[np.newaxis, :, :] * 0.6,
            total_counts=np.ones(n_samples) * 100,
            effective_n=np.ones(n_samples) * 95,
        )

    def test_fdr_correction(self):
        """Test FDR correction on multiple modules."""
        psi_list = [self.create_module_psi(f"mod_{i}") for i in range(5)]

        group_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        results = test_heterogeneous_splicing(psi_list, group_labels)

        # All FDR values should be in [0, 1]
        for result in results:
            self.assertTrue(0 <= result.fdr <= 1)
            # FDR >= p-value
            self.assertGreaterEqual(result.fdr, result.ttest_pvalue - 1e-10)


if __name__ == "__main__":
    unittest.main()
