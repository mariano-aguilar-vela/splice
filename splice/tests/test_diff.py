"""
Test suite for Module 17: core/diff.py

Tests differential splicing testing with DM-GLM and covariate support.
"""

import unittest

import numpy as np

from splice.core.diff import DiffResult, differential_splicing
from splice.core.evidence import ModuleEvidence
from splice.core.psi import ModulePSI
from splice.core.splicegraph import SplicingModule
from splice.utils.genomic import Junction


class TestDiffResultDataclass(unittest.TestCase):
    """Test DiffResult dataclass."""

    def test_diff_result_creation(self):
        """Test basic DiffResult creation."""
        result = DiffResult(
            module_id="mod_1",
            gene_id="gene_1",
            gene_name="GENE1",
            chrom="chr1",
            strand="+",
            event_type="SE",
            n_junctions=2,
            junction_coords=["chr1:100-200:+", "chr1:200-300:+"],
            junction_confidence=[0.8, 0.9],
            is_annotated=[True, False],
            psi_group1=np.array([0.3, 0.7]),
            psi_group2=np.array([0.6, 0.4]),
            delta_psi=np.array([-0.3, 0.3]),
            max_abs_delta_psi=0.3,
            delta_psi_ci_low=np.array([-0.5, 0.1]),
            delta_psi_ci_high=np.array([-0.1, 0.5]),
            log_likelihood_null=-100.0,
            log_likelihood_full=-80.0,
            degrees_of_freedom=1,
            p_value=0.05,
            fdr=0.1,
            null_converged=True,
            full_converged=True,
            null_refit_used=False,
            null_iterations=50,
            full_iterations=60,
            null_gradient_norm=1e-5,
            full_gradient_norm=1e-5,
        )

        self.assertEqual(result.module_id, "mod_1")
        self.assertEqual(result.event_type, "SE")
        self.assertAlmostEqual(result.p_value, 0.05)
        self.assertEqual(result.n_junctions, 2)

    def test_diff_result_frozen(self):
        """Test that DiffResult is frozen."""
        result = DiffResult(
            module_id="test",
            gene_id="g1",
            gene_name="G1",
            chrom="chr1",
            strand="+",
            event_type="SE",
            n_junctions=1,
            junction_coords=["chr1:100-200:+"],
            junction_confidence=[0.5],
            is_annotated=[False],
            psi_group1=np.array([0.5]),
            psi_group2=np.array([0.5]),
            delta_psi=np.array([0.0]),
            max_abs_delta_psi=0.0,
            delta_psi_ci_low=np.array([-0.1]),
            delta_psi_ci_high=np.array([0.1]),
            log_likelihood_null=-10.0,
            log_likelihood_full=-10.0,
            degrees_of_freedom=1,
            p_value=1.0,
            fdr=1.0,
            null_converged=True,
            full_converged=True,
            null_refit_used=False,
            null_iterations=1,
            full_iterations=1,
            null_gradient_norm=0.01,
            full_gradient_norm=0.01,
        )

        with self.assertRaises(AttributeError):
            result.p_value = 0.01


class TestDifferentialSplicingBasic(unittest.TestCase):
    """Test basic differential splicing testing."""

    def create_test_data(self, n_samples=10, n_junctions=2):
        """Create test ModuleEvidence and ModulePSI."""
        # Create SplicingModule
        junctions = [
            Junction(chrom="chr1", start=100 + i * 100, end=200 + i * 100, strand="+")
            for i in range(n_junctions)
        ]
        module = SplicingModule(
            module_id="test_module",
            gene_id="gene1",
            gene_name="GENE1",
            chrom="chr1",
            strand="+",
            start=100,
            end=300 + n_junctions * 100,
            junctions=junctions,
            junction_indices=list(range(n_junctions)),
            n_connections=n_junctions,
        )

        # Create ModuleEvidence
        count_matrix = np.random.poisson(50, size=(n_junctions, n_samples))
        count_matrix = np.maximum(count_matrix, 1)

        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=count_matrix,
            junction_weighted_matrix=count_matrix.astype(float) * 0.95,
            junction_mapq_matrix=np.full((n_junctions, n_samples), 30.0),
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=np.ones(n_junctions) * 100.0,
            normalized_count_matrix=count_matrix.astype(float),
            total_counts=np.sum(count_matrix, axis=0),
            total_weighted=np.sum(count_matrix.astype(float) * 0.95, axis=0),
            junction_confidence=np.ones(n_junctions),
            is_annotated=np.zeros(n_junctions, dtype=bool),
        )

        # Create ModulePSI (simplified)
        normalized_counts = count_matrix.astype(float)
        col_sums = np.sum(normalized_counts, axis=0)
        psi_matrix = normalized_counts / col_sums[np.newaxis, :]

        psi = ModulePSI(
            module_id="test_module",
            psi_matrix=psi_matrix,
            ci_low_matrix=psi_matrix * 0.8,
            ci_high_matrix=psi_matrix * 1.2,
            bootstrap_psi=np.random.rand(30, n_junctions, n_samples) * 0.5
            + psi_matrix[np.newaxis, :, :] * 0.5,
            total_counts=np.sum(count_matrix, axis=0),
            effective_n=np.sum(count_matrix.astype(float) * 0.95, axis=0),
        )

        return evidence, psi

    def test_diff_splicing_two_group(self):
        """Test differential splicing with two groups."""
        evidence, psi = self.create_test_data(n_samples=12, n_junctions=2)

        group_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        results = differential_splicing(
            [evidence], [psi], group_labels, min_total_reads_per_group=10
        )

        self.assertEqual(len(results), 1)
        result = results[0]

        # Check result properties
        self.assertEqual(result.module_id, "test_module")
        self.assertEqual(result.n_junctions, 2)
        self.assertTrue(0 <= result.p_value <= 1)
        self.assertTrue(0 <= result.fdr <= 1)
        self.assertEqual(result.degrees_of_freedom, 1)

    def test_diff_splicing_psi_shapes(self):
        """Test that PSI arrays have correct shapes."""
        evidence, psi = self.create_test_data(n_samples=10, n_junctions=3)

        group_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        results = differential_splicing(
            [evidence], [psi], group_labels, min_total_reads_per_group=5
        )

        result = results[0]

        self.assertEqual(result.psi_group1.shape, (3,))
        self.assertEqual(result.psi_group2.shape, (3,))
        self.assertEqual(result.delta_psi.shape, (3,))
        self.assertEqual(result.delta_psi_ci_low.shape, (3,))
        self.assertEqual(result.delta_psi_ci_high.shape, (3,))

    def test_diff_splicing_delta_psi_bounds(self):
        """Test that delta-PSI is in valid range."""
        evidence, psi = self.create_test_data(n_samples=10, n_junctions=2)

        group_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        results = differential_splicing(
            [evidence], [psi], group_labels, min_total_reads_per_group=5
        )

        result = results[0]

        # Delta-PSI should be in [-1, 1]
        self.assertTrue(np.all(result.delta_psi >= -1))
        self.assertTrue(np.all(result.delta_psi <= 1))

        # CI bounds should be ordered
        self.assertTrue(np.all(result.delta_psi_ci_low <= result.delta_psi_ci_high))

    def test_diff_splicing_fdr_correction(self):
        """Test FDR correction on multiple modules."""
        # Create multiple modules
        evidence_list = []
        psi_list = []

        for i in range(3):
            evidence, psi = self.create_test_data(n_samples=10, n_junctions=2)
            evidence_list.append(evidence)
            psi_list.append(psi)

        group_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        results = differential_splicing(
            evidence_list, psi_list, group_labels, min_total_reads_per_group=5
        )

        # All FDR values should be in [0, 1]
        for result in results:
            self.assertTrue(0 <= result.fdr <= 1)
            # FDR >= p-value
            self.assertGreaterEqual(result.fdr, result.p_value)

    def test_diff_splicing_convergence(self):
        """Test convergence information in results."""
        evidence, psi = self.create_test_data(n_samples=10, n_junctions=2)

        group_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        results = differential_splicing(
            [evidence], [psi], group_labels, min_total_reads_per_group=5
        )

        result = results[0]

        # Check convergence flags
        self.assertIsInstance(result.null_converged, (bool, np.bool_))
        self.assertIsInstance(result.full_converged, (bool, np.bool_))
        self.assertGreaterEqual(result.null_iterations, 0)
        self.assertGreaterEqual(result.full_iterations, 0)
        self.assertGreaterEqual(result.null_gradient_norm, 0)
        self.assertGreaterEqual(result.full_gradient_norm, 0)


class TestDifferentialSplicingMultipleModules(unittest.TestCase):
    """Test with multiple modules."""

    def create_module_data(self, module_id, n_samples=12):
        """Create test data for one module."""
        junctions = [
            Junction(chrom="chr1", start=100 + i * 100, end=200 + i * 100, strand="+")
            for i in range(2)
        ]
        module = SplicingModule(
            module_id=module_id,
            gene_id=f"gene_{module_id}",
            gene_name=f"GENE_{module_id}",
            chrom="chr1",
            strand="+",
            start=100,
            end=300,
            junctions=junctions,
            junction_indices=[0, 1],
            n_connections=2,
        )

        count_matrix = np.random.poisson(50, size=(2, n_samples))
        count_matrix = np.maximum(count_matrix, 1)

        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=count_matrix,
            junction_weighted_matrix=count_matrix.astype(float) * 0.95,
            junction_mapq_matrix=np.full((2, n_samples), 30.0),
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=np.ones(2) * 100.0,
            normalized_count_matrix=count_matrix.astype(float),
            total_counts=np.sum(count_matrix, axis=0),
            total_weighted=np.sum(count_matrix.astype(float) * 0.95, axis=0),
            junction_confidence=np.ones(2),
            is_annotated=np.zeros(2, dtype=bool),
        )

        psi_matrix = count_matrix.astype(float) / np.sum(
            count_matrix, axis=0, keepdims=True
        )

        psi = ModulePSI(
            module_id=module_id,
            psi_matrix=psi_matrix,
            ci_low_matrix=psi_matrix * 0.8,
            ci_high_matrix=psi_matrix * 1.2,
            bootstrap_psi=np.random.rand(20, 2, n_samples) * 0.5
            + psi_matrix[np.newaxis, :, :] * 0.5,
            total_counts=np.sum(count_matrix, axis=0),
            effective_n=np.sum(count_matrix.astype(float) * 0.95, axis=0),
        )

        return evidence, psi

    def test_multiple_modules(self):
        """Test with multiple modules."""
        evidence_list = []
        psi_list = []

        for i in range(5):
            evidence, psi = self.create_module_data(f"mod_{i}", n_samples=12)
            evidence_list.append(evidence)
            psi_list.append(psi)

        group_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        results = differential_splicing(
            evidence_list, psi_list, group_labels, min_total_reads_per_group=10
        )

        # Should process all modules
        self.assertGreaterEqual(len(results), 1)
        self.assertLessEqual(len(results), 5)

        # All results should have valid p-values and FDR
        for result in results:
            self.assertTrue(0 <= result.p_value <= 1)
            self.assertTrue(0 <= result.fdr <= 1)


class TestDifferentialSplicingWithCovariates(unittest.TestCase):
    """Test with covariates."""

    def create_test_data(self, n_samples=12):
        """Create test data."""
        junctions = [
            Junction(chrom="chr1", start=100 + i * 100, end=200 + i * 100, strand="+")
            for i in range(2)
        ]
        module = SplicingModule(
            module_id="test",
            gene_id="gene1",
            gene_name="GENE1",
            chrom="chr1",
            strand="+",
            start=100,
            end=300,
            junctions=junctions,
            junction_indices=[0, 1],
            n_connections=2,
        )

        count_matrix = np.random.poisson(50, size=(2, n_samples))
        count_matrix = np.maximum(count_matrix, 1)

        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=count_matrix,
            junction_weighted_matrix=count_matrix.astype(float) * 0.95,
            junction_mapq_matrix=np.full((2, n_samples), 30.0),
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=np.ones(2) * 100.0,
            normalized_count_matrix=count_matrix.astype(float),
            total_counts=np.sum(count_matrix, axis=0),
            total_weighted=np.sum(count_matrix.astype(float) * 0.95, axis=0),
            junction_confidence=np.ones(2),
            is_annotated=np.zeros(2, dtype=bool),
        )

        psi_matrix = count_matrix.astype(float) / np.sum(
            count_matrix, axis=0, keepdims=True
        )

        psi = ModulePSI(
            module_id="test",
            psi_matrix=psi_matrix,
            ci_low_matrix=psi_matrix * 0.8,
            ci_high_matrix=psi_matrix * 1.2,
            bootstrap_psi=np.random.rand(20, 2, n_samples) * 0.5
            + psi_matrix[np.newaxis, :, :] * 0.5,
            total_counts=np.sum(count_matrix, axis=0),
            effective_n=np.sum(count_matrix.astype(float) * 0.95, axis=0),
        )

        return evidence, psi

    def test_with_numeric_covariate(self):
        """Test with numeric covariate."""
        evidence, psi = self.create_test_data(n_samples=12)

        group_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        covariates = np.random.randn(12)

        results = differential_splicing(
            [evidence],
            [psi],
            group_labels,
            covariates=covariates,
            min_total_reads_per_group=10,
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(0 <= results[0].p_value <= 1)


if __name__ == "__main__":
    unittest.main()
