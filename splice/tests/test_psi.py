"""
Test suite for Module 15: core/psi.py

Tests per-sample PSI quantification with bootstrap confidence intervals.
"""

import unittest

import numpy as np

from splicekit.core.evidence import ModuleEvidence
from splicekit.core.psi import ModulePSI, quantify_psi
from splicekit.core.splicegraph import SplicingModule
from splicekit.utils.genomic import Junction


class TestModulePSIDataclass(unittest.TestCase):
    """Test ModulePSI dataclass."""

    def test_module_psi_creation(self):
        """Test basic ModulePSI creation."""
        module_id = "module_1"
        psi_matrix = np.array([[0.3, 0.4], [0.7, 0.6]])
        ci_low = np.array([[0.2, 0.3], [0.6, 0.5]])
        ci_high = np.array([[0.4, 0.5], [0.8, 0.7]])
        bootstrap_psi = np.random.rand(30, 2, 2)
        total_counts = np.array([100, 150])
        effective_n = np.array([95.0, 140.0])

        module_psi = ModulePSI(
            module_id=module_id,
            psi_matrix=psi_matrix,
            ci_low_matrix=ci_low,
            ci_high_matrix=ci_high,
            bootstrap_psi=bootstrap_psi,
            total_counts=total_counts,
            effective_n=effective_n,
        )

        self.assertEqual(module_psi.module_id, "module_1")
        self.assertEqual(module_psi.psi_matrix.shape, (2, 2))
        self.assertEqual(module_psi.ci_low_matrix.shape, (2, 2))
        self.assertEqual(module_psi.ci_high_matrix.shape, (2, 2))
        self.assertEqual(module_psi.bootstrap_psi.shape, (30, 2, 2))
        self.assertEqual(len(module_psi.total_counts), 2)
        self.assertEqual(len(module_psi.effective_n), 2)

    def test_module_psi_frozen(self):
        """Test that ModulePSI is frozen (immutable)."""
        module_psi = ModulePSI(
            module_id="test",
            psi_matrix=np.array([[0.5]]),
            ci_low_matrix=np.array([[0.4]]),
            ci_high_matrix=np.array([[0.6]]),
            bootstrap_psi=np.array([[[0.5]]]),
            total_counts=np.array([100]),
            effective_n=np.array([95.0]),
        )

        with self.assertRaises(AttributeError):
            module_psi.module_id = "new_id"


class TestQuantifyPSIBasic(unittest.TestCase):
    """Test quantify_psi function with basic inputs."""

    def create_test_module_evidence(
        self,
        module_id: str = "test_module",
        n_junctions: int = 2,
        n_samples: int = 3,
    ) -> ModuleEvidence:
        """Create a test ModuleEvidence object."""
        # Create a dummy SplicingModule
        junctions = [
            Junction(chrom="chr1", start=100, end=200, strand="+"),
            Junction(chrom="chr1", start=200, end=300, strand="+"),
        ][:n_junctions]

        module = SplicingModule(
            module_id=module_id,
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

        # Create evidence matrices dynamically
        junction_count_matrix = np.random.poisson(
            150, size=(n_junctions, n_samples)
        ).astype(int)
        junction_count_matrix = np.maximum(junction_count_matrix, 10)  # Ensure minimum counts

        junction_weighted_matrix = junction_count_matrix.astype(float) * 0.95
        junction_mapq_matrix = np.full((n_junctions, n_samples), 30.0)

        junction_effective_lengths = np.ones(n_junctions) * 100.0
        normalized_count_matrix = junction_count_matrix.astype(float)
        total_counts = np.sum(junction_count_matrix, axis=0)
        total_weighted = np.sum(junction_weighted_matrix, axis=0)

        junction_confidence = np.ones(n_junctions)
        is_annotated = np.zeros(n_junctions, dtype=bool)

        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=junction_count_matrix,
            junction_weighted_matrix=junction_weighted_matrix,
            junction_mapq_matrix=junction_mapq_matrix,
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=junction_effective_lengths,
            normalized_count_matrix=normalized_count_matrix,
            total_counts=total_counts,
            total_weighted=total_weighted,
            junction_confidence=junction_confidence,
            is_annotated=is_annotated,
        )

        return evidence

    def test_quantify_psi_shape(self):
        """Test that quantify_psi returns correct shapes."""
        evidence = self.create_test_module_evidence(n_junctions=3, n_samples=5)
        psi_list = quantify_psi([evidence], n_bootstraps=10, seed=42)

        self.assertEqual(len(psi_list), 1)
        module_psi = psi_list[0]

        self.assertEqual(module_psi.psi_matrix.shape, (3, 5))
        self.assertEqual(module_psi.ci_low_matrix.shape, (3, 5))
        self.assertEqual(module_psi.ci_high_matrix.shape, (3, 5))
        self.assertEqual(module_psi.bootstrap_psi.shape, (10, 3, 5))
        self.assertEqual(len(module_psi.total_counts), 5)
        self.assertEqual(len(module_psi.effective_n), 5)

    def test_quantify_psi_sums_to_one(self):
        """Test that PSI values sum to 1 per sample."""
        evidence = self.create_test_module_evidence(n_junctions=2, n_samples=4)
        psi_list = quantify_psi([evidence], n_bootstraps=15, seed=42)

        module_psi = psi_list[0]
        psi_sums = np.sum(module_psi.psi_matrix, axis=0)

        # PSI should sum to 1 per sample
        np.testing.assert_array_almost_equal(psi_sums, 1.0, decimal=5)

    def test_quantify_psi_values_in_range(self):
        """Test that PSI values are in [0, 1]."""
        evidence = self.create_test_module_evidence(n_junctions=3, n_samples=3)
        psi_list = quantify_psi([evidence], n_bootstraps=20, seed=42)

        module_psi = psi_list[0]

        self.assertTrue(np.all(module_psi.psi_matrix >= 0))
        self.assertTrue(np.all(module_psi.psi_matrix <= 1))

    def test_quantify_psi_ci_bounds(self):
        """Test that CI bounds are valid."""
        evidence = self.create_test_module_evidence(n_junctions=2, n_samples=3)
        psi_list = quantify_psi([evidence], n_bootstraps=30, seed=42)

        module_psi = psi_list[0]

        # CI low should be <= CI high
        self.assertTrue(np.all(module_psi.ci_low_matrix <= module_psi.ci_high_matrix))

        # CIs should be in [0, 1]
        self.assertTrue(np.all(module_psi.ci_low_matrix >= 0))
        self.assertTrue(np.all(module_psi.ci_high_matrix <= 1))

    def test_quantify_psi_bootstrap_coverage(self):
        """Test that point estimate is within bootstrap CI."""
        evidence = self.create_test_module_evidence(n_junctions=2, n_samples=3)
        psi_list = quantify_psi([evidence], n_bootstraps=30, seed=42)

        module_psi = psi_list[0]

        # Point estimate should typically be within CI
        # (May occasionally fail for extreme quantiles, but should usually pass)
        within_ci = (
            (module_psi.psi_matrix >= module_psi.ci_low_matrix - 1e-6)
            & (module_psi.psi_matrix <= module_psi.ci_high_matrix + 1e-6)
        )
        # Most (but not all) point estimates should be within CI
        self.assertGreater(np.mean(within_ci), 0.5)

    def test_quantify_psi_reproducibility(self):
        """Test that same seed produces same results."""
        evidence = self.create_test_module_evidence(n_junctions=2, n_samples=3)

        psi_list1 = quantify_psi([evidence], n_bootstraps=10, seed=42)
        psi_list2 = quantify_psi([evidence], n_bootstraps=10, seed=42)

        np.testing.assert_array_equal(
            psi_list1[0].psi_matrix, psi_list2[0].psi_matrix
        )
        np.testing.assert_array_equal(
            psi_list1[0].ci_low_matrix, psi_list2[0].ci_low_matrix
        )
        np.testing.assert_array_equal(
            psi_list1[0].ci_high_matrix, psi_list2[0].ci_high_matrix
        )

    def test_quantify_psi_different_seeds(self):
        """Test that different seeds produce different bootstrap results."""
        evidence = self.create_test_module_evidence(n_junctions=2, n_samples=3)

        psi_list1 = quantify_psi([evidence], n_bootstraps=10, seed=42)
        psi_list2 = quantify_psi([evidence], n_bootstraps=10, seed=123)

        # Bootstrap results should differ (with very high probability)
        self.assertFalse(
            np.array_equal(psi_list1[0].bootstrap_psi, psi_list2[0].bootstrap_psi)
        )


class TestQuantifyPSIMultiple(unittest.TestCase):
    """Test quantify_psi with multiple modules."""

    def create_test_module_evidence(
        self,
        module_id: str,
        n_junctions: int,
        n_samples: int,
    ) -> ModuleEvidence:
        """Create a test ModuleEvidence object."""
        junctions = [
            Junction(chrom="chr1", start=100 + i * 100, end=200 + i * 100, strand="+")
            for i in range(n_junctions)
        ]

        module = SplicingModule(
            module_id=module_id,
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

        # Create evidence matrices with different values per module
        base_counts = 50 * (int(module_id.split("_")[1]) + 1)
        junction_count_matrix = np.random.poisson(
            base_counts, size=(n_junctions, n_samples)
        )
        junction_count_matrix = np.maximum(junction_count_matrix, 1)

        junction_weighted_matrix = junction_count_matrix.astype(float) * 0.9
        junction_mapq_matrix = np.full((n_junctions, n_samples), 30.0)
        junction_effective_lengths = np.ones(n_junctions) * 100.0
        normalized_count_matrix = junction_count_matrix.astype(float)
        total_counts = np.sum(junction_count_matrix, axis=0)
        total_weighted = np.sum(junction_weighted_matrix, axis=0)
        junction_confidence = np.ones(n_junctions)
        is_annotated = np.zeros(n_junctions, dtype=bool)

        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=junction_count_matrix,
            junction_weighted_matrix=junction_weighted_matrix,
            junction_mapq_matrix=junction_mapq_matrix,
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=junction_effective_lengths,
            normalized_count_matrix=normalized_count_matrix,
            total_counts=total_counts,
            total_weighted=total_weighted,
            junction_confidence=junction_confidence,
            is_annotated=is_annotated,
        )

        return evidence

    def test_quantify_psi_multiple_modules(self):
        """Test quantify_psi with multiple modules."""
        evidence_list = [
            self.create_test_module_evidence("module_0", 2, 4),
            self.create_test_module_evidence("module_1", 3, 4),
            self.create_test_module_evidence("module_2", 2, 4),
        ]

        psi_list = quantify_psi(evidence_list, n_bootstraps=15, seed=42)

        self.assertEqual(len(psi_list), 3)

        # Check each module's PSI
        for i, module_psi in enumerate(psi_list):
            n_junctions = evidence_list[i].junction_count_matrix.shape[0]
            self.assertEqual(module_psi.psi_matrix.shape, (n_junctions, 4))
            self.assertEqual(module_psi.module_id, f"module_{i}")

    def test_quantify_psi_consistency_across_modules(self):
        """Test that all modules are processed consistently."""
        evidence_list = [
            self.create_test_module_evidence("module_0", 2, 5),
            self.create_test_module_evidence("module_1", 2, 5),
        ]

        psi_list = quantify_psi(evidence_list, n_bootstraps=20, seed=42)

        # Both modules should have same number of samples
        self.assertEqual(psi_list[0].psi_matrix.shape[1], 5)
        self.assertEqual(psi_list[1].psi_matrix.shape[1], 5)

        # Both modules should have same total_counts length
        self.assertEqual(len(psi_list[0].total_counts), 5)
        self.assertEqual(len(psi_list[1].total_counts), 5)


class TestQuantifyPSIEdgeCases(unittest.TestCase):
    """Test quantify_psi with edge cases."""

    def test_quantify_psi_single_junction(self):
        """Test with single junction (PSI should be 1.0)."""
        junctions = [Junction(chrom="chr1", start=100, end=200, strand="+")]
        module = SplicingModule(
            module_id="single",
            gene_id="gene1",
            gene_name="GENE1",
            chrom="chr1",
            strand="+",
            start=100,
            end=200,
            junctions=junctions,
            junction_indices=[0],
            n_connections=1,
        )

        junction_count_matrix = np.array([[100, 200, 150]], dtype=int)
        junction_weighted_matrix = junction_count_matrix.astype(float) * 0.9
        junction_mapq_matrix = np.array([[30.0, 30.0, 30.0]])
        junction_effective_lengths = np.array([100.0])
        normalized_count_matrix = junction_count_matrix.astype(float)
        total_counts = np.sum(junction_count_matrix, axis=0)
        total_weighted = np.sum(junction_weighted_matrix, axis=0)
        junction_confidence = np.array([1.0])
        is_annotated = np.array([False])

        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=junction_count_matrix,
            junction_weighted_matrix=junction_weighted_matrix,
            junction_mapq_matrix=junction_mapq_matrix,
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=junction_effective_lengths,
            normalized_count_matrix=normalized_count_matrix,
            total_counts=total_counts,
            total_weighted=total_weighted,
            junction_confidence=junction_confidence,
            is_annotated=is_annotated,
        )

        psi_list = quantify_psi([evidence], n_bootstraps=10, seed=42)
        module_psi = psi_list[0]

        # Single junction should have PSI = 1.0 in all samples
        np.testing.assert_array_almost_equal(module_psi.psi_matrix, 1.0)

    def test_quantify_psi_balanced_junctions(self):
        """Test with equally-sized junctions."""
        junctions = [
            Junction(chrom="chr1", start=100, end=200, strand="+"),
            Junction(chrom="chr1", start=200, end=300, strand="+"),
        ]
        module = SplicingModule(
            module_id="balanced",
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

        # Equal counts for both junctions -> PSI = [0.5, 0.5]
        junction_count_matrix = np.array([
            [100, 100],
            [100, 100],
        ], dtype=int)
        junction_weighted_matrix = junction_count_matrix.astype(float) * 0.9
        junction_mapq_matrix = np.array([
            [30.0, 30.0],
            [30.0, 30.0],
        ])
        junction_effective_lengths = np.array([100.0, 100.0])
        normalized_count_matrix = junction_count_matrix.astype(float)
        total_counts = np.array([200, 200])
        total_weighted = np.sum(junction_weighted_matrix, axis=0)
        junction_confidence = np.array([1.0, 1.0])
        is_annotated = np.array([False, False])

        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=junction_count_matrix,
            junction_weighted_matrix=junction_weighted_matrix,
            junction_mapq_matrix=junction_mapq_matrix,
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=junction_effective_lengths,
            normalized_count_matrix=normalized_count_matrix,
            total_counts=total_counts,
            total_weighted=total_weighted,
            junction_confidence=junction_confidence,
            is_annotated=is_annotated,
        )

        psi_list = quantify_psi([evidence], n_bootstraps=20, seed=42)
        module_psi = psi_list[0]

        # Both junctions should have PSI = 0.5
        np.testing.assert_array_almost_equal(module_psi.psi_matrix, 0.5, decimal=5)


class TestQuantifyPSIIntegration(unittest.TestCase):
    """Integration tests for quantify_psi."""

    def test_quantify_psi_ci_width_with_bootstrap_replicates(self):
        """Test that CI width correlates with variability."""
        junctions = [
            Junction(chrom="chr1", start=100, end=200, strand="+"),
            Junction(chrom="chr1", start=200, end=300, strand="+"),
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

        junction_count_matrix = np.array([
            [100, 200, 150],
            [300, 400, 350],
        ], dtype=int)
        junction_weighted_matrix = junction_count_matrix.astype(float) * 0.9
        junction_mapq_matrix = np.array([
            [30.0, 30.0, 30.0],
            [30.0, 30.0, 30.0],
        ])
        junction_effective_lengths = np.array([100.0, 100.0])
        normalized_count_matrix = junction_count_matrix.astype(float)
        total_counts = np.sum(junction_count_matrix, axis=0)
        total_weighted = np.sum(junction_weighted_matrix, axis=0)
        junction_confidence = np.array([1.0, 1.0])
        is_annotated = np.array([False, False])

        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=junction_count_matrix,
            junction_weighted_matrix=junction_weighted_matrix,
            junction_mapq_matrix=junction_mapq_matrix,
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=junction_effective_lengths,
            normalized_count_matrix=normalized_count_matrix,
            total_counts=total_counts,
            total_weighted=total_weighted,
            junction_confidence=junction_confidence,
            is_annotated=is_annotated,
        )

        psi_list = quantify_psi([evidence], n_bootstraps=50, seed=42)
        module_psi = psi_list[0]

        # CI widths should be reasonable (> 0 and < 1)
        ci_widths = module_psi.ci_high_matrix - module_psi.ci_low_matrix
        self.assertTrue(np.all(ci_widths > 0))
        self.assertTrue(np.all(ci_widths < 1))

    def test_quantify_psi_effective_n(self):
        """Test that effective_n matches total_weighted."""
        junctions = [Junction(chrom="chr1", start=100, end=200, strand="+")]
        module = SplicingModule(
            module_id="test",
            gene_id="gene1",
            gene_name="GENE1",
            chrom="chr1",
            strand="+",
            start=100,
            end=200,
            junctions=junctions,
            junction_indices=[0],
            n_connections=1,
        )

        junction_count_matrix = np.array([[100, 200]], dtype=int)
        junction_weighted_matrix = np.array([[95.0, 190.0]])
        junction_mapq_matrix = np.array([[30.0, 30.0]])
        junction_effective_lengths = np.array([100.0])
        normalized_count_matrix = junction_count_matrix.astype(float)
        total_counts = np.array([100, 200])
        total_weighted = np.array([95.0, 190.0])
        junction_confidence = np.array([1.0])
        is_annotated = np.array([False])

        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=junction_count_matrix,
            junction_weighted_matrix=junction_weighted_matrix,
            junction_mapq_matrix=junction_mapq_matrix,
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=junction_effective_lengths,
            normalized_count_matrix=normalized_count_matrix,
            total_counts=total_counts,
            total_weighted=total_weighted,
            junction_confidence=junction_confidence,
            is_annotated=is_annotated,
        )

        psi_list = quantify_psi([evidence], n_bootstraps=20, seed=42)
        module_psi = psi_list[0]

        # effective_n should match total_weighted
        np.testing.assert_array_almost_equal(
            module_psi.effective_n, total_weighted, decimal=5
        )


if __name__ == "__main__":
    unittest.main()
