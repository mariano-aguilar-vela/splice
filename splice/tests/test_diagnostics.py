"""
Test suite for Module 19: core/diagnostics.py

Tests diagnostic record computation and confidence tier assignment.
"""

import unittest

import numpy as np

from splice.core.diagnostics import EventDiagnostic, compute_diagnostics
from splice.core.diff import DiffResult
from splice.core.evidence import ModuleEvidence
from splice.core.psi import ModulePSI
from splice.core.splicegraph import SplicingModule
from splice.utils.genomic import Junction


class TestEventDiagnosticDataclass(unittest.TestCase):
    """Test EventDiagnostic dataclass."""

    def test_event_diagnostic_creation(self):
        """Test basic EventDiagnostic creation."""
        diag = EventDiagnostic(
            module_id="mod_1",
            confidence_tier="HIGH",
            null_converged=True,
            full_converged=True,
            null_refit_used=False,
            mean_mapq=30.0,
            median_mapq=31.0,
            frac_high_mapq=0.9,
            frac_multi_mapped=0.05,
            min_group_total_reads=50.0,
            effective_n_min=15.0,
            mean_junction_confidence=0.8,
            min_junction_confidence=0.7,
            frac_annotated_junctions=1.0,
            prior_dominance=0.1,
            bootstrap_cv=0.2,
            has_novel_junctions=False,
            has_low_confidence_junction=False,
            has_convergence_issue=False,
            reason="All quality criteria met",
        )

        self.assertEqual(diag.module_id, "mod_1")
        self.assertEqual(diag.confidence_tier, "HIGH")
        self.assertTrue(diag.null_converged)

    def test_event_diagnostic_frozen(self):
        """Test that EventDiagnostic is frozen."""
        diag = EventDiagnostic(
            module_id="test",
            confidence_tier="MEDIUM",
            null_converged=True,
            full_converged=True,
            null_refit_used=False,
            mean_mapq=25.0,
            median_mapq=26.0,
            frac_high_mapq=0.75,
            frac_multi_mapped=0.1,
            min_group_total_reads=25.0,
            effective_n_min=8.0,
            mean_junction_confidence=0.6,
            min_junction_confidence=0.5,
            frac_annotated_junctions=0.8,
            prior_dominance=0.15,
            bootstrap_cv=0.4,
            has_novel_junctions=True,
            has_low_confidence_junction=False,
            has_convergence_issue=False,
            reason="Failed one criterion",
        )

        with self.assertRaises(AttributeError):
            diag.confidence_tier = "LOW"


class TestDiagnosticsBasic(unittest.TestCase):
    """Test basic diagnostics computation."""

    def create_test_data(self, n_samples=10, n_junctions=2):
        """Create test data for diagnostics."""
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

        mapq_matrix = np.full((n_junctions, n_samples), 30.0)

        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=count_matrix,
            junction_weighted_matrix=count_matrix.astype(float) * 0.95,
            junction_mapq_matrix=mapq_matrix,
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=np.ones(n_junctions) * 100.0,
            normalized_count_matrix=count_matrix.astype(float),
            total_counts=np.sum(count_matrix, axis=0),
            total_weighted=np.sum(count_matrix.astype(float) * 0.95, axis=0),
            junction_confidence=np.ones(n_junctions) * 0.8,
            is_annotated=np.ones(n_junctions, dtype=bool),
        )

        # Create ModulePSI
        psi_matrix = count_matrix.astype(float) / np.sum(
            count_matrix, axis=0, keepdims=True
        )

        psi = ModulePSI(
            module_id="test_module",
            psi_matrix=psi_matrix,
            ci_low_matrix=psi_matrix * 0.8,
            ci_high_matrix=psi_matrix * 1.2,
            bootstrap_psi=np.random.rand(15, n_junctions, n_samples) * 0.4
            + psi_matrix[np.newaxis, :, :] * 0.6,
            total_counts=np.sum(count_matrix, axis=0),
            effective_n=np.sum(count_matrix.astype(float) * 0.95, axis=0),
        )

        # Create DiffResult
        diff_result = DiffResult(
            module_id="test_module",
            gene_id="gene1",
            gene_name="GENE1",
            chrom="chr1",
            strand="+",
            event_type="SE",
            n_junctions=n_junctions,
            junction_coords=[f"chr1:{100+i*100}-{200+i*100}:+" for i in range(n_junctions)],
            junction_confidence=[0.8] * n_junctions,
            is_annotated=[True] * n_junctions,
            psi_group1=psi_matrix[:, :n_samples // 2].mean(axis=1),
            psi_group2=psi_matrix[:, n_samples // 2 :].mean(axis=1),
            delta_psi=psi_matrix[:, n_samples // 2 :].mean(axis=1)
            - psi_matrix[:, :n_samples // 2].mean(axis=1),
            max_abs_delta_psi=0.3,
            delta_psi_ci_low=np.full(n_junctions, -0.2),
            delta_psi_ci_high=np.full(n_junctions, 0.2),
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

        return evidence, psi, diff_result

    def test_compute_diagnostics_basic(self):
        """Test basic diagnostics computation."""
        evidence, psi, diff_result = self.create_test_data()

        diagnostics = compute_diagnostics([evidence], [psi], [diff_result])

        self.assertEqual(len(diagnostics), 1)
        diag = diagnostics[0]

        self.assertEqual(diag.module_id, "test_module")
        self.assertIn(
            diag.confidence_tier, ["HIGH", "MEDIUM", "LOW", "FAIL"]
        )
        self.assertTrue(diag.null_converged)
        self.assertTrue(diag.full_converged)

    def test_diagnostics_metrics(self):
        """Test that diagnostic metrics are computed."""
        evidence, psi, diff_result = self.create_test_data()

        diagnostics = compute_diagnostics([evidence], [psi], [diff_result])

        diag = diagnostics[0]

        # Check that all metrics are finite
        self.assertTrue(np.isfinite(diag.mean_mapq))
        self.assertTrue(np.isfinite(diag.median_mapq))
        self.assertTrue(np.isfinite(diag.frac_high_mapq))
        self.assertTrue(np.isfinite(diag.min_group_total_reads))
        self.assertTrue(np.isfinite(diag.effective_n_min))
        self.assertTrue(np.isfinite(diag.mean_junction_confidence))
        self.assertTrue(np.isfinite(diag.bootstrap_cv))

    def test_diagnostics_flags(self):
        """Test diagnostic flags."""
        evidence, psi, diff_result = self.create_test_data()

        diagnostics = compute_diagnostics([evidence], [psi], [diff_result])

        diag = diagnostics[0]

        # Check flag types
        self.assertIsInstance(diag.has_novel_junctions, (bool, np.bool_))
        self.assertIsInstance(diag.has_low_confidence_junction, (bool, np.bool_))
        self.assertIsInstance(diag.has_convergence_issue, (bool, np.bool_))

    def test_tier_high_quality(self):
        """Test HIGH tier assignment with high-quality data."""
        evidence, psi, diff_result = self.create_test_data(n_samples=20)

        # Modify to ensure HIGH quality
        evidence.junction_mapq_matrix[:] = 35.0

        diagnostics = compute_diagnostics([evidence], [psi], [diff_result])

        diag = diagnostics[0]

        # With high MAPQ and good convergence, should be HIGH or MEDIUM
        self.assertIn(diag.confidence_tier, ["HIGH", "MEDIUM"])


class TestDiagnosticsMultiple(unittest.TestCase):
    """Test with multiple modules."""

    def test_multiple_modules(self):
        """Test diagnostics for multiple modules."""
        evidence_list = []
        psi_list = []
        diff_results = []

        for i in range(3):
            # Simplified module creation
            junctions = [
                Junction(chrom="chr1", start=100 + i * 100, end=200 + i * 100, strand="+"),
                Junction(chrom="chr1", start=200 + i * 100, end=300 + i * 100, strand="+"),
            ]
            module = SplicingModule(
                module_id=f"mod_{i}",
                gene_id=f"gene_{i}",
                gene_name=f"GENE_{i}",
                chrom="chr1",
                strand="+",
                start=100 + i * 100,
                end=300 + i * 100,
                junctions=junctions,
                junction_indices=[0, 1],
                n_connections=2,
            )

            count_matrix = np.random.poisson(50, size=(2, 10))
            count_matrix = np.maximum(count_matrix, 1)

            evidence = ModuleEvidence(
                module=module,
                junction_count_matrix=count_matrix,
                junction_weighted_matrix=count_matrix.astype(float) * 0.95,
                junction_mapq_matrix=np.full((2, 10), 30.0),
                exon_body_count_matrix=None,
                exon_body_weighted_matrix=None,
                junction_effective_lengths=np.ones(2) * 100.0,
                normalized_count_matrix=count_matrix.astype(float),
                total_counts=np.sum(count_matrix, axis=0),
                total_weighted=np.sum(count_matrix.astype(float) * 0.95, axis=0),
                junction_confidence=np.ones(2) * 0.8,
                is_annotated=np.ones(2, dtype=bool),
            )

            psi_matrix = count_matrix.astype(float) / np.sum(count_matrix, axis=0, keepdims=True)

            psi = ModulePSI(
                module_id=f"mod_{i}",
                psi_matrix=psi_matrix,
                ci_low_matrix=psi_matrix * 0.8,
                ci_high_matrix=psi_matrix * 1.2,
                bootstrap_psi=np.random.rand(10, 2, 10),
                total_counts=np.sum(count_matrix, axis=0),
                effective_n=np.sum(count_matrix.astype(float) * 0.95, axis=0),
            )

            diff_result = DiffResult(
                module_id=f"mod_{i}",
                gene_id=f"gene_{i}",
                gene_name=f"GENE_{i}",
                chrom="chr1",
                strand="+",
                event_type="SE",
                n_junctions=2,
                junction_coords=[f"chr1:100-200:+", f"chr1:200-300:+"],
                junction_confidence=[0.8, 0.8],
                is_annotated=[True, True],
                psi_group1=np.array([0.3, 0.7]),
                psi_group2=np.array([0.5, 0.5]),
                delta_psi=np.array([0.2, -0.2]),
                max_abs_delta_psi=0.2,
                delta_psi_ci_low=np.array([-0.1, -0.3]),
                delta_psi_ci_high=np.array([0.5, 0.1]),
                log_likelihood_null=-100.0,
                log_likelihood_full=-85.0,
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

            evidence_list.append(evidence)
            psi_list.append(psi)
            diff_results.append(diff_result)

        diagnostics = compute_diagnostics(evidence_list, psi_list, diff_results)

        self.assertEqual(len(diagnostics), 3)

        for diag in diagnostics:
            self.assertIn(diag.confidence_tier, ["HIGH", "MEDIUM", "LOW", "FAIL"])


class TestDiagnosticsEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_low_convergence(self):
        """Test LOW tier due to convergence failure."""
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

        count_matrix = np.ones((1, 10)) * 50

        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=count_matrix.astype(int),
            junction_weighted_matrix=count_matrix * 0.95,
            junction_mapq_matrix=np.ones((1, 10)) * 30.0,
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=np.ones(1) * 100.0,
            normalized_count_matrix=count_matrix,
            total_counts=np.sum(count_matrix, axis=0),
            total_weighted=np.sum(count_matrix * 0.95, axis=0),
            junction_confidence=np.array([0.8]),
            is_annotated=np.array([True]),
        )

        psi = ModulePSI(
            module_id="test",
            psi_matrix=np.ones((1, 10)),
            ci_low_matrix=np.ones((1, 10)) * 0.9,
            ci_high_matrix=np.ones((1, 10)),
            bootstrap_psi=np.ones((10, 1, 10)),
            total_counts=np.ones(10) * 50,
            effective_n=np.ones(10) * 47.5,
        )

        diff_result = DiffResult(
            module_id="test",
            gene_id="gene1",
            gene_name="GENE1",
            chrom="chr1",
            strand="+",
            event_type="SE",
            n_junctions=1,
            junction_coords=["chr1:100-200:+"],
            junction_confidence=[0.8],
            is_annotated=[True],
            psi_group1=np.array([1.0]),
            psi_group2=np.array([1.0]),
            delta_psi=np.array([0.0]),
            max_abs_delta_psi=0.0,
            delta_psi_ci_low=np.array([-0.1]),
            delta_psi_ci_high=np.array([0.1]),
            log_likelihood_null=-100.0,
            log_likelihood_full=-100.0,
            degrees_of_freedom=1,
            p_value=1.0,
            fdr=1.0,
            null_converged=False,  # Convergence failed
            full_converged=False,
            null_refit_used=False,
            null_iterations=50,
            full_iterations=50,
            null_gradient_norm=0.1,
            full_gradient_norm=0.1,
        )

        diagnostics = compute_diagnostics([evidence], [psi], [diff_result])

        diag = diagnostics[0]

        # Should be LOW or FAIL due to convergence failure
        self.assertIn(diag.confidence_tier, ["LOW", "FAIL"])
        self.assertTrue(diag.has_convergence_issue)


if __name__ == "__main__":
    unittest.main()
