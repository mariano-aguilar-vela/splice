"""
Test suite for Module 12: core/evidence.py

Tests evidence matrix building and filtering.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from splice.core.evidence import (
    ModuleEvidence,
    build_evidence_matrices,
    filter_evidence_by_depth,
    filter_evidence_by_size,
    get_module_psi_matrix,
)
from splice.core.splicegraph import SplicingModule
from splice.utils.genomic import Junction


class TestModuleEvidence(unittest.TestCase):
    """Test ModuleEvidence dataclass."""

    def test_evidence_creation(self):
        """Test creating a ModuleEvidence object."""
        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
            junctions=[
                Junction(chrom="chr1", start=1000, end=2000, strand="+"),
                Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            ],
            junction_indices=[0, 1],
            n_connections=2,
        )

        count_matrix = np.array([[100, 200], [300, 400]])
        weighted_matrix = np.array([[95.0, 190.0], [285.0, 380.0]])
        mapq_matrix = np.array([[50.0, 55.0], [60.0, 58.0]])
        effective_lengths = np.array([100.0, 100.0])
        normalized = np.array([[1.0, 2.0], [3.0, 4.0]])
        total_counts = np.array([400, 600])
        total_weighted = np.array([380.0, 570.0])
        confidence = np.array([0.8, 0.9])
        annotated = np.array([True, False])

        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=count_matrix,
            junction_weighted_matrix=weighted_matrix,
            junction_mapq_matrix=mapq_matrix,
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=effective_lengths,
            normalized_count_matrix=normalized,
            total_counts=total_counts,
            total_weighted=total_weighted,
            junction_confidence=confidence,
            is_annotated=annotated,
        )

        self.assertEqual(evidence.module.module_id, "mod1")
        self.assertEqual(evidence.junction_count_matrix.shape, (2, 2))
        self.assertEqual(evidence.total_counts.shape, (2,))


class TestBuildEvidenceMatrices(unittest.TestCase):
    """Test build_evidence_matrices function."""

    def test_empty_modules(self):
        """Test with empty module list."""
        result = build_evidence_matrices([], {}, n_samples=5)
        self.assertEqual(len(result), 0)

    def test_single_module_two_junctions(self):
        """Test building evidence for a single module."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=2500, strand="+")

        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
            junctions=[j1, j2],
            junction_indices=[0, 1],
            n_connections=2,
        )

        # Mock junction evidence
        evidence1 = MagicMock()
        evidence1.sample_counts = np.array([100, 200])
        evidence1.sample_weighted_counts = np.array([95.0, 190.0])
        evidence1.sample_mapq_mean = np.array([50.0, 55.0])
        evidence1.is_annotated = True

        evidence2 = MagicMock()
        evidence2.sample_counts = np.array([300, 400])
        evidence2.sample_weighted_counts = np.array([285.0, 380.0])
        evidence2.sample_mapq_mean = np.array([60.0, 58.0])
        evidence2.is_annotated = False

        junction_evidence = {j1: evidence1, j2: evidence2}

        result = build_evidence_matrices([module], junction_evidence, n_samples=2)

        self.assertEqual(len(result), 1)
        evidence = result[0]

        self.assertEqual(evidence.junction_count_matrix.shape, (2, 2))
        self.assertEqual(evidence.module.module_id, "mod1")
        self.assertTrue(evidence.is_annotated[0])
        self.assertFalse(evidence.is_annotated[1])

    def test_total_counts_computation(self):
        """Test that total counts are computed correctly."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")

        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
            junctions=[j1],
            junction_indices=[0],
            n_connections=1,
        )

        evidence1 = MagicMock()
        evidence1.sample_counts = np.array([100, 200, 300])
        evidence1.sample_weighted_counts = np.array([95.0, 190.0, 285.0])
        evidence1.sample_mapq_mean = np.array([50.0, 55.0, 60.0])
        evidence1.is_annotated = True

        junction_evidence = {j1: evidence1}

        result = build_evidence_matrices([module], junction_evidence, n_samples=3)

        evidence = result[0]
        np.testing.assert_array_equal(
            evidence.total_counts, [100, 200, 300]
        )


class TestFilterEvidenceByDepth(unittest.TestCase):
    """Test filter_evidence_by_depth function."""

    def test_filter_empty_list(self):
        """Test filtering empty evidence list."""
        result = filter_evidence_by_depth([])
        self.assertEqual(len(result), 0)

    def test_filter_by_total_reads(self):
        """Test filtering by minimum total reads."""
        # Evidence with sufficient reads
        module1 = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
            junctions=[Junction(chrom="chr1", start=1000, end=2000, strand="+")],
            junction_indices=[0],
            n_connections=1,
        )

        evidence1 = ModuleEvidence(
            module=module1,
            junction_count_matrix=np.array([[100, 200]]),
            junction_weighted_matrix=np.array([[95.0, 190.0]]),
            junction_mapq_matrix=np.array([[50.0, 55.0]]),
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=np.array([100.0]),
            normalized_count_matrix=np.array([[1.0, 2.0]]),
            total_counts=np.array([100, 200]),
            total_weighted=np.array([95.0, 190.0]),
            junction_confidence=np.array([0.8]),
            is_annotated=np.array([True]),
        )

        # Evidence with insufficient reads
        module2 = SplicingModule(
            module_id="mod2",
            gene_id="GENE2",
            gene_name="Gene2",
            chrom="chr1",
            strand="+",
            start=3000,
            end=4000,
            junctions=[Junction(chrom="chr1", start=3000, end=4000, strand="+")],
            junction_indices=[1],
            n_connections=1,
        )

        evidence2 = ModuleEvidence(
            module=module2,
            junction_count_matrix=np.array([[5, 5]]),
            junction_weighted_matrix=np.array([[4.5, 4.5]]),
            junction_mapq_matrix=np.array([[50.0, 50.0]]),
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=np.array([100.0]),
            normalized_count_matrix=np.array([[0.05, 0.05]]),
            total_counts=np.array([5, 5]),
            total_weighted=np.array([4.5, 4.5]),
            junction_confidence=np.array([0.5]),
            is_annotated=np.array([True]),
        )

        result = filter_evidence_by_depth(
            [evidence1, evidence2], min_total_reads=20, min_samples_with_reads=2
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].module.module_id, "mod1")

    def test_filter_by_samples_with_reads(self):
        """Test filtering by minimum samples with reads."""
        # Evidence with enough samples
        module1 = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
            junctions=[Junction(chrom="chr1", start=1000, end=2000, strand="+")],
            junction_indices=[0],
            n_connections=1,
        )

        evidence1 = ModuleEvidence(
            module=module1,
            junction_count_matrix=np.array([[100, 200, 150, 50]]),
            junction_weighted_matrix=np.array([[95.0, 190.0, 142.5, 47.5]]),
            junction_mapq_matrix=np.array([[50.0, 55.0, 52.0, 48.0]]),
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=np.array([100.0]),
            normalized_count_matrix=np.array([[1.0, 2.0, 1.5, 0.5]]),
            total_counts=np.array([100, 200, 150, 50]),
            total_weighted=np.array([95.0, 190.0, 142.5, 47.5]),
            junction_confidence=np.array([0.8]),
            is_annotated=np.array([True]),
        )

        # Evidence with not enough samples
        module2 = SplicingModule(
            module_id="mod2",
            gene_id="GENE2",
            gene_name="Gene2",
            chrom="chr1",
            strand="+",
            start=3000,
            end=4000,
            junctions=[Junction(chrom="chr1", start=3000, end=4000, strand="+")],
            junction_indices=[1],
            n_connections=1,
        )

        evidence2 = ModuleEvidence(
            module=module2,
            junction_count_matrix=np.array([[50, 0, 0, 0]]),
            junction_weighted_matrix=np.array([[47.5, 0, 0, 0]]),
            junction_mapq_matrix=np.array([[50.0, 0, 0, 0]]),
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=np.array([100.0]),
            normalized_count_matrix=np.array([[0.5, 0, 0, 0]]),
            total_counts=np.array([50, 0, 0, 0]),
            total_weighted=np.array([47.5, 0, 0, 0]),
            junction_confidence=np.array([0.5]),
            is_annotated=np.array([True]),
        )

        result = filter_evidence_by_depth(
            [evidence1, evidence2], min_samples_with_reads=3
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].module.module_id, "mod1")


class TestFilterEvidenceBySize(unittest.TestCase):
    """Test filter_evidence_by_size function."""

    def test_filter_by_junctions(self):
        """Test filtering by minimum junctions."""
        # Module with 2 junctions
        module1 = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
            junctions=[
                Junction(chrom="chr1", start=1000, end=2000, strand="+"),
                Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            ],
            junction_indices=[0, 1],
            n_connections=2,
        )

        evidence1 = ModuleEvidence(
            module=module1,
            junction_count_matrix=np.array([[100, 200], [300, 400]]),
            junction_weighted_matrix=np.array([[95.0, 190.0], [285.0, 380.0]]),
            junction_mapq_matrix=np.array([[50.0, 55.0], [60.0, 58.0]]),
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=np.array([100.0, 100.0]),
            normalized_count_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
            total_counts=np.array([400, 600]),
            total_weighted=np.array([380.0, 570.0]),
            junction_confidence=np.array([0.8, 0.9]),
            is_annotated=np.array([True, False]),
        )

        # Module with 1 junction
        module2 = SplicingModule(
            module_id="mod2",
            gene_id="GENE2",
            gene_name="Gene2",
            chrom="chr1",
            strand="+",
            start=3000,
            end=4000,
            junctions=[Junction(chrom="chr1", start=3000, end=4000, strand="+")],
            junction_indices=[2],
            n_connections=1,
        )

        evidence2 = ModuleEvidence(
            module=module2,
            junction_count_matrix=np.array([[100, 200]]),
            junction_weighted_matrix=np.array([[95.0, 190.0]]),
            junction_mapq_matrix=np.array([[50.0, 55.0]]),
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=np.array([100.0]),
            normalized_count_matrix=np.array([[1.0, 2.0]]),
            total_counts=np.array([100, 200]),
            total_weighted=np.array([95.0, 190.0]),
            junction_confidence=np.array([0.8]),
            is_annotated=np.array([True]),
        )

        result = filter_evidence_by_size([evidence1, evidence2], min_junctions=2)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].module.module_id, "mod1")


class TestGetModulePSIMatrix(unittest.TestCase):
    """Test get_module_psi_matrix function."""

    def test_psi_computation(self):
        """Test PSI computation from normalized counts."""
        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
            junctions=[
                Junction(chrom="chr1", start=1000, end=2000, strand="+"),
                Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            ],
            junction_indices=[0, 1],
            n_connections=2,
        )

        # Normalized counts: [1, 2] and [3, 4]
        # Sample 1: PSI = [1/(1+3), 3/(1+3)] = [0.25, 0.75]
        # Sample 2: PSI = [2/(2+4), 4/(2+4)] = [0.333, 0.667]
        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=np.array([[100, 200], [300, 400]]),
            junction_weighted_matrix=np.array([[95.0, 190.0], [285.0, 380.0]]),
            junction_mapq_matrix=np.array([[50.0, 55.0], [60.0, 58.0]]),
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=np.array([100.0, 100.0]),
            normalized_count_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
            total_counts=np.array([400, 600]),
            total_weighted=np.array([380.0, 570.0]),
            junction_confidence=np.array([0.8, 0.9]),
            is_annotated=np.array([True, False]),
        )

        psi = get_module_psi_matrix(evidence)

        self.assertEqual(psi.shape, (2, 2))
        np.testing.assert_array_almost_equal(
            psi[:, 0], [0.25, 0.75]
        )
        np.testing.assert_array_almost_equal(
            psi[:, 1], [2.0/6.0, 4.0/6.0], decimal=4
        )

    def test_psi_zero_counts(self):
        """Test PSI computation with zero counts."""
        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
            junctions=[Junction(chrom="chr1", start=1000, end=2000, strand="+")],
            junction_indices=[0],
            n_connections=1,
        )

        evidence = ModuleEvidence(
            module=module,
            junction_count_matrix=np.array([[0, 100]]),
            junction_weighted_matrix=np.array([[0, 95.0]]),
            junction_mapq_matrix=np.array([[0, 50.0]]),
            exon_body_count_matrix=None,
            exon_body_weighted_matrix=None,
            junction_effective_lengths=np.array([100.0]),
            normalized_count_matrix=np.array([[0.0, 1.0]]),
            total_counts=np.array([0, 100]),
            total_weighted=np.array([0, 95.0]),
            junction_confidence=np.array([0.8]),
            is_annotated=np.array([True]),
        )

        psi = get_module_psi_matrix(evidence)

        # First sample: zero counts, PSI should be 0
        self.assertEqual(psi[0, 0], 0.0)
        # Second sample: full PSI
        self.assertEqual(psi[0, 1], 1.0)


if __name__ == "__main__":
    unittest.main()
