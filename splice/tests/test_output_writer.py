"""
Test suite for Module 21: io/output_writer.py

Tests TSV writing functions for results, junction details, and summaries.
"""

import os
import tempfile
import unittest

import numpy as np

from splice.core.diagnostics import EventDiagnostic
from splice.core.diff import DiffResult
from splice.core.nmd_classifier import NMDClassification
from splice.io.output_writer import (
    write_junction_details_tsv,
    write_results_tsv,
    write_summary_tsv,
)
from splice.utils.genomic import Junction


class TestWriteResultsTSV(unittest.TestCase):
    """Test write_results_tsv function."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "results.tsv")

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def create_test_diff_result(self):
        """Create a test DiffResult."""
        return DiffResult(
            module_id="test_module",
            gene_id="gene1",
            gene_name="GENE1",
            chrom="chr1",
            strand="+",
            event_type="SE",
            n_junctions=2,
            junction_coords=["chr1:100-200:+", "chr1:300-400:+"],
            junction_confidence=[0.8, 0.7],
            is_annotated=[True, False],
            psi_group1=np.array([0.3, 0.7]),
            psi_group2=np.array([0.5, 0.5]),
            delta_psi=np.array([0.2, -0.2]),
            max_abs_delta_psi=0.2,
            delta_psi_ci_low=np.array([-0.1, -0.3]),
            delta_psi_ci_high=np.array([0.5, 0.1]),
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

    def create_test_diagnostic(self):
        """Create a test EventDiagnostic."""
        return EventDiagnostic(
            module_id="test_module",
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

    def test_write_results_single_event(self):
        """Test writing a single event to results TSV."""
        diff_results = [self.create_test_diff_result()]
        diagnostics = [self.create_test_diagnostic()]

        write_results_tsv(diff_results, diagnostics, self.output_path)

        self.assertTrue(os.path.exists(self.output_path))

        with open(self.output_path) as f:
            lines = f.readlines()

        # Header + 1 data row
        self.assertEqual(len(lines), 2)
        self.assertIn("module_id", lines[0])
        self.assertIn("test_module", lines[1])

    def test_write_results_multiple_events(self):
        """Test writing multiple events."""
        diff_results = [self.create_test_diff_result() for _ in range(3)]
        diagnostics = [self.create_test_diagnostic() for _ in range(3)]

        write_results_tsv(diff_results, diagnostics, self.output_path)

        with open(self.output_path) as f:
            lines = f.readlines()

        # Header + 3 data rows
        self.assertEqual(len(lines), 4)

    def test_write_results_empty(self):
        """Test writing empty results."""
        write_results_tsv([], [], self.output_path)

        with open(self.output_path) as f:
            lines = f.readlines()

        # Header only
        self.assertEqual(len(lines), 1)
        self.assertIn("module_id", lines[0])

    def test_write_results_columns(self):
        """Test that all expected columns are present."""
        diff_results = [self.create_test_diff_result()]
        diagnostics = [self.create_test_diagnostic()]

        write_results_tsv(diff_results, diagnostics, self.output_path)

        with open(self.output_path) as f:
            header = f.readline().strip()

        expected_columns = [
            "module_id",
            "gene_id",
            "event_type",
            "p_value",
            "fdr",
            "confidence_tier",
        ]
        for col in expected_columns:
            self.assertIn(col, header)

    def test_write_results_values(self):
        """Test that values are written correctly."""
        diff_results = [self.create_test_diff_result()]
        diagnostics = [self.create_test_diagnostic()]

        write_results_tsv(diff_results, diagnostics, self.output_path)

        with open(self.output_path) as f:
            lines = f.readlines()
            data_row = lines[1].strip().split("\t")

        # Find column index for known fields
        header = lines[0].strip().split("\t")
        module_idx = header.index("module_id")
        fdr_idx = header.index("fdr")
        tier_idx = header.index("confidence_tier")

        self.assertEqual(data_row[module_idx], "test_module")
        self.assertEqual(float(data_row[fdr_idx]), 0.1)
        self.assertEqual(data_row[tier_idx], "HIGH")


class TestWriteJunctionDetailsTSV(unittest.TestCase):
    """Test write_junction_details_tsv function."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "junctions.tsv")

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_write_junction_details_basic(self):
        """Test writing junction details."""
        junction = Junction(chrom="chr1", start=100, end=200, strand="+")
        junction_evidence = {
            "junc_1": {
                "junction": junction,
                "gene_id": "gene1",
                "gene_name": "GENE1",
                "is_annotated": True,
                "motif": "GT/AG",
                "motif_score": 0.95,
                "total_reads": 1000,
                "mean_mapq": 30.0,
                "sample_counts": [10, 20, 15, 25],
            }
        }
        junction_confidence = {"junc_1": 0.85}
        nmd_classifications = {
            "junc_1": NMDClassification(
                junction=junction,
                classification="PR",
                n_productive_paths=5,
                n_unproductive_paths=0,
                confidence=1.0,
                ptc_position=None,
                last_ejc_position=200,
            )
        }

        write_junction_details_tsv(
            junction_evidence, junction_confidence, nmd_classifications, self.output_path
        )

        self.assertTrue(os.path.exists(self.output_path))

        with open(self.output_path) as f:
            lines = f.readlines()

        # Header + 1 data row
        self.assertEqual(len(lines), 2)

    def test_write_junction_details_columns(self):
        """Test that all expected columns are present."""
        junction = Junction(chrom="chr1", start=100, end=200, strand="+")
        junction_evidence = {
            "junc_1": {
                "junction": junction,
                "gene_id": "gene1",
                "gene_name": "GENE1",
                "is_annotated": True,
                "motif": "GT/AG",
                "motif_score": 0.95,
                "total_reads": 1000,
                "mean_mapq": 30.0,
                "sample_counts": [10, 20, 15, 25],
            }
        }
        junction_confidence = {"junc_1": 0.85}
        nmd_classifications = {}

        write_junction_details_tsv(
            junction_evidence, junction_confidence, nmd_classifications, self.output_path
        )

        with open(self.output_path) as f:
            header = f.readline().strip()

        expected_columns = [
            "junction_id",
            "chrom",
            "start",
            "end",
            "strand",
            "gene_id",
            "gene_name",
            "is_annotated",
            "motif",
            "motif_score",
            "confidence_score",
            "nmd_class",
            "total_reads",
            "sample_counts",
        ]
        for col in expected_columns:
            self.assertIn(col, header)

    def test_write_junction_details_values(self):
        """Test that values are written correctly."""
        junction = Junction(chrom="chr1", start=100, end=200, strand="+")
        junction_evidence = {
            "junc_1": {
                "junction": junction,
                "gene_id": "gene1",
                "gene_name": "GENE1",
                "is_annotated": True,
                "motif": "GT/AG",
                "motif_score": 0.95,
                "total_reads": 1000,
                "mean_mapq": 30.0,
                "sample_counts": [10, 20, 15, 25],
            }
        }
        junction_confidence = {"junc_1": 0.85}
        nmd_classifications = {
            "junc_1": NMDClassification(
                junction=junction,
                classification="PR",
                n_productive_paths=5,
                n_unproductive_paths=0,
                confidence=1.0,
                ptc_position=None,
                last_ejc_position=200,
            )
        }

        write_junction_details_tsv(
            junction_evidence, junction_confidence, nmd_classifications, self.output_path
        )

        with open(self.output_path) as f:
            lines = f.readlines()
            data_row = lines[1].strip().split("\t")

        header = lines[0].strip().split("\t")
        chrom_idx = header.index("chrom")
        motif_idx = header.index("motif")
        nmd_class_idx = header.index("nmd_class")

        self.assertEqual(data_row[chrom_idx], "chr1")
        self.assertEqual(data_row[motif_idx], "GT/AG")
        self.assertEqual(data_row[nmd_class_idx], "PR")

    def test_write_junction_details_missing_nmd(self):
        """Test with missing NMD classification."""
        junction = Junction(chrom="chr1", start=100, end=200, strand="+")
        junction_evidence = {
            "junc_1": {
                "junction": junction,
                "gene_id": "gene1",
                "gene_name": "GENE1",
                "is_annotated": True,
                "motif": "GT/AG",
                "motif_score": 0.95,
                "total_reads": 1000,
                "mean_mapq": 30.0,
                "sample_counts": [10, 20, 15],
            }
        }
        junction_confidence = {"junc_1": 0.85}
        nmd_classifications = {}  # Missing NMD classification

        write_junction_details_tsv(
            junction_evidence, junction_confidence, nmd_classifications, self.output_path
        )

        with open(self.output_path) as f:
            lines = f.readlines()
            data_row = lines[1].strip().split("\t")

        header = lines[0].strip().split("\t")
        nmd_class_idx = header.index("nmd_class")

        self.assertEqual(data_row[nmd_class_idx], "NA")

    def test_write_junction_details_sample_counts(self):
        """Test sample_counts formatting."""
        junction = Junction(chrom="chr1", start=100, end=200, strand="+")
        junction_evidence = {
            "junc_1": {
                "junction": junction,
                "gene_id": "gene1",
                "gene_name": "GENE1",
                "is_annotated": True,
                "motif": "GT/AG",
                "motif_score": 0.95,
                "total_reads": 1000,
                "mean_mapq": 30.0,
                "sample_counts": [10, 20, 15, 25],
            }
        }
        junction_confidence = {"junc_1": 0.85}
        nmd_classifications = {}

        write_junction_details_tsv(
            junction_evidence, junction_confidence, nmd_classifications, self.output_path
        )

        with open(self.output_path) as f:
            lines = f.readlines()
            data_row = lines[1].strip().split("\t")

        header = lines[0].strip().split("\t")
        counts_idx = header.index("sample_counts")
        recurrence_idx = header.index("cross_sample_recurrence")

        self.assertEqual(data_row[counts_idx], "10,20,15,25")
        self.assertEqual(int(data_row[recurrence_idx]), 4)

    def test_write_junction_details_empty(self):
        """Test writing empty junction details."""
        write_junction_details_tsv({}, {}, {}, self.output_path)

        with open(self.output_path) as f:
            lines = f.readlines()

        # Header only
        self.assertEqual(len(lines), 1)


class TestWriteSummaryTSV(unittest.TestCase):
    """Test write_summary_tsv function."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "summary.tsv")

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def create_test_diff_result(self, fdr=0.05):
        """Create a test DiffResult."""
        return DiffResult(
            module_id="test_module",
            gene_id="gene1",
            gene_name="GENE1",
            chrom="chr1",
            strand="+",
            event_type="SE",
            n_junctions=2,
            junction_coords=["chr1:100-200:+", "chr1:300-400:+"],
            junction_confidence=[0.8, 0.7],
            is_annotated=[True, False],
            psi_group1=np.array([0.3, 0.7]),
            psi_group2=np.array([0.5, 0.5]),
            delta_psi=np.array([0.2, -0.2]),
            max_abs_delta_psi=0.2,
            delta_psi_ci_low=np.array([-0.1, -0.3]),
            delta_psi_ci_high=np.array([0.5, 0.1]),
            log_likelihood_null=-100.0,
            log_likelihood_full=-80.0,
            degrees_of_freedom=1,
            p_value=0.05,
            fdr=fdr,
            null_converged=True,
            full_converged=True,
            null_refit_used=False,
            null_iterations=50,
            full_iterations=60,
            null_gradient_norm=1e-5,
            full_gradient_norm=1e-5,
        )

    def create_test_diagnostic(self, tier="HIGH"):
        """Create a test EventDiagnostic."""
        return EventDiagnostic(
            module_id="test_module",
            confidence_tier=tier,
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

    def test_write_summary_basic(self):
        """Test writing a summary."""
        diff_results = [self.create_test_diff_result()]
        diagnostics = [self.create_test_diagnostic()]
        event_types = {"SE": 1}

        write_summary_tsv(diff_results, diagnostics, event_types, self.output_path)

        self.assertTrue(os.path.exists(self.output_path))

        with open(self.output_path) as f:
            content = f.read()

        self.assertIn("Total events", content)
        self.assertIn("1", content)

    def test_write_summary_event_types(self):
        """Test event type breakdown in summary."""
        diff_results = [
            self.create_test_diff_result(),
            self.create_test_diff_result(),
        ]
        diagnostics = [
            self.create_test_diagnostic(),
            self.create_test_diagnostic(),
        ]
        event_types = {"SE": 2}

        write_summary_tsv(diff_results, diagnostics, event_types, self.output_path)

        with open(self.output_path) as f:
            content = f.read()

        self.assertIn("SE\t2", content)

    def test_write_summary_significance(self):
        """Test significance counts in summary."""
        diff_results = [
            self.create_test_diff_result(fdr=0.005),  # Significant at 0.05 and 0.01
            self.create_test_diff_result(fdr=0.03),  # Significant at 0.05 only
            self.create_test_diff_result(fdr=0.1),  # Not significant
        ]
        diagnostics = [
            self.create_test_diagnostic(),
            self.create_test_diagnostic(),
            self.create_test_diagnostic(),
        ]
        event_types = {"SE": 3}

        write_summary_tsv(diff_results, diagnostics, event_types, self.output_path)

        with open(self.output_path) as f:
            content = f.read()

        self.assertIn("Significant (FDR < 0.05)\t2", content)
        self.assertIn("Significant (FDR < 0.01)\t1", content)

    def test_write_summary_confidence_tiers(self):
        """Test confidence tier breakdown in summary."""
        diff_results = [
            self.create_test_diff_result(),
            self.create_test_diff_result(),
            self.create_test_diff_result(),
        ]
        diagnostics = [
            self.create_test_diagnostic(tier="HIGH"),
            self.create_test_diagnostic(tier="MEDIUM"),
            self.create_test_diagnostic(tier="LOW"),
        ]
        event_types = {"SE": 3}

        write_summary_tsv(diff_results, diagnostics, event_types, self.output_path)

        with open(self.output_path) as f:
            content = f.read()

        self.assertIn("Confidence HIGH\t1", content)
        self.assertIn("Confidence MEDIUM\t1", content)
        self.assertIn("Confidence LOW\t1", content)

    def test_write_summary_quality_metrics(self):
        """Test quality metrics in summary."""
        diff_results = [self.create_test_diff_result()]
        diagnostics = [self.create_test_diagnostic()]
        event_types = {"SE": 1}

        write_summary_tsv(diff_results, diagnostics, event_types, self.output_path)

        with open(self.output_path) as f:
            content = f.read()

        self.assertIn("Mean MAPQ", content)
        self.assertIn("Mean junction confidence", content)
        self.assertIn("Mean bootstrap CV", content)

    def test_write_summary_convergence(self):
        """Test convergence statistics in summary."""
        diff_results = [self.create_test_diff_result()]
        diagnostics = [self.create_test_diagnostic()]
        event_types = {"SE": 1}

        write_summary_tsv(diff_results, diagnostics, event_types, self.output_path)

        with open(self.output_path) as f:
            content = f.read()

        self.assertIn("Null model converged\t1/1", content)
        self.assertIn("Full model converged\t1/1", content)

    def test_write_summary_empty(self):
        """Test writing empty summary."""
        write_summary_tsv([], [], {}, self.output_path)

        with open(self.output_path) as f:
            content = f.read()

        self.assertIn("Total events\t0", content)


class TestOutputWriterIntegration(unittest.TestCase):
    """Integration tests for output writer functions."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_write_all_outputs(self):
        """Test writing all output files together."""
        # Create test data
        diff_results = [
            DiffResult(
                module_id=f"mod_{i}",
                gene_id=f"gene{i}",
                gene_name=f"GENE{i}",
                chrom="chr1",
                strand="+",
                event_type="SE",
                n_junctions=2,
                junction_coords=[f"chr1:{100*i}-{200*i}:+"],
                junction_confidence=[0.8],
                is_annotated=[True],
                psi_group1=np.array([0.3, 0.7]),
                psi_group2=np.array([0.5, 0.5]),
                delta_psi=np.array([0.2, -0.2]),
                max_abs_delta_psi=0.2,
                delta_psi_ci_low=np.array([-0.1, -0.3]),
                delta_psi_ci_high=np.array([0.5, 0.1]),
                log_likelihood_null=-100.0,
                log_likelihood_full=-80.0,
                degrees_of_freedom=1,
                p_value=0.01 * (i + 1),
                fdr=0.05 * (i + 1),
                null_converged=True,
                full_converged=True,
                null_refit_used=False,
                null_iterations=50,
                full_iterations=60,
                null_gradient_norm=1e-5,
                full_gradient_norm=1e-5,
            )
            for i in range(3)
        ]

        diagnostics = [
            EventDiagnostic(
                module_id=f"mod_{i}",
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
            for i in range(3)
        ]

        event_types = {"SE": 3}

        # Write all outputs
        results_path = os.path.join(self.temp_dir.name, "results.tsv")
        summary_path = os.path.join(self.temp_dir.name, "summary.tsv")
        junctions_path = os.path.join(self.temp_dir.name, "junctions.tsv")

        write_results_tsv(diff_results, diagnostics, results_path)
        write_summary_tsv(diff_results, diagnostics, event_types, summary_path)

        # Write junction details
        junction_evidence = {
            "junc_1": {
                "junction": Junction(chrom="chr1", start=100, end=200, strand="+"),
                "gene_id": "gene0",
                "gene_name": "GENE0",
                "is_annotated": True,
                "motif": "GT/AG",
                "motif_score": 0.95,
                "total_reads": 1000,
                "mean_mapq": 30.0,
                "sample_counts": [10, 20, 15],
            }
        }
        junction_confidence = {"junc_1": 0.85}
        nmd_classifications = {}

        write_junction_details_tsv(
            junction_evidence, junction_confidence, nmd_classifications, junctions_path
        )

        # Verify files exist
        self.assertTrue(os.path.exists(results_path))
        self.assertTrue(os.path.exists(summary_path))
        self.assertTrue(os.path.exists(junctions_path))

        # Verify content
        with open(results_path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 4)  # Header + 3 data rows

        with open(summary_path) as f:
            content = f.read()
        self.assertIn("Total events\t3", content)

        with open(junctions_path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)  # Header + 1 data row


if __name__ == "__main__":
    unittest.main()
