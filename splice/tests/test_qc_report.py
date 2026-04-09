"""
Test suite for Module 24: io/qc_report.py

Tests QC report generation with embedded figures.
"""

import os
import tempfile
import unittest

import numpy as np

from splice.core.diagnostics import EventDiagnostic
from splice.core.diff import DiffResult
from splice.core.diff_het import HetResult
from splice.core.nmd_classifier import NMDClassification
from splice.io.qc_report import generate_qc_report
from splice.utils.genomic import Junction


class TestGenerateQCReport(unittest.TestCase):
    """Test generate_qc_report function."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "report.html")

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def create_test_diff_result(self, fdr=0.01):
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
            p_value=0.001,
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

    def test_generate_report_basic(self):
        """Test basic report generation."""
        diff_results = [self.create_test_diff_result()]
        diagnostics = [self.create_test_diagnostic()]
        event_types = {"SE": 1}

        generate_qc_report(
            diff_results=diff_results,
            het_results=[],
            diagnostics=diagnostics,
            event_types=event_types,
            junction_evidence={},
            nmd_classifications={},
            output_path=self.output_path,
        )

        self.assertTrue(os.path.exists(self.output_path))

        with open(self.output_path) as f:
            content = f.read()

        self.assertIn("<!DOCTYPE html>", content)
        self.assertIn("SPLICE Quality Control Report", content)

    def test_report_contains_sections(self):
        """Test that report contains all expected sections."""
        diff_results = [self.create_test_diff_result()]
        diagnostics = [self.create_test_diagnostic()]
        event_types = {"SE": 1}

        generate_qc_report(
            diff_results=diff_results,
            het_results=[],
            diagnostics=diagnostics,
            event_types=event_types,
            junction_evidence={},
            nmd_classifications={},
            output_path=self.output_path,
        )

        with open(self.output_path) as f:
            content = f.read()

        # Check for section headers
        self.assertIn("Data Summary", content)
        self.assertIn("Junction Discovery", content)
        self.assertIn("Clustering", content)
        self.assertIn("Differential Splicing", content)
        self.assertIn("Diagnostics", content)

    def test_report_html_structure(self):
        """Test HTML structure of report."""
        diff_results = [self.create_test_diff_result()]
        diagnostics = [self.create_test_diagnostic()]
        event_types = {"SE": 1}

        generate_qc_report(
            diff_results=diff_results,
            het_results=[],
            diagnostics=diagnostics,
            event_types=event_types,
            junction_evidence={},
            nmd_classifications={},
            output_path=self.output_path,
        )

        with open(self.output_path) as f:
            content = f.read()

        # Check HTML tags
        self.assertIn("<html", content)
        self.assertIn("</html>", content)
        self.assertIn("<head>", content)
        self.assertIn("</head>", content)
        self.assertIn("<body>", content)
        self.assertIn("</body>", content)
        self.assertIn("<style>", content)

    def test_report_contains_base64_images(self):
        """Test that report contains embedded base64 images."""
        diff_results = [
            self.create_test_diff_result(fdr=0.01),
            self.create_test_diff_result(fdr=0.1),
        ]
        diagnostics = [self.create_test_diagnostic("HIGH"), self.create_test_diagnostic("MEDIUM")]
        event_types = {"SE": 2}

        generate_qc_report(
            diff_results=diff_results,
            het_results=[],
            diagnostics=diagnostics,
            event_types=event_types,
            junction_evidence={},
            nmd_classifications={},
            output_path=self.output_path,
        )

        with open(self.output_path) as f:
            content = f.read()

        # Check for base64 encoded images
        self.assertIn("data:image/png;base64,", content)

    def test_report_with_junction_evidence(self):
        """Test report generation with junction evidence."""
        diff_results = [self.create_test_diff_result()]
        diagnostics = [self.create_test_diagnostic()]
        event_types = {"SE": 1}

        junction_evidence = {
            "junc_1": {
                "junction": Junction(chrom="chr1", start=100, end=200, strand="+"),
                "gene_id": "gene1",
                "gene_name": "GENE1",
                "is_annotated": True,
                "motif": "GT/AG",
                "motif_score": 0.95,
                "total_reads": 1000,
                "mean_mapq": 30.0,
                "sample_counts": [10, 20, 15, 25],
            },
            "junc_2": {
                "junction": Junction(chrom="chr1", start=300, end=400, strand="+"),
                "gene_id": "gene1",
                "gene_name": "GENE1",
                "is_annotated": False,
                "motif": "GC/AG",
                "motif_score": 0.75,
                "total_reads": 500,
                "mean_mapq": 25.0,
                "sample_counts": [5, 10, 8],
            },
        }

        generate_qc_report(
            diff_results=diff_results,
            het_results=[],
            diagnostics=diagnostics,
            event_types=event_types,
            junction_evidence=junction_evidence,
            nmd_classifications={},
            output_path=self.output_path,
        )

        with open(self.output_path) as f:
            content = f.read()

        # Should contain junction statistics
        self.assertIn("Total junctions", content)
        self.assertIn("Annotated", content)
        self.assertIn("Novel", content)

    def test_report_with_nmd_classifications(self):
        """Test report generation with NMD classifications."""
        diff_results = [self.create_test_diff_result()]
        diagnostics = [self.create_test_diagnostic()]
        event_types = {"SE": 1}

        junction = Junction(chrom="chr1", start=100, end=200, strand="+")
        nmd_classifications = {
            "junc_1": NMDClassification(
                junction=junction,
                classification="PR",
                n_productive_paths=5,
                n_unproductive_paths=0,
                confidence=1.0,
                ptc_position=None,
                last_ejc_position=200,
            ),
            "junc_2": NMDClassification(
                junction=junction,
                classification="UP",
                n_productive_paths=0,
                n_unproductive_paths=3,
                confidence=0.0,
                ptc_position=150,
                last_ejc_position=200,
            ),
        }

        generate_qc_report(
            diff_results=diff_results,
            het_results=[],
            diagnostics=diagnostics,
            event_types=event_types,
            junction_evidence={},
            nmd_classifications=nmd_classifications,
            output_path=self.output_path,
        )

        with open(self.output_path) as f:
            content = f.read()

        # Should contain NMD section
        self.assertIn("Functional Annotation", content)
        self.assertIn("NMD Classification", content)

    def test_report_empty_data(self):
        """Test report generation with minimal data."""
        generate_qc_report(
            diff_results=[],
            het_results=[],
            diagnostics=[],
            event_types={},
            junction_evidence={},
            nmd_classifications={},
            output_path=self.output_path,
        )

        self.assertTrue(os.path.exists(self.output_path))

        with open(self.output_path) as f:
            content = f.read()

        self.assertIn("SPLICE Quality Control Report", content)


class TestQCReportIntegration(unittest.TestCase):
    """Integration tests for QC report generation."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "report.html")

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_complete_report(self):
        """Test generation of complete report with all components."""
        # Create comprehensive test data
        diff_results = [
            DiffResult(
                module_id=f"mod_{i}",
                gene_id=f"gene{i}",
                gene_name=f"GENE{i}",
                chrom="chr1",
                strand="+",
                event_type=["SE", "A3SS", "A5SS"][i % 3],
                n_junctions=2,
                junction_coords=[f"chr1:{100+i*100}-{200+i*100}:+"],
                junction_confidence=[0.8],
                is_annotated=[True],
                psi_group1=np.array([0.3, 0.7]),
                psi_group2=np.array([0.5, 0.5]),
                delta_psi=np.array([0.2, -0.2]),
                max_abs_delta_psi=0.1 + i * 0.05,
                delta_psi_ci_low=np.array([-0.1, -0.3]),
                delta_psi_ci_high=np.array([0.5, 0.1]),
                log_likelihood_null=-100.0,
                log_likelihood_full=-80.0,
                degrees_of_freedom=1,
                p_value=0.001 * (i + 1),
                fdr=0.01 * (i + 1),
                null_converged=True,
                full_converged=True,
                null_refit_used=False,
                null_iterations=50,
                full_iterations=60,
                null_gradient_norm=1e-5,
                full_gradient_norm=1e-5,
            )
            for i in range(5)
        ]

        diagnostics = [
            EventDiagnostic(
                module_id=f"mod_{i}",
                confidence_tier=["HIGH", "MEDIUM", "LOW"][i % 3],
                null_converged=True,
                full_converged=True,
                null_refit_used=False,
                mean_mapq=30.0 - i,
                median_mapq=31.0 - i,
                frac_high_mapq=0.9 - i * 0.1,
                frac_multi_mapped=0.05 + i * 0.01,
                min_group_total_reads=50.0,
                effective_n_min=15.0,
                mean_junction_confidence=0.8 - i * 0.05,
                min_junction_confidence=0.7 - i * 0.05,
                frac_annotated_junctions=1.0 - i * 0.1,
                prior_dominance=0.1,
                bootstrap_cv=0.2 + i * 0.05,
                has_novel_junctions=i % 2 == 0,
                has_low_confidence_junction=False,
                has_convergence_issue=False,
                reason="Test event",
            )
            for i in range(5)
        ]

        event_types = {"SE": 2, "A3SS": 2, "A5SS": 1}

        junction_evidence = {
            f"junc_{i}": {
                "junction": Junction(
                    chrom="chr1", start=100 + i, end=200 + i, strand="+"
                ),
                "gene_id": f"gene_{i}",
                "gene_name": f"GENE_{i}",
                "is_annotated": i % 2 == 0,
                "motif": ["GT/AG", "GC/AG"][i % 2],
                "motif_score": 0.9 - i * 0.05,
                "total_reads": 1000 - i * 100,
                "mean_mapq": 30.0 - i * 0.5,
                "sample_counts": [10 + j for j in range(min(5, 5 - i))],
            }
            for i in range(8)
        }

        nmd_classifications = {
            f"junc_{i}": NMDClassification(
                junction=Junction(
                    chrom="chr1", start=100 + i, end=200 + i, strand="+"
                ),
                classification=["PR", "UP", "NE"][i % 3],
                n_productive_paths=5 - i,
                n_unproductive_paths=i,
                confidence=1.0 - i * 0.1,
                ptc_position=100 + i if i % 2 == 0 else None,
                last_ejc_position=200 + i,
            )
            for i in range(8)
        }

        generate_qc_report(
            diff_results=diff_results,
            het_results=[],
            diagnostics=diagnostics,
            event_types=event_types,
            junction_evidence=junction_evidence,
            nmd_classifications=nmd_classifications,
            output_path=self.output_path,
        )

        # Verify file exists and has content
        self.assertTrue(os.path.exists(self.output_path))
        file_size = os.path.getsize(self.output_path)
        self.assertGreater(file_size, 1000)  # Should be at least 1KB

        with open(self.output_path) as f:
            content = f.read()

        # Verify all major sections are present
        sections = [
            "Data Summary",
            "Junction Discovery",
            "Clustering",
            "Differential Splicing",
            "Diagnostics",
            "Functional Annotation",
        ]
        for section in sections:
            self.assertIn(section, content, f"Section '{section}' not found in report")

        # Verify HTML is well-formed
        self.assertIn("<!DOCTYPE html>", content)
        self.assertIn("</html>", content)
        self.assertTrue(content.count("<img") > 0, "Report should contain images")


if __name__ == "__main__":
    unittest.main()