"""
Tests for PDF report generation.
"""

import os

import numpy as np
import pytest

from splice.core.diff import DiffResult
from splice.core.diagnostics import EventDiagnostic
from splice.io.pdf_report import generate_pdf_report


def _make_diff_result(module_id, gene_name, event_type, p_value, fdr, delta=0.3):
    return DiffResult(
        module_id=module_id, gene_id=f"G_{module_id}", gene_name=gene_name,
        chrom="chr1", strand="+", event_type=event_type, n_junctions=2,
        junction_coords=["chr1:100-200:+", "chr1:300-400:+"],
        junction_confidence=[0.9, 0.8], is_annotated=[True, False],
        psi_group1=np.array([0.8, 0.2]), psi_group2=np.array([0.5, 0.5]),
        delta_psi=np.array([-delta, delta]), max_abs_delta_psi=delta,
        delta_psi_ci_low=np.array([-0.4, 0.1]),
        delta_psi_ci_high=np.array([-0.2, 0.4]),
        log_likelihood_null=-100.0, log_likelihood_full=-90.0,
        degrees_of_freedom=1, p_value=p_value, fdr=fdr,
        null_converged=True, full_converged=True, null_refit_used=False,
        null_iterations=10, full_iterations=10,
        null_gradient_norm=0.01, full_gradient_norm=0.01,
    )


def _make_diagnostic(module_id, tier="HIGH"):
    return EventDiagnostic(
        module_id=module_id, confidence_tier=tier,
        null_converged=True, full_converged=True, null_refit_used=False,
        mean_mapq=55.0, median_mapq=55.0,
        frac_high_mapq=0.9, frac_multi_mapped=0.05,
        min_group_total_reads=50, effective_n_min=3.0,
        mean_junction_confidence=0.85, min_junction_confidence=0.7,
        frac_annotated_junctions=0.8,
        has_novel_junctions=True, has_low_confidence_junction=False,
        has_convergence_issue=False, reason="All criteria met",
        bootstrap_cv=0.05, prior_dominance=0.1,
    )


class TestPdfReport:

    def test_creates_pdf(self, tmp_path):
        """PDF file should be created."""
        diff = [
            _make_diff_result("m1", "GeneA", "SE", 0.001, 0.01),
            _make_diff_result("m2", "GeneB", "A3SS", 0.05, 0.08),
        ]
        diag = [_make_diagnostic("m1"), _make_diagnostic("m2", "MEDIUM")]
        event_counts = {"SE": 1, "A3SS": 1}

        pdf_path = str(tmp_path / "report.pdf")
        generate_pdf_report(
            diff, diag, event_counts, pdf_path, str(tmp_path),
            sample_info={"Samples": "6"},
            parameters={"Bootstraps": "100"},
        )

        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0

    def test_creates_svg_figures(self, tmp_path):
        """SVG figures should be created in figures/ subdirectory."""
        diff = [_make_diff_result("m1", "GeneA", "SE", 0.001, 0.01)]
        diag = [_make_diagnostic("m1")]
        event_counts = {"SE": 1}

        pdf_path = str(tmp_path / "report.pdf")
        generate_pdf_report(diff, diag, event_counts, pdf_path, str(tmp_path))

        figures_dir = tmp_path / "figures"
        assert figures_dir.exists()
        assert (figures_dir / "volcano_plot.svg").exists()
        assert (figures_dir / "event_types.svg").exists()
        assert (figures_dir / "diagnostics.svg").exists()

    def test_empty_results(self, tmp_path):
        """Should handle empty results without error."""
        pdf_path = str(tmp_path / "report.pdf")
        generate_pdf_report([], [], {}, pdf_path, str(tmp_path))
        assert os.path.exists(pdf_path)

    def test_no_significant_events(self, tmp_path):
        """Should handle case with no significant events."""
        diff = [_make_diff_result("m1", "GeneA", "SE", 0.5, 0.8)]
        diag = [_make_diagnostic("m1", "LOW")]

        pdf_path = str(tmp_path / "report.pdf")
        generate_pdf_report(diff, diag, {"SE": 1}, pdf_path, str(tmp_path))
        assert os.path.exists(pdf_path)
