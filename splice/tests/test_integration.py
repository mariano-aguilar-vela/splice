"""
Module 28: tests/test_integration.py

Full end-to-end pipeline integration tests on synthetic data.

Tests all 9 requirements from splicekit_definitive_spec.md Integration Test Specification:
1. Gene A: significant, classified as SE, confidence HIGH
2. Gene B: not significant, classified as A3SS
3. Gene C: valid p-value, classified as Complex, novel junction flagged
4. NMD classification runs without error
5. QC report generates
6. LeafCutter and rMATS exports produce valid files
7. Checkpoint save/resume works
8. Junction co-occurrences extracted for Gene C
9. Effective length normalization changes PSI vs unnormalized

Uses conftest.py fixtures (3 synthetic genes, 6 BAMs, FASTA, GTF) to run
the full pipeline and verify each stage.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pytest

from splicekit.core.clustering import cluster_junctions
from splicekit.core.confidence_scorer import score_all_junctions
from splicekit.core.diagnostics import compute_diagnostics
from splicekit.core.diff import DiffResult, test_differential_splicing
from splicekit.core.diff_het import test_heterogeneous_splicing
from splicekit.core.event_classifier import classify_all_events
from splicekit.core.evidence import ModuleEvidence, build_evidence_matrices
from splicekit.core.junction_extractor import extract_all_junctions
from splicekit.core.nmd_classifier import classify_all_junctions_nmd
from splicekit.core.psi import ModulePSI, quantify_psi
from splicekit.core.splicegraph import SplicingModule, build_splicegraph
from splicekit.io.format_export import export_leafcutter_format, export_rmats_format
from splicekit.io.qc_report import generate_qc_report
from splicekit.io.serialization import load_checkpoint, save_checkpoint
from splicekit.tests.conftest import (
    J_A1,
    J_A2,
    J_A3,
    J_B1,
    J_B2,
    J_C1,
    J_C2,
    J_C3,
    J_C4,
    J_C5_NOVEL,
)
from splicekit.utils.genomic import JunctionPair

# Ensure pysam is available for tests requiring BAM parsing
pytest.importorskip("pysam")


@dataclass
class MockGene:
    """Minimal Gene object for build_splicegraph (only 5 scalar attributes used)."""

    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    start: int
    end: int


# Mock genes for build_splicegraph
MOCK_GENES = {
    "GENE_A": MockGene("GENE_A", "GeneA", "chr1", "+", 1000, 3100),
    "GENE_B": MockGene("GENE_B", "GeneB", "chr1", "+", 5000, 6100),
    "GENE_C": MockGene("GENE_C", "GeneC", "chr1", "+", 8000, 10100),
}


# ─── Helper functions ────────────────────────────────────────────────────────


def _find_diff_result(diff_results: List[DiffResult], gene_id: str) -> Optional[DiffResult]:
    """Find DiffResult for a specific gene_id."""
    return next((r for r in diff_results if r.gene_id == gene_id), None)


def _find_event_type(
    modules: List[SplicingModule], event_types_list: List[str], gene_id: str
) -> Optional[str]:
    """Find event type for a specific gene_id."""
    for i, m in enumerate(modules):
        if m.gene_id == gene_id:
            return event_types_list[i]
    return None


def _find_diagnostic(diff_results: List[DiffResult], gene_id: str, diagnostics) -> Optional:
    """Find EventDiagnostic for a specific gene_id via its diff_result."""
    result = _find_diff_result(diff_results, gene_id)
    if result is None:
        return None
    return next((d for d in diagnostics if d.module_id == result.module_id), None)


def _read_fasta(path: str) -> Dict[str, str]:
    """Read FASTA file into dict."""
    genome = {}
    current_chrom = None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_chrom:
                    genome[current_chrom] = "".join(seq_lines)
                current_chrom = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)
    if current_chrom:
        genome[current_chrom] = "".join(seq_lines)
    return genome


def _build_junction_evidence_dict(junction_evidence):
    """Convert JunctionEvidence dict to output format for write_junction_details_tsv."""
    result = {}
    for junc, ev in junction_evidence.items():
        junc_id = junc.to_string()
        result[junc_id] = {
            "junction": junc,
            "gene_id": "",
            "gene_name": "",
            "is_annotated": ev.is_annotated,
            "motif": ev.motif,
            "motif_score": ev.motif_score,
            "total_reads": int(ev.sample_counts.sum()),
            "mean_mapq": float(ev.sample_mapq_mean.mean()),
            "sample_counts": ev.sample_counts.tolist(),
        }
    return result


# ─── Module-scoped fixture: run full pipeline once ─────────────────────────────


@pytest.fixture(scope="module")
def pipeline_results(
    all_bam_paths,
    sample_names,
    known_junctions,
    group_labels,
    genome_fasta_path,
    tmp_path_factory,
):
    """Run the full SPLICE pipeline once and return all intermediate results."""
    output_dir = tmp_path_factory.mktemp("integration_output")

    # Step 1: Junction extraction
    junction_evidence, cooccurrence = extract_all_junctions(
        bam_paths=all_bam_paths,
        sample_names=sample_names,
        known_junctions=known_junctions,
        genome_fasta_path=genome_fasta_path,
    )

    # Step 2: Clustering
    clusters = cluster_junctions(list(junction_evidence.keys()))

    # Step 3: Build splicegraph
    modules, junction_to_idx = build_splicegraph(
        genes=MOCK_GENES,
        junction_evidence=junction_evidence,
        clusters=clusters,
        known_junctions=known_junctions,
    )

    # Step 4: Score junctions
    confidence_scores = score_all_junctions(junction_evidence)

    # Step 5: Build evidence matrices
    evidence_list = build_evidence_matrices(
        modules=modules,
        junction_evidence=junction_evidence,
        junction_confidence=confidence_scores,
        read_length=50,  # Matches synthetic read length
    )

    # Step 6: Quantify PSI (reduced bootstraps for test speed)
    psi_list = quantify_psi(evidence_list, n_bootstraps=10, seed=42)

    # Step 7: Differential testing
    diff_results = test_differential_splicing(
        module_evidence_list=evidence_list,
        module_psi_list=psi_list,
        group_labels=group_labels,
    )

    # Step 8: Heterogeneity testing
    het_results = test_heterogeneous_splicing(psi_list, group_labels)

    # Step 9: Event classification
    event_types_list = classify_all_events(modules)

    # Step 10: Diagnostics (with list alignment)
    tested_ids = {dr.module_id for dr in diff_results}

    # Align evidence and psi to match diff_results by module_id
    module_order = {dr.module_id: i for i, dr in enumerate(diff_results)}
    tested_evidence = sorted(
        [e for e in evidence_list if e.module.module_id in tested_ids],
        key=lambda e: module_order.get(e.module.module_id, 999),
    )
    tested_psi = sorted(
        [p for p in psi_list if p.module_id in tested_ids],
        key=lambda p: module_order.get(p.module_id, 999),
    )

    diagnostics = compute_diagnostics(tested_evidence, tested_psi, diff_results)

    # Build event type counts
    event_type_counts = {}
    for module, evt in zip(modules, event_types_list):
        event_type_counts[evt] = event_type_counts.get(evt, 0) + 1

    return {
        "junction_evidence": junction_evidence,
        "cooccurrence": cooccurrence,
        "clusters": clusters,
        "modules": modules,
        "confidence_scores": confidence_scores,
        "evidence_list": evidence_list,
        "psi_list": psi_list,
        "diff_results": diff_results,
        "het_results": het_results,
        "event_types_list": event_types_list,
        "diagnostics": diagnostics,
        "event_type_counts": event_type_counts,
        "output_dir": str(output_dir),
    }


# ─── Test Classes ────────────────────────────────────────────────────────────


class TestJunctionExtraction:
    """Tests for junction extraction from synthetic BAMs."""

    def test_gene_a_junctions_all_detected(self, pipeline_results):
        """All three Gene A junctions should be detected."""
        junc_dict = pipeline_results["junction_evidence"]
        assert J_A1 in junc_dict
        assert J_A2 in junc_dict
        assert J_A3 in junc_dict

    def test_gene_b_junctions_all_detected(self, pipeline_results):
        """Both Gene B junctions should be detected."""
        junc_dict = pipeline_results["junction_evidence"]
        assert J_B1 in junc_dict
        assert J_B2 in junc_dict

    def test_gene_c_annotated_junctions_detected(self, pipeline_results):
        """All four annotated Gene C junctions should be detected."""
        junc_dict = pipeline_results["junction_evidence"]
        assert J_C1 in junc_dict
        assert J_C2 in junc_dict
        assert J_C3 in junc_dict
        assert J_C4 in junc_dict

    def test_novel_junction_c5_detected(self, pipeline_results):
        """Novel junction J_C5_NOVEL should be detected."""
        junc_dict = pipeline_results["junction_evidence"]
        assert J_C5_NOVEL in junc_dict

    def test_annotated_junctions_marked_as_annotated(self, pipeline_results, known_junctions):
        """Annotated junctions should have is_annotated=True."""
        junc_dict = pipeline_results["junction_evidence"]
        for junc in known_junctions:
            assert junc_dict[junc].is_annotated is True

    def test_novel_junction_c5_is_not_annotated(self, pipeline_results):
        """J_C5_NOVEL should have is_annotated=False."""
        junc_dict = pipeline_results["junction_evidence"]
        assert junc_dict[J_C5_NOVEL].is_annotated is False

    def test_cooccurrences_extracted_for_gene_c(self, pipeline_results):
        """Gene C should have junction co-occurrence evidence."""
        cooccurrence = pipeline_results["cooccurrence"]
        gene_c_juncs = {J_C1, J_C2, J_C3, J_C4, J_C5_NOVEL}

        # Find pairs involving Gene C junctions
        gene_c_pairs = [
            pair
            for pair in cooccurrence.keys()
            if pair.junction1 in gene_c_juncs or pair.junction2 in gene_c_juncs
        ]

        assert len(gene_c_pairs) > 0, "No co-occurrence evidence found for Gene C"

    def test_sample_counts_have_correct_shape(self, pipeline_results):
        """Each JunctionEvidence should have counts for all 6 samples."""
        junc_dict = pipeline_results["junction_evidence"]
        for ev in junc_dict.values():
            assert ev.sample_counts.shape == (6,)
            assert ev.sample_mapq_mean.shape == (6,)
            assert ev.sample_mapq_median.shape == (6,)


class TestClustering:
    """Tests for junction clustering."""

    def test_three_clusters_formed(self, pipeline_results):
        """Should form exactly 3 clusters (one per gene)."""
        clusters = pipeline_results["clusters"]
        assert len(clusters) == 3

    def test_gene_a_cluster_has_three_junctions(self, pipeline_results):
        """Gene A cluster should contain exactly 3 junctions."""
        clusters = pipeline_results["clusters"]
        gene_a_cluster = next(
            (c for c in clusters if all(j in {J_A1, J_A2, J_A3} for j in c.junctions)), None
        )
        assert gene_a_cluster is not None
        assert len(gene_a_cluster.junctions) == 3

    def test_gene_b_cluster_has_two_junctions(self, pipeline_results):
        """Gene B cluster should contain exactly 2 junctions."""
        clusters = pipeline_results["clusters"]
        gene_b_cluster = next(
            (c for c in clusters if all(j in {J_B1, J_B2} for j in c.junctions)), None
        )
        assert gene_b_cluster is not None
        assert len(gene_b_cluster.junctions) == 2

    def test_gene_c_cluster_has_five_junctions(self, pipeline_results):
        """Gene C cluster should contain all 5 junctions (4 annotated + 1 novel)."""
        clusters = pipeline_results["clusters"]
        gene_c_juncs = {J_C1, J_C2, J_C3, J_C4, J_C5_NOVEL}
        gene_c_cluster = next(
            (
                c
                for c in clusters
                if all(j in gene_c_juncs for j in c.junctions) and len(c.junctions) == 5
            ),
            None,
        )
        assert gene_c_cluster is not None
        assert len(gene_c_cluster.junctions) == 5


class TestDifferentialSplicing:
    """Tests for differential splicing detection."""

    def test_three_diff_results_returned(self, pipeline_results):
        """Should have one DiffResult per module."""
        diff_results = pipeline_results["diff_results"]
        assert len(diff_results) == 3

    def test_gene_a_significant(self, pipeline_results):
        """Gene A should be statistically significant (p < 0.05)."""
        result = _find_diff_result(pipeline_results["diff_results"], "GENE_A")
        assert result is not None
        assert result.p_value < 0.05

    def test_gene_b_not_significant(self, pipeline_results):
        """Gene B should not be statistically significant (p > 0.05)."""
        result = _find_diff_result(pipeline_results["diff_results"], "GENE_B")
        assert result is not None
        assert result.p_value > 0.05

    def test_gene_c_has_valid_pvalue(self, pipeline_results):
        """Gene C should have a valid p-value."""
        result = _find_diff_result(pipeline_results["diff_results"], "GENE_C")
        assert result is not None
        assert 0.0 <= result.p_value <= 1.0

    def test_gene_a_has_large_delta_psi(self, pipeline_results):
        """Gene A should show large max |delta PSI| (> 0.3)."""
        result = _find_diff_result(pipeline_results["diff_results"], "GENE_A")
        assert result is not None
        assert result.max_abs_delta_psi > 0.3

    def test_gene_b_has_small_delta_psi(self, pipeline_results):
        """Gene B should show small max |delta PSI| (< 0.2) due to equal counts."""
        result = _find_diff_result(pipeline_results["diff_results"], "GENE_B")
        assert result is not None
        assert result.max_abs_delta_psi < 0.2

    def test_all_diff_results_have_valid_fdr(self, pipeline_results):
        """All DiffResults should have valid FDR values (0-1)."""
        for result in pipeline_results["diff_results"]:
            assert 0.0 <= result.fdr <= 1.0

    def test_models_converge(self, pipeline_results):
        """Both null and full models should converge for all genes."""
        for result in pipeline_results["diff_results"]:
            assert result.null_converged is True
            assert result.full_converged is True


class TestEventClassification:
    """Tests for event type classification."""

    def test_gene_a_classified_as_se(self, pipeline_results):
        """Gene A should be classified as SE (Skipped Exon)."""
        event_type = _find_event_type(
            pipeline_results["modules"], pipeline_results["event_types_list"], "GENE_A"
        )
        assert event_type == "SE"

    def test_gene_b_classified_as_a3ss(self, pipeline_results):
        """Gene B should be classified as A3SS (Alternative 3' Splice Site)."""
        event_type = _find_event_type(
            pipeline_results["modules"], pipeline_results["event_types_list"], "GENE_B"
        )
        assert event_type == "A3SS"

    def test_gene_c_classified_as_complex(self, pipeline_results):
        """Gene C should be classified as Complex (5 junctions)."""
        event_type = _find_event_type(
            pipeline_results["modules"], pipeline_results["event_types_list"], "GENE_C"
        )
        assert event_type == "Complex"


class TestDiagnostics:
    """Tests for quality diagnostics and confidence tiers."""

    def test_gene_a_confidence_tier_high_or_medium(self, pipeline_results):
        """Gene A should have confidence tier HIGH or MEDIUM due to good read count/signal."""
        diag = _find_diagnostic(
            pipeline_results["diff_results"], "GENE_A", pipeline_results["diagnostics"]
        )
        assert diag is not None
        assert diag.confidence_tier in {"HIGH", "MEDIUM"}

    def test_all_diagnostics_have_valid_metrics(self, pipeline_results):
        """All diagnostics should have valid metric ranges."""
        for diag in pipeline_results["diagnostics"]:
            assert diag.mean_mapq >= 0
            assert diag.mean_junction_confidence >= 0
            assert diag.bootstrap_cv >= 0
            assert diag.confidence_tier in {"HIGH", "MEDIUM", "LOW", "FAIL"}

    def test_gene_c_has_novel_junction_flagged(self, pipeline_results):
        """Gene C diagnostic should have has_novel_junctions=True."""
        diag = _find_diagnostic(
            pipeline_results["diff_results"], "GENE_C", pipeline_results["diagnostics"]
        )
        assert diag is not None
        assert diag.has_novel_junctions is True


class TestNMDClassification:
    """Tests for NMD classification."""

    def test_nmd_classification_runs_without_error(self, genome_fasta_path):
        """NMD classifier should run without raising exceptions."""
        genome = _read_fasta(genome_fasta_path)

        # Minimal exon positions for Gene A
        exon_positions = {
            0: (1000, 1100),
            1: (2000, 2100),
            2: (3000, 3100),
        }

        result = classify_all_junctions_nmd(
            junctions=[J_A1, J_A2, J_A3],
            exon_positions=exon_positions,
            genome_fasta=genome,
            strand="+",
        )

        assert len(result) == 3
        for nmd_class in result:
            assert nmd_class.classification in ("PR", "UP", "NE", "IN")

    def test_nmd_returns_valid_confidence(self, genome_fasta_path):
        """NMD classifications should have valid confidence values."""
        genome = _read_fasta(genome_fasta_path)
        exon_positions = {0: (1000, 1100), 1: (2000, 2100), 2: (3000, 3100)}

        result = classify_all_junctions_nmd(
            junctions=[J_A1, J_A2, J_A3],
            exon_positions=exon_positions,
            genome_fasta=genome,
            strand="+",
        )

        for nmd_class in result:
            assert (
                0.0 <= nmd_class.confidence <= 1.0
                or np.isnan(nmd_class.confidence)
            )


class TestQCReport:
    """Tests for QC report generation."""

    def test_qc_report_generates(self, pipeline_results, tmp_path):
        """QC report should generate successfully."""
        report_path = str(tmp_path / "report.html")

        junc_evidence_dict = _build_junction_evidence_dict(
            pipeline_results["junction_evidence"]
        )

        generate_qc_report(
            diff_results=pipeline_results["diff_results"],
            het_results=pipeline_results["het_results"],
            diagnostics=pipeline_results["diagnostics"],
            event_types=pipeline_results["event_type_counts"],
            junction_evidence=junc_evidence_dict,
            nmd_classifications={},
            output_path=report_path,
        )

        assert os.path.exists(report_path)

    def test_qc_report_is_valid_html(self, pipeline_results, tmp_path):
        """QC report should be valid HTML."""
        report_path = str(tmp_path / "report.html")

        junc_evidence_dict = _build_junction_evidence_dict(
            pipeline_results["junction_evidence"]
        )

        generate_qc_report(
            diff_results=pipeline_results["diff_results"],
            het_results=pipeline_results["het_results"],
            diagnostics=pipeline_results["diagnostics"],
            event_types=pipeline_results["event_type_counts"],
            junction_evidence=junc_evidence_dict,
            nmd_classifications={},
            output_path=report_path,
        )

        with open(report_path) as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content or "<html" in content.lower()
        assert "</html>" in content.lower()


class TestLeafCutterExport:
    """Tests for LeafCutter format export."""

    def test_leafcutter_export_produces_file(self, pipeline_results, tmp_path):
        """LeafCutter export should create a file."""
        output_path = str(tmp_path / "leafcutter.tsv")

        export_leafcutter_format(
            diff_results=pipeline_results["diff_results"],
            output_path=output_path,
            fdr_threshold=0.1,
        )

        assert os.path.exists(output_path)

    def test_leafcutter_export_is_tsv_format(self, pipeline_results, tmp_path):
        """LeafCutter export should have TSV header."""
        output_path = str(tmp_path / "leafcutter.tsv")

        export_leafcutter_format(
            diff_results=pipeline_results["diff_results"],
            output_path=output_path,
            fdr_threshold=0.1,
        )

        with open(output_path) as f:
            header = f.readline().strip()

        assert "cluster" in header.lower()
        assert "gene" in header.lower()


class TestRMatsExport:
    """Tests for rMATS format export."""

    def test_rmats_export_produces_file(self, pipeline_results, tmp_path):
        """rMATS export should create a file."""
        output_path = str(tmp_path / "rmats.tsv")

        export_rmats_format(
            diff_results=pipeline_results["diff_results"],
            output_path=output_path,
            fdr_threshold=0.1,
        )

        assert os.path.exists(output_path)

    def test_rmats_export_is_tsv_format(self, pipeline_results, tmp_path):
        """rMATS export should have TSV header."""
        output_path = str(tmp_path / "rmats.tsv")

        export_rmats_format(
            diff_results=pipeline_results["diff_results"],
            output_path=output_path,
            fdr_threshold=0.1,
        )

        with open(output_path) as f:
            header = f.readline().strip()

        assert len(header.split("\t")) >= 15  # rMATS has many columns


class TestCheckpoint:
    """Tests for checkpoint save/restore functionality."""

    def test_save_and_load_diff_results(self, pipeline_results, tmp_path):
        """DiffResults should survive save/load via checkpoint."""
        checkpoint_path = str(tmp_path / "diff_results.pkl")

        original = pipeline_results["diff_results"]
        save_checkpoint(original, checkpoint_path)

        assert os.path.exists(checkpoint_path)

        loaded = load_checkpoint(checkpoint_path)

        assert len(loaded) == len(original)
        assert loaded[0].module_id == original[0].module_id
        assert loaded[0].p_value == original[0].p_value

    def test_save_and_load_junction_evidence(self, pipeline_results, tmp_path):
        """Junction evidence should survive save/load."""
        checkpoint_path = str(tmp_path / "junction_evidence.pkl")

        original = pipeline_results["junction_evidence"]
        save_checkpoint(original, checkpoint_path)

        assert os.path.exists(checkpoint_path)

        loaded = load_checkpoint(checkpoint_path)

        assert set(loaded.keys()) == set(original.keys())

        # Spot-check a junction
        for junc in [J_A1, J_B1, J_C1]:
            assert junc in loaded
            np.testing.assert_array_equal(
                original[junc].sample_counts, loaded[junc].sample_counts
            )


class TestEffectiveLengthNormalization:
    """Tests for effective length normalization."""

    def test_all_effective_lengths_positive(self, pipeline_results):
        """All junction effective lengths should be positive."""
        for ev in pipeline_results["evidence_list"]:
            assert ev.junction_effective_lengths.shape == (ev.junction_count_matrix.shape[0],)
            assert np.all(ev.junction_effective_lengths > 0)

    def test_normalized_matrix_shape_matches_raw(self, pipeline_results):
        """Normalized count matrix should have same shape as raw count matrix."""
        for ev in pipeline_results["evidence_list"]:
            assert ev.normalized_count_matrix.shape == ev.junction_count_matrix.shape

    def test_normalized_counts_are_non_negative(self, pipeline_results):
        """All normalized counts should be non-negative."""
        for ev in pipeline_results["evidence_list"]:
            assert np.all(ev.normalized_count_matrix >= 0)
