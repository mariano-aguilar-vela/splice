"""
Tests for chromosome-level pipeline parallelism.

Tests process_chromosome(), merge_chromosome_results(), and the
chromosome-parallel execution path.
"""

import numpy as np
import pytest

from splice.core.chromosome_pipeline import (
    ChromosomeResult,
    merge_chromosome_results,
    process_chromosome,
)
from splice.core.diff import DiffResult
from splice.core.junction_extractor import JunctionEvidence
from splice.utils.genomic import Junction
from splice.utils.stats import benjamini_hochberg


# Mock Gene compatible with splicegraph.build_splicegraph
class _MockGene:
    def __init__(self, gene_id, gene_name, chrom, strand, start, end):
        self.gene_id = gene_id
        self.gene_name = gene_name
        self.chrom = chrom
        self.strand = strand
        self.start = start
        self.end = end
        self.exons = []
        self.transcripts = {}


_MOCK_GENES = {
    "GENE_A": _MockGene("GENE_A", "GeneA", "chr1", "+", 1000, 3100),
    "GENE_B": _MockGene("GENE_B", "GeneB", "chr1", "+", 5000, 6100),
    "GENE_C": _MockGene("GENE_C", "GeneC", "chr1", "+", 8000, 10100),
}


class TestChrPipelineSmoke:
    """Smoke tests for chromosome pipeline dataclasses."""

    def test_empty_chromosome_result(self):
        """ChromosomeResult with empty data should have correct defaults."""
        cr = ChromosomeResult(
            chrom="chrZ",
            junction_evidence={},
            cooccurrence={},
            modules=[],
            evidence_list=[],
            psi_list=[],
            diff_results=[],
            het_results=[],
            event_types=[],
            diagnostics=[],
            n_junctions_raw=0,
            n_junctions_filtered=0,
            n_clusters=0,
            n_modules=0,
            n_tested=0,
            elapsed_seconds=0.0,
        )
        assert cr.chrom == "chrZ"
        assert cr.n_junctions_raw == 0
        assert len(cr.modules) == 0


class TestMergeChromosomeResults:
    """Tests for merge_chromosome_results()."""

    def test_merge_empty_list(self):
        """Merging empty list should return empty collections."""
        result = merge_chromosome_results([])
        (junc_ev, cooc, modules, evidence, psi,
         diff, het, event_types, diagnostics) = result

        assert len(junc_ev) == 0
        assert len(modules) == 0
        assert len(diff) == 0

    def test_merge_single_empty_chromosome(self):
        """Merging a single empty ChromosomeResult."""
        cr = ChromosomeResult(
            chrom="chr1",
            junction_evidence={},
            cooccurrence={},
            modules=[],
            evidence_list=[],
            psi_list=[],
            diff_results=[],
            het_results=[],
            event_types=[],
            diagnostics=[],
            n_junctions_raw=0,
            n_junctions_filtered=0,
            n_clusters=0,
            n_modules=0,
            n_tested=0,
            elapsed_seconds=0.1,
        )
        result = merge_chromosome_results([cr])
        (junc_ev, cooc, modules, evidence, psi,
         diff, het, event_types, diagnostics) = result

        assert len(junc_ev) == 0
        assert len(diff) == 0

    def test_merge_preserves_junction_evidence(self):
        """Merging should combine junction evidence from multiple chromosomes."""
        j1 = Junction("chr1", 100, 200, "+")
        j2 = Junction("chr2", 300, 400, "+")

        ev1 = JunctionEvidence(
            junction=j1,
            sample_counts=np.array([10, 5]),
            sample_weighted_counts=np.array([10.0, 5.0]),
            sample_mapq_mean=np.array([60.0, 55.0]),
            sample_mapq_median=np.array([60.0, 55.0]),
            sample_nh_distribution=np.array([1.0, 1.0]),
            is_annotated=True, motif="GT/AG", motif_score=1.0,
            max_anchor=50, n_samples_detected=2, cross_sample_recurrence=1.0,
        )
        ev2 = JunctionEvidence(
            junction=j2,
            sample_counts=np.array([8, 3]),
            sample_weighted_counts=np.array([8.0, 3.0]),
            sample_mapq_mean=np.array([50.0, 45.0]),
            sample_mapq_median=np.array([50.0, 45.0]),
            sample_nh_distribution=np.array([1.0, 1.0]),
            is_annotated=False, motif="", motif_score=0.0,
            max_anchor=30, n_samples_detected=2, cross_sample_recurrence=1.0,
        )

        cr1 = ChromosomeResult(
            chrom="chr1", junction_evidence={j1: ev1}, cooccurrence={},
            modules=[], evidence_list=[], psi_list=[], diff_results=[],
            het_results=[], event_types=[], diagnostics=[],
            n_junctions_raw=1, n_junctions_filtered=1, n_clusters=0,
            n_modules=0, n_tested=0, elapsed_seconds=0.1,
        )
        cr2 = ChromosomeResult(
            chrom="chr2", junction_evidence={j2: ev2}, cooccurrence={},
            modules=[], evidence_list=[], psi_list=[], diff_results=[],
            het_results=[], event_types=[], diagnostics=[],
            n_junctions_raw=1, n_junctions_filtered=1, n_clusters=0,
            n_modules=0, n_tested=0, elapsed_seconds=0.1,
        )

        result = merge_chromosome_results([cr1, cr2])
        junc_ev = result[0]
        assert len(junc_ev) == 2
        assert j1 in junc_ev
        assert j2 in junc_ev

    def test_global_fdr_correction(self):
        """FDR should be recomputed globally, not per-chromosome."""
        def _make_diff(module_id, chrom, p_value):
            return DiffResult(
                module_id=module_id, gene_id="G", gene_name="Gene",
                chrom=chrom, strand="+", event_type="SE", n_junctions=2,
                junction_coords=["chr1:100-200:+"], junction_confidence=[0.9],
                is_annotated=[True],
                psi_group1=np.array([0.8]), psi_group2=np.array([0.5]),
                delta_psi=np.array([-0.3]), max_abs_delta_psi=0.3,
                delta_psi_ci_low=np.array([-0.4]),
                delta_psi_ci_high=np.array([-0.2]),
                log_likelihood_null=-100.0, log_likelihood_full=-90.0,
                degrees_of_freedom=1, p_value=p_value, fdr=1.0,
                null_converged=True, full_converged=True,
                null_refit_used=False, null_iterations=10,
                full_iterations=10, null_gradient_norm=0.01,
                full_gradient_norm=0.01,
            )

        cr1 = ChromosomeResult(
            chrom="chr1", junction_evidence={}, cooccurrence={},
            modules=[], evidence_list=[], psi_list=[],
            diff_results=[
                _make_diff("mod1", "chr1", 0.01),
                _make_diff("mod2", "chr1", 0.05),
            ],
            het_results=[], event_types=[], diagnostics=[],
            n_junctions_raw=10, n_junctions_filtered=5, n_clusters=2,
            n_modules=2, n_tested=2, elapsed_seconds=1.0,
        )
        cr2 = ChromosomeResult(
            chrom="chr2", junction_evidence={}, cooccurrence={},
            modules=[], evidence_list=[], psi_list=[],
            diff_results=[
                _make_diff("mod3", "chr2", 0.03),
                _make_diff("mod4", "chr2", 0.10),
            ],
            het_results=[], event_types=[], diagnostics=[],
            n_junctions_raw=8, n_junctions_filtered=4, n_clusters=1,
            n_modules=2, n_tested=2, elapsed_seconds=0.8,
        )

        result = merge_chromosome_results([cr1, cr2])
        diff_results = result[5]

        assert len(diff_results) == 4

        for dr in diff_results:
            assert 0.0 <= dr.fdr <= 1.0
            assert dr.fdr >= dr.p_value

        # FDR values should be computed from all 4 p-values together
        all_p = np.array([0.01, 0.05, 0.03, 0.10])
        expected_fdr = benjamini_hochberg(all_p)
        actual_fdr = np.array([dr.fdr for dr in diff_results])
        np.testing.assert_array_almost_equal(actual_fdr, expected_fdr)


class TestProcessChromosomeIntegration:
    """Integration tests using conftest synthetic BAMs (on chr1)."""

    def test_process_chromosome_with_data(self, all_bam_paths,
                                          sample_names,
                                          known_junctions,
                                          group_labels):
        """process_chromosome should return results for chr1 which has data."""
        result = process_chromosome(
            chrom="chr1",
            bam_paths=all_bam_paths,
            sample_names=sample_names,
            genes=_MOCK_GENES,
            known_junctions=known_junctions,
            group_labels=group_labels,
            min_cluster_reads=1,
            n_bootstraps=5,
            read_length=50,
            run_het=False,
        )

        assert result.chrom == "chr1"
        assert result.n_junctions_raw > 0
        assert result.elapsed_seconds > 0

    def test_process_chromosome_no_data(self, all_bam_paths,
                                        sample_names,
                                        known_junctions,
                                        group_labels):
        """process_chromosome should return empty result for chromosome with no data."""
        result = process_chromosome(
            chrom="chrZ",
            bam_paths=all_bam_paths,
            sample_names=sample_names,
            genes=_MOCK_GENES,
            known_junctions=known_junctions,
            group_labels=group_labels,
            min_cluster_reads=1,
            n_bootstraps=5,
            read_length=50,
            run_het=False,
        )

        assert result.chrom == "chrZ"
        assert result.n_junctions_raw == 0
        assert result.modules == []
        assert result.diff_results == []

    def test_process_chromosome_returns_correct_chrom(self, all_bam_paths,
                                                      sample_names,
                                                      known_junctions,
                                                      group_labels):
        """All junctions should be on the requested chromosome."""
        result = process_chromosome(
            chrom="chr1",
            bam_paths=all_bam_paths,
            sample_names=sample_names,
            genes=_MOCK_GENES,
            known_junctions=known_junctions,
            group_labels=group_labels,
            min_cluster_reads=1,
            n_bootstraps=5,
            read_length=50,
            run_het=False,
        )

        for junc in result.junction_evidence:
            assert junc.chrom == "chr1"
