"""
Tests for cross-tool comparison module.
"""

import os

import numpy as np
import pandas as pd
import pytest

from splice.analysis.cross_tool_comparison import (
    compute_concordance_stats,
    generate_comparison_report,
    load_majiq_results,
    load_rmats_results,
    load_splice_results,
    load_suppa2_results,
    match_events_by_gene,
)


def _make_splice_tsv(path):
    """Create a minimal splice_results.tsv for testing."""
    df = pd.DataFrame({
        "module_id": ["mod1", "mod2", "mod3", "mod4"],
        "gene_id": ["ENSG00000001.1", "ENSG00000002.1", "ENSG00000003.1", "ENSG00000004.1"],
        "gene_name": ["GeneA", "GeneB", "GeneC", "GeneD"],
        "chrom": ["chr1"] * 4,
        "strand": ["+"] * 4,
        "event_type": ["SE", "A3SS", "A5SS", "Complex"],
        "max_abs_delta_psi": [0.3, 0.2, 0.4, 0.1],
        "p_value": [0.001, 0.04, 0.0001, 0.5],
        "fdr": [0.01, 0.04, 0.001, 0.6],
    })
    df.to_csv(path, sep="\t", index=False)


def _make_rmats_files(directory):
    """Create minimal rMATS output files."""
    os.makedirs(directory, exist_ok=True)
    se_df = pd.DataFrame({
        "ID": [1, 2],
        "GeneID": ['"ENSG00000001"', '"ENSG00000005"'],
        "geneSymbol": ['"GeneA"', '"GeneE"'],
        "chr": ["chr1", "chr1"],
        "strand": ["+", "+"],
        "PValue": [0.005, 0.01],
        "FDR": [0.02, 0.04],
        "IncLevelDifference": [0.25, -0.18],
    })
    se_df.to_csv(os.path.join(directory, "SE.MATS.JC.txt"), sep="\t", index=False)

    a3ss_df = pd.DataFrame({
        "ID": [1],
        "GeneID": ['"ENSG00000002"'],
        "geneSymbol": ['"GeneB"'],
        "chr": ["chr1"],
        "strand": ["+"],
        "PValue": [0.03],
        "FDR": [0.04],
        "IncLevelDifference": [0.22],
    })
    a3ss_df.to_csv(os.path.join(directory, "A3SS.MATS.JC.txt"), sep="\t", index=False)


def _make_majiq_tsv(path):
    """Create a minimal MAJIQ deltapsi.tsv."""
    df = pd.DataFrame({
        "Gene Name": ["GeneA", "GeneC", "GeneF"],
        "Gene ID": ["ENSG00000001", "ENSG00000003", "ENSG00000006"],
        "LSV ID": ["lsv1", "lsv2", "lsv3"],
        "mean_dpsi_per_lsv_junction": ["0.25;-0.25", "0.40;-0.40", "0.05;-0.05"],
        "P(|dPSI|>=0.20) per LSV junction": ["0.98;0.98", "0.99;0.99", "0.20;0.20"],
        "chr": ["chr1", "chr1", "chr1"],
        "strand": ["+", "+", "+"],
    })
    df.to_csv(path, sep="\t", index=False)


def _make_suppa2_dpsi(path):
    """Create a minimal SUPPA2 .dpsi file."""
    df = pd.DataFrame({
        "groups": [0.3, 0.15, 0.05],
        "p-value": [0.001, 0.03, 0.5],
    }, index=[
        "ENSG00000001;SE:chr1:100-200:300-400:+",
        "ENSG00000002;A3SS:chr1:500-600:+",
        "ENSG00000007;RI:chr1:700-800:+",
    ])
    df.to_csv(path, sep="\t")


class TestLoaders:

    def test_load_splice(self, tmp_path):
        path = str(tmp_path / "splice_results.tsv")
        _make_splice_tsv(path)
        df = load_splice_results(path)
        assert len(df) == 4
        assert "delta_psi" in df.columns
        assert "significant" in df.columns
        assert df["significant"].sum() == 3  # First three have FDR < 0.05

    def test_load_rmats(self, tmp_path):
        directory = str(tmp_path / "rmats")
        _make_rmats_files(directory)
        df = load_rmats_results(directory)
        assert len(df) == 3  # 2 SE + 1 A3SS
        assert "delta_psi" in df.columns
        assert all(df["tool"] == "rMATS")
        assert df["significant"].sum() == 3  # All have FDR < 0.05

    def test_load_rmats_missing_dir(self, tmp_path):
        df = load_rmats_results(str(tmp_path / "nonexistent"))
        assert len(df) == 0

    def test_load_majiq(self, tmp_path):
        path = str(tmp_path / "deltapsi.tsv")
        _make_majiq_tsv(path)
        df = load_majiq_results(path)
        assert len(df) == 3
        assert df["significant"].sum() == 2  # First two have prob > 0.95

    def test_load_suppa2(self, tmp_path):
        path = str(tmp_path / "test.dpsi")
        _make_suppa2_dpsi(path)
        df = load_suppa2_results(path)
        assert len(df) == 3
        assert df["significant"].sum() == 2  # First two have p < 0.05


class TestMatching:

    def test_match_events_by_gene(self, tmp_path):
        splice_path = str(tmp_path / "splice.tsv")
        _make_splice_tsv(splice_path)
        rmats_dir = str(tmp_path / "rmats")
        _make_rmats_files(rmats_dir)

        splice_df = load_splice_results(splice_path)
        rmats_df = load_rmats_results(rmats_dir)
        merged = match_events_by_gene(splice_df, rmats_df, "rMATS")

        assert "gene_id" in merged.columns
        assert "splice_significant" in merged.columns
        assert "rMATS_significant" in merged.columns
        assert "concordant" in merged.columns
        # GeneA and GeneB are in both as significant
        concordant_count = merged["concordant"].sum()
        assert concordant_count >= 2

    def test_match_empty(self):
        empty_df = pd.DataFrame(columns=[
            "gene_id", "significant", "delta_psi", "fdr",
        ])
        result = match_events_by_gene(empty_df, empty_df, "rMATS")
        assert len(result) == 0


class TestConcordanceStats:

    def test_compute_concordance(self, tmp_path):
        splice_path = str(tmp_path / "splice.tsv")
        _make_splice_tsv(splice_path)
        rmats_dir = str(tmp_path / "rmats")
        _make_rmats_files(rmats_dir)
        majiq_path = str(tmp_path / "deltapsi.tsv")
        _make_majiq_tsv(majiq_path)

        splice_df = load_splice_results(splice_path)
        rmats_df = load_rmats_results(rmats_dir)
        majiq_df = load_majiq_results(majiq_path)

        stats = compute_concordance_stats(splice_df, rmats_df, majiq_df)

        assert "splice_n_significant" in stats
        assert "rMATS" in stats
        assert "MAJIQ" in stats
        assert stats["splice_n_significant"] == 3
        assert stats["rMATS"]["n_significant"] == 3
        assert 0.0 <= stats["rMATS"]["jaccard"] <= 1.0


class TestComparisonReport:

    def test_generate_full_report(self, tmp_path):
        splice_dir = tmp_path / "splice_out"
        splice_dir.mkdir()
        _make_splice_tsv(str(splice_dir / "splice_results.tsv"))

        rmats_dir = tmp_path / "rmats_out"
        _make_rmats_files(str(rmats_dir))

        majiq_dir = tmp_path / "majiq_out"
        majiq_dir.mkdir()
        _make_majiq_tsv(str(majiq_dir / "deltapsi.tsv"))

        suppa2_dir = tmp_path / "suppa2_out"
        suppa2_dir.mkdir()
        _make_suppa2_dpsi(str(suppa2_dir / "test.dpsi"))

        output_dir = tmp_path / "comparison"
        stats = generate_comparison_report(
            splice_dir=str(splice_dir),
            rmats_dir=str(rmats_dir),
            majiq_dir=str(majiq_dir),
            suppa2_dir=str(suppa2_dir),
            output_dir=str(output_dir),
        )

        assert (output_dir / "concordance_summary.tsv").exists()
        assert (output_dir / "venn_diagram.svg").exists()
        assert (output_dir / "upset_plot.svg").exists()
        assert (output_dir / "concordance_heatmap.svg").exists()
        assert (output_dir / "delta_psi_correlation.svg").exists()
        assert "splice_n_significant" in stats

    def test_report_splice_only(self, tmp_path):
        """Should work with only SPLICE results."""
        splice_dir = tmp_path / "splice_out"
        splice_dir.mkdir()
        _make_splice_tsv(str(splice_dir / "splice_results.tsv"))

        output_dir = tmp_path / "comparison"
        stats = generate_comparison_report(
            splice_dir=str(splice_dir),
            output_dir=str(output_dir),
        )

        assert (output_dir / "concordance_summary.tsv").exists()
        assert stats["splice_n_significant"] == 3
