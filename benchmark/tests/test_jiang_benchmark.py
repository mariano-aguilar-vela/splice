"""
Tests for Jiang et al. 2023 benchmark module.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Allow importing benchmark module from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from benchmark.jiang2023_benchmark import (
    PUBLISHED_TOOLS,
    compare_with_published_results,
    evaluate_splice_results,
    generate_benchmark_report,
    load_ground_truth,
    prepare_splice_input,
    _compute_metrics,
    _normalize_gene_id,
)


def _make_truth_tsv(path):
    """Create a minimal ground truth TSV."""
    df = pd.DataFrame({
        "gene_id": [
            "ENSG00000001", "ENSG00000002", "ENSG00000003",
            "ENSG00000004", "ENSG00000005", "ENSG00000006",
        ],
        "event_type": ["SE", "A3SS", "A5SS", "MXE", "RI", "SE"],
        "chrom": ["chr1"] * 6,
        "start": [100, 200, 300, 400, 500, 600],
        "end": [200, 300, 400, 500, 600, 700],
        "strand": ["+"] * 6,
        "is_differential": [True, True, True, True, False, False],
    })
    df.to_csv(path, sep="\t", index=False)


def _make_splice_results_tsv(path):
    """Create a minimal splice_results.tsv mirroring the truth structure."""
    df = pd.DataFrame({
        "module_id": [f"mod{i}" for i in range(6)],
        "gene_id": [
            "ENSG00000001.1", "ENSG00000002.1", "ENSG00000003.1",
            "ENSG00000007.1", "ENSG00000005.1", "ENSG00000008.1",
        ],
        "gene_name": [f"Gene{i}" for i in range(6)],
        "chrom": ["chr1"] * 6,
        "strand": ["+"] * 6,
        "event_type": ["SE", "A3SS", "A5SS", "SE", "RI", "Complex"],
        "max_abs_delta_psi": [0.3, 0.4, 0.5, 0.2, 0.1, 0.05],
        "p_value": [0.001, 0.002, 0.003, 0.04, 0.5, 0.7],
        "fdr": [0.005, 0.01, 0.02, 0.04, 0.6, 0.8],
    })
    df.to_csv(path, sep="\t", index=False)


class TestComputeMetrics:

    def test_perfect_classification(self):
        m = _compute_metrics(tp=10, fp=0, fn=0)
        assert m["tpr"] == 1.0
        assert m["fdr"] == 0.0
        assert m["f_score"] == 1.0

    def test_zero_tp(self):
        m = _compute_metrics(tp=0, fp=5, fn=10)
        assert m["tpr"] == 0.0
        assert m["fdr"] == 1.0
        assert m["f_score"] == 0.0

    def test_partial_classification(self):
        m = _compute_metrics(tp=8, fp=2, fn=2)
        assert m["tpr"] == 0.8
        assert m["fdr"] == 0.2
        assert m["f_score"] == pytest.approx(0.8)


class TestNormalizeGeneId:

    def test_strips_version(self):
        assert _normalize_gene_id("ENSG00000001.5") == "ENSG00000001"

    def test_no_version(self):
        assert _normalize_gene_id("ENSG00000001") == "ENSG00000001"

    def test_strips_quotes(self):
        assert _normalize_gene_id('"ENSG00000001.5"') == "ENSG00000001"

    def test_empty(self):
        assert _normalize_gene_id("") == ""


class TestLoadGroundTruth:

    def test_load_from_tsv(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_truth_tsv(str(data_dir / "ground_truth.tsv"))

        df = load_ground_truth(str(tmp_path))
        assert len(df) == 6
        assert df["is_differential"].sum() == 4
        assert "event_type" in df.columns
        assert "chrom" in df.columns

    def test_missing_truth_file(self, tmp_path):
        df = load_ground_truth(str(tmp_path))
        assert len(df) == 0
        assert list(df.columns) == [
            "gene_id", "event_type", "chrom", "start", "end",
            "strand", "is_differential",
        ]


class TestEvaluateSpliceResults:

    def test_evaluation_with_overlap(self, tmp_path):
        truth_path = tmp_path / "truth.tsv"
        _make_truth_tsv(str(truth_path))
        truth_df = pd.read_csv(truth_path, sep="\t")
        truth_df["is_differential"] = truth_df["is_differential"].astype(bool)

        splice_path = tmp_path / "splice_results.tsv"
        _make_splice_results_tsv(str(splice_path))

        metrics = evaluate_splice_results(
            str(splice_path), truth_df, fdr_threshold=0.05,
        )

        assert "gene_level" in metrics
        assert "event_level" in metrics
        assert "per_event_type" in metrics

        gene = metrics["gene_level"]
        # Truth differential genes: 1, 2, 3, 4
        # SPLICE significant (FDR<0.05): 1, 2, 3, 7 (gene 4 has FDR=0.04, gene 7 is FP)
        # TP: 1, 2, 3 (3); FP: 7 (1); FN: 4 (1)
        assert gene["tp"] == 3
        assert gene["fp"] == 1
        assert gene["fn"] == 1

    def test_per_event_type(self, tmp_path):
        truth_path = tmp_path / "truth.tsv"
        _make_truth_tsv(str(truth_path))
        truth_df = pd.read_csv(truth_path, sep="\t")
        truth_df["is_differential"] = truth_df["is_differential"].astype(bool)

        splice_path = tmp_path / "splice_results.tsv"
        _make_splice_results_tsv(str(splice_path))

        metrics = evaluate_splice_results(
            str(splice_path), truth_df, fdr_threshold=0.05,
        )

        per_type = metrics["per_event_type"]
        assert "SE" in per_type
        assert "A3SS" in per_type
        assert "A5SS" in per_type
        assert "MXE" in per_type
        assert "RI" in per_type


class TestCompareWithPublished:

    def test_includes_splice(self):
        splice_metrics = {
            "gene_level": {
                "tpr": 0.85, "fdr": 0.04, "f_score": 0.88,
                "tp": 100, "fp": 4, "fn": 18,
            },
            "event_level": {
                "tpr": 0.80, "fdr": 0.05, "f_score": 0.82,
                "tp": 100, "fp": 5, "fn": 25,
            },
            "per_event_type": {},
        }
        df = compare_with_published_results(splice_metrics)

        assert "SPLICE" in df["tool"].values
        # Should be sorted by f_score descending
        assert df["f_score"].is_monotonic_decreasing
        # Should have published tools + SPLICE
        assert len(df) == len(PUBLISHED_TOOLS) + 1

    def test_ranking_column(self):
        splice_metrics = {
            "gene_level": {
                "tpr": 0.85, "fdr": 0.04, "f_score": 0.88,
                "tp": 100, "fp": 4, "fn": 18,
            },
            "event_level": {
                "tpr": 0.80, "fdr": 0.05, "f_score": 0.82,
                "tp": 100, "fp": 5, "fn": 25,
            },
            "per_event_type": {},
        }
        df = compare_with_published_results(splice_metrics)
        assert "rank" in df.columns
        assert df["rank"].tolist() == list(range(1, len(df) + 1))


class TestGenerateBenchmarkReport:

    def test_creates_outputs(self, tmp_path):
        splice_metrics = {
            "gene_level": {
                "tpr": 0.85, "fdr": 0.04, "f_score": 0.88,
                "tp": 100, "fp": 4, "fn": 18,
            },
            "event_level": {
                "tpr": 0.80, "fdr": 0.05, "f_score": 0.82,
                "tp": 100, "fp": 5, "fn": 25,
            },
            "per_event_type": {
                "SE": {"tp": 50, "fp": 2, "fn": 8, "tpr": 0.86, "fdr": 0.04, "f_score": 0.87},
                "A3SS": {"tp": 20, "fp": 1, "fn": 5, "tpr": 0.80, "fdr": 0.05, "f_score": 0.81},
                "A5SS": {"tp": 18, "fp": 1, "fn": 4, "tpr": 0.82, "fdr": 0.05, "f_score": 0.83},
                "MXE": {"tp": 8, "fp": 1, "fn": 4, "tpr": 0.67, "fdr": 0.11, "f_score": 0.71},
                "RI": {"tp": 4, "fp": 0, "fn": 4, "tpr": 0.50, "fdr": 0.0, "f_score": 0.67},
            },
        }
        comparison = compare_with_published_results(splice_metrics)

        output_dir = tmp_path / "report"
        generate_benchmark_report(splice_metrics, comparison, str(output_dir))

        assert (output_dir / "benchmark_results.tsv").exists()
        assert (output_dir / "benchmark_comparison.tsv").exists()
        assert (output_dir / "tpr_fdr_plot.svg").exists()
        assert (output_dir / "fscore_barplot.svg").exists()
        assert (output_dir / "per_event_type.svg").exists()


class TestPrepareSpliceInput:

    def test_with_missing_files(self, tmp_path):
        result = prepare_splice_input(str(tmp_path), str(tmp_path / "out"))
        assert "splice_command" in result
        assert "bams_group1" in result
        assert "bams_group2" in result
        assert result["bams_group1"] == []
        assert result["bams_group2"] == []
