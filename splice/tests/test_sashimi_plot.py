"""
Tests for sashimi plot generation.
"""

import os

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from splice.visualization.sashimi_plot import (
    draw_coverage_track,
    draw_gene_model,
    draw_junction_arcs,
    generate_sashimi_plot,
    get_coverage_for_region,
    get_junction_reads,
)


class TestCoverageExtraction:

    def test_coverage_for_region(self, all_bam_paths):
        """Coverage extraction should return non-empty array for region with reads."""
        cov = get_coverage_for_region(all_bam_paths[0], "chr1", 0, 5000)
        assert isinstance(cov, np.ndarray)
        assert len(cov) == 5000
        assert cov.sum() > 0

    def test_coverage_empty_region(self, all_bam_paths):
        """Coverage outside data range should be zeros."""
        cov = get_coverage_for_region(all_bam_paths[0], "chr1", 100000, 100100)
        assert isinstance(cov, np.ndarray)
        assert len(cov) == 100
        assert cov.sum() == 0

    def test_coverage_invalid_chrom(self, all_bam_paths):
        """Invalid chromosome should return zeros, not crash."""
        cov = get_coverage_for_region(all_bam_paths[0], "chrZZZ", 0, 100)
        assert isinstance(cov, np.ndarray)
        assert cov.sum() == 0


class TestJunctionExtraction:

    def test_junction_reads(self, all_bam_paths):
        """Junction extraction should find junctions in synthetic data."""
        junctions = get_junction_reads(all_bam_paths[0], "chr1", 0, 5000)
        assert isinstance(junctions, list)
        assert len(junctions) > 0
        for j_start, j_end, count in junctions:
            assert isinstance(j_start, int)
            assert isinstance(j_end, int)
            assert isinstance(count, int)
            assert j_end > j_start
            assert count > 0

    def test_junction_reads_invalid_chrom(self, all_bam_paths):
        """Invalid chromosome should return empty list."""
        junctions = get_junction_reads(all_bam_paths[0], "chrZZZ", 0, 100)
        assert junctions == []


class TestDrawing:

    def test_draw_gene_model(self):
        """Gene model drawing should not raise."""
        fig, ax = plt.subplots()
        exons = [(100, 200), (300, 400), (500, 600)]
        draw_gene_model(ax, exons, "+", y_position=0.0)
        # Verify patches were added
        assert len(ax.patches) >= len(exons)
        plt.close(fig)

    def test_draw_gene_model_negative_strand(self):
        """Gene model on negative strand should add arrows in reverse."""
        fig, ax = plt.subplots()
        exons = [(100, 200), (300, 400)]
        draw_gene_model(ax, exons, "-", y_position=0.0)
        plt.close(fig)

    def test_draw_gene_model_empty(self):
        """Empty exon list should not crash."""
        fig, ax = plt.subplots()
        draw_gene_model(ax, [], "+", y_position=0.0)
        plt.close(fig)

    def test_draw_coverage_track(self):
        """Coverage track drawing should fill collection."""
        fig, ax = plt.subplots()
        coverage = np.array([5, 10, 15, 12, 8, 3])
        draw_coverage_track(ax, coverage, start=100, color="#3498DB", label="Test")
        # Should have a fill collection
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_draw_coverage_empty(self):
        """Empty coverage should not crash."""
        fig, ax = plt.subplots()
        draw_coverage_track(ax, np.array([]), start=0, color="#000000")
        plt.close(fig)

    def test_draw_junction_arcs(self):
        """Junction arc drawing should add path patches."""
        fig, ax = plt.subplots()
        coverage = np.array([10, 10, 10, 10, 10])
        draw_coverage_track(ax, coverage, start=100, color="#3498DB")
        junctions = [(105, 200, 50), (200, 305, 30)]
        draw_junction_arcs(ax, junctions, y_base=10, color="#3498DB")
        # Each junction adds a PathPatch
        assert len(ax.patches) >= len(junctions)
        plt.close(fig)

    def test_draw_junction_arcs_empty(self):
        """Empty junction list should not crash."""
        fig, ax = plt.subplots()
        draw_junction_arcs(ax, [], y_base=0)
        plt.close(fig)


class TestSashimiPlot:

    def test_generate_sashimi_plot(self, all_bam_paths, tmp_path):
        """Full sashimi plot generation should produce SVG and PNG files."""
        bams_g1 = all_bam_paths[:3]
        bams_g2 = all_bam_paths[3:]
        exons = [(1000, 1100), (2000, 2100), (3000, 3100)]

        output_path = str(tmp_path / "test_sashimi")
        generate_sashimi_plot(
            bam_paths_group1=bams_g1,
            bam_paths_group2=bams_g2,
            chrom="chr1",
            start=900,
            end=3200,
            exons=exons,
            strand="+",
            gene_name="GeneA",
            event_type="SE",
            delta_psi=0.35,
            fdr=0.001,
            output_path=output_path,
        )

        assert os.path.exists(f"{output_path}.svg")
        assert os.path.exists(f"{output_path}.png")
        assert os.path.getsize(f"{output_path}.svg") > 0
        assert os.path.getsize(f"{output_path}.png") > 0

    def test_generate_sashimi_empty_groups(self, tmp_path):
        """Should handle empty BAM lists without crashing."""
        output_path = str(tmp_path / "empty_sashimi")
        generate_sashimi_plot(
            bam_paths_group1=[],
            bam_paths_group2=[],
            chrom="chr1",
            start=100, end=500,
            exons=[(150, 250), (300, 400)],
            strand="+",
            gene_name="GeneX",
            event_type="SE",
            delta_psi=0.0,
            fdr=1.0,
            output_path=output_path,
        )
        assert os.path.exists(f"{output_path}.svg")
