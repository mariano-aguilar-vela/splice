"""
Test suite for Module 22: io/format_export.py

Tests format export functions for rMATS, LeafCutter, MAJIQ, BED, and GTF.
"""

import json
import os
import tempfile
import unittest

import numpy as np

from splice.core.diff import DiffResult
from splice.core.psi import ModulePSI
from splice.io.format_export import (
    export_bed_format,
    export_event_gtf,
    export_leafcutter_format,
    export_majiq_like_format,
    export_rmats_format,
)


class TestExportRMATSFormat(unittest.TestCase):
    """Test export_rmats_format function."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "events.txt")

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

    def test_export_rmats_basic(self):
        """Test basic rMATS export."""
        diff_results = [self.create_test_diff_result()]

        export_rmats_format(diff_results, self.output_path)

        self.assertTrue(os.path.exists(self.output_path))

        with open(self.output_path) as f:
            lines = f.readlines()

        # Header + 1 data row
        self.assertEqual(len(lines), 2)
        self.assertIn("ID", lines[0])
        self.assertIn("GeneID", lines[0])

    def test_export_rmats_filtering(self):
        """Test FDR threshold filtering."""
        diff_results = [
            self.create_test_diff_result(fdr=0.01),  # Include
            self.create_test_diff_result(fdr=0.1),  # Exclude
        ]

        export_rmats_format(diff_results, self.output_path, fdr_threshold=0.05)

        with open(self.output_path) as f:
            lines = f.readlines()

        # Header + 1 data row (only first result)
        self.assertEqual(len(lines), 2)

    def test_export_rmats_columns(self):
        """Test that expected columns are present."""
        diff_results = [self.create_test_diff_result()]

        export_rmats_format(diff_results, self.output_path)

        with open(self.output_path) as f:
            header = f.readline().strip()

        expected_cols = [
            "ID",
            "GeneID",
            "GeneName",
            "chr",
            "strand",
            "PValue",
            "FDR",
            "IncLevel1",
            "IncLevel2",
        ]
        for col in expected_cols:
            self.assertIn(col, header)

    def test_export_rmats_values(self):
        """Test that values are exported correctly."""
        diff_results = [self.create_test_diff_result()]

        export_rmats_format(diff_results, self.output_path)

        with open(self.output_path) as f:
            lines = f.readlines()
            data_row = lines[1].strip().split("\t")

        header = lines[0].strip().split("\t")
        gene_idx = header.index("GeneID")
        chr_idx = header.index("chr")

        self.assertEqual(data_row[gene_idx], "gene1")
        self.assertEqual(data_row[chr_idx], "chr1")


class TestExportLeafCutterFormat(unittest.TestCase):
    """Test export_leafcutter_format function."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "clusters.txt")

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

    def test_export_leafcutter_basic(self):
        """Test basic LeafCutter export."""
        diff_results = [self.create_test_diff_result()]

        export_leafcutter_format(diff_results, self.output_path)

        self.assertTrue(os.path.exists(self.output_path))

        with open(self.output_path) as f:
            lines = f.readlines()

        # Header + 1 data row
        self.assertEqual(len(lines), 2)
        self.assertIn("cluster", lines[0])
        self.assertIn("gene", lines[0])

    def test_export_leafcutter_filtering(self):
        """Test FDR threshold filtering."""
        diff_results = [
            self.create_test_diff_result(fdr=0.01),
            self.create_test_diff_result(fdr=0.1),
        ]

        export_leafcutter_format(
            diff_results, self.output_path, fdr_threshold=0.05
        )

        with open(self.output_path) as f:
            lines = f.readlines()

        # Header + 1 data row
        self.assertEqual(len(lines), 2)

    def test_export_leafcutter_columns(self):
        """Test that expected columns are present."""
        diff_results = [self.create_test_diff_result()]

        export_leafcutter_format(diff_results, self.output_path)

        with open(self.output_path) as f:
            header = f.readline().strip()

        expected_cols = [
            "cluster",
            "gene",
            "chr",
            "start",
            "end",
            "n_junctions",
            "fdr",
        ]
        for col in expected_cols:
            self.assertIn(col, header)


class TestExportMAJIQFormat(unittest.TestCase):
    """Test export_majiq_like_format function."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

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
            delta_psi_ci_low=np.array([0.0, -0.3]),  # CI doesn't cross zero
            delta_psi_ci_high=np.array([0.4, 0.0]),  # for first junction
            log_likelihood_null=-100.0,
            log_likelihood_full=-80.0,
            degrees_of_freedom=1,
            p_value=0.001,
            fdr=0.01,
            null_converged=True,
            full_converged=True,
            null_refit_used=False,
            null_iterations=50,
            full_iterations=60,
            null_gradient_norm=1e-5,
            full_gradient_norm=1e-5,
        )

    def create_test_module_psi(self, n_samples=10):
        """Create a test ModulePSI."""
        return ModulePSI(
            module_id="test_module",
            psi_matrix=np.random.rand(2, n_samples) * 0.5,
            ci_low_matrix=np.random.rand(2, n_samples) * 0.3,
            ci_high_matrix=np.random.rand(2, n_samples) * 0.7,
            bootstrap_psi=np.random.rand(10, 2, n_samples) * 0.5,
            total_counts=np.ones(n_samples) * 1000,
            effective_n=np.ones(n_samples) * 100,
        )

    def test_export_majiq_basic(self):
        """Test basic MAJIQ export."""
        diff_results = [self.create_test_diff_result()]
        module_psi_list = [self.create_test_module_psi()]

        export_majiq_like_format(diff_results, module_psi_list, self.output_dir)

        # Check that LSV file exists
        lsv_file = os.path.join(self.output_dir, "lsv_results.json")
        self.assertTrue(os.path.exists(lsv_file))

    def test_export_majiq_lsv_structure(self):
        """Test LSV JSON structure."""
        diff_results = [self.create_test_diff_result()]
        module_psi_list = [self.create_test_module_psi(n_samples=5)]

        export_majiq_like_format(diff_results, module_psi_list, self.output_dir)

        lsv_file = os.path.join(self.output_dir, "lsv_results.json")
        with open(lsv_file) as f:
            data = json.load(f)

        self.assertIn("lsvs", data)
        self.assertIn("n_samples", data)
        self.assertEqual(data["n_samples"], 5)
        self.assertEqual(len(data["lsvs"]), 1)

        lsv = data["lsvs"][0]
        self.assertIn("lsv_id", lsv)
        self.assertIn("gene_id", lsv)
        self.assertIn("probability_of_change", lsv)
        self.assertIn("fdr", lsv)

    def test_export_majiq_psi_files(self):
        """Test per-sample PSI JSON files."""
        diff_results = [self.create_test_diff_result()]
        module_psi_list = [self.create_test_module_psi()]

        export_majiq_like_format(diff_results, module_psi_list, self.output_dir)

        psi_file = os.path.join(self.output_dir, "psi_lsv_000000.json")
        self.assertTrue(os.path.exists(psi_file))

        with open(psi_file) as f:
            data = json.load(f)

        self.assertIn("lsv_id", data)
        self.assertIn("psi_matrix", data)
        self.assertIn("ci_low", data)
        self.assertIn("ci_high", data)

    def test_export_majiq_multiple_lsvs(self):
        """Test with multiple LSVs."""
        diff_results = [
            self.create_test_diff_result(),
            self.create_test_diff_result(),
        ]
        module_psi_list = [
            self.create_test_module_psi(),
            self.create_test_module_psi(),
        ]

        export_majiq_like_format(diff_results, module_psi_list, self.output_dir)

        lsv_file = os.path.join(self.output_dir, "lsv_results.json")
        with open(lsv_file) as f:
            data = json.load(f)

        self.assertEqual(len(data["lsvs"]), 2)

        # Check that both PSI files exist
        psi_file_0 = os.path.join(self.output_dir, "psi_lsv_000000.json")
        psi_file_1 = os.path.join(self.output_dir, "psi_lsv_000001.json")
        self.assertTrue(os.path.exists(psi_file_0))
        self.assertTrue(os.path.exists(psi_file_1))


class TestExportBEDFormat(unittest.TestCase):
    """Test export_bed_format function."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "events.bed")

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

    def test_export_bed_basic(self):
        """Test basic BED export."""
        diff_results = [self.create_test_diff_result()]

        export_bed_format(diff_results, self.output_path)

        self.assertTrue(os.path.exists(self.output_path))

        with open(self.output_path) as f:
            lines = f.readlines()

        # Header comment + 2 junctions
        self.assertGreaterEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("#"))

    def test_export_bed_filtering(self):
        """Test FDR threshold filtering."""
        diff_results = [
            self.create_test_diff_result(fdr=0.01),
            self.create_test_diff_result(fdr=0.1),
        ]

        export_bed_format(diff_results, self.output_path, fdr_threshold=0.05)

        with open(self.output_path) as f:
            lines = f.readlines()

        # Header + 2 junctions (from first result only)
        data_lines = [l for l in lines if not l.startswith("#")]
        self.assertEqual(len(data_lines), 2)

    def test_export_bed_format(self):
        """Test BED format structure."""
        diff_results = [self.create_test_diff_result()]

        export_bed_format(diff_results, self.output_path)

        with open(self.output_path) as f:
            lines = f.readlines()

        # Get first data line
        data_line = None
        for line in lines:
            if not line.startswith("#"):
                data_line = line.strip()
                break

        self.assertIsNotNone(data_line)
        fields = data_line.split("\t")

        # BED format: chrom, start, end, name, score, strand
        self.assertEqual(len(fields), 6)
        self.assertEqual(fields[0], "chr1")  # chrom
        self.assertEqual(fields[5], "+")  # strand


class TestExportGTFFormat(unittest.TestCase):
    """Test export_event_gtf function."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "events.gtf")

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

    def test_export_gtf_basic(self):
        """Test basic GTF export."""
        diff_results = [self.create_test_diff_result()]
        event_types = {"SE": 1}

        export_event_gtf(diff_results, event_types, self.output_path)

        self.assertTrue(os.path.exists(self.output_path))

        with open(self.output_path) as f:
            lines = f.readlines()

        # Header comments + 1 data line
        self.assertGreaterEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("#"))

    def test_export_gtf_filtering(self):
        """Test FDR threshold filtering."""
        diff_results = [
            self.create_test_diff_result(fdr=0.01),
            self.create_test_diff_result(fdr=0.1),
        ]
        event_types = {"SE": 2}

        export_event_gtf(
            diff_results, event_types, self.output_path, fdr_threshold=0.05
        )

        with open(self.output_path) as f:
            lines = f.readlines()

        # Count data lines
        data_lines = [l for l in lines if not l.startswith("#")]
        self.assertEqual(len(data_lines), 1)

    def test_export_gtf_format(self):
        """Test GTF format structure."""
        diff_results = [self.create_test_diff_result()]
        event_types = {"SE": 1}

        export_event_gtf(diff_results, event_types, self.output_path)

        with open(self.output_path) as f:
            lines = f.readlines()

        # Get first data line
        data_line = None
        for line in lines:
            if not line.startswith("#"):
                data_line = line.strip()
                break

        self.assertIsNotNone(data_line)
        fields = data_line.split("\t")

        # GTF format: seqname, source, feature, start, end, score, strand, frame, attributes
        self.assertEqual(len(fields), 9)
        self.assertEqual(fields[0], "chr1")  # seqname
        self.assertEqual(fields[1], "SPLICE")  # source
        self.assertEqual(fields[6], "+")  # strand
        self.assertTrue(fields[8].startswith('gene_id'))  # attributes

    def test_export_gtf_attributes(self):
        """Test GTF attributes."""
        diff_results = [self.create_test_diff_result()]
        event_types = {"SE": 1}

        export_event_gtf(diff_results, event_types, self.output_path)

        with open(self.output_path) as f:
            lines = f.readlines()

        # Get first data line
        data_line = None
        for line in lines:
            if not line.startswith("#"):
                data_line = line.strip()
                break

        fields = data_line.split("\t")
        attributes = fields[8]

        self.assertIn("gene_id", attributes)
        self.assertIn("gene_name", attributes)
        self.assertIn("module_id", attributes)
        self.assertIn("event_type", attributes)


class TestFormatExportIntegration(unittest.TestCase):
    """Integration tests for all format exports."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_export_all_formats(self):
        """Test exporting in all formats together."""
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
                junction_coords=[f"chr1:{100+i*100}-{200+i*100}:+"],
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
                p_value=0.001,
                fdr=0.01 * (i + 1),
                null_converged=True,
                full_converged=True,
                null_refit_used=False,
                null_iterations=50,
                full_iterations=60,
                null_gradient_norm=1e-5,
                full_gradient_norm=1e-5,
            )
            for i in range(2)
        ]

        module_psi_list = [
            ModulePSI(
                module_id=f"mod_{i}",
                psi_matrix=np.random.rand(2, 10) * 0.5,
                ci_low_matrix=np.random.rand(2, 10) * 0.3,
                ci_high_matrix=np.random.rand(2, 10) * 0.7,
                bootstrap_psi=np.random.rand(10, 2, 10) * 0.5,
                total_counts=np.ones(10) * 1000,
                effective_n=np.ones(10) * 100,
            )
            for i in range(2)
        ]

        event_types = {"SE": 2}

        # Export to all formats
        export_rmats_format(
            diff_results,
            os.path.join(self.output_dir, "rmats.txt"),
        )
        export_leafcutter_format(
            diff_results,
            os.path.join(self.output_dir, "leafcutter.txt"),
        )
        export_majiq_like_format(diff_results, module_psi_list, self.output_dir)
        export_bed_format(
            diff_results,
            os.path.join(self.output_dir, "events.bed"),
        )
        export_event_gtf(
            diff_results,
            event_types,
            os.path.join(self.output_dir, "events.gtf"),
        )

        # Verify all files exist
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "rmats.txt"))
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.output_dir, "leafcutter.txt")
            )
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "lsv_results.json"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "events.bed"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "events.gtf"))
        )


if __name__ == "__main__":
    unittest.main()