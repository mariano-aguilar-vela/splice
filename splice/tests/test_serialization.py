"""
Test suite for Module 23: io/serialization.py

Tests checkpoint and junction evidence serialization functions.
"""

import os
import tempfile
import unittest

import numpy as np

from splicekit.core.diff import DiffResult
from splicekit.core.diagnostics import EventDiagnostic
from splicekit.core.psi import ModulePSI
from splicekit.io.serialization import (
    load_checkpoint,
    load_junction_evidence,
    save_checkpoint,
    save_junction_evidence,
)
from splicekit.utils.genomic import Junction


class TestSaveLoadCheckpoint(unittest.TestCase):
    """Test save_checkpoint and load_checkpoint functions."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_checkpoint_basic_types(self):
        """Test checkpoint with basic Python types."""
        data = {
            "int": 42,
            "float": 3.14,
            "str": "hello",
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2},
        }

        checkpoint_path = os.path.join(self.temp_path, "test.pkl")
        save_checkpoint(data, checkpoint_path)

        loaded = load_checkpoint(checkpoint_path)

        self.assertEqual(loaded["int"], 42)
        self.assertEqual(loaded["float"], 3.14)
        self.assertEqual(loaded["str"], "hello")
        self.assertEqual(loaded["list"], [1, 2, 3])
        self.assertEqual(loaded["dict"], {"a": 1, "b": 2})

    def test_checkpoint_numpy(self):
        """Test checkpoint with NumPy arrays."""
        data = {
            "array_1d": np.array([1, 2, 3, 4, 5]),
            "array_2d": np.array([[1, 2], [3, 4]]),
            "float_array": np.random.rand(10, 5),
        }

        checkpoint_path = os.path.join(self.temp_path, "numpy.pkl")
        save_checkpoint(data, checkpoint_path)

        loaded = load_checkpoint(checkpoint_path)

        np.testing.assert_array_equal(loaded["array_1d"], data["array_1d"])
        np.testing.assert_array_equal(loaded["array_2d"], data["array_2d"])
        np.testing.assert_array_almost_equal(
            loaded["float_array"], data["float_array"]
        )

    def test_checkpoint_diff_result(self):
        """Test checkpoint with DiffResult objects."""
        diff_result = DiffResult(
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

        checkpoint_path = os.path.join(self.temp_path, "diff_result.pkl")
        save_checkpoint(diff_result, checkpoint_path)

        loaded = load_checkpoint(checkpoint_path)

        self.assertEqual(loaded.module_id, "test_module")
        self.assertEqual(loaded.gene_id, "gene1")
        self.assertEqual(loaded.fdr, 0.1)
        np.testing.assert_array_equal(loaded.delta_psi, diff_result.delta_psi)

    def test_checkpoint_list(self):
        """Test checkpoint with list of complex objects."""
        data = [
            {"id": 1, "value": 10.5},
            {"id": 2, "value": 20.3},
            {"id": 3, "value": 30.1},
        ]

        checkpoint_path = os.path.join(self.temp_path, "list.pkl")
        save_checkpoint(data, checkpoint_path)

        loaded = load_checkpoint(checkpoint_path)

        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0]["id"], 1)
        self.assertEqual(loaded[2]["value"], 30.1)

    def test_checkpoint_file_not_found(self):
        """Test loading non-existent checkpoint."""
        checkpoint_path = os.path.join(self.temp_path, "nonexistent.pkl")

        with self.assertRaises(FileNotFoundError):
            load_checkpoint(checkpoint_path)

    def test_checkpoint_nested_directories(self):
        """Test checkpoint creation with nested directories."""
        checkpoint_path = os.path.join(
            self.temp_path, "nested", "dirs", "test.pkl"
        )
        data = {"key": "value"}

        save_checkpoint(data, checkpoint_path)

        self.assertTrue(os.path.exists(checkpoint_path))
        loaded = load_checkpoint(checkpoint_path)
        self.assertEqual(loaded["key"], "value")


class TestSaveLoadJunctionEvidence(unittest.TestCase):
    """Test save_junction_evidence and load_junction_evidence functions."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def create_test_evidence(self):
        """Create test junction evidence dict."""
        return {
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
                "junction": Junction(chrom="chr2", start=500, end=600, strand="-"),
                "gene_id": "gene2",
                "gene_name": "GENE2",
                "is_annotated": False,
                "motif": "GC/AG",
                "motif_score": 0.75,
                "total_reads": 500,
                "mean_mapq": 25.0,
                "sample_counts": [5, 10, 8, 12],
            },
        }

    def test_save_load_junction_evidence_basic(self):
        """Test basic junction evidence save and load."""
        evidence = self.create_test_evidence()

        evidence_path = os.path.join(self.temp_path, "evidence")
        save_junction_evidence(evidence, evidence_path)

        loaded = load_junction_evidence(evidence_path)

        self.assertEqual(len(loaded), 2)
        self.assertIn("junc_1", loaded)
        self.assertIn("junc_2", loaded)

    def test_save_load_junction_values(self):
        """Test that junction evidence values are preserved."""
        evidence = self.create_test_evidence()

        evidence_path = os.path.join(self.temp_path, "evidence")
        save_junction_evidence(evidence, evidence_path)

        loaded = load_junction_evidence(evidence_path)

        junc_1 = loaded["junc_1"]
        self.assertEqual(junc_1["junction"].chrom, "chr1")
        self.assertEqual(junc_1["junction"].start, 100)
        self.assertEqual(junc_1["gene_id"], "gene1")
        self.assertEqual(junc_1["is_annotated"], True)
        self.assertEqual(junc_1["motif"], "GT/AG")
        self.assertAlmostEqual(junc_1["motif_score"], 0.95)
        self.assertEqual(junc_1["total_reads"], 1000)
        self.assertEqual(junc_1["sample_counts"], [10, 20, 15, 25])

    def test_save_load_strand_and_coords(self):
        """Test that strand and coordinates are preserved."""
        evidence = self.create_test_evidence()

        evidence_path = os.path.join(self.temp_path, "evidence")
        save_junction_evidence(evidence, evidence_path)

        loaded = load_junction_evidence(evidence_path)

        junc_2 = loaded["junc_2"]
        self.assertEqual(junc_2["junction"].strand, "-")
        self.assertEqual(junc_2["junction"].end, 600)

    def test_save_load_empty_evidence(self):
        """Test with empty evidence dict."""
        evidence = {}

        evidence_path = os.path.join(self.temp_path, "empty")
        save_junction_evidence(evidence, evidence_path)

        # Should not raise error
        loaded = load_junction_evidence(evidence_path)
        self.assertEqual(len(loaded), 0)

    def test_save_load_variable_sample_counts(self):
        """Test with different sample count lengths."""
        evidence = {
            "junc_1": {
                "junction": Junction(chrom="chr1", start=100, end=200, strand="+"),
                "gene_id": "gene1",
                "gene_name": "GENE1",
                "is_annotated": True,
                "motif": "GT/AG",
                "motif_score": 0.95,
                "total_reads": 1000,
                "mean_mapq": 30.0,
                "sample_counts": [10, 20, 15],  # 3 samples
            },
            "junc_2": {
                "junction": Junction(chrom="chr2", start=500, end=600, strand="-"),
                "gene_id": "gene2",
                "gene_name": "GENE2",
                "is_annotated": False,
                "motif": "GC/AG",
                "motif_score": 0.75,
                "total_reads": 500,
                "mean_mapq": 25.0,
                "sample_counts": [5, 10, 8, 12, 7, 3],  # 6 samples
            },
        }

        evidence_path = os.path.join(self.temp_path, "variable")
        save_junction_evidence(evidence, evidence_path)

        loaded = load_junction_evidence(evidence_path)

        # Shorter sample counts should be padded with zeros
        self.assertEqual(len(loaded["junc_1"]["sample_counts"]), 3)
        self.assertEqual(len(loaded["junc_2"]["sample_counts"]), 6)

    def test_save_load_missing_fields(self):
        """Test with missing optional fields."""
        evidence = {
            "junc_1": {
                "junction": Junction(chrom="chr1", start=100, end=200, strand="+"),
                # Missing most fields
            }
        }

        evidence_path = os.path.join(self.temp_path, "minimal")
        save_junction_evidence(evidence, evidence_path)

        loaded = load_junction_evidence(evidence_path)

        junc_1 = loaded["junc_1"]
        self.assertEqual(junc_1["junction"].chrom, "chr1")
        self.assertEqual(junc_1["gene_id"], "NA")
        self.assertEqual(junc_1["motif"], "NA")

    def test_save_load_nested_directories(self):
        """Test evidence save with nested directories."""
        evidence = self.create_test_evidence()

        evidence_path = os.path.join(
            self.temp_path, "nested", "dirs", "evidence"
        )
        save_junction_evidence(evidence, evidence_path)

        loaded = load_junction_evidence(evidence_path)

        self.assertEqual(len(loaded), 2)

    def test_save_load_file_not_found(self):
        """Test loading non-existent evidence."""
        evidence_path = os.path.join(self.temp_path, "nonexistent")

        with self.assertRaises(FileNotFoundError):
            load_junction_evidence(evidence_path)


class TestSerializationIntegration(unittest.TestCase):
    """Integration tests for serialization functions."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_roundtrip_pipeline_state(self):
        """Test saving and loading complete pipeline state."""
        # Create a complete pipeline state
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

        pipeline_state = {
            "diff_results": diff_results,
            "module_psi_list": module_psi_list,
            "config": {"threshold": 0.05, "version": "1.0"},
        }

        checkpoint_path = os.path.join(self.temp_path, "pipeline.pkl")
        save_checkpoint(pipeline_state, checkpoint_path)

        loaded_state = load_checkpoint(checkpoint_path)

        self.assertEqual(len(loaded_state["diff_results"]), 2)
        self.assertEqual(len(loaded_state["module_psi_list"]), 2)
        self.assertEqual(loaded_state["config"]["threshold"], 0.05)

    def test_roundtrip_junction_evidence_with_checkpoint(self):
        """Test saving/loading both evidence and checkpoint together."""
        evidence = {
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
            }
        }

        metadata = {
            "n_samples": 4,
            "n_junctions": 1,
            "timestamp": "2024-01-01",
        }

        # Save both
        evidence_path = os.path.join(self.temp_path, "evidence")
        checkpoint_path = os.path.join(self.temp_path, "metadata.pkl")

        save_junction_evidence(evidence, evidence_path)
        save_checkpoint(metadata, checkpoint_path)

        # Load both
        loaded_evidence = load_junction_evidence(evidence_path)
        loaded_metadata = load_checkpoint(checkpoint_path)

        self.assertEqual(len(loaded_evidence), 1)
        self.assertEqual(loaded_metadata["n_samples"], 4)

    def test_large_evidence_dataset(self):
        """Test with a larger evidence dataset."""
        evidence = {}
        for i in range(100):
            evidence[f"junc_{i}"] = {
                "junction": Junction(
                    chrom="chr1", start=100 + i, end=200 + i, strand="+"
                ),
                "gene_id": f"gene_{i % 10}",
                "gene_name": f"GENE_{i % 10}",
                "is_annotated": i % 2 == 0,
                "motif": "GT/AG" if i % 2 == 0 else "GC/AG",
                "motif_score": 0.9 - (i * 0.001),
                "total_reads": 1000 - i * 5,
                "mean_mapq": 30.0 - (i * 0.1),
                "sample_counts": [10 + j for j in range(max(1, i % 5))],
            }

        evidence_path = os.path.join(self.temp_path, "large_evidence")
        save_junction_evidence(evidence, evidence_path)

        loaded = load_junction_evidence(evidence_path)

        self.assertEqual(len(loaded), 100)
        self.assertEqual(loaded["junc_50"]["gene_id"], "gene_0")


if __name__ == "__main__":
    unittest.main()
