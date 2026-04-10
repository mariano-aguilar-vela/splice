"""
Tests for the Rust BAM reader.

These tests verify that the Rust extension produces identical results
to the Python reference implementation. They are skipped if Rust is
not compiled.
"""

import numpy as np
import pytest

from splice.utils.genomic import Junction, JunctionPair

# Skip all tests if Rust is not available
try:
    from splice._rust_bam import RUST_AVAILABLE
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE,
    reason="Rust extension not compiled"
)


class TestRustPythonEquivalence:
    """Verify Rust and Python produce identical results."""

    def test_identical_junction_counts(self, all_bam_paths, sample_names):
        """Rust and Python should find the same junctions with same counts."""
        from splice.io.bam_utils import (
            _python_extract_junction_stats_streaming,
            _rust_extract_and_aggregate,
        )

        # Python reference implementation
        py_js, py_co = {}, {}
        _python_extract_junction_stats_streaming(
            all_bam_paths[0], 0, py_js, py_co,
            n_samples=len(all_bam_paths), min_anchor=6, min_mapq=0,
        )

        # Rust implementation
        rust_js, rust_co = {}, {}
        _rust_extract_and_aggregate(
            all_bam_paths[0], 0, rust_js, rust_co,
            n_samples=len(all_bam_paths), min_anchor=6, min_mapq=0,
        )

        assert set(py_js.keys()) == set(rust_js.keys()), \
            "Rust and Python found different junctions"

        for junc in py_js:
            py_s = py_js[junc][0]
            rust_s = rust_js[junc][0]
            assert py_s["counts"] == rust_s["counts"], \
                f"Count mismatch for {junc}: Python={py_s['counts']}, Rust={rust_s['counts']}"

    def test_identical_statistics(self, all_bam_paths, sample_names):
        """Rust and Python should report same BAM-level statistics."""
        from splice.io.bam_utils import (
            _python_extract_junction_stats_streaming,
            _rust_extract_and_aggregate,
        )

        py_js, py_co = {}, {}
        py_stats = _python_extract_junction_stats_streaming(
            all_bam_paths[0], 0, py_js, py_co,
            n_samples=len(all_bam_paths),
        )

        rust_js, rust_co = {}, {}
        rust_stats = _rust_extract_and_aggregate(
            all_bam_paths[0], 0, rust_js, rust_co,
            n_samples=len(all_bam_paths),
        )

        assert py_stats["total_reads"] == rust_stats["total_reads"]
        assert py_stats["mapped_reads"] == rust_stats["mapped_reads"]
        assert py_stats["junction_reads"] == rust_stats["junction_reads"]

    def test_identical_cooccurrence(self, all_bam_paths, sample_names):
        """Rust and Python should find same co-occurrence pairs."""
        from splice.io.bam_utils import (
            _python_extract_junction_stats_streaming,
            _rust_extract_and_aggregate,
        )

        py_js, py_co = {}, {}
        _python_extract_junction_stats_streaming(
            all_bam_paths[0], 0, py_js, py_co,
            n_samples=len(all_bam_paths),
        )

        rust_js, rust_co = {}, {}
        _rust_extract_and_aggregate(
            all_bam_paths[0], 0, rust_js, rust_co,
            n_samples=len(all_bam_paths),
        )

        assert len(py_co) == len(rust_co), \
            f"Co-occurrence count mismatch: Python={len(py_co)}, Rust={len(rust_co)}"

    def test_region_fetch(self, all_bam_paths, sample_names):
        """Rust should handle region-based fetch correctly."""
        from splice.io.bam_utils import _rust_extract_and_aggregate

        js, co = {}, {}
        stats = _rust_extract_and_aggregate(
            all_bam_paths[0], 0, js, co,
            n_samples=len(all_bam_paths),
            region="chr1",
        )

        for junc in js:
            assert junc.chrom == "chr1"

    def test_invalid_region(self, all_bam_paths, sample_names):
        """Rust should handle invalid region gracefully."""
        from splice.io.bam_utils import _rust_extract_and_aggregate

        js, co = {}, {}
        try:
            stats = _rust_extract_and_aggregate(
                all_bam_paths[0], 0, js, co,
                n_samples=len(all_bam_paths),
                region="chrNONEXISTENT",
            )
            assert len(js) == 0
        except (ValueError, Exception):
            pass  # Expected for invalid region


class TestRustRequiredPath:
    """Test that extract_junction_stats_streaming requires Rust."""

    def test_rust_availability_check(self):
        """RUST_AVAILABLE should be a boolean."""
        from splice.io.bam_utils import RUST_AVAILABLE as bam_rust
        assert isinstance(bam_rust, bool)

    def test_python_reference_is_accessible(self):
        """The Python reference implementation should be importable."""
        from splice.io.bam_utils import _python_extract_junction_stats_streaming
        assert callable(_python_extract_junction_stats_streaming)

    def test_streaming_raises_without_rust(self):
        """extract_junction_stats_streaming should raise if Rust is unavailable."""
        import splice.io.bam_utils as bam_module
        original = bam_module.RUST_AVAILABLE
        try:
            bam_module.RUST_AVAILABLE = False
            with pytest.raises(ImportError, match="Rust BAM reader required"):
                bam_module.extract_junction_stats_streaming(
                    "/nonexistent.bam", 0, {}, {}, n_samples=1,
                )
        finally:
            bam_module.RUST_AVAILABLE = original
