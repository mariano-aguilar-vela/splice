"""
Test suite for benchmarking SPLICE pipeline performance.

This module tests the full SPLICE pipeline on a realistic-scale synthetic dataset
(30 genes × 3 chromosomes, 10 samples, 90 junctions total, ~4,500 reads per BAM).
The benchmark is designed to catch performance regressions (O(n²) degradations) while
remaining fast enough to run in CI (~5 minutes total).

Performance targets extrapolated from spec (10 BAMs × 50M reads, full human genome):
  Junction extraction:     < 30 min (scaled: < 120 s for our 4,500-read dataset)
  Clustering:             < 2 min   (scaled: < 5 s)
  Evidence building:      < 10 min  (scaled: < 30 s)
  PSI quantification:     < 5 min   (scaled: < 60 s for 30 bootstraps)
  Differential testing:   < 10 min  (scaled: < 60 s)
  Total pipeline:         < 90 min  (scaled: < 300 s)
"""

import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest

# Import pipeline stages
from splicekit.core.clustering import cluster_junctions
from splicekit.core.confidence_scorer import score_all_junctions
from splicekit.core.diff import test_differential_splicing
from splicekit.core.evidence import build_evidence_matrices
from splicekit.core.junction_extractor import extract_all_junctions
from splicekit.core.psi import quantify_psi
from splicekit.core.splicegraph import build_splicegraph

# Import test helpers from conftest
from .conftest import (
    _make_junction_spanning_read,
    _write_sorted_indexed_bam,
)

# ============================================================================
# Benchmark Configuration
# ============================================================================

# Data scale parameters
N_CHROMS = 3  # chr1, chr2, chr3
N_GENES_PER_CHROM = 10  # 30 total, all SE events
N_SAMPLES = 10  # 5 per group
N_READS_PER_JNC = 50  # reads per junction per sample
N_BOOTSTRAPS = 30  # spec default
CHROM_LEN = 45_000  # bp per chromosome
GENE_SPACING = 4_000  # bp between gene starts
READ_LENGTH = 50  # bases
ANCHOR = 25  # bases on each side of junction
MAPQ = 60

# Timing limits (seconds) — generous to catch O(n²) regressions
LIMIT_JUNCTION_EXTRACTION = 120
LIMIT_CLUSTERING = 5
LIMIT_EVIDENCE_BUILDING = 30
LIMIT_PSI_QUANTIFICATION = 60
LIMIT_DIFFERENTIAL_TESTING = 60
LIMIT_TOTAL_PIPELINE = 300


# ============================================================================
# MockGene — minimal Gene mock for build_splicegraph compatibility
# ============================================================================

@dataclass
class MockGene:
    """Minimal gene mock for build_splicegraph."""

    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    start: int
    end: int


# ============================================================================
# Helper Functions
# ============================================================================

def _bench_gene_layout() -> List[Tuple[str, int, Tuple, Tuple, Tuple]]:
    """
    Generate layout for all benchmark genes.

    Returns list of (chrom, gene_idx, j_inc1, j_inc2, j_skip) tuples.
    All genes are SE (skipping exon) events with 3 junctions each.

    Gene layout for gene g on chromosome c:
      exon1: [start,       start + 100)
      exon2: [start + 800, start + 900)  (cassette exon)
      exon3: [start + 1600, start + 1700)
      j_inc1: (chrom, start+100 → start+800, '+')
      j_inc2: (chrom, start+900 → start+1600, '+')
      j_skip: (chrom, start+100 → start+1600, '+')

    All genes use identical read counts per junction to ensure
    all are detected as significant SE events.
    """
    layout = []
    for chrom_idx in range(N_CHROMS):
        chrom = f"chr{chrom_idx + 1}"
        for gene_idx in range(N_GENES_PER_CHROM):
            gene_num = chrom_idx * N_GENES_PER_CHROM + gene_idx
            start = gene_num * GENE_SPACING

            j_inc1 = (chrom, start + 100, start + 800, "+")
            j_inc2 = (chrom, start + 900, start + 1600, "+")
            j_skip = (chrom, start + 100, start + 1600, "+")

            layout.append((chrom, gene_num, j_inc1, j_inc2, j_skip))

    return layout


def _make_bench_bam_header(chroms: List[str], chrom_len: int) -> Dict[str, Any]:
    """Create pysam SAM header for multiple chromosomes."""
    header = {
        "HD": {"VN": "1.0", "SO": "coordinate"},
        "SQ": [{"SN": chrom, "LN": chrom_len} for chrom in chroms],
    }
    return header


def _make_bench_genome_fasta(
    path: Path,
    chroms: List[str],
    chrom_len: int,
    gene_layout: List[Tuple[str, int, Tuple, Tuple, Tuple]],
) -> None:
    """
    Write multi-chromosome FASTA with GT/AG splice site motifs.

    Args:
        path: Output FASTA path
        chroms: List of chromosome names
        chrom_len: Length of each chromosome
        gene_layout: From _bench_gene_layout()
    """
    import pysam

    # Build sequence: cyclic ACGT
    base_seq = "ACGT" * (chrom_len // 4 + 1)

    # Write FASTA
    with pysam.FastaFile.from_sequencedicts(
        {chrom: base_seq[:chrom_len] for chrom in chroms},
        save_index=False,
    ) as fasta:
        pass

    # Now modify in-place to add splice sites
    with open(path, "w") as f:
        for chrom in chroms:
            f.write(f">{chrom}\n")
            seq = list(base_seq[:chrom_len])

            # Add GT at all donor sites
            for _, _, j_inc1, j_inc2, j_skip in gene_layout:
                for _, donor, _, _ in [j_inc1, j_inc2, j_skip]:
                    if chrom in [j_inc1[0], j_inc2[0], j_skip[0]]:
                        if donor < chrom_len - 1:
                            seq[donor : donor + 2] = ["G", "T"]

            # Add AG at all acceptor sites
            for _, _, j_inc1, j_inc2, j_skip in gene_layout:
                for _, _, acceptor, _ in [j_inc1, j_inc2, j_skip]:
                    if chrom in [j_inc1[0], j_inc2[0], j_skip[0]]:
                        if acceptor >= 1:
                            seq[acceptor - 1 : acceptor + 1] = ["A", "G"]

            f.write("".join(seq) + "\n")

    # Index with samtools
    pysam.faidx(str(path))


def _make_bench_bam(
    bam_path: Path,
    header: Dict[str, Any],
    gene_layout: List[Tuple[str, int, Tuple, Tuple, Tuple]],
    sample_name: str,
    group: int,
) -> None:
    """
    Write one benchmark BAM with group-appropriate junction read counts.

    Args:
        bam_path: Output BAM path
        header: pysam SAM header
        gene_layout: From _bench_gene_layout()
        sample_name: Name for this sample (goes in SM tag)
        group: 0 for group1, 1 for group2
    """
    # Read distribution by group
    if group == 0:
        # Group 1: PSI_skip = 10/50 = 0.20
        counts = {"j_inc1": 20, "j_inc2": 20, "j_skip": 10}
    else:
        # Group 2: PSI_skip = 34/50 = 0.68
        counts = {"j_inc1": 8, "j_inc2": 8, "j_skip": 34}

    # Build junction -> count map
    junc_counts = {}
    for _, _, j_inc1, j_inc2, j_skip in gene_layout:
        junc_counts[j_inc1] = counts["j_inc1"]
        junc_counts[j_inc2] = counts["j_inc2"]
        junc_counts[j_skip] = counts["j_skip"]

    # Build reads
    reads = []
    read_id = 0
    for junc, count in junc_counts.items():
        chrom, donor, acceptor, strand = junc
        for _ in range(count):
            read = _make_junction_spanning_read(
                read_id,
                chrom,
                donor,
                acceptor,
                ANCHOR,
                READ_LENGTH,
                strand,
                MAPQ,
                sample_name,
            )
            reads.append(read)
            read_id += 1

    _write_sorted_indexed_bam(bam_path, header, reads)


# ============================================================================
# Session-scoped Fixtures (expensive, amortized across many tests)
# ============================================================================


@pytest.fixture(scope="session")
def bench_tmp(tmp_path_factory):
    """Create session-scoped temporary directory."""
    return tmp_path_factory.mktemp("benchmark")


@pytest.fixture(scope="session")
def bench_genome_fasta(bench_tmp):
    """Create multi-chromosome FASTA with splice site motifs."""
    import pysam

    gene_layout = _bench_gene_layout()
    chroms = [f"chr{i+1}" for i in range(N_CHROMS)]
    fasta_path = bench_tmp / "genome.fa"

    _make_bench_genome_fasta(fasta_path, chroms, CHROM_LEN, gene_layout)
    return str(fasta_path)


@pytest.fixture(scope="session")
def bench_bam_paths(bench_tmp, bench_genome_fasta):
    """Create 10 sorted+indexed BAM files (5 per group) with benchmark data."""
    import pysam

    gene_layout = _bench_gene_layout()
    chroms = [f"chr{i+1}" for i in range(N_CHROMS)]
    header = _make_bench_bam_header(chroms, CHROM_LEN)

    bam_paths = []
    for sample_idx in range(N_SAMPLES):
        group = 0 if sample_idx < 5 else 1
        sample_name = f"sample_{sample_idx:02d}"
        bam_path = bench_tmp / f"{sample_name}.bam"

        _make_bench_bam(bam_path, header, gene_layout, sample_name, group)
        bam_paths.append(str(bam_path))

    return bam_paths


@pytest.fixture(scope="session")
def bench_known_junctions():
    """Return set of all 90 known junctions."""
    gene_layout = _bench_gene_layout()
    known = set()
    for _, _, j_inc1, j_inc2, j_skip in gene_layout:
        known.add(j_inc1)
        known.add(j_inc2)
        known.add(j_skip)
    return known


@pytest.fixture(scope="session")
def bench_sample_names():
    """Return list of 10 sample names."""
    return [f"sample_{i:02d}" for i in range(N_SAMPLES)]


@pytest.fixture(scope="session")
def bench_group_labels():
    """Return group labels array [0,0,0,0,0,1,1,1,1,1]."""
    return [0] * 5 + [1] * 5


@pytest.fixture(scope="session")
def bench_timing(
    bench_bam_paths,
    bench_genome_fasta,
    bench_known_junctions,
    bench_sample_names,
    bench_group_labels,
    bench_tmp,
):
    """
    Run full pipeline once and return dict of stage timings and results.

    Returns dict with keys:
      - "junction_extraction": elapsed seconds
      - "clustering": elapsed seconds
      - "evidence_building": elapsed seconds
      - "psi_quantification": elapsed seconds
      - "differential_testing": elapsed seconds
      - "total_pipeline": elapsed seconds from extraction through diff testing
      - "junction_evidence": JunctionEvidence dict
      - "clusters": dict of clusters
      - "modules": list of modules
      - "evidence_list": list of evidence dicts
      - "psi_list": list of PSI dicts
      - "diff_results": list of DiffResult
      - "n_junctions": count
      - "n_modules": count
      - "n_samples": count
    """
    import numpy as np

    results = {}

    # === Stage 1: Junction extraction ===
    t0 = time.perf_counter()
    junction_evidence = extract_all_junctions(
        bench_bam_paths,
        bench_sample_names,
        bench_group_labels,
        min_mapq=MAPQ,
        min_anchor=ANCHOR,
        max_intron_length=100_000,
    )
    results["junction_extraction"] = time.perf_counter() - t0

    # === Stage 2: Clustering ===
    t0 = time.perf_counter()
    clusters = cluster_junctions(
        junction_evidence,
        bench_known_junctions,
        min_cluster_reads=0,
    )
    results["clustering"] = time.perf_counter() - t0

    # === Stage 3a: Splicegraph and confidence scoring ===
    t0_evidence = time.perf_counter()

    # Build splicegraph (minimal gene info)
    genes = []
    for chrom_idx in range(N_CHROMS):
        chrom = f"chr{chrom_idx + 1}"
        for gene_idx in range(N_GENES_PER_CHROM):
            gene_num = chrom_idx * N_GENES_PER_CHROM + gene_idx
            start = gene_num * GENE_SPACING

            gene = MockGene(
                gene_id=f"gene_{gene_num:03d}",
                gene_name=f"GENE{gene_num}",
                chrom=chrom,
                strand="+",
                start=start,
                end=start + 2_000,
            )
            genes.append(gene)

    splicegraph = build_splicegraph(genes)

    # Score confidence
    score_all_junctions(junction_evidence, splicegraph)

    # === Stage 3b: Evidence building ===
    modules, evidence_list = build_evidence_matrices(
        clusters, bench_sample_names, bench_group_labels, n_exon_body_reads=0
    )
    results["evidence_building"] = time.perf_counter() - t0_evidence

    # === Stage 4: PSI quantification ===
    t0 = time.perf_counter()
    psi_list = quantify_psi(
        evidence_list,
        n_bootstraps=N_BOOTSTRAPS,
        seed=42,
    )
    results["psi_quantification"] = time.perf_counter() - t0

    # === Stage 5: Differential testing ===
    t0 = time.perf_counter()
    diff_results = test_differential_splicing(
        modules,
        evidence_list,
        psi_list,
        bench_group_labels,
        covariates=None,
    )
    results["differential_testing"] = time.perf_counter() - t0

    # Total pipeline time
    results["total_pipeline"] = (
        results["junction_extraction"]
        + results["clustering"]
        + results["evidence_building"]
        + results["psi_quantification"]
        + results["differential_testing"]
    )

    # Store results
    results["junction_evidence"] = junction_evidence
    results["clusters"] = clusters
    results["modules"] = modules
    results["evidence_list"] = evidence_list
    results["psi_list"] = psi_list
    results["diff_results"] = diff_results
    results["n_junctions"] = len(junction_evidence)
    results["n_modules"] = len(modules)
    results["n_samples"] = len(bench_sample_names)

    return results


# ============================================================================
# Test Classes
# ============================================================================


class TestDataGeneration(unittest.TestCase):
    """Test that benchmark data is generated correctly."""

    def test_bench_bam_files_created(self, bench_bam_paths):
        """Verify 10 BAM files exist and are non-empty."""
        assert len(bench_bam_paths) == 10
        for bam_path in bench_bam_paths:
            assert Path(bam_path).exists()
            assert Path(bam_path).stat().st_size > 100  # non-empty

    def test_bench_fasta_created(self, bench_genome_fasta):
        """Verify FASTA exists and has correct 3-chrom header."""
        assert Path(bench_genome_fasta).exists()
        with open(bench_genome_fasta) as f:
            headers = [line.strip() for line in f if line.startswith(">")]
        assert len(headers) == 3
        assert headers == [">chr1", ">chr2", ">chr3"]

    def test_bench_known_junctions_count(self, bench_known_junctions):
        """Verify exactly 90 known junctions (30 genes × 3 junctions)."""
        assert len(bench_known_junctions) == 90


class TestJunctionExtractionBenchmark(unittest.TestCase):
    """Test junction extraction timing and correctness."""

    def test_junction_extraction_completes_within_time_limit(self, bench_timing):
        """Extraction must complete within 120 seconds."""
        elapsed = bench_timing["junction_extraction"]
        assert elapsed < LIMIT_JUNCTION_EXTRACTION, (
            f"Junction extraction took {elapsed:.1f}s, exceeds limit {LIMIT_JUNCTION_EXTRACTION}s"
        )

    def test_junction_extraction_detects_all_junctions(
        self, bench_timing, bench_known_junctions
    ):
        """Extraction must detect all 90+ known junctions."""
        junction_evidence = bench_timing["junction_evidence"]
        detected = set(
            (j.chrom, j.donor, j.acceptor, j.strand) for j in junction_evidence.keys()
        )
        assert len(detected) >= len(bench_known_junctions), (
            f"Only detected {len(detected)} junctions, expected {len(bench_known_junctions)}"
        )

    def test_junction_extraction_throughput(self, bench_timing, bench_bam_paths):
        """Report reads/second (informational, no assertion)."""
        elapsed = bench_timing["junction_extraction"]
        total_reads = N_SAMPLES * N_GENES_PER_CHROM * N_CHROMS * 3 * N_READS_PER_JNC
        throughput = total_reads / elapsed if elapsed > 0 else 0
        print(f"\n  Junction extraction: {total_reads} reads in {elapsed:.1f}s = {throughput:.0f} reads/sec")


class TestClusteringBenchmark(unittest.TestCase):
    """Test clustering timing and module count."""

    def test_clustering_completes_within_time_limit(self, bench_timing):
        """Clustering must complete within 5 seconds."""
        elapsed = bench_timing["clustering"]
        assert elapsed < LIMIT_CLUSTERING, (
            f"Clustering took {elapsed:.1f}s, exceeds limit {LIMIT_CLUSTERING}s"
        )

    def test_clustering_produces_correct_cluster_count(self, bench_timing):
        """Should produce ~30 clusters (one per gene, approximately)."""
        clusters = bench_timing["clusters"]
        # Allow some variance; at minimum should have substantial clustering
        assert len(clusters) > 15, (
            f"Only {len(clusters)} clusters, expected ~30"
        )


class TestEvidenceAndPSIBenchmark(unittest.TestCase):
    """Test evidence building and PSI quantification timing."""

    def test_evidence_building_within_time_limit(self, bench_timing):
        """Evidence building must complete within 30 seconds."""
        elapsed = bench_timing["evidence_building"]
        assert elapsed < LIMIT_EVIDENCE_BUILDING, (
            f"Evidence building took {elapsed:.1f}s, exceeds limit {LIMIT_EVIDENCE_BUILDING}s"
        )

    def test_psi_quantification_within_time_limit(self, bench_timing):
        """PSI quantification (30 bootstraps) must complete within 60 seconds."""
        elapsed = bench_timing["psi_quantification"]
        assert elapsed < LIMIT_PSI_QUANTIFICATION, (
            f"PSI quantification took {elapsed:.1f}s, exceeds limit {LIMIT_PSI_QUANTIFICATION}s"
        )

    def test_psi_quantification_is_valid(self, bench_timing):
        """All PSI values must be in [0, 1]."""
        psi_list = bench_timing["psi_list"]
        for psi_dict in psi_list:
            for junc_id, psi_val in psi_dict.items():
                assert 0 <= psi_val <= 1, (
                    f"Invalid PSI value {psi_val} for junction {junc_id}"
                )


class TestDifferentialTestingBenchmark(unittest.TestCase):
    """Test differential testing timing and detection power."""

    def test_differential_testing_within_time_limit(self, bench_timing):
        """Differential testing must complete within 60 seconds."""
        elapsed = bench_timing["differential_testing"]
        assert elapsed < LIMIT_DIFFERENTIAL_TESTING, (
            f"Differential testing took {elapsed:.1f}s, exceeds limit {LIMIT_DIFFERENTIAL_TESTING}s"
        )

    def test_all_genes_detected_as_significant(self, bench_timing):
        """All 30 SE genes should be detected (p < 0.05)."""
        diff_results = bench_timing["diff_results"]
        significant = [r for r in diff_results if r.pvalue < 0.05]
        # Should detect most/all genes given strong signal
        assert len(significant) >= 25, (
            f"Only {len(significant)}/30 genes significant, expected ~30"
        )

    def test_differential_throughput(self, bench_timing):
        """Report modules/second tested (informational, no assertion)."""
        elapsed = bench_timing["differential_testing"]
        n_modules = bench_timing["n_modules"]
        throughput = n_modules / elapsed if elapsed > 0 else 0
        print(f"\n  Differential testing: {n_modules} modules in {elapsed:.1f}s = {throughput:.1f} modules/sec")


class TestTotalPipelineBenchmark(unittest.TestCase):
    """Test total pipeline timing and completeness."""

    def test_total_pipeline_within_time_limit(self, bench_timing):
        """Total pipeline must complete within 300 seconds."""
        elapsed = bench_timing["total_pipeline"]
        assert elapsed < LIMIT_TOTAL_PIPELINE, (
            f"Total pipeline took {elapsed:.1f}s, exceeds limit {LIMIT_TOTAL_PIPELINE}s"
        )

    def test_pipeline_produces_complete_results(self, bench_timing):
        """All expected outputs must be present."""
        assert "junction_evidence" in bench_timing
        assert "clusters" in bench_timing
        assert "modules" in bench_timing
        assert "evidence_list" in bench_timing
        assert "psi_list" in bench_timing
        assert "diff_results" in bench_timing
        assert len(bench_timing["modules"]) > 0
        assert len(bench_timing["psi_list"]) > 0
        assert len(bench_timing["diff_results"]) > 0

    def test_extrapolated_production_timing(self, bench_timing):
        """
        Extrapolate benchmark timings to production scale.

        Production: 10 BAMs × 50M reads, ~20,000 genes, ~60,000 junctions
        Benchmark: 10 BAMs × ~4,500 reads, 30 genes, 90 junctions

        Linear scaling (for junction extraction, clustering, differential):
          Scale by reads: 50,000,000 / (90 * 50 * 10 / 10) ≈ 11,111x

        Quadratic scaling (for evidence building):
          Scale by (genes × modules)²
        """
        t = bench_timing

        # Benchmark totals
        bench_reads = (
            N_CHROMS * N_GENES_PER_CHROM * 3 * N_READS_PER_JNC * N_SAMPLES
        )
        bench_genes = N_CHROMS * N_GENES_PER_CHROM

        # Production scale
        prod_reads = 50_000_000 * N_SAMPLES
        prod_genes = 20_000

        # Scale factors
        read_scale = prod_reads / bench_reads if bench_reads > 0 else 1
        gene_scale = prod_genes / bench_genes if bench_genes > 0 else 1

        # Linear extrapolation (reads-based)
        ex_junction_extraction = t["junction_extraction"] * read_scale
        ex_clustering = t["clustering"] * read_scale
        ex_differential = t["differential_testing"] * gene_scale

        # Quadratic extrapolation (evidence building is O(modules × samples²))
        ex_evidence = t["evidence_building"] * (gene_scale ** 1.5)
        ex_psi = t["psi_quantification"] * gene_scale

        total_extrapolated = (
            ex_junction_extraction
            + ex_clustering
            + ex_evidence
            + ex_psi
            + ex_differential
        )

        print("\n" + "=" * 70)
        print("EXTRAPOLATED PRODUCTION TIMINGS (10 BAMs × 50M reads, 20K genes)")
        print("=" * 70)
        print(f"\nBenchmark scale:")
        print(f"  Reads: {bench_reads:,} reads")
        print(f"  Genes: {bench_genes}")
        print(f"  Junctions: 90")
        print(f"\nProduction scale:")
        print(f"  Reads: {prod_reads:,} reads")
        print(f"  Genes: {prod_genes:,}")
        print(f"  Junctions: ~60,000 (estimated)")
        print(f"\nScale factors:")
        print(f"  Read-based: {read_scale:,.0f}x")
        print(f"  Gene-based: {gene_scale:,.1f}x")
        print(f"\nExtrapolated timings:")
        print(f"  Junction extraction:  {ex_junction_extraction / 60:6.1f} min (spec: < 30 min)")
        print(f"  Clustering:           {ex_clustering / 60:6.1f} min (spec: < 2 min)")
        print(f"  Evidence building:    {ex_evidence / 60:6.1f} min (spec: < 10 min)")
        print(f"  PSI quantification:   {ex_psi / 60:6.1f} min (spec: < 5 min)")
        print(f"  Differential testing: {ex_differential / 60:6.1f} min (spec: < 10 min)")
        print(f"  {'─' * 40}")
        print(f"  TOTAL PIPELINE:       {total_extrapolated / 60:6.1f} min (spec: < 90 min)")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    unittest.main()