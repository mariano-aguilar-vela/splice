"""
tests/conftest.py

Shared pytest fixtures for SPLICE integration and benchmark tests.

Creates synthetic BAM, GTF, and genome FASTA files on disk using pysam
and tempfile. All fixtures are session-scoped and reused across test files.

Synthetic data layout (all on chr1):
  - Gene A: SE (skipped exon) event, differential (PSI_skip ≈ 0.2 vs 0.72)
  - Gene B: A3SS (alt 3' splice site) event, no differential
  - Gene C: Complex event (5 junctions), moderate differential + 1 novel junction
"""

import os
import tempfile
from typing import Dict, List, Set

import numpy as np
import pysam
import pytest

from splicekit.utils.genomic import Junction

# ─── Read and sequence parameters ─────────────────────────────────────────────
READ_LENGTH = 50      # synthetic read length (bp)
ANCHOR = 25          # bases on each side of junction for spanning reads
MAPQ = 60             # mapping quality for synthetic reads
SEQ_LEN = 11000      # total length of synthetic chr1 sequence

# ─── Gene A: Skipped Exon (SE) event ─────────────────────────────────────────
# Exons (0-based half-open): 1000-1100, 2000-2100 (cassette), 3000-3100
# Junctions: exon1->exon2, exon2->exon3, exon1->exon3 (skip)
# Differential: group1 PSI_skip ≈ 0.2, group2 PSI_skip ≈ 0.72
A_CHROM = "chr1"
A_STRAND = "+"
J_A1 = Junction("chr1", 1100, 2000, "+")   # exon1->exon2 (inclusion)
J_A2 = Junction("chr1", 2100, 3000, "+")   # exon2->exon3 (inclusion)
J_A3 = Junction("chr1", 1100, 3000, "+")   # exon1->exon3 (skipping)

# Group 1: group 1 PSI_skip = 10/50 = 0.20
_A_G1_COUNTS = {J_A1: 20, J_A2: 20, J_A3: 10}
# Group 2: group 2 PSI_skip = 36/50 = 0.72
_A_G2_COUNTS = {J_A1: 7, J_A2: 7, J_A3: 36}

# ─── Gene B: Alternative 3' Splice Site (A3SS) event ────────────────────────
# Exons: 5000-5100, then either 5950-6100 (long form) or 6000-6100 (short form)
# Junctions: 5100->5950 (long), 5100->6000 (short)
# No differential: equal usage in both groups
B_CHROM = "chr1"
B_STRAND = "+"
J_B1 = Junction("chr1", 5100, 5950, "+")   # to long form
J_B2 = Junction("chr1", 5100, 6000, "+")   # to short form

# Equal usage in both groups (PSI = 0.5 for each junction)
_B_G1_COUNTS = {J_B1: 20, J_B2: 20}
_B_G2_COUNTS = {J_B1: 20, J_B2: 20}

# ─── Gene C: Complex event with 5 junctions (4 annotated + 1 novel) ─────────
# Exons: 8000-8100, 8500-8600, 9000-9100, 9500-9600, 10000-10100
# Annotated junctions: C1 (exon1->2), C2 (exon2->3), C3 (exon3->4), C4 (exon4->5)
# Novel junction: C5 (exon1->3, skip, intentionally absent from GTF)
# Moderate differential distribution + novel junction present in both groups
C_CHROM = "chr1"
C_STRAND = "+"
J_C1 = Junction("chr1", 8100, 8500, "+")     # exon1->exon2
J_C2 = Junction("chr1", 8600, 9000, "+")     # exon2->exon3
J_C3 = Junction("chr1", 9100, 9500, "+")     # exon3->exon4
J_C4 = Junction("chr1", 9600, 10000, "+")    # exon4->exon5
J_C5_NOVEL = Junction("chr1", 8100, 9000, "+")  # novel: exon1->exon3 skip

# Group 1: more reads for C1/C2 pathway
_C_G1_COUNTS = {J_C1: 20, J_C2: 20, J_C3: 5, J_C4: 5, J_C5_NOVEL: 10}
# Group 2: more reads for C3/C4 pathway
_C_G2_COUNTS = {J_C1: 5, J_C2: 5, J_C3: 20, J_C4: 20, J_C5_NOVEL: 10}


# ─── Helper functions ─────────────────────────────────────────────────────────


def _make_genome_seq() -> str:
    """Create synthetic chr1 sequence with GT/AG motifs at all splice sites."""
    # Base sequence using fixed cyclic pattern
    seq = list("ACGT" * (SEQ_LEN // 4 + 1))
    seq = seq[:SEQ_LEN]

    # Place GT at all donor sites (intron 5' end)
    # Gene A: 1100, 2100; Gene B: 5100; Gene C: 8100, 8600, 9100, 9600
    donor_positions = [1100, 2100, 5100, 8100, 8600, 9100, 9600]
    for pos in donor_positions:
        seq[pos] = "G"
        seq[pos + 1] = "T"

    # Place AG at all acceptor sites (intron 3' end, 2 bp before exon start)
    # Gene A: 1998 (2000-2), 2998 (3000-2)
    # Gene B: 5948 (5950-2), 5998 (6000-2)
    # Gene C: 8498 (8500-2), 8998 (9000-2), 9498 (9500-2), 9998 (10000-2)
    acceptor_positions = [
        1998, 2998, 5948, 5998, 8498, 8998, 9498, 9998
    ]
    for pos in acceptor_positions:
        seq[pos] = "A"
        seq[pos + 1] = "G"

    return "".join(seq)


def _make_bam_header() -> pysam.AlignmentHeader:
    """Create pysam BAM header for synthetic chr1."""
    return pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": SEQ_LEN}],
    })


def _make_aligned_read(
    header: pysam.AlignmentHeader,
    read_name: str,
    ref_start: int,
    cigartuples: list,
    mapq: int = MAPQ,
) -> pysam.AlignedSegment:
    """Create a synthetic pysam AlignedSegment with the given parameters."""
    a = pysam.AlignedSegment(header)
    a.query_name = read_name
    a.query_sequence = "A" * READ_LENGTH
    a.flag = 0
    a.reference_id = 0  # chr1 (reference index 0)
    a.reference_start = ref_start
    a.mapping_quality = mapq
    a.cigartuples = cigartuples
    a.query_qualities = pysam.qualitystring_to_array("I" * READ_LENGTH)
    a.set_tag("NH", 1)  # single alignment
    a.set_tag("AS", READ_LENGTH * 2)  # alignment score
    return a


def _make_junction_spanning_read(
    header: pysam.AlignmentHeader,
    read_name: str,
    junction: Junction,
    anchor: int = ANCHOR,
) -> pysam.AlignedSegment:
    """
    Create a synthetic read spanning a junction.

    Positions the read with `anchor` bases in the upstream exon and the rest
    in the downstream exon. CIGAR will be: anchor M + intron N + downstream M

    Args:
        header: pysam AlignmentHeader
        read_name: read identifier
        junction: Junction to span
        anchor: bases on each side of junction (default 25)

    Returns:
        pysam.AlignedSegment spanning the junction
    """
    exon_end = junction.start      # intron starts at exon end (0-based)
    intron_len = junction.end - junction.start
    ref_start = exon_end - anchor  # start position in genome
    downstream = READ_LENGTH - anchor

    cigar = [(0, anchor), (3, intron_len), (0, downstream)]
    return _make_aligned_read(header, read_name, ref_start, cigar)


def _write_sorted_indexed_bam(
    bam_path: str,
    header: pysam.AlignmentHeader,
    reads: List[pysam.AlignedSegment],
) -> None:
    """
    Write reads to BAM file, sort by coordinate, and create index.

    Args:
        bam_path: output BAM file path
        header: pysam AlignmentHeader
        reads: list of AlignedSegments
    """
    # Write unsorted BAM
    unsorted_path = bam_path + ".unsorted"
    with pysam.AlignmentFile(unsorted_path, "wb", header=header) as bam:
        for read in reads:
            bam.write(read)

    # Sort by coordinate
    pysam.sort("-o", bam_path, unsorted_path)

    # Create index
    pysam.index(bam_path)

    # Clean up unsorted file
    os.unlink(unsorted_path)


def _build_sample_reads(
    header: pysam.AlignmentHeader,
    sample_name: str,
    junction_counts: Dict[Junction, int],
) -> List[pysam.AlignedSegment]:
    """
    Build list of reads for one sample based on junction counts.

    Args:
        header: pysam AlignmentHeader
        sample_name: sample identifier for read names
        junction_counts: dict mapping Junction -> count of spanning reads

    Returns:
        List of AlignedSegments (not yet sorted)
    """
    reads = []
    for junction, count in junction_counts.items():
        for i in range(count):
            read_name = f"{sample_name}_{junction.start}_{junction.end}_{i}"
            reads.append(_make_junction_spanning_read(header, read_name, junction))

    # Sort by reference_start for better coordinate order
    reads.sort(key=lambda r: r.reference_start)
    return reads


# ─── Pytest fixtures (session-scoped) ─────────────────────────────────────────


@pytest.fixture(scope="session")
def tmp_dir():
    """Session-scoped temporary directory for all synthetic test files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture(scope="session")
def genome_fasta_path(tmp_dir: str) -> str:
    """
    Create synthetic genome FASTA with chr1 and GT/AG at all splice sites.

    The FASTA is written with 60 bases per line. pyfastx will build a .fxi
    index on first access.

    Args:
        tmp_dir: temporary directory path

    Yields:
        path to genome.fa
    """
    fasta_path = os.path.join(tmp_dir, "genome.fa")
    seq = _make_genome_seq()

    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        # Write 60 bases per line (standard FASTA wrapping)
        for i in range(0, len(seq), 60):
            f.write(seq[i:i + 60] + "\n")

    return fasta_path


@pytest.fixture(scope="session")
def gtf_path(tmp_dir: str) -> str:
    """
    Create synthetic GTF with Gene A (SE), Gene B (A3SS), Gene C (Complex).

    GTF uses 1-based closed intervals. All genes on chr1 with + strand.
    Gene C's novel junction (J_C5_NOVEL) is intentionally omitted from GTF.

    Args:
        tmp_dir: temporary directory path

    Yields:
        path to genes.gtf
    """
    gtf_path = os.path.join(tmp_dir, "genes.gtf")

    def gtf_line(chrom: str, feature: str, start_0b: int, end_0b: int,
                 strand: str, attrs: Dict[str, str]) -> str:
        """
        Convert 0-based half-open coordinates to 1-based closed (GTF format).

        0-based [start, end) → 1-based [start+1, end]
        """
        start_1b = start_0b + 1
        end_1b = end_0b  # end stays the same
        attr_str = " ".join(f'{k} "{v}";' for k, v in attrs.items())
        return f"{chrom}\tsynthetic\t{feature}\t{start_1b}\t{end_1b}\t.\t{strand}\t.\t{attr_str}\n"

    lines = []

    # ── Gene A: SE event (exons at 1000-1100, 2000-2100, 3000-3100) ────────
    lines.append(
        gtf_line("chr1", "gene", 1000, 3100, "+",
                 {"gene_id": "GENE_A", "gene_name": "GeneA",
                  "gene_type": "protein_coding"})
    )

    # Gene A, Transcript 1: inclusion (exon1 + exon2 + exon3)
    lines.append(
        gtf_line("chr1", "transcript", 1000, 3100, "+",
                 {"gene_id": "GENE_A", "gene_name": "GeneA",
                  "transcript_id": "GENE_A_t1", "transcript_type": "protein_coding"})
    )
    lines.append(
        gtf_line("chr1", "exon", 1000, 1100, "+",
                 {"gene_id": "GENE_A", "gene_name": "GeneA",
                  "transcript_id": "GENE_A_t1"})
    )
    lines.append(
        gtf_line("chr1", "exon", 2000, 2100, "+",
                 {"gene_id": "GENE_A", "gene_name": "GeneA",
                  "transcript_id": "GENE_A_t1"})
    )
    lines.append(
        gtf_line("chr1", "exon", 3000, 3100, "+",
                 {"gene_id": "GENE_A", "gene_name": "GeneA",
                  "transcript_id": "GENE_A_t1"})
    )

    # Gene A, Transcript 2: skipping (exon1 + exon3, skip exon2)
    lines.append(
        gtf_line("chr1", "transcript", 1000, 3100, "+",
                 {"gene_id": "GENE_A", "gene_name": "GeneA",
                  "transcript_id": "GENE_A_t2", "transcript_type": "protein_coding"})
    )
    lines.append(
        gtf_line("chr1", "exon", 1000, 1100, "+",
                 {"gene_id": "GENE_A", "gene_name": "GeneA",
                  "transcript_id": "GENE_A_t2"})
    )
    lines.append(
        gtf_line("chr1", "exon", 3000, 3100, "+",
                 {"gene_id": "GENE_A", "gene_name": "GeneA",
                  "transcript_id": "GENE_A_t2"})
    )

    # ── Gene B: A3SS event (exon1 at 5000-5100, exon2 forms at 5950-6100 or 6000-6100) ─
    lines.append(
        gtf_line("chr1", "gene", 5000, 6100, "+",
                 {"gene_id": "GENE_B", "gene_name": "GeneB",
                  "gene_type": "protein_coding"})
    )

    # Gene B, Transcript 1: long form (acceptor at 5950)
    lines.append(
        gtf_line("chr1", "transcript", 5000, 6100, "+",
                 {"gene_id": "GENE_B", "gene_name": "GeneB",
                  "transcript_id": "GENE_B_t1", "transcript_type": "protein_coding"})
    )
    lines.append(
        gtf_line("chr1", "exon", 5000, 5100, "+",
                 {"gene_id": "GENE_B", "gene_name": "GeneB",
                  "transcript_id": "GENE_B_t1"})
    )
    lines.append(
        gtf_line("chr1", "exon", 5950, 6100, "+",
                 {"gene_id": "GENE_B", "gene_name": "GeneB",
                  "transcript_id": "GENE_B_t1"})
    )

    # Gene B, Transcript 2: short form (acceptor at 6000)
    lines.append(
        gtf_line("chr1", "transcript", 5000, 6100, "+",
                 {"gene_id": "GENE_B", "gene_name": "GeneB",
                  "transcript_id": "GENE_B_t2", "transcript_type": "protein_coding"})
    )
    lines.append(
        gtf_line("chr1", "exon", 5000, 5100, "+",
                 {"gene_id": "GENE_B", "gene_name": "GeneB",
                  "transcript_id": "GENE_B_t2"})
    )
    lines.append(
        gtf_line("chr1", "exon", 6000, 6100, "+",
                 {"gene_id": "GENE_B", "gene_name": "GeneB",
                  "transcript_id": "GENE_B_t2"})
    )

    # ── Gene C: Complex event (5 exons, linear transcript, 4 annotated + 1 novel junction) ─
    lines.append(
        gtf_line("chr1", "gene", 8000, 10100, "+",
                 {"gene_id": "GENE_C", "gene_name": "GeneC",
                  "gene_type": "protein_coding"})
    )

    # Gene C, Transcript 1: linear path (exon1 -> exon2 -> exon3 -> exon4 -> exon5)
    # Note: J_C5_NOVEL (8100->9000) is intentionally NOT in this transcript
    lines.append(
        gtf_line("chr1", "transcript", 8000, 10100, "+",
                 {"gene_id": "GENE_C", "gene_name": "GeneC",
                  "transcript_id": "GENE_C_t1", "transcript_type": "protein_coding"})
    )
    for i, (exon_start, exon_end) in enumerate(
        [(8000, 8100), (8500, 8600), (9000, 9100), (9500, 9600), (10000, 10100)],
        start=1,
    ):
        lines.append(
            gtf_line("chr1", "exon", exon_start, exon_end, "+",
                     {"gene_id": "GENE_C", "gene_name": "GeneC",
                      "transcript_id": "GENE_C_t1"})
        )

    with open(gtf_path, "w") as f:
        f.writelines(lines)

    return gtf_path


@pytest.fixture(scope="session")
def group1_bam_paths(tmp_dir: str) -> List[str]:
    """
    Create three BAM files for group 1 (PSI_skip ≈ 0.2 for Gene A).

    BAM files are named g1_s1.bam, g1_s2.bam, g1_s3.bam and contain:
      - Gene A: 20 reads J_A1, 20 reads J_A2, 10 reads J_A3 (PSI_skip = 0.2)
      - Gene B: 20 reads J_B1, 20 reads J_B2 (equal usage)
      - Gene C: 20 reads J_C1, 20 reads J_C2, 5 reads J_C3, 5 reads J_C4, 10 reads J_C5_NOVEL

    Args:
        tmp_dir: temporary directory path

    Yields:
        list of 3 BAM file paths
    """
    header = _make_bam_header()
    paths = []

    for i in range(3):
        sample_name = f"g1_s{i + 1}"
        bam_path = os.path.join(tmp_dir, f"{sample_name}.bam")

        # Merge junction counts for all genes
        junction_counts = {}
        junction_counts.update(_A_G1_COUNTS)
        junction_counts.update(_B_G1_COUNTS)
        junction_counts.update(_C_G1_COUNTS)

        # Build and write reads
        reads = _build_sample_reads(header, sample_name, junction_counts)
        _write_sorted_indexed_bam(bam_path, header, reads)

        paths.append(bam_path)

    return paths


@pytest.fixture(scope="session")
def group2_bam_paths(tmp_dir: str) -> List[str]:
    """
    Create three BAM files for group 2 (PSI_skip ≈ 0.72 for Gene A).

    BAM files are named g2_s1.bam, g2_s2.bam, g2_s3.bam and contain:
      - Gene A: 7 reads J_A1, 7 reads J_A2, 36 reads J_A3 (PSI_skip = 0.72)
      - Gene B: 20 reads J_B1, 20 reads J_B2 (equal usage)
      - Gene C: 5 reads J_C1, 5 reads J_C2, 20 reads J_C3, 20 reads J_C4, 10 reads J_C5_NOVEL

    Args:
        tmp_dir: temporary directory path

    Yields:
        list of 3 BAM file paths
    """
    header = _make_bam_header()
    paths = []

    for i in range(3):
        sample_name = f"g2_s{i + 1}"
        bam_path = os.path.join(tmp_dir, f"{sample_name}.bam")

        # Merge junction counts for all genes
        junction_counts = {}
        junction_counts.update(_A_G2_COUNTS)
        junction_counts.update(_B_G2_COUNTS)
        junction_counts.update(_C_G2_COUNTS)

        # Build and write reads
        reads = _build_sample_reads(header, sample_name, junction_counts)
        _write_sorted_indexed_bam(bam_path, header, reads)

        paths.append(bam_path)

    return paths


@pytest.fixture(scope="session")
def all_bam_paths(group1_bam_paths: List[str], group2_bam_paths: List[str]) -> List[str]:
    """
    Combined list of all 6 BAM files: group1 (3) + group2 (3).

    Sample order: [g1_s1, g1_s2, g1_s3, g2_s1, g2_s2, g2_s3]

    Args:
        group1_bam_paths: list of 3 group 1 BAM paths
        group2_bam_paths: list of 3 group 2 BAM paths

    Yields:
        list of 6 BAM paths
    """
    return group1_bam_paths + group2_bam_paths


@pytest.fixture(scope="session")
def sample_names() -> List[str]:
    """
    Sample names matching all_bam_paths order.

    Returns:
        ["g1_s1", "g1_s2", "g1_s3", "g2_s1", "g2_s2", "g2_s3"]
    """
    return ["g1_s1", "g1_s2", "g1_s3", "g2_s1", "g2_s2", "g2_s3"]


@pytest.fixture(scope="session")
def group1_indices() -> List[int]:
    """
    Indices of group 1 samples in all_bam_paths / sample_names.

    Returns:
        [0, 1, 2]
    """
    return [0, 1, 2]


@pytest.fixture(scope="session")
def group2_indices() -> List[int]:
    """
    Indices of group 2 samples in all_bam_paths / sample_names.

    Returns:
        [3, 4, 5]
    """
    return [3, 4, 5]


@pytest.fixture(scope="session")
def known_junctions() -> Set[Junction]:
    """
    Set of all annotated junctions from GTF (excludes J_C5_NOVEL).

    Returns 9 junctions:
      - Gene A: J_A1, J_A2, J_A3
      - Gene B: J_B1, J_B2
      - Gene C: J_C1, J_C2, J_C3, J_C4

    Returns:
        Set[Junction] of annotated junctions
    """
    return {J_A1, J_A2, J_A3, J_B1, J_B2, J_C1, J_C2, J_C3, J_C4}


@pytest.fixture(scope="session")
def gene_a_junctions() -> List[Junction]:
    """
    All junctions in Gene A cluster (including skipping junction J_A3).

    Returns:
        [J_A1, J_A2, J_A3]
    """
    return [J_A1, J_A2, J_A3]


@pytest.fixture(scope="session")
def gene_b_junctions() -> List[Junction]:
    """
    All junctions in Gene B cluster (A3SS alternatives).

    Returns:
        [J_B1, J_B2]
    """
    return [J_B1, J_B2]


@pytest.fixture(scope="session")
def gene_c_junctions() -> List[Junction]:
    """
    All junctions in Gene C cluster (including novel J_C5_NOVEL).

    Returns:
        [J_C1, J_C2, J_C3, J_C4, J_C5_NOVEL]
    """
    return [J_C1, J_C2, J_C3, J_C4, J_C5_NOVEL]


@pytest.fixture(scope="session")
def group_labels() -> np.ndarray:
    """
    Group label array for all 6 samples.

    Returns numpy array: [0, 0, 0, 1, 1, 1]
      - 0: group 1 (samples 0-2)
      - 1: group 2 (samples 3-5)

    Returns:
        numpy.ndarray of shape (6,) with dtype int
    """
    return np.array([0, 0, 0, 1, 1, 1])
