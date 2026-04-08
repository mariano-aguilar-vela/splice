"""
Module 2: utils/motif.py

Splice site motif classification and scoring, following STAR's motif hierarchy.
Supports extraction from genome FASTA.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pyfastx


# Motif scores following STAR's hierarchy
# (stitchAlignToTranscript.cpp lines 125-144)
MOTIF_SCORES = {
    "GT/AG": 1.0,           # canonical, no penalty
    "GC/AG": 0.8,           # semi-canonical, STAR penalty -4
    "AT/AC": 0.6,           # semi-canonical, STAR penalty -8
    "non-canonical": 0.2    # non-canonical, STAR penalty -8
}


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence.

    Args:
        seq: DNA sequence (case-insensitive).

    Returns:
        Reverse complement in uppercase.
    """
    complement_map = {"A": "T", "T": "A", "G": "C", "C": "G"}
    seq_upper = seq.upper()
    return "".join(complement_map.get(base, "N") for base in reversed(seq_upper))


def classify_motif(
    donor_dinucleotide: str,
    acceptor_dinucleotide: str
) -> str:
    """Classify a splice junction by its donor/acceptor dinucleotides.

    Args:
        donor_dinucleotide: 2 bases at the intron 5' end (e.g., "GT").
        acceptor_dinucleotide: 2 bases at the intron 3' end (e.g., "AG").

    Returns:
        One of: "GT/AG", "GC/AG", "AT/AC", "non-canonical".

    Notes:
        Also handles reverse complement: "CT/AC" is recognized as the reverse
        complement of "GT/AG" and classified as such. This ensures that junctions
        extracted from opposite strands are correctly classified.
    """
    # Normalize to uppercase
    d = donor_dinucleotide.upper()
    a = acceptor_dinucleotide.upper()

    # Check canonical forms
    if d == "GT" and a == "AG":
        return "GT/AG"
    if d == "GC" and a == "AG":
        return "GC/AG"
    if d == "AT" and a == "AC":
        return "AT/AC"

    # Check reverse complements
    # (important for handling strand-specific extraction or data from reverse strand)
    rc_d = _reverse_complement(d)
    rc_a = _reverse_complement(a)

    if rc_d == "GT" and rc_a == "AG":
        return "GT/AG"
    if rc_d == "GC" and rc_a == "AG":
        return "GC/AG"
    if rc_d == "AT" and rc_a == "AC":
        return "AT/AC"

    return "non-canonical"


def score_motif(motif: str) -> float:
    """Return the confidence score for a motif class.

    Args:
        motif: One of the motif classes returned by classify_motif.

    Returns:
        Score in [0.2, 1.0], with 0.2 as default for unknown motifs.
    """
    return MOTIF_SCORES.get(motif, 0.2)


def extract_motif_from_genome(
    genome_fasta_path: str,
    chrom: str,
    intron_start: int,
    intron_end: int,
    strand: str
) -> Tuple[str, str, str]:
    """Extract donor and acceptor dinucleotides from genome FASTA.

    Extracts the 2 bases at the intron boundaries using pyfastx for fast
    indexed FASTA access. For + strand junctions, donor is at intron_start
    and acceptor is at intron_end - 2. For - strand junctions, positions are
    swapped but extraction is still from the genome.

    Args:
        genome_fasta_path: Path to indexed genome FASTA file.
        chrom: Chromosome name (must match FASTA header).
        intron_start: Intron start position (0-based).
        intron_end: Intron end position (0-based exclusive).
        strand: '+', '-', or '.'.

    Returns:
        Tuple of (donor_dinuc, acceptor_dinuc, motif_class).
        Both dinucleotides are uppercase.

    Raises:
        pyfastx.FetchError: If chrom is not found in FASTA.
        IOError: If FASTA file cannot be opened.
    """
    fasta = pyfastx.Fasta(genome_fasta_path)
    if strand == "+":
        # Donor at intron start, acceptor at intron end
        donor_dinuc = fasta[chrom][intron_start:intron_start+2].seq.upper()
        acceptor_dinuc = fasta[chrom][intron_end-2:intron_end].seq.upper()
    else:
        # "-" or ".": donor at intron_end-1, acceptor at intron_start
        # Extract the 2 bases before the end and 2 bases at the start
        donor_dinuc = fasta[chrom][intron_end-2:intron_end].seq.upper()
        acceptor_dinuc = fasta[chrom][intron_start:intron_start+2].seq.upper()

    motif_class = classify_motif(donor_dinuc, acceptor_dinuc)
    return (donor_dinuc, acceptor_dinuc, motif_class)
