"""
Module 20: core/nmd_classifier.py

Graph-based NMD/PTC functional classification.
Detects premature termination codons (PTCs) and nonsense-mediated decay (NMD) eligibility.
Uses polynomial-time transcript-based analysis instead of exponential path enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from Bio.Seq import Seq

from splice.utils.genomic import Junction


@dataclass(frozen=True, slots=True)
class NMDClassification:
    """NMD/PTC classification for a junction.

    Attributes:
        junction: Junction object being classified.
        classification: "PR" (productive), "UP" (unproductive), "NE" (ambiguous), "IN" (intergenic).
        n_productive_paths: Number of paths avoiding PTC.
        n_unproductive_paths: Number of paths with PTC.
        confidence: n_productive / (n_productive + n_unproductive) or NaN.
        ptc_position: Genomic position of earliest PTC if unproductive.
        last_ejc_position: Position of last exon-exon junction.
    """

    junction: Junction
    classification: str
    n_productive_paths: int
    n_unproductive_paths: int
    confidence: float
    ptc_position: Optional[int]
    last_ejc_position: Optional[int]


# Constants
STOP_CODONS = {"TAA", "TAG", "TGA"}
NMD_THRESHOLD = 55  # bp upstream of EJC for NMD trigger


def _reverse_complement(seq: str) -> str:
    """Get reverse complement of DNA sequence."""
    complement = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(complement.get(base, "N") for base in reversed(seq))


def _get_sequence_at_position(
    genome_fasta: Dict[str, str],
    chrom: str,
    start: int,
    end: int,
    strand: str,
) -> str:
    """Extract sequence from genome."""
    if chrom not in genome_fasta:
        return ""

    seq = genome_fasta[chrom][start:end].upper()

    if strand == "-":
        seq = _reverse_complement(seq)

    return seq


def _find_stop_codon_in_frame(
    sequence: str, frame: int = 0
) -> Optional[int]:
    """Find first stop codon in given reading frame.

    Returns offset in sequence, or None if no stop codon found.
    """
    for i in range(frame, len(sequence) - 2, 3):
        codon = sequence[i : i + 3].upper()
        if len(codon) == 3 and codon in STOP_CODONS:
            return i
    return None


def _substitute_junction_in_transcript(
    exons: List[Tuple[int, int]],
    junction: Junction,
) -> Optional[List[Tuple[int, int]]]:
    """Build a modified exon list by substituting a junction into a transcript.

    Finds the transcript junction that shares a splice site with the given
    junction and modifies the exon boundaries accordingly.

    Returns None if the junction cannot be substituted.
    """
    modified = list(exons)

    for i in range(len(exons) - 1):
        tx_donor = exons[i][1]
        tx_acceptor = exons[i + 1][0]

        if junction.start == tx_donor and junction.end == tx_acceptor:
            # Exact match: no modification needed
            return modified

        if junction.start == tx_donor and junction.end != tx_acceptor:
            # Same donor, different acceptor: modify downstream exon start
            modified[i + 1] = (junction.end, exons[i + 1][1])
            if modified[i + 1][0] >= modified[i + 1][1]:
                return None
            return modified

        if junction.end == tx_acceptor and junction.start != tx_donor:
            # Same acceptor, different donor: modify upstream exon end
            modified[i] = (exons[i][0], junction.start)
            if modified[i][0] >= modified[i][1]:
                return None
            return modified

    return None


def _extract_mrna_sequence(
    exons: List[Tuple[int, int]],
    genome_fasta: Dict[str, str],
    chrom: str,
    strand: str,
) -> str:
    """Concatenate exon sequences to form the spliced mRNA."""
    parts = []
    sorted_exons = sorted(exons, key=lambda e: e[0])

    for start, end in sorted_exons:
        seq = _get_sequence_at_position(genome_fasta, chrom, start, end, "+")
        if not seq:
            return ""
        parts.append(seq)

    mrna = "".join(parts)

    if strand == "-":
        mrna = _reverse_complement(mrna)

    return mrna


def _compute_ejc_positions(exons: List[Tuple[int, int]]) -> List[int]:
    """Compute exon-exon junction positions in mRNA coordinates.

    EJC positions are the cumulative exon lengths at each junction boundary.
    """
    sorted_exons = sorted(exons, key=lambda e: e[0])
    positions = []
    cumulative = 0

    for i, (start, end) in enumerate(sorted_exons):
        cumulative += end - start
        if i < len(sorted_exons) - 1:
            positions.append(cumulative)

    return positions


def classify_junction_nmd(
    junction: Junction,
    gene_transcripts: Dict[str, List[Tuple[int, int]]],
    genome_fasta: Dict[str, str],
    chrom: str,
    strand: str,
) -> NMDClassification:
    """Classify a junction for NMD.

    For each annotated transcript that shares a splice site with this junction:
    1. Create a modified transcript by substituting this junction.
    2. Extract and translate the modified mRNA.
    3. Check for PTC > 55 nt upstream of last EJC.

    If no transcripts share a splice site, classify as "NE".

    Args:
        junction: Junction to classify.
        gene_transcripts: Dict mapping transcript_id to sorted exon list [(start, end), ...].
        genome_fasta: Dict mapping chromosome to sequence string.
        chrom: Chromosome of the gene.
        strand: Strand of the gene.

    Returns:
        NMDClassification object.
    """
    n_productive = 0
    n_unproductive = 0
    earliest_ptc = None
    last_ejc = None

    for tx_id, exons in gene_transcripts.items():
        # Check if this junction shares a splice site with any junction in this transcript
        tx_junctions = []
        for i in range(len(exons) - 1):
            tx_junctions.append((exons[i][1], exons[i + 1][0]))

        shares_site = False
        for tx_donor, tx_acceptor in tx_junctions:
            if junction.start == tx_donor or junction.end == tx_acceptor:
                shares_site = True
                break

        if not shares_site:
            continue

        # Build modified exon list by substituting this junction
        modified_exons = _substitute_junction_in_transcript(exons, junction)
        if modified_exons is None:
            continue

        # Extract and translate
        mrna_seq = _extract_mrna_sequence(modified_exons, genome_fasta, chrom, strand)
        if not mrna_seq or len(mrna_seq) < 3:
            continue

        # Find start codon
        start_pos = mrna_seq.find("ATG")
        if start_pos < 0:
            continue

        # Translate from start codon
        coding_seq = mrna_seq[start_pos:]
        protein = str(Seq(coding_seq).translate())

        # Find first stop codon position in mRNA coordinates
        stop_idx = protein.find("*")
        if stop_idx < 0:
            # No stop codon found: readthrough, classify as productive
            n_productive += 1
            continue

        stop_mrna_pos = start_pos + stop_idx * 3

        # Find last EJC position in mRNA coordinates
        ejc_positions = _compute_ejc_positions(modified_exons)
        if not ejc_positions:
            n_productive += 1
            continue

        last_ejc_mrna = ejc_positions[-1]

        # NMD rule: PTC is > 55 nt upstream of last EJC
        distance_to_last_ejc = last_ejc_mrna - stop_mrna_pos
        if distance_to_last_ejc > NMD_THRESHOLD:
            n_unproductive += 1
            if earliest_ptc is None or stop_mrna_pos < earliest_ptc:
                earliest_ptc = stop_mrna_pos
        else:
            n_productive += 1

        last_ejc = last_ejc_mrna

    # Determine classification
    total = n_productive + n_unproductive
    if total == 0:
        classification = "NE"
        confidence = float('nan')
    elif n_productive > 0:
        classification = "PR"
        confidence = n_productive / total
    else:
        classification = "UP"
        confidence = 0.0

    return NMDClassification(
        junction=junction,
        classification=classification,
        n_productive_paths=n_productive,
        n_unproductive_paths=n_unproductive,
        confidence=confidence,
        ptc_position=earliest_ptc,
        last_ejc_position=last_ejc,
    )


def build_translation_graph(
    exon_positions: Dict[int, Tuple[int, int]],
    observed_junctions: Set[Junction],
    genome_fasta: Dict[str, str],
    strand: str = "+",
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """Build translation graph for analyzing reading frames.

    Nodes are (exon_index, reading_frame) tuples.
    Edges connect exons through observed junctions, tracking frame shifts.

    Args:
        exon_positions: Dict mapping exon index to (start, end) positions.
        observed_junctions: Set of Junction objects.
        genome_fasta: Dict mapping chromosome to sequence.
        strand: Genomic strand.

    Returns:
        Adjacency list representation of translation graph.
    """
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for exon_idx in exon_positions:
        for frame in range(3):
            node = (exon_idx, frame)
            graph[node] = []

    for junction in observed_junctions:
        intron_length = junction.end - junction.start
        frame_shift = intron_length % 3

        for src_exon_idx, (exon_start, exon_end) in exon_positions.items():
            if exon_end == junction.start:
                for dst_exon_idx, (dst_start, dst_end) in exon_positions.items():
                    if dst_start == junction.end:
                        for src_frame in range(3):
                            dst_frame = (src_frame + frame_shift) % 3
                            graph[(src_exon_idx, src_frame)].append(
                                (dst_exon_idx, dst_frame)
                            )

    return graph


def classify_all_junctions_nmd(
    junctions: List[Junction],
    gene_transcripts: Dict[str, List[Tuple[int, int]]],
    genome_fasta: Dict[str, str],
    chrom: str,
    strand: str,
) -> List[NMDClassification]:
    """Classify multiple junctions for NMD eligibility.

    Args:
        junctions: List of Junction objects.
        gene_transcripts: Dict mapping transcript_id to sorted exon list.
        genome_fasta: Dict mapping chromosome to sequence.
        chrom: Chromosome.
        strand: Genomic strand.

    Returns:
        List of NMDClassification objects.
    """
    classifications = []
    for junction in junctions:
        classification = classify_junction_nmd(
            junction, gene_transcripts, genome_fasta, chrom, strand
        )
        classifications.append(classification)

    return classifications
