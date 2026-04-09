"""
Module 3: core/gtf_parser.py

Parse GTF/GFF3 annotations into gene models with exon structures.
Extracts known splice junctions from transcript exon chains.
Designed for GENCODE annotations but works with any standard GTF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from splice.utils.genomic import Junction


@dataclass
class Gene:
    """A gene parsed from GTF with its exon structure.

    Attributes:
        gene_id: Ensembl gene ID (e.g., ENSG00000134323.14).
        gene_name: Gene symbol (e.g., MYCN).
        chrom: Chromosome (e.g., chr2).
        strand: Strand (+ or -).
        start: Gene start (0-based).
        end: Gene end (0-based, exclusive).
        transcripts: Dict mapping transcript_id to list of (start, end) exon tuples.
    """

    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    start: int
    end: int
    transcripts: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)


def parse_gtf(
    gtf_path: str,
    feature_filter: str = "exon",
    gene_type_filter: str = "protein_coding",
) -> Dict[str, Gene]:
    """Parse a GTF file into Gene objects.

    Reads the GTF line by line, extracting gene and exon features.
    Builds a dict of Gene objects with per-transcript exon lists.

    GTF coordinates are 1-based closed [start, end].
    Internally we convert to 0-based half-open [start, end) for consistency
    with BAM coordinates and the rest of the SPLICE pipeline.

    Args:
        gtf_path: Path to GTF file.
        feature_filter: Feature type to extract exons from (default "exon").
        gene_type_filter: Only include genes of this type. Set to None to
            include all genes.

    Returns:
        Dict mapping gene_id -> Gene object.
    """
    genes: Dict[str, Gene] = {}

    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue

            feature = fields[2]
            chrom = fields[0]
            start_1based = int(fields[3])
            end_1based = int(fields[4])
            strand = fields[6]
            attrs_str = fields[8]

            # Convert to 0-based half-open
            start = start_1based - 1
            end = end_1based

            # Parse attributes
            attrs = _parse_attributes(attrs_str)
            gene_id = attrs.get("gene_id", "")
            gene_name = attrs.get("gene_name", "")
            gene_type = attrs.get("gene_type", attrs.get("gene_biotype", ""))
            transcript_id = attrs.get("transcript_id", "")

            if not gene_id:
                continue

            # Gene feature: create Gene object if not exists
            if feature == "gene":
                if gene_type_filter and gene_type != gene_type_filter:
                    continue
                if gene_id not in genes:
                    genes[gene_id] = Gene(
                        gene_id=gene_id,
                        gene_name=gene_name,
                        chrom=chrom,
                        strand=strand,
                        start=start,
                        end=end,
                    )

            # Exon feature: add to transcript
            elif feature == feature_filter:
                if gene_id not in genes:
                    if gene_type_filter and gene_type and gene_type != gene_type_filter:
                        continue
                    genes[gene_id] = Gene(
                        gene_id=gene_id,
                        gene_name=gene_name,
                        chrom=chrom,
                        strand=strand,
                        start=start,
                        end=end,
                    )

                gene = genes[gene_id]

                # Update gene boundaries
                if start < gene.start:
                    gene.start = start
                if end > gene.end:
                    gene.end = end

                # Add exon to transcript
                if transcript_id:
                    if transcript_id not in gene.transcripts:
                        gene.transcripts[transcript_id] = []
                    gene.transcripts[transcript_id].append((start, end))

    # Sort exons within each transcript by start position
    for gene in genes.values():
        for tx_id in gene.transcripts:
            gene.transcripts[tx_id].sort(key=lambda x: x[0])

    return genes


def extract_known_junctions(genes: Dict[str, Gene]) -> Set[Junction]:
    """Extract all annotated splice junctions from gene models.

    For each transcript, consecutive exons define a junction:
    junction = (chrom, exon_i_end, exon_i+1_start, strand).

    Args:
        genes: Dict mapping gene_id -> Gene object (from parse_gtf).

    Returns:
        Set of Junction objects representing all annotated splice junctions.
    """
    junctions: Set[Junction] = set()

    for gene in genes.values():
        for tx_id, exons in gene.transcripts.items():
            if len(exons) < 2:
                continue
            for i in range(len(exons) - 1):
                donor = exons[i][1]
                acceptor = exons[i + 1][0]
                if acceptor > donor:
                    junctions.add(Junction(gene.chrom, donor, acceptor, gene.strand))

    return junctions


def _parse_attributes(attrs_str: str) -> Dict[str, str]:
    """Parse GTF attribute string into a dict.

    Args:
        attrs_str: The 9th column of a GTF line.

    Returns:
        Dict mapping attribute names to values (quotes stripped).
    """
    attrs = {}
    for item in attrs_str.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(None, 1)
        if len(parts) == 2:
            key = parts[0]
            value = parts[1].strip().strip('"')
            attrs[key] = value
    return attrs
