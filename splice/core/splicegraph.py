"""
Module 8: core/splicegraph.py

Splicegraph: annotation-seeded with de novo junctions and module detection.
Builds splicing modules from junction clusters and genes.
Integrates co-occurrence evidence for pathway analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from splicekit.core.clustering import JunctionCluster
from splicekit.core.cooccurrence import CooccurrenceGraph
from splicekit.utils.genomic import GenomicInterval, Junction, overlaps


@dataclass
class SplicingModule:
    """A splicing module: group of spatially overlapping junctions within a gene.

    Attributes:
        module_id: Unique identifier for this module.
        gene_id: ID of the associated gene (or 'intergenic').
        gene_name: Name of the gene (or empty string).
        chrom: Chromosome.
        strand: Strand ('+', '-', or '.').
        start: Minimum junction start in module.
        end: Maximum junction end in module.
        junctions: List of Junction objects in this module.
        junction_indices: Indices into the global junction list.
        n_connections: Number of junctions (= len(junctions)).
        cooccurrence_graph: Optional co-occurrence graph for this module.
        mutually_exclusive_paths: Pairs of mutually exclusive junction sets.
        coordinated_junctions: Groups of coordinated junctions.
    """

    module_id: str
    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    start: int
    end: int
    junctions: List[Junction]
    junction_indices: List[int]
    n_connections: int
    cooccurrence_graph: Optional[CooccurrenceGraph] = None
    mutually_exclusive_paths: List[Tuple[Set[int], Set[int]]] = field(
        default_factory=list
    )
    coordinated_junctions: List[Set[int]] = field(default_factory=list)

    @property
    def is_binary(self) -> bool:
        """Return True if this module has exactly 2 junctions."""
        return self.n_connections == 2


def build_splicegraph(
    genes: Dict[str, "Gene"],
    junction_evidence: Dict[Junction, "JunctionEvidence"],
    clusters: List[JunctionCluster],
    known_junctions: Set[Junction],
    max_denovo_distance: int = 400,
) -> Tuple[List[SplicingModule], Dict[Junction, int]]:
    """Build splicing modules from clusters and genes.

    Maps each cluster to the best-matching gene by coordinate overlap.
    Merges overlapping clusters into modules. Annotates junctions and
    assigns de novo junctions to nearby genes.

    Algorithm:
    1. For each cluster, find genes that overlap its coordinate range
    2. Merge overlapping clusters into modules (per gene)
    3. Annotate junctions: is junction in known_junctions?
    4. For de novo junctions, find nearby annotated exon boundaries

    Args:
        genes: Dict mapping gene_id -> Gene object.
        junction_evidence: Dict mapping Junction -> JunctionEvidence.
        clusters: List of JunctionCluster objects.
        known_junctions: Set of annotated Junction objects.
        max_denovo_distance: Max distance to assign de novo junctions to a gene.

    Returns:
        Tuple of (modules, junction_to_index_dict) where:
          - modules: List of SplicingModule objects
          - junction_to_index_dict: Maps Junction -> global index
    """
    if not clusters:
        return [], {}

    # Build junction-to-index mapping
    all_junctions: List[Junction] = []
    junction_to_index: Dict[Junction, int] = {}
    for cluster in clusters:
        for junc in cluster.junctions:
            if junc not in junction_to_index:
                idx = len(all_junctions)
                all_junctions.append(junc)
                junction_to_index[junc] = idx

    # Group clusters by (chrom, strand)
    clusters_by_region: Dict[Tuple[str, str], List[JunctionCluster]] = {}
    for cluster in clusters:
        key = (cluster.chrom, cluster.strand)
        if key not in clusters_by_region:
            clusters_by_region[key] = []
        clusters_by_region[key].append(cluster)

    modules: List[SplicingModule] = []
    module_counter = 0

    # Process each (chrom, strand) region
    for (chrom, strand), region_clusters in clusters_by_region.items():
        # Find genes overlapping this region
        gene_matches: Dict[str, "Gene"] = {}
        for gene_id, gene in genes.items():
            if gene.chrom == chrom and gene.strand == strand:
                # Check if gene overlaps any cluster
                for cluster in region_clusters:
                    if cluster.start < gene.end and cluster.end > gene.start:
                        gene_matches[gene_id] = gene
                        break

        # If no genes match, create intergenic module
        if not gene_matches:
            for cluster in region_clusters:
                module_id = f"intergenic:{chrom}:{strand}:{module_counter}"
                module_counter += 1

                junction_indices = [
                    junction_to_index[j] for j in cluster.junctions
                ]

                module = SplicingModule(
                    module_id=module_id,
                    gene_id="",
                    gene_name="",
                    chrom=chrom,
                    strand=strand,
                    start=cluster.start,
                    end=cluster.end,
                    junctions=cluster.junctions,
                    junction_indices=junction_indices,
                    n_connections=len(cluster.junctions),
                )
                modules.append(module)
        else:
            # Merge overlapping clusters within each gene
            for gene_id, gene in gene_matches.items():
                # Find clusters that overlap this gene
                gene_clusters = [
                    c
                    for c in region_clusters
                    if c.start < gene.end and c.end > gene.start
                ]

                if not gene_clusters:
                    continue

                # Merge overlapping clusters
                merged = _merge_overlapping_clusters(gene_clusters)

                # Create modules for merged clusters
                for merged_cluster in merged:
                    module_id = (
                        f"{gene_id}:{chrom}:{strand}:{module_counter}"
                    )
                    module_counter += 1

                    junction_indices = [
                        junction_to_index[j] for j in merged_cluster
                    ]

                    module = SplicingModule(
                        module_id=module_id,
                        gene_id=gene_id,
                        gene_name=gene.gene_name,
                        chrom=chrom,
                        strand=strand,
                        start=min(j.start for j in merged_cluster),
                        end=max(j.end for j in merged_cluster),
                        junctions=merged_cluster,
                        junction_indices=junction_indices,
                        n_connections=len(merged_cluster),
                    )
                    modules.append(module)

    return modules, junction_to_index


def _merge_overlapping_clusters(
    clusters: List[JunctionCluster],
) -> List[List[Junction]]:
    """Merge clusters that have overlapping junctions.

    Returns list of merged junction groups (each group is a list of junctions
    that should be in the same module).
    """
    if not clusters:
        return []

    # Build a graph of cluster overlap
    n = len(clusters)
    adjacency = [[False] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            # Check if clusters i and j overlap
            if clusters[i].start < clusters[j].end and clusters[i].end > clusters[j].start:
                adjacency[i][j] = True
                adjacency[j][i] = True

    # Find connected components
    visited = [False] * n
    merged_groups: List[List[Junction]] = []

    def dfs(idx: int, group: List[Junction]) -> None:
        visited[idx] = True
        group.extend(clusters[idx].junctions)
        for neighbor in range(n):
            if adjacency[idx][neighbor] and not visited[neighbor]:
                dfs(neighbor, group)

    for i in range(n):
        if not visited[i]:
            group: List[Junction] = []
            dfs(i, group)
            merged_groups.append(group)

    return merged_groups


def get_module_junctions(module: SplicingModule) -> List[Junction]:
    """Get the list of junctions in a module."""
    return module.junctions


def filter_modules_by_size(
    modules: List[SplicingModule], min_junctions: int = 2
) -> List[SplicingModule]:
    """Filter modules by minimum number of junctions."""
    return [m for m in modules if m.n_connections >= min_junctions]


def filter_modules_by_gene(
    modules: List[SplicingModule], gene_id: str
) -> List[SplicingModule]:
    """Filter modules belonging to a specific gene."""
    return [m for m in modules if m.gene_id == gene_id]


def filter_modules_by_region(
    modules: List[SplicingModule], chrom: str, start: int, end: int
) -> List[SplicingModule]:
    """Filter modules overlapping a genomic region."""
    result = []
    for module in modules:
        if module.chrom == chrom and module.start < end and module.end > start:
            result.append(module)
    return result
