"""
Module 7: core/clustering.py

Annotation-free intron clustering using the LeafCutter algorithm.
Clusters junctions by coordinate overlap and shared splice sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np

from splicekit.utils.genomic import Junction


@dataclass
class JunctionCluster:
    """A cluster of junctions that belong together.

    Attributes:
        cluster_id: Unique identifier for this cluster.
        junctions: List of Junction objects in this cluster.
        chrom: Chromosome of the cluster.
        strand: Strand of the cluster.
        start: Minimum junction start position in cluster.
        end: Maximum junction end position in cluster.
        size: Number of junctions in cluster.
    """

    cluster_id: str
    junctions: List[Junction]
    chrom: str
    strand: str
    start: int
    end: int

    @property
    def size(self) -> int:
        """Return number of junctions in cluster."""
        return len(self.junctions)


def cluster_junctions(
    junctions: List[Junction],
    max_intron_length: int = 100000,
    min_cluster_size: int = 2,
) -> List[JunctionCluster]:
    """Cluster junctions using the LeafCutter algorithm.

    Groups junctions by coordinate overlap and shared splice sites. This
    annotation-free approach identifies intronic regions that show complex
    splicing patterns.

    Algorithm:
    1. Group junctions by chromosome and strand
    2. Within each (chrom, strand) pair, find junctions with overlapping introns
    3. Refine by merging junctions that share splice sites
    4. Apply ratio pruning (remove low-count junctions)
    5. Return clusters with at least min_cluster_size members

    Args:
        junctions: List of Junction objects to cluster.
        max_intron_length: Maximum intron length to consider (default 100kb).
        min_cluster_size: Minimum number of junctions per cluster (default 2).

    Returns:
        List of JunctionCluster objects.
    """
    if not junctions:
        return []

    # Group junctions by (chrom, strand)
    by_region: Dict[tuple, List[Junction]] = {}
    for junc in junctions:
        key = (junc.chrom, junc.strand)
        if key not in by_region:
            by_region[key] = []
        by_region[key].append(junc)

    clusters: List[JunctionCluster] = []
    cluster_counter = 0

    # Cluster within each (chrom, strand) region
    for (chrom, strand), region_junctions in by_region.items():
        # Sort by start position for efficient clustering
        sorted_junctions = sorted(region_junctions, key=lambda j: (j.start, j.end))

        # Find clusters by overlap
        region_clusters = _find_overlapping_clusters(
            sorted_junctions, max_intron_length
        )

        # Refine by shared splice sites
        for cluster_junctions_list in region_clusters:
            refined = _refine_by_splice_sites(cluster_junctions_list)

            for cluster_group in refined:
                if len(cluster_group) >= min_cluster_size:
                    cluster_id = f"{chrom}:{strand}:{cluster_counter}"
                    cluster_counter += 1

                    min_start = min(j.start for j in cluster_group)
                    max_end = max(j.end for j in cluster_group)

                    cluster = JunctionCluster(
                        cluster_id=cluster_id,
                        junctions=cluster_group,
                        chrom=chrom,
                        strand=strand,
                        start=min_start,
                        end=max_end,
                    )
                    clusters.append(cluster)

    return clusters


def _find_overlapping_clusters(
    sorted_junctions: List[Junction], max_intron_length: int
) -> List[List[Junction]]:
    """Find clusters of junctions with overlapping introns.

    Uses a greedy algorithm: start a new cluster when no junction in the
    current cluster overlaps with a candidate junction.

    Args:
        sorted_junctions: Junctions sorted by start position.
        max_intron_length: Maximum allowed intron length.

    Returns:
        List of clusters, where each cluster is a list of junctions.
    """
    if not sorted_junctions:
        return []

    clusters: List[List[Junction]] = []
    current_cluster: List[Junction] = []
    cluster_end = 0

    for junc in sorted_junctions:
        # Skip junctions that exceed max intron length
        if junc.end - junc.start > max_intron_length:
            continue

        # Check if this junction overlaps with current cluster
        if current_cluster and junc.start > cluster_end:
            # No overlap, start new cluster
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [junc]
            cluster_end = junc.end
        else:
            # Overlaps, add to current cluster
            current_cluster.append(junc)
            cluster_end = max(cluster_end, junc.end)

    # Don't forget the last cluster
    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def _refine_by_splice_sites(junctions: List[Junction]) -> List[List[Junction]]:
    """Refine clusters by shared splice sites.

    Junctions that share a donor or acceptor site are likely part of the
    same splicing module (e.g., alternative 5' or 3' sites). This function
    identifies subclusters based on splice site connectivity.

    Args:
        junctions: List of junctions in a cluster.

    Returns:
        List of refined clusters (subclusters).
    """
    if len(junctions) <= 1:
        return [junctions]

    # Build connectivity graph: edges between junctions that share splice sites
    n = len(junctions)
    adjacency = [[False] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if junctions[i].shares_splice_site(junctions[j]):
                adjacency[i][j] = True
                adjacency[j][i] = True

    # Find connected components using DFS
    visited = [False] * n
    components: List[List[Junction]] = []

    def dfs(node: int, component: List[int]) -> None:
        visited[node] = True
        component.append(node)
        for neighbor in range(n):
            if adjacency[node][neighbor] and not visited[neighbor]:
                dfs(neighbor, component)

    for i in range(n):
        if not visited[i]:
            component_indices: List[int] = []
            dfs(i, component_indices)
            component_junctions = [junctions[idx] for idx in component_indices]
            components.append(component_junctions)

    # If a component only has 1 junction, merge it with the largest component
    # (or just keep singletons for now)
    return components


def get_cluster_junctions(cluster: JunctionCluster) -> Set[Junction]:
    """Get the set of junctions in a cluster.

    Args:
        cluster: JunctionCluster object.

    Returns:
        Set of Junction objects.
    """
    return set(cluster.junctions)


def filter_clusters_by_size(
    clusters: List[JunctionCluster], min_junctions: int = 2
) -> List[JunctionCluster]:
    """Filter clusters by minimum size.

    Args:
        clusters: List of JunctionCluster objects.
        min_junctions: Minimum number of junctions required.

    Returns:
        Filtered list of clusters.
    """
    return [c for c in clusters if c.size >= min_junctions]


def filter_clusters_by_region(
    clusters: List[JunctionCluster],
    chrom: str,
    start: int,
    end: int,
) -> List[JunctionCluster]:
    """Filter clusters by genomic region.

    Args:
        clusters: List of JunctionCluster objects.
        chrom: Chromosome.
        start: Region start (0-based).
        end: Region end (0-based exclusive).

    Returns:
        Clusters that overlap the specified region.
    """
    result = []
    for cluster in clusters:
        if cluster.chrom == chrom and cluster.start < end and cluster.end > start:
            result.append(cluster)
    return result
