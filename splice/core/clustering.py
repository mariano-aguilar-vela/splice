"""
Module 7: core/clustering.py

Annotation-free intron clustering using bipartite-graph union-find.
Clusters junctions by shared splice sites in O(N) time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np

from splice.utils.genomic import Junction


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


class _UnionFind:
    """Disjoint set with path compression and union by rank."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def cluster_junctions(
    junctions: List[Junction],
    max_intron_length: int = 100000,
    min_cluster_size: int = 2,
) -> List[JunctionCluster]:
    """Cluster junctions using bipartite-graph union-find.

    Each junction is an edge connecting its donor site to its acceptor site
    in a bipartite graph. Two junctions sharing a splice site (donor or acceptor)
    are in the same connected component. Finding connected components with
    union-find is O(N * alpha(N)), effectively O(N) linear time.

    Algorithm:
    1. Filter junctions exceeding max_intron_length.
    2. For each junction, create a site key for its donor and acceptor.
       Union the junction's donor site key with its acceptor site key.
       Since all junctions sharing a donor will union through that donor's
       site key, and all junctions sharing an acceptor will union through
       that acceptor's site key, connected components emerge automatically.
    3. Group junctions by their root in the union-find.
    4. Discard clusters with fewer than min_cluster_size junctions.

    Args:
        junctions: List of Junction objects.
        max_intron_length: Maximum intron length (default 100kb).
        min_cluster_size: Minimum junctions per cluster (default 2).

    Returns:
        List of JunctionCluster objects.
    """
    if not junctions:
        return []

    uf = _UnionFind()

    junction_keys = {}
    valid_junctions = []

    for junc in junctions:
        if junc.end - junc.start > max_intron_length:
            continue
        valid_junctions.append(junc)

        junc_key = f"J:{junc.chrom}:{junc.strand}:{junc.start}:{junc.end}"
        donor_key = f"S:{junc.chrom}:{junc.strand}:{junc.start}"
        acceptor_key = f"S:{junc.chrom}:{junc.strand}:{junc.end}"

        junction_keys[junc] = junc_key

        uf.union(junc_key, donor_key)
        uf.union(junc_key, acceptor_key)

    # Group junctions by their component root
    components = defaultdict(list)
    for junc in valid_junctions:
        root = uf.find(junction_keys[junc])
        components[root].append(junc)

    # Build JunctionCluster objects
    clusters = []
    cluster_counter = 0
    for root, cluster_juncs in components.items():
        if len(cluster_juncs) < min_cluster_size:
            continue
        chrom = cluster_juncs[0].chrom
        strand = cluster_juncs[0].strand
        cluster_id = f"{chrom}:{strand}:{cluster_counter}"
        cluster_counter += 1
        clusters.append(JunctionCluster(
            cluster_id=cluster_id,
            junctions=cluster_juncs,
            chrom=chrom,
            strand=strand,
            start=min(j.start for j in cluster_juncs),
            end=max(j.end for j in cluster_juncs),
        ))

    return clusters


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
