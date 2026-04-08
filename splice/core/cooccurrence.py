"""
Module 6: core/cooccurrence.py

Junction co-occurrence analysis: build graphs, identify mutually exclusive
and coordinated splice paths from co-occurrence patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

from splicekit.core.junction_extractor import CooccurrenceEvidence
from splicekit.utils.genomic import Junction, JunctionPair


@dataclass
class CooccurrenceGraph:
    """Graph where nodes are junctions and edges are co-occurrence counts.

    Attributes:
        junctions: List of Junction objects (nodes in graph).
        adjacency: (n_junctions, n_junctions) array of total co-occurrence counts
                   across all samples (symmetric matrix).
        sample_adjacency: (n_junctions, n_junctions, n_samples) array of
                         co-occurrence counts per sample.
    """

    junctions: List[Junction]
    adjacency: np.ndarray
    sample_adjacency: np.ndarray


def build_cooccurrence_graph(
    cooccurrence_evidence: Dict[JunctionPair, CooccurrenceEvidence],
    module_junctions: List[Junction],
    n_samples: int,
) -> CooccurrenceGraph:
    """Build co-occurrence graph for junctions within a module.

    Constructs adjacency matrices from CooccurrenceEvidence, capturing which
    junctions appear together in the same read. The adjacency matrix is
    symmetric: if (j1, j2) co-occurs, so does (j2, j1).

    Args:
        cooccurrence_evidence: Dict mapping JunctionPair -> CooccurrenceEvidence.
        module_junctions: List of Junction objects to include in graph.
        n_samples: Number of samples.

    Returns:
        CooccurrenceGraph with populated adjacency matrices.
    """
    n_junctions = len(module_junctions)

    # Create junction-to-index mapping for fast lookup
    junction_to_idx: Dict[Junction, int] = {j: i for i, j in enumerate(module_junctions)}

    # Initialize adjacency matrices
    adjacency = np.zeros((n_junctions, n_junctions), dtype=int)
    sample_adjacency = np.zeros((n_junctions, n_junctions, n_samples), dtype=int)

    # Populate from cooccurrence evidence
    for pair, evidence in cooccurrence_evidence.items():
        j1, j2 = pair.junction1, pair.junction2

        # Only include if both junctions are in the module
        if j1 not in junction_to_idx or j2 not in junction_to_idx:
            continue

        idx1 = junction_to_idx[j1]
        idx2 = junction_to_idx[j2]

        # Add counts (symmetric matrix)
        for sample_idx in range(n_samples):
            count = evidence.sample_counts[sample_idx]
            adjacency[idx1, idx2] += count
            adjacency[idx2, idx1] += count
            sample_adjacency[idx1, idx2, sample_idx] = count
            sample_adjacency[idx2, idx1, sample_idx] = count

    return CooccurrenceGraph(
        junctions=module_junctions,
        adjacency=adjacency,
        sample_adjacency=sample_adjacency,
    )


def identify_mutually_exclusive_paths(
    graph: CooccurrenceGraph,
) -> List[Tuple[Set[int], Set[int]]]:
    """Identify pairs of junction sets that never co-occur in the same read.

    Finds junctions with zero co-occurrence (never appear in the same read),
    indicating mutually exclusive splice events. Returns pairs of such junction
    sets as tuples of index sets.

    Args:
        graph: CooccurrenceGraph object.

    Returns:
        List of (path_a_indices, path_b_indices) tuples where each tuple
        contains two sets of junction indices that are mutually exclusive
        (zero co-occurrence).
    """
    n_junctions = len(graph.junctions)
    mutually_exclusive_pairs: List[Tuple[Set[int], Set[int]]] = []

    # Find all pairs with zero co-occurrence
    for i in range(n_junctions):
        for j in range(i + 1, n_junctions):
            if graph.adjacency[i, j] == 0:
                # Junctions i and j never co-occur
                mutually_exclusive_pairs.append((set([i]), set([j])))

    return mutually_exclusive_pairs


def identify_coordinated_junctions(
    graph: CooccurrenceGraph, min_cooccurrence: int = 3
) -> List[Set[int]]:
    """Identify groups of junctions that frequently co-occur.

    Uses connected components to find groups of junctions that frequently
    appear together in the same reads (co-occurrence >= min_cooccurrence).
    These represent coordinated splice paths like tandem cassettes.

    Args:
        graph: CooccurrenceGraph object.
        min_cooccurrence: Minimum co-occurrence count to consider junctions
                         as connected (default 3).

    Returns:
        List of sets, where each set contains indices of coordinated junctions.
    """
    n_junctions = len(graph.junctions)

    # Build adjacency list with thresholding: edge exists if cooccurrence >= min
    adjacency_list: Dict[int, Set[int]] = {i: set() for i in range(n_junctions)}

    for i in range(n_junctions):
        for j in range(i + 1, n_junctions):
            if graph.adjacency[i, j] >= min_cooccurrence:
                adjacency_list[i].add(j)
                adjacency_list[j].add(i)

    # Find connected components using BFS
    visited = set()
    components: List[Set[int]] = []

    for start in range(n_junctions):
        if start in visited:
            continue

        # BFS from this starting node
        component = set()
        queue = [start]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue

            visited.add(node)
            component.add(node)

            # Add unvisited neighbors
            for neighbor in adjacency_list[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        components.append(component)

    return components


# Additional analysis functions for junction relationships


def compute_cooccurrence_similarity(
    graph: CooccurrenceGraph, threshold: float = 0.5
) -> Dict[int, Set[int]]:
    """Compute similarity between junctions based on co-occurrence patterns.

    Two junctions are considered similar if they co-occur with a consistent
    set of other junctions. This helps identify functionally equivalent
    junctions (e.g., alternative 5' splice sites with similar usage).

    Args:
        graph: CooccurrenceGraph object.
        threshold: Jaccard similarity threshold (0.0 to 1.0) to consider
                  junctions as similar.

    Returns:
        Dict mapping junction index -> set of similar junction indices.
    """
    n_junctions = len(graph.junctions)
    similarity_groups: Dict[int, Set[int]] = {i: set() for i in range(n_junctions)}

    for i in range(n_junctions):
        for j in range(i + 1, n_junctions):
            # Get neighbors of i and j (junctions they co-occur with)
            neighbors_i = set(np.where(graph.adjacency[i, :] > 0)[0]) - {i}
            neighbors_j = set(np.where(graph.adjacency[j, :] > 0)[0]) - {j}

            # Compute Jaccard similarity
            if len(neighbors_i | neighbors_j) > 0:
                intersection = len(neighbors_i & neighbors_j)
                union = len(neighbors_i | neighbors_j)
                similarity = intersection / union if union > 0 else 0.0

                if similarity >= threshold:
                    similarity_groups[i].add(j)
                    similarity_groups[j].add(i)

    return similarity_groups
