"""
Tests for core/cooccurrence.py

Covers: CooccurrenceGraph, build_cooccurrence_graph, identify_mutually_exclusive_paths,
identify_coordinated_junctions, and compute_cooccurrence_similarity.
"""

import numpy as np
import pytest

from splice.core.cooccurrence import (
    CooccurrenceGraph,
    build_cooccurrence_graph,
    compute_cooccurrence_similarity,
    identify_coordinated_junctions,
    identify_mutually_exclusive_paths,
)
from splice.core.junction_extractor import CooccurrenceEvidence
from splice.utils.genomic import Junction, JunctionPair


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_junctions():
    """Create simple test junctions."""
    return [
        Junction("chr1", 100, 200, "+"),  # 0
        Junction("chr1", 300, 400, "+"),  # 1
        Junction("chr1", 500, 600, "+"),  # 2
    ]


@pytest.fixture
def simple_cooccurrence_evidence():
    """Create simple co-occurrence evidence."""
    j0 = Junction("chr1", 100, 200, "+")
    j1 = Junction("chr1", 300, 400, "+")
    j2 = Junction("chr1", 500, 600, "+")

    pair_01 = JunctionPair(j0, j1)
    pair_12 = JunctionPair(j1, j2)

    n_samples = 2

    evidence = {
        pair_01: CooccurrenceEvidence(
            pair=pair_01, sample_counts=np.array([5, 3], dtype=int)
        ),
        pair_12: CooccurrenceEvidence(
            pair=pair_12, sample_counts=np.array([4, 2], dtype=int)
        ),
    }

    return evidence


# ---------------------------------------------------------------------------
# CooccurrenceGraph tests
# ---------------------------------------------------------------------------


class TestCooccurrenceGraph:
    def test_construction(self, simple_junctions):
        """Test basic CooccurrenceGraph construction."""
        n_junctions = len(simple_junctions)
        n_samples = 2

        adjacency = np.zeros((n_junctions, n_junctions), dtype=int)
        sample_adjacency = np.zeros((n_junctions, n_junctions, n_samples), dtype=int)

        graph = CooccurrenceGraph(
            junctions=simple_junctions,
            adjacency=adjacency,
            sample_adjacency=sample_adjacency,
        )

        assert len(graph.junctions) == 3
        assert graph.adjacency.shape == (3, 3)
        assert graph.sample_adjacency.shape == (3, 3, 2)

    def test_adjacency_shape(self, simple_junctions):
        """Test that adjacency matrices have correct shapes."""
        n_junctions = len(simple_junctions)
        n_samples = 3

        graph = CooccurrenceGraph(
            junctions=simple_junctions,
            adjacency=np.zeros((n_junctions, n_junctions), dtype=int),
            sample_adjacency=np.zeros((n_junctions, n_junctions, n_samples), dtype=int),
        )

        assert graph.adjacency.shape == (3, 3)
        assert graph.sample_adjacency.shape == (3, 3, 3)


# ---------------------------------------------------------------------------
# build_cooccurrence_graph tests
# ---------------------------------------------------------------------------


class TestBuildCooccurrenceGraph:
    def test_simple_graph(self, simple_junctions, simple_cooccurrence_evidence):
        """Test building a simple co-occurrence graph."""
        graph = build_cooccurrence_graph(
            simple_cooccurrence_evidence, simple_junctions, n_samples=2
        )

        assert len(graph.junctions) == 3
        assert graph.adjacency.shape == (3, 3)
        assert graph.sample_adjacency.shape == (3, 3, 2)

    def test_adjacency_symmetry(self, simple_junctions, simple_cooccurrence_evidence):
        """Test that adjacency matrix is symmetric."""
        graph = build_cooccurrence_graph(
            simple_cooccurrence_evidence, simple_junctions, n_samples=2
        )

        # Check symmetry: adjacency[i, j] == adjacency[j, i]
        assert graph.adjacency[0, 1] == graph.adjacency[1, 0]
        assert graph.adjacency[1, 2] == graph.adjacency[2, 1]

    def test_co_occurrence_counts(self, simple_junctions, simple_cooccurrence_evidence):
        """Test that co-occurrence counts are correctly populated."""
        graph = build_cooccurrence_graph(
            simple_cooccurrence_evidence, simple_junctions, n_samples=2
        )

        # j0-j1 co-occur [5, 3] -> total 8
        assert graph.adjacency[0, 1] == 8
        assert graph.adjacency[1, 0] == 8

        # j1-j2 co-occur [4, 2] -> total 6
        assert graph.adjacency[1, 2] == 6
        assert graph.adjacency[2, 1] == 6

        # j0-j2 don't co-occur -> 0
        assert graph.adjacency[0, 2] == 0
        assert graph.adjacency[2, 0] == 0

    def test_sample_adjacency(self, simple_junctions, simple_cooccurrence_evidence):
        """Test that per-sample co-occurrence counts are correct."""
        graph = build_cooccurrence_graph(
            simple_cooccurrence_evidence, simple_junctions, n_samples=2
        )

        # j0-j1: sample 0 has 5, sample 1 has 3
        assert graph.sample_adjacency[0, 1, 0] == 5
        assert graph.sample_adjacency[0, 1, 1] == 3

    def test_filtered_junctions(self, simple_cooccurrence_evidence):
        """Test with subset of junctions (filtering)."""
        j0 = Junction("chr1", 100, 200, "+")
        j1 = Junction("chr1", 300, 400, "+")
        # Only include j0 and j1, not j2

        module_junctions = [j0, j1]

        graph = build_cooccurrence_graph(
            simple_cooccurrence_evidence, module_junctions, n_samples=2
        )

        assert len(graph.junctions) == 2
        assert graph.adjacency.shape == (2, 2)


# ---------------------------------------------------------------------------
# identify_mutually_exclusive_paths tests
# ---------------------------------------------------------------------------


class TestIdentifyMutuallyExclusivePaths:
    def test_no_mutual_exclusion(self, simple_junctions, simple_cooccurrence_evidence):
        """Test when junctions frequently co-occur (no mutual exclusion)."""
        graph = build_cooccurrence_graph(
            simple_cooccurrence_evidence, simple_junctions, n_samples=2
        )

        paths = identify_mutually_exclusive_paths(graph)

        # j0-j1 and j1-j2 co-occur, so only j0-j2 pair is mutually exclusive
        assert len(paths) == 1
        # paths should contain a pair like ({0}, {2}) or ({2}, {0})
        combined = [p[0] | p[1] for p in paths]
        assert {0, 2} in combined

    def test_mutual_exclusion_detection(self):
        """Test detection of truly mutually exclusive junctions."""
        j0 = Junction("chr1", 100, 200, "+")
        j1 = Junction("chr1", 300, 400, "+")
        j2 = Junction("chr1", 500, 600, "+")

        # j0-j1 co-occur, j1-j2 co-occur, but j0-j2 don't
        pair_01 = JunctionPair(j0, j1)
        pair_12 = JunctionPair(j1, j2)

        evidence = {
            pair_01: CooccurrenceEvidence(pair=pair_01, sample_counts=np.array([5])),
            pair_12: CooccurrenceEvidence(pair=pair_12, sample_counts=np.array([3])),
        }

        graph = build_cooccurrence_graph(evidence, [j0, j1, j2], n_samples=1)
        paths = identify_mutually_exclusive_paths(graph)

        # Should identify j0 and j2 as mutually exclusive
        assert len(paths) == 1

    def test_empty_graph(self):
        """Test with no co-occurrences at all."""
        j0 = Junction("chr1", 100, 200, "+")
        j1 = Junction("chr1", 300, 400, "+")
        j2 = Junction("chr1", 500, 600, "+")

        # No co-occurrence evidence
        graph = build_cooccurrence_graph({}, [j0, j1, j2], n_samples=1)
        paths = identify_mutually_exclusive_paths(graph)

        # All pairs are mutually exclusive
        assert len(paths) == 3  # C(3,2) = 3 pairs


# ---------------------------------------------------------------------------
# identify_coordinated_junctions tests
# ---------------------------------------------------------------------------


class TestIdentifyCoordinatedJunctions:
    def test_simple_coordinated_junctions(self):
        """Test identification of coordinated junctions."""
        j0 = Junction("chr1", 100, 200, "+")
        j1 = Junction("chr1", 300, 400, "+")
        j2 = Junction("chr1", 500, 600, "+")

        # j0-j1 and j1-j2 strongly co-occur
        pair_01 = JunctionPair(j0, j1)
        pair_12 = JunctionPair(j1, j2)

        evidence = {
            pair_01: CooccurrenceEvidence(pair=pair_01, sample_counts=np.array([10])),
            pair_12: CooccurrenceEvidence(pair=pair_12, sample_counts=np.array([8])),
        }

        graph = build_cooccurrence_graph(evidence, [j0, j1, j2], n_samples=1)
        components = identify_coordinated_junctions(graph, min_cooccurrence=3)

        # j0, j1, j2 should form a single coordinated group
        assert len(components) == 1
        assert components[0] == {0, 1, 2}

    def test_multiple_coordinated_groups(self):
        """Test identification of multiple independent groups."""
        j0 = Junction("chr1", 100, 200, "+")
        j1 = Junction("chr1", 300, 400, "+")
        j2 = Junction("chr1", 500, 600, "+")
        j3 = Junction("chr1", 700, 800, "+")

        # j0-j1 strongly co-occur; j2-j3 strongly co-occur
        # but j0,j1 don't co-occur with j2,j3
        pair_01 = JunctionPair(j0, j1)
        pair_23 = JunctionPair(j2, j3)

        evidence = {
            pair_01: CooccurrenceEvidence(pair=pair_01, sample_counts=np.array([10])),
            pair_23: CooccurrenceEvidence(pair=pair_23, sample_counts=np.array([10])),
        }

        graph = build_cooccurrence_graph(
            evidence, [j0, j1, j2, j3], n_samples=1
        )
        components = identify_coordinated_junctions(graph, min_cooccurrence=3)

        # Should find two separate groups
        assert len(components) == 2
        assert {0, 1} in components
        assert {2, 3} in components

    def test_min_cooccurrence_threshold(self):
        """Test that min_cooccurrence threshold is respected."""
        j0 = Junction("chr1", 100, 200, "+")
        j1 = Junction("chr1", 300, 400, "+")
        j2 = Junction("chr1", 500, 600, "+")

        # j0-j1 co-occur [5], j1-j2 co-occur [2]
        pair_01 = JunctionPair(j0, j1)
        pair_12 = JunctionPair(j1, j2)

        evidence = {
            pair_01: CooccurrenceEvidence(pair=pair_01, sample_counts=np.array([5])),
            pair_12: CooccurrenceEvidence(pair=pair_12, sample_counts=np.array([2])),
        }

        graph = build_cooccurrence_graph(evidence, [j0, j1, j2], n_samples=1)

        # With threshold=3: j0-j1 connected, j1-j2 not connected
        components_t3 = identify_coordinated_junctions(graph, min_cooccurrence=3)
        assert len(components_t3) == 2  # {0,1} and {2}

        # With threshold=1: all connected
        components_t1 = identify_coordinated_junctions(graph, min_cooccurrence=1)
        assert len(components_t1) == 1  # {0,1,2}

    def test_single_junction_components(self):
        """Test that isolated junctions form singleton components."""
        j0 = Junction("chr1", 100, 200, "+")
        j1 = Junction("chr1", 300, 400, "+")

        # No co-occurrence
        graph = build_cooccurrence_graph({}, [j0, j1], n_samples=1)
        components = identify_coordinated_junctions(graph, min_cooccurrence=1)

        # Two isolated junctions
        assert len(components) == 2
        assert {0} in components
        assert {1} in components


# ---------------------------------------------------------------------------
# compute_cooccurrence_similarity tests
# ---------------------------------------------------------------------------


class TestComputeCooccurrenceSimilarity:
    def test_simple_similarity(self):
        """Test basic similarity computation."""
        j0 = Junction("chr1", 100, 200, "+")
        j1 = Junction("chr1", 300, 400, "+")
        j2 = Junction("chr1", 500, 600, "+")

        # j0 co-occurs with j1 and j2
        # j1 co-occurs with j0 and j2
        # j2 co-occurs with j0 and j1
        pair_01 = JunctionPair(j0, j1)
        pair_02 = JunctionPair(j0, j2)
        pair_12 = JunctionPair(j1, j2)

        evidence = {
            pair_01: CooccurrenceEvidence(pair=pair_01, sample_counts=np.array([5])),
            pair_02: CooccurrenceEvidence(pair=pair_02, sample_counts=np.array([5])),
            pair_12: CooccurrenceEvidence(pair=pair_12, sample_counts=np.array([5])),
        }

        graph = build_cooccurrence_graph(evidence, [j0, j1, j2], n_samples=1)
        similarity = compute_cooccurrence_similarity(graph, threshold=0.33)

        # j0: neighbors {1, 2}, j1: neighbors {0, 2}
        # Jaccard(j0,j1) = 1/{0,1,2} = 1/3 = 0.33
        # So with threshold 0.33 (inclusive), should find similarities
        assert len(similarity[0]) > 0 or len(similarity[1]) > 0 or len(similarity[2]) > 0

    def test_no_similarity(self):
        """Test when no junctions are similar."""
        j0 = Junction("chr1", 100, 200, "+")
        j1 = Junction("chr1", 300, 400, "+")
        j2 = Junction("chr1", 500, 600, "+")

        # Each junction only co-occurs with one other
        pair_01 = JunctionPair(j0, j1)
        pair_12 = JunctionPair(j1, j2)

        evidence = {
            pair_01: CooccurrenceEvidence(pair=pair_01, sample_counts=np.array([5])),
            pair_12: CooccurrenceEvidence(pair=pair_12, sample_counts=np.array([5])),
        }

        graph = build_cooccurrence_graph(evidence, [j0, j1, j2], n_samples=1)
        similarity = compute_cooccurrence_similarity(graph, threshold=0.8)

        # Low similarity threshold (0.8): should find few or no similar pairs
        total_similar = sum(len(s) for s in similarity.values())
        assert total_similar < 6  # Not all pairwise similar

    def test_threshold_effect(self):
        """Test that threshold affects similarity grouping."""
        j0 = Junction("chr1", 100, 200, "+")
        j1 = Junction("chr1", 300, 400, "+")
        j2 = Junction("chr1", 500, 600, "+")

        pair_01 = JunctionPair(j0, j1)
        pair_02 = JunctionPair(j0, j2)
        pair_12 = JunctionPair(j1, j2)

        evidence = {
            pair_01: CooccurrenceEvidence(pair=pair_01, sample_counts=np.array([5])),
            pair_02: CooccurrenceEvidence(pair=pair_02, sample_counts=np.array([5])),
            pair_12: CooccurrenceEvidence(pair=pair_12, sample_counts=np.array([5])),
        }

        graph = build_cooccurrence_graph(evidence, [j0, j1, j2], n_samples=1)

        # Low threshold: more similar groups
        similarity_low = compute_cooccurrence_similarity(graph, threshold=0.3)
        count_low = sum(len(s) for s in similarity_low.values())

        # High threshold: fewer similar groups
        similarity_high = compute_cooccurrence_similarity(graph, threshold=0.9)
        count_high = sum(len(s) for s in similarity_high.values())

        assert count_low >= count_high


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_workflow(self):
        """Test complete workflow from evidence to pathways."""
        # Create a realistic splicing scenario:
        # - Exon 1 (j0: 100-200)
        # - Exon 2a (j1: 300-400) or Exon 2b (j2: 300-450) [alternative]
        # - Exon 3 (j3: 500-600)

        j0 = Junction("chr1", 100, 200, "+")  # 1->2
        j1 = Junction("chr1", 300, 400, "+")  # 2a->3
        j2 = Junction("chr1", 300, 450, "+")  # 2b->3

        pair_01 = JunctionPair(j0, j1)  # Exon 2a
        pair_02 = JunctionPair(j0, j2)  # Exon 2b
        # j1 and j2 don't co-occur (mutually exclusive)

        evidence = {
            pair_01: CooccurrenceEvidence(pair=pair_01, sample_counts=np.array([20])),
            pair_02: CooccurrenceEvidence(pair=pair_02, sample_counts=np.array([15])),
        }

        graph = build_cooccurrence_graph(evidence, [j0, j1, j2], n_samples=1)

        # Analyze pathways
        coordinated = identify_coordinated_junctions(graph, min_cooccurrence=5)
        exclusive = identify_mutually_exclusive_paths(graph)

        # j0 is coordinated with both j1 and j2, forming a single connected component
        # (star graph with j0 at center)
        assert len(coordinated) == 1  # All connected through j0
        assert coordinated[0] == {0, 1, 2}
        # j1 and j2 are mutually exclusive (no direct edge)
        assert len(exclusive) == 1  # j1-j2 are exclusive
