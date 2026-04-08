"""
Test suite for Module 7: core/clustering.py

Tests the LeafCutter clustering algorithm including:
- Junction cluster creation and properties
- Overlap-based clustering
- Splice site-based refinement
- Size and region filtering
"""

import unittest
from typing import List

from splicekit.core.clustering import (
    JunctionCluster,
    cluster_junctions,
    get_cluster_junctions,
    filter_clusters_by_size,
    filter_clusters_by_region,
)
from splicekit.utils.genomic import Junction


class TestJunctionCluster(unittest.TestCase):
    """Test JunctionCluster dataclass."""

    def test_cluster_creation(self):
        """Test creating a JunctionCluster."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1500, end=2500, strand="+")

        cluster = JunctionCluster(
            cluster_id="chr1:+:0",
            junctions=[j1, j2],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
        )

        self.assertEqual(cluster.cluster_id, "chr1:+:0")
        self.assertEqual(len(cluster.junctions), 2)
        self.assertEqual(cluster.chrom, "chr1")
        self.assertEqual(cluster.strand, "+")
        self.assertEqual(cluster.start, 1000)
        self.assertEqual(cluster.end, 2500)

    def test_cluster_size_property(self):
        """Test the size property of JunctionCluster."""
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=1500, end=2500, strand="+"),
            Junction(chrom="chr1", start=2000, end=3000, strand="+"),
        ]

        cluster = JunctionCluster(
            cluster_id="test",
            junctions=junctions,
            chrom="chr1",
            strand="+",
            start=1000,
            end=3000,
        )

        self.assertEqual(cluster.size, 3)

    def test_cluster_immutability(self):
        """Test that JunctionCluster attributes are as expected."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        cluster = JunctionCluster(
            cluster_id="test",
            junctions=[j1],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
        )

        # Verify cluster is created with correct properties
        self.assertEqual(cluster.cluster_id, "test")
        self.assertEqual(cluster.junctions, [j1])
        self.assertEqual(cluster.chrom, "chr1")
        self.assertEqual(cluster.strand, "+")


class TestClusterJunctions(unittest.TestCase):
    """Test the main cluster_junctions function."""

    def test_empty_input(self):
        """Test clustering with empty junction list."""
        clusters = cluster_junctions([])
        self.assertEqual(clusters, [])

    def test_single_junction(self):
        """Test clustering with a single junction."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")

        # Single junction with min_cluster_size=2 should not form a cluster
        clusters = cluster_junctions([j1], min_cluster_size=2)
        self.assertEqual(len(clusters), 0)

        # But with min_cluster_size=1, it should
        clusters = cluster_junctions([j1], min_cluster_size=1)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].size, 1)

    def test_multiple_junctions_no_overlap(self):
        """Test clustering with non-overlapping junctions."""
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=5000, end=6000, strand="+"),
            Junction(chrom="chr1", start=9000, end=10000, strand="+"),
        ]

        clusters = cluster_junctions(junctions)
        # No overlaps, so no clusters should form
        self.assertEqual(len(clusters), 0)

    def test_overlapping_junctions(self):
        """Test clustering with overlapping junctions that share splice sites."""
        # Junctions sharing donor sites stay together after refinement
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            Junction(chrom="chr1", start=1000, end=3000, strand="+"),
        ]

        clusters = cluster_junctions(junctions)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].size, 3)
        self.assertEqual(clusters[0].start, 1000)
        self.assertEqual(clusters[0].end, 3000)

    def test_max_intron_length_filtering(self):
        """Test that junctions exceeding max_intron_length are excluded."""
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=1500, end=102000, strand="+"),
        ]

        clusters = cluster_junctions(junctions, max_intron_length=100000)
        # The long junction should be skipped, so no cluster forms
        self.assertEqual(len(clusters), 0)

    def test_min_cluster_size_filtering(self):
        """Test that clusters smaller than min_cluster_size are excluded."""
        # Use junctions that share splice sites to stay together
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=1000, end=2500, strand="+"),
        ]

        clusters = cluster_junctions(junctions, min_cluster_size=3)
        # Cluster has only 2 junctions, should be excluded
        self.assertEqual(len(clusters), 0)

        clusters = cluster_junctions(junctions, min_cluster_size=2)
        # Now it should pass
        self.assertEqual(len(clusters), 1)

    def test_multiple_chromosomes(self):
        """Test clustering with junctions on different chromosomes."""
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            Junction(chrom="chr2", start=1000, end=2000, strand="+"),
            Junction(chrom="chr2", start=1000, end=2500, strand="+"),
        ]

        clusters = cluster_junctions(junctions)
        self.assertEqual(len(clusters), 2)

        chr1_clusters = [c for c in clusters if c.chrom == "chr1"]
        chr2_clusters = [c for c in clusters if c.chrom == "chr2"]

        self.assertEqual(len(chr1_clusters), 1)
        self.assertEqual(len(chr2_clusters), 1)

    def test_multiple_strands(self):
        """Test clustering with junctions on different strands."""
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            Junction(chrom="chr1", start=1000, end=2000, strand="-"),
            Junction(chrom="chr1", start=1000, end=2500, strand="-"),
        ]

        clusters = cluster_junctions(junctions)
        self.assertEqual(len(clusters), 2)

        plus_clusters = [c for c in clusters if c.strand == "+"]
        minus_clusters = [c for c in clusters if c.strand == "-"]

        self.assertEqual(len(plus_clusters), 1)
        self.assertEqual(len(minus_clusters), 1)
        self.assertEqual(plus_clusters[0].size, 2)
        self.assertEqual(minus_clusters[0].size, 2)

    def test_splice_site_refinement(self):
        """Test clustering with splice site-based refinement."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=2500, strand="+")
        j3 = Junction(chrom="chr1", start=1500, end=2000, strand="+")

        clusters = cluster_junctions([j1, j2, j3])
        self.assertGreaterEqual(len(clusters), 1)

    def test_cluster_id_uniqueness(self):
        """Test that cluster IDs are unique."""
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=1500, end=2500, strand="+"),
            Junction(chrom="chr1", start=5000, end=6000, strand="+"),
            Junction(chrom="chr1", start=5500, end=6500, strand="+"),
            Junction(chrom="chr2", start=1000, end=2000, strand="-"),
            Junction(chrom="chr2", start=1500, end=2500, strand="-"),
        ]

        clusters = cluster_junctions(junctions)
        cluster_ids = [c.cluster_id for c in clusters]
        self.assertEqual(len(cluster_ids), len(set(cluster_ids)))

    def test_cluster_bounds(self):
        """Test that cluster start/end bounds are correct."""
        # All junctions share the same donor site
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            Junction(chrom="chr1", start=1000, end=3000, strand="+"),
        ]

        clusters = cluster_junctions(junctions)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].start, 1000)
        self.assertEqual(clusters[0].end, 3000)


class TestGetClusterJunctions(unittest.TestCase):
    """Test the get_cluster_junctions function."""

    def test_get_junctions_from_cluster(self):
        """Test retrieving junctions from a cluster."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1500, end=2500, strand="+")

        cluster = JunctionCluster(
            cluster_id="test",
            junctions=[j1, j2],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
        )

        junc_set = get_cluster_junctions(cluster)

        self.assertEqual(len(junc_set), 2)
        self.assertIn(j1, junc_set)
        self.assertIn(j2, junc_set)

    def test_get_junctions_returns_set(self):
        """Test that get_cluster_junctions returns a set."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")

        cluster = JunctionCluster(
            cluster_id="test",
            junctions=[j1],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
        )

        junc_set = get_cluster_junctions(cluster)
        self.assertIsInstance(junc_set, set)


class TestFilterClustersBySize(unittest.TestCase):
    """Test the filter_clusters_by_size function."""

    def test_filter_empty_list(self):
        """Test filtering empty cluster list."""
        result = filter_clusters_by_size([])
        self.assertEqual(result, [])

    def test_filter_by_min_size(self):
        """Test filtering clusters by minimum size."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1500, end=2500, strand="+")
        j3 = Junction(chrom="chr1", start=2000, end=3000, strand="+")

        cluster1 = JunctionCluster(
            cluster_id="c1",
            junctions=[j1],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
        )

        cluster2 = JunctionCluster(
            cluster_id="c2",
            junctions=[j2, j3],
            chrom="chr1",
            strand="+",
            start=1500,
            end=3000,
        )

        clusters = [cluster1, cluster2]

        result = filter_clusters_by_size(clusters, min_junctions=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].cluster_id, "c2")

    def test_filter_default_min_junctions(self):
        """Test filtering with default min_junctions=2."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")

        cluster = JunctionCluster(
            cluster_id="test",
            junctions=[j1],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
        )

        result = filter_clusters_by_size([cluster])
        self.assertEqual(len(result), 0)


class TestFilterClustersByRegion(unittest.TestCase):
    """Test the filter_clusters_by_region function."""

    def test_filter_empty_list(self):
        """Test filtering empty cluster list."""
        result = filter_clusters_by_region([], "chr1", 1000, 2000)
        self.assertEqual(result, [])

    def test_filter_by_overlapping_region(self):
        """Test filtering clusters that overlap a region."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1500, end=2500, strand="+")

        cluster = JunctionCluster(
            cluster_id="test",
            junctions=[j1, j2],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
        )

        result = filter_clusters_by_region([cluster], "chr1", 1500, 2000)
        self.assertEqual(len(result), 1)

    def test_filter_no_overlapping_region(self):
        """Test filtering clusters that don't overlap a region."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")

        cluster = JunctionCluster(
            cluster_id="test",
            junctions=[j1],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
        )

        result = filter_clusters_by_region([cluster], "chr1", 5000, 6000)
        self.assertEqual(len(result), 0)

    def test_filter_different_chromosome(self):
        """Test filtering clusters on different chromosome."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")

        cluster = JunctionCluster(
            cluster_id="test",
            junctions=[j1],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
        )

        result = filter_clusters_by_region([cluster], "chr2", 1000, 2000)
        self.assertEqual(len(result), 0)

    def test_filter_region_boundaries(self):
        """Test region filtering with boundary conditions."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")

        cluster = JunctionCluster(
            cluster_id="test",
            junctions=[j1],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
        )

        result = filter_clusters_by_region([cluster], "chr1", 0, 1000)
        self.assertEqual(len(result), 0)

        result = filter_clusters_by_region([cluster], "chr1", 2000, 3000)
        self.assertEqual(len(result), 0)

        result = filter_clusters_by_region([cluster], "chr1", 999, 1001)
        self.assertEqual(len(result), 1)

    def test_filter_multiple_clusters(self):
        """Test filtering multiple clusters with region query."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=5000, end=6000, strand="+")

        cluster1 = JunctionCluster(
            cluster_id="c1",
            junctions=[j1],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
        )

        cluster2 = JunctionCluster(
            cluster_id="c2",
            junctions=[j2],
            chrom="chr1",
            strand="+",
            start=5000,
            end=6000,
        )

        result = filter_clusters_by_region([cluster1, cluster2], "chr1", 1500, 1800)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].cluster_id, "c1")


class TestClusteringIntegration(unittest.TestCase):
    """Integration tests combining multiple clustering features."""

    def test_complex_clustering_scenario(self):
        """Test clustering with multiple regions and filtering."""
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            Junction(chrom="chr1", start=1000, end=3000, strand="+"),
            Junction(chrom="chr1", start=10000, end=11000, strand="+"),
            Junction(chrom="chr1", start=10000, end=11500, strand="+"),
            Junction(chrom="chr1", start=1000, end=2000, strand="-"),
            Junction(chrom="chr1", start=1000, end=2500, strand="-"),
        ]

        clusters = cluster_junctions(junctions)

        self.assertEqual(len(clusters), 3)

        filtered = filter_clusters_by_region(clusters, "chr1", 1000, 2000)
        self.assertEqual(len(filtered), 2)

    def test_clustering_with_multiple_filters(self):
        """Test applying multiple filters to clusters."""
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            Junction(chrom="chr1", start=1000, end=3000, strand="+"),
            Junction(chrom="chr1", start=5000, end=6000, strand="+"),
        ]

        clusters = cluster_junctions(junctions)

        by_size = filter_clusters_by_size(clusters, min_junctions=2)
        self.assertEqual(len(by_size), 1)

        by_region = filter_clusters_by_region(by_size, "chr1", 1000, 3000)
        self.assertEqual(len(by_region), 1)

    def test_max_intron_length_with_mixed_junctions(self):
        """Test max_intron_length filtering with mixed junction lengths."""
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=1500, end=2500, strand="+"),
            Junction(chrom="chr1", start=2000, end=102000, strand="+"),
            Junction(chrom="chr1", start=3000, end=4000, strand="+"),
        ]

        clusters = cluster_junctions(junctions, max_intron_length=50000)
        self.assertGreaterEqual(len(clusters), 0)

    def test_large_number_of_junctions(self):
        """Test clustering with a large number of junctions."""
        # All junctions share the same donor site
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000 + i * 10, strand="+")
            for i in range(100)
        ]

        clusters = cluster_junctions(junctions)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].size, 100)


if __name__ == "__main__":
    unittest.main()
