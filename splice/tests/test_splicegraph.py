"""
Test suite for Module 8: core/splicegraph.py

Tests splicegraph module building including:
- SplicingModule creation and properties
- Cluster-to-gene mapping
- Module merging and refinement
- Gene assignment for de novo junctions
"""

import unittest
from dataclasses import dataclass

from splice.core.clustering import JunctionCluster
from splice.core.splicegraph import (
    SplicingModule,
    build_splicegraph,
    filter_modules_by_size,
    filter_modules_by_gene,
    filter_modules_by_region,
    get_module_junctions,
)
from splice.utils.genomic import Junction


@dataclass
class MockGene:
    """Mock Gene class for testing."""

    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    start: int
    end: int


class TestSplicingModule(unittest.TestCase):
    """Test SplicingModule dataclass."""

    def test_module_creation(self):
        """Test creating a SplicingModule."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=2500, strand="+")

        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
            junctions=[j1, j2],
            junction_indices=[0, 1],
            n_connections=2,
        )

        self.assertEqual(module.module_id, "mod1")
        self.assertEqual(module.gene_id, "GENE1")
        self.assertEqual(module.gene_name, "Gene1")
        self.assertEqual(len(module.junctions), 2)
        self.assertEqual(module.n_connections, 2)

    def test_module_is_binary(self):
        """Test the is_binary property."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=2500, strand="+")

        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
            junctions=[j1, j2],
            junction_indices=[0, 1],
            n_connections=2,
        )

        self.assertTrue(module.is_binary)

    def test_module_is_not_binary(self):
        """Test is_binary when module has >2 junctions."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=2500, strand="+")
        j3 = Junction(chrom="chr1", start=1000, end=3000, strand="+")

        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=3000,
            junctions=[j1, j2, j3],
            junction_indices=[0, 1, 2],
            n_connections=3,
        )

        self.assertFalse(module.is_binary)


class TestBuildSplicegraph(unittest.TestCase):
    """Test the build_splicegraph function."""

    def test_empty_clusters(self):
        """Test building splicegraph with empty clusters."""
        genes = {}
        junction_evidence = {}
        clusters = []
        known_junctions = set()

        modules, j_to_idx = build_splicegraph(
            genes, junction_evidence, clusters, known_junctions
        )

        self.assertEqual(len(modules), 0)
        self.assertEqual(len(j_to_idx), 0)

    def test_single_cluster_intergenic(self):
        """Test single cluster with no gene match."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=2500, strand="+")

        cluster = JunctionCluster(
            cluster_id="c1",
            junctions=[j1, j2],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
        )

        genes = {}
        junction_evidence = {}
        known_junctions = {j1}

        modules, j_to_idx = build_splicegraph(
            genes, junction_evidence, [cluster], known_junctions
        )

        self.assertEqual(len(modules), 1)
        self.assertEqual(modules[0].gene_id, "")
        self.assertEqual(modules[0].n_connections, 2)
        self.assertEqual(len(j_to_idx), 2)

    def test_single_cluster_with_gene(self):
        """Test single cluster assigned to a gene."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=2500, strand="+")

        cluster = JunctionCluster(
            cluster_id="c1",
            junctions=[j1, j2],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
        )

        gene = MockGene(
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=500,
            end=5000,
        )

        genes = {"GENE1": gene}
        junction_evidence = {}
        known_junctions = {j1}

        modules, j_to_idx = build_splicegraph(
            genes, junction_evidence, [cluster], known_junctions
        )

        self.assertEqual(len(modules), 1)
        self.assertEqual(modules[0].gene_id, "GENE1")
        self.assertEqual(modules[0].gene_name, "Gene1")
        self.assertEqual(modules[0].n_connections, 2)

    def test_multiple_clusters_same_gene(self):
        """Test multiple overlapping clusters assigned to the same gene."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=2500, strand="+")
        j3 = Junction(chrom="chr1", start=2000, end=3000, strand="+")
        j4 = Junction(chrom="chr1", start=2000, end=3500, strand="+")

        cluster1 = JunctionCluster(
            cluster_id="c1",
            junctions=[j1, j2],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
        )

        cluster2 = JunctionCluster(
            cluster_id="c2",
            junctions=[j3, j4],
            chrom="chr1",
            strand="+",
            start=2000,
            end=3500,
        )

        gene = MockGene(
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=500,
            end=5000,
        )

        genes = {"GENE1": gene}
        junction_evidence = {}
        known_junctions = {j1, j3}

        modules, j_to_idx = build_splicegraph(
            genes, junction_evidence, [cluster1, cluster2], known_junctions
        )

        # Two overlapping clusters should merge into one module
        self.assertEqual(len(modules), 1)
        self.assertEqual(modules[0].n_connections, 4)
        self.assertEqual(len(j_to_idx), 4)

    def test_multiple_clusters_different_genes(self):
        """Test clusters assigned to different genes."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=2500, strand="+")
        j3 = Junction(chrom="chr1", start=50000, end=51000, strand="+")
        j4 = Junction(chrom="chr1", start=50000, end=51500, strand="+")

        cluster1 = JunctionCluster(
            cluster_id="c1",
            junctions=[j1, j2],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
        )

        cluster2 = JunctionCluster(
            cluster_id="c2",
            junctions=[j3, j4],
            chrom="chr1",
            strand="+",
            start=50000,
            end=51500,
        )

        gene1 = MockGene(
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=500,
            end=10000,
        )

        gene2 = MockGene(
            gene_id="GENE2",
            gene_name="Gene2",
            chrom="chr1",
            strand="+",
            start=40000,
            end=60000,
        )

        genes = {"GENE1": gene1, "GENE2": gene2}
        junction_evidence = {}
        known_junctions = {j1, j3}

        modules, j_to_idx = build_splicegraph(
            genes, junction_evidence, [cluster1, cluster2], known_junctions
        )

        self.assertEqual(len(modules), 2)

        gene1_modules = [m for m in modules if m.gene_id == "GENE1"]
        gene2_modules = [m for m in modules if m.gene_id == "GENE2"]

        self.assertEqual(len(gene1_modules), 1)
        self.assertEqual(len(gene2_modules), 1)

    def test_strands_separated(self):
        """Test that junctions on different strands are in separate modules."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=2500, strand="+")
        j3 = Junction(chrom="chr1", start=1000, end=2000, strand="-")
        j4 = Junction(chrom="chr1", start=1000, end=2500, strand="-")

        cluster1 = JunctionCluster(
            cluster_id="c1",
            junctions=[j1, j2],
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
        )

        cluster2 = JunctionCluster(
            cluster_id="c2",
            junctions=[j3, j4],
            chrom="chr1",
            strand="-",
            start=1000,
            end=2500,
        )

        gene = MockGene(
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=500,
            end=5000,
        )

        genes = {"GENE1": gene}
        junction_evidence = {}
        known_junctions = {j1, j3}

        modules, j_to_idx = build_splicegraph(
            genes, junction_evidence, [cluster1, cluster2], known_junctions
        )

        # Plus strand cluster should be assigned to gene
        # Minus strand cluster should be intergenic
        plus_modules = [m for m in modules if m.strand == "+"]
        minus_modules = [m for m in modules if m.strand == "-"]

        self.assertEqual(len(plus_modules), 1)
        self.assertEqual(len(minus_modules), 1)
        self.assertEqual(plus_modules[0].gene_id, "GENE1")
        self.assertEqual(minus_modules[0].gene_id, "")


class TestGetModuleJunctions(unittest.TestCase):
    """Test the get_module_junctions function."""

    def test_get_junctions(self):
        """Test retrieving junctions from a module."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=2500, strand="+")

        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
            junctions=[j1, j2],
            junction_indices=[0, 1],
            n_connections=2,
        )

        junctions = get_module_junctions(module)

        self.assertEqual(len(junctions), 2)
        self.assertEqual(junctions, [j1, j2])


class TestFilterModulesBySize(unittest.TestCase):
    """Test the filter_modules_by_size function."""

    def test_filter_empty_list(self):
        """Test filtering empty module list."""
        result = filter_modules_by_size([])
        self.assertEqual(result, [])

    def test_filter_by_min_junctions(self):
        """Test filtering modules by minimum junction count."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=2500, strand="+")
        j3 = Junction(chrom="chr1", start=1000, end=3000, strand="+")

        module1 = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
            junctions=[j1],
            junction_indices=[0],
            n_connections=1,
        )

        module2 = SplicingModule(
            module_id="mod2",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=3000,
            junctions=[j2, j3],
            junction_indices=[1, 2],
            n_connections=2,
        )

        modules = [module1, module2]

        result = filter_modules_by_size(modules, min_junctions=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].module_id, "mod2")


class TestFilterModulesByGene(unittest.TestCase):
    """Test the filter_modules_by_gene function."""

    def test_filter_by_gene_id(self):
        """Test filtering modules by gene ID."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")

        module1 = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
            junctions=[j1],
            junction_indices=[0],
            n_connections=1,
        )

        module2 = SplicingModule(
            module_id="mod2",
            gene_id="GENE2",
            gene_name="Gene2",
            chrom="chr1",
            strand="+",
            start=5000,
            end=6000,
            junctions=[j1],
            junction_indices=[0],
            n_connections=1,
        )

        modules = [module1, module2]

        result = filter_modules_by_gene(modules, "GENE1")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].gene_id, "GENE1")


class TestFilterModulesByRegion(unittest.TestCase):
    """Test the filter_modules_by_region function."""

    def test_filter_by_region(self):
        """Test filtering modules by genomic region."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=5000, end=6000, strand="+")

        module1 = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
            junctions=[j1],
            junction_indices=[0],
            n_connections=1,
        )

        module2 = SplicingModule(
            module_id="mod2",
            gene_id="GENE2",
            gene_name="Gene2",
            chrom="chr1",
            strand="+",
            start=5000,
            end=6000,
            junctions=[j2],
            junction_indices=[1],
            n_connections=1,
        )

        modules = [module1, module2]

        result = filter_modules_by_region(modules, "chr1", 1000, 3000)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].module_id, "mod1")


if __name__ == "__main__":
    unittest.main()
