"""
Test suite for Module 10: core/event_classifier.py

Tests event classification including:
- SE (skipped exon) detection
- A3SS (alternative 3' site) detection
- A5SS (alternative 5' site) detection
- MXE (mutually exclusive) detection
- RI (retained intron) detection
- Complex event classification
"""

import unittest

from splicekit.core.event_classifier import (
    classify_all_events,
    classify_event,
    filter_modules_by_event_type,
    get_event_type_counts,
)
from splicekit.core.splicegraph import SplicingModule
from splicekit.utils.genomic import Junction


class TestA3SS(unittest.TestCase):
    """Test A3SS (Alternative 3' Splice Site) classification."""

    def test_a3ss_detection(self):
        """Test detection of A3SS events."""
        # A3SS: same donor (1000), different acceptors (2000, 2500)
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

        event_type = classify_event(module)
        self.assertEqual(event_type, "A3SS")


class TestA5SS(unittest.TestCase):
    """Test A5SS (Alternative 5' Splice Site) classification."""

    def test_a5ss_detection(self):
        """Test detection of A5SS events."""
        # A5SS: same acceptor (2000), different donors (1000, 1500)
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1500, end=2000, strand="+")

        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
            junctions=[j1, j2],
            junction_indices=[0, 1],
            n_connections=2,
        )

        event_type = classify_event(module)
        self.assertEqual(event_type, "A5SS")


class TestSE(unittest.TestCase):
    """Test SE (Skipped Exon) classification."""

    def test_se_detection_shared_donor(self):
        """Test detection of SE with two junctions sharing donor."""
        # SE: j1 and j2 share donor 1000
        # j3 has different donor but shares an endpoint
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=3000, strand="+")
        j3 = Junction(chrom="chr1", start=2100, end=3000, strand="+")

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

        event_type = classify_event(module)
        self.assertEqual(event_type, "SE")

    def test_se_detection_shared_acceptor(self):
        """Test detection of SE with two junctions sharing acceptor."""
        # SE: j1 and j2 share acceptor 3000
        j1 = Junction(chrom="chr1", start=1000, end=3000, strand="+")
        j2 = Junction(chrom="chr1", start=1500, end=3000, strand="+")
        j3 = Junction(chrom="chr1", start=1000, end=2000, strand="+")

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

        event_type = classify_event(module)
        self.assertEqual(event_type, "SE")


class TestRI(unittest.TestCase):
    """Test RI (Retained Intron) classification."""

    def test_ri_detection(self):
        """Test detection of RI events."""
        # RI: j2 fully contains j1 (retained intron)
        j1 = Junction(chrom="chr1", start=1500, end=2500, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=3000, strand="+")

        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=3000,
            junctions=[j1, j2],
            junction_indices=[0, 1],
            n_connections=2,
        )

        event_type = classify_event(module)
        self.assertEqual(event_type, "RI")

    def test_ri_detection_reverse_order(self):
        """Test RI detection with contained junction first."""
        # Same as above but j1 and j2 swapped
        j1 = Junction(chrom="chr1", start=1000, end=3000, strand="+")
        j2 = Junction(chrom="chr1", start=1500, end=2500, strand="+")

        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=3000,
            junctions=[j1, j2],
            junction_indices=[0, 1],
            n_connections=2,
        )

        event_type = classify_event(module)
        self.assertEqual(event_type, "RI")


class TestComplex(unittest.TestCase):
    """Test Complex event classification."""

    def test_complex_4junctions(self):
        """Test that 4 junctions without clear MXE pattern are Complex."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1000, end=2500, strand="+")
        j3 = Junction(chrom="chr1", start=1500, end=3000, strand="+")
        j4 = Junction(chrom="chr1", start=2000, end=3000, strand="+")

        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=3000,
            junctions=[j1, j2, j3, j4],
            junction_indices=[0, 1, 2, 3],
            n_connections=4,
        )

        event_type = classify_event(module)
        self.assertEqual(event_type, "Complex")

    def test_complex_5junctions(self):
        """Test that >4 junctions are classified as Complex."""
        junctions = [
            Junction(chrom="chr1", start=1000, end=2000, strand="+"),
            Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            Junction(chrom="chr1", start=1500, end=3000, strand="+"),
            Junction(chrom="chr1", start=2000, end=3500, strand="+"),
            Junction(chrom="chr1", start=2500, end=4000, strand="+"),
        ]

        module = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=4000,
            junctions=junctions,
            junction_indices=list(range(len(junctions))),
            n_connections=len(junctions),
        )

        event_type = classify_event(module)
        self.assertEqual(event_type, "Complex")

    def test_complex_3junctions_no_pattern(self):
        """Test that 3 junctions without SE pattern are Complex."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=1500, end=2500, strand="+")
        j3 = Junction(chrom="chr1", start=2000, end=3000, strand="+")

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

        event_type = classify_event(module)
        self.assertEqual(event_type, "Complex")


class TestClassifyAllEvents(unittest.TestCase):
    """Test classify_all_events function."""

    def test_classify_multiple_modules(self):
        """Test classifying multiple modules at once."""
        # A3SS module
        a3ss = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
            junctions=[
                Junction(chrom="chr1", start=1000, end=2000, strand="+"),
                Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            ],
            junction_indices=[0, 1],
            n_connections=2,
        )

        # A5SS module
        a5ss = SplicingModule(
            module_id="mod2",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
            junctions=[
                Junction(chrom="chr1", start=1000, end=2000, strand="+"),
                Junction(chrom="chr1", start=1500, end=2000, strand="+"),
            ],
            junction_indices=[2, 3],
            n_connections=2,
        )

        # RI module
        ri = SplicingModule(
            module_id="mod3",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=3000,
            junctions=[
                Junction(chrom="chr1", start=1500, end=2500, strand="+"),
                Junction(chrom="chr1", start=1000, end=3000, strand="+"),
            ],
            junction_indices=[4, 5],
            n_connections=2,
        )

        modules = [a3ss, a5ss, ri]
        event_types = classify_all_events(modules)

        self.assertEqual(len(event_types), 3)
        self.assertEqual(event_types[0], "A3SS")
        self.assertEqual(event_types[1], "A5SS")
        self.assertEqual(event_types[2], "RI")


class TestFilterModulesByEventType(unittest.TestCase):
    """Test filter_modules_by_event_type function."""

    def test_filter_by_a3ss(self):
        """Test filtering modules by A3SS event type."""
        a3ss = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
            junctions=[
                Junction(chrom="chr1", start=1000, end=2000, strand="+"),
                Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            ],
            junction_indices=[0, 1],
            n_connections=2,
        )

        a5ss = SplicingModule(
            module_id="mod2",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2000,
            junctions=[
                Junction(chrom="chr1", start=1000, end=2000, strand="+"),
                Junction(chrom="chr1", start=1500, end=2000, strand="+"),
            ],
            junction_indices=[2, 3],
            n_connections=2,
        )

        modules = [a3ss, a5ss]
        filtered = filter_modules_by_event_type(modules, "A3SS")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].module_id, "mod1")


class TestGetEventTypeCounts(unittest.TestCase):
    """Test get_event_type_counts function."""

    def test_count_event_types(self):
        """Test counting modules by event type."""
        a3ss1 = SplicingModule(
            module_id="mod1",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=1000,
            end=2500,
            junctions=[
                Junction(chrom="chr1", start=1000, end=2000, strand="+"),
                Junction(chrom="chr1", start=1000, end=2500, strand="+"),
            ],
            junction_indices=[0, 1],
            n_connections=2,
        )

        a3ss2 = SplicingModule(
            module_id="mod2",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=3000,
            end=4500,
            junctions=[
                Junction(chrom="chr1", start=3000, end=4000, strand="+"),
                Junction(chrom="chr1", start=3000, end=4500, strand="+"),
            ],
            junction_indices=[2, 3],
            n_connections=2,
        )

        a5ss = SplicingModule(
            module_id="mod3",
            gene_id="GENE1",
            gene_name="Gene1",
            chrom="chr1",
            strand="+",
            start=5000,
            end=6000,
            junctions=[
                Junction(chrom="chr1", start=5000, end=6000, strand="+"),
                Junction(chrom="chr1", start=5500, end=6000, strand="+"),
            ],
            junction_indices=[4, 5],
            n_connections=2,
        )

        modules = [a3ss1, a3ss2, a5ss]
        counts = get_event_type_counts(modules)

        self.assertEqual(counts["A3SS"], 2)
        self.assertEqual(counts["A5SS"], 1)
        self.assertEqual(len(counts), 2)


if __name__ == "__main__":
    unittest.main()
