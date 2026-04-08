"""
Test suite for Module 20: core/nmd_classifier.py

Tests NMD/PTC functional classification of junctions.
"""

import unittest

from splicekit.core.nmd_classifier import (
    NMDClassification,
    build_translation_graph,
    classify_all_junctions_nmd,
    classify_junction_nmd,
)
from splicekit.utils.genomic import Junction


class TestNMDClassificationDataclass(unittest.TestCase):
    """Test NMDClassification dataclass."""

    def test_nmd_classification_creation(self):
        """Test basic NMDClassification creation."""
        junction = Junction(chrom="chr1", start=100, end=200, strand="+")

        classification = NMDClassification(
            junction=junction,
            classification="PR",
            n_productive_paths=5,
            n_unproductive_paths=0,
            confidence=1.0,
            ptc_position=None,
            last_ejc_position=200,
        )

        self.assertEqual(classification.classification, "PR")
        self.assertEqual(classification.n_productive_paths, 5)
        self.assertEqual(classification.confidence, 1.0)

    def test_nmd_classification_frozen(self):
        """Test that NMDClassification is frozen."""
        junction = Junction(chrom="chr1", start=100, end=200, strand="+")

        classification = NMDClassification(
            junction=junction,
            classification="UP",
            n_productive_paths=0,
            n_unproductive_paths=3,
            confidence=0.0,
            ptc_position=150,
            last_ejc_position=200,
        )

        with self.assertRaises(AttributeError):
            classification.classification = "PR"

    def test_nmd_classification_types(self):
        """Test NMDClassification with all classification types."""
        junction = Junction(chrom="chr1", start=100, end=200, strand="+")

        for cls_type in ["PR", "UP", "NE", "IN"]:
            classification = NMDClassification(
                junction=junction,
                classification=cls_type,
                n_productive_paths=1,
                n_unproductive_paths=0,
                confidence=0.5,
                ptc_position=None,
                last_ejc_position=200,
            )

            self.assertEqual(classification.classification, cls_type)


class TestClassifyJunctionBasic(unittest.TestCase):
    """Test basic junction classification."""

    def test_classify_single_junction(self):
        """Test classification of single junction."""
        junction = Junction(chrom="chr1", start=100, end=200, strand="+")

        exon_positions = {0: (0, 100), 1: (200, 300)}
        genome_fasta = {"chr1": "A" * 500}

        classification = classify_junction_nmd(
            junction, exon_positions, genome_fasta
        )

        self.assertIsInstance(classification, NMDClassification)
        self.assertEqual(classification.junction, junction)
        self.assertIn(classification.classification, ["PR", "UP", "NE", "IN"])

    def test_classify_junction_confidence(self):
        """Test that confidence is in [0, 1] or NaN."""
        junction = Junction(chrom="chr1", start=100, end=200, strand="+")

        exon_positions = {0: (0, 100), 1: (200, 300)}
        genome_fasta = {"chr1": "A" * 500}

        classification = classify_junction_nmd(
            junction, exon_positions, genome_fasta
        )

        # Confidence should be between 0 and 1, or NaN
        import math

        if not math.isnan(classification.confidence):
            self.assertGreaterEqual(classification.confidence, 0.0)
            self.assertLessEqual(classification.confidence, 1.0)

    def test_classify_junction_paths(self):
        """Test that path counts are non-negative."""
        junction = Junction(chrom="chr1", start=100, end=200, strand="+")

        exon_positions = {0: (0, 100), 1: (200, 300)}
        genome_fasta = {"chr1": "A" * 500}

        classification = classify_junction_nmd(
            junction, exon_positions, genome_fasta
        )

        self.assertGreaterEqual(classification.n_productive_paths, 0)
        self.assertGreaterEqual(classification.n_unproductive_paths, 0)


class TestClassifyMultipleJunctions(unittest.TestCase):
    """Test classification of multiple junctions."""

    def test_classify_all_junctions(self):
        """Test classification of multiple junctions."""
        junctions = [
            Junction(chrom="chr1", start=100, end=200, strand="+"),
            Junction(chrom="chr1", start=300, end=400, strand="+"),
            Junction(chrom="chr1", start=500, end=600, strand="+"),
        ]

        exon_positions = {
            0: (0, 100),
            1: (200, 300),
            2: (400, 500),
            3: (600, 700),
        }
        genome_fasta = {"chr1": "A" * 1000}

        classifications = classify_all_junctions_nmd(
            junctions, exon_positions, genome_fasta
        )

        self.assertEqual(len(classifications), 3)

        for classification in classifications:
            self.assertIsInstance(classification, NMDClassification)
            self.assertIn(classification.classification, ["PR", "UP", "NE", "IN"])

    def test_classify_same_result_reproducible(self):
        """Test that same junctions get consistent classification."""
        junction = Junction(chrom="chr1", start=100, end=200, strand="+")

        exon_positions = {0: (0, 100), 1: (200, 300)}
        genome_fasta = {"chr1": "A" * 500}

        # Classify twice
        result1 = classify_junction_nmd(
            junction, exon_positions, genome_fasta
        )
        result2 = classify_junction_nmd(
            junction, exon_positions, genome_fasta
        )

        # Should get same classification
        self.assertEqual(result1.classification, result2.classification)
        self.assertEqual(result1.n_productive_paths, result2.n_productive_paths)


class TestBuildTranslationGraph(unittest.TestCase):
    """Test translation graph building."""

    def test_build_empty_graph(self):
        """Test building graph with no junctions."""
        exon_positions = {0: (0, 100), 1: (200, 300)}
        observed_junctions = set()
        genome_fasta = {"chr1": "A" * 500}

        graph = build_translation_graph(
            exon_positions, observed_junctions, genome_fasta
        )

        # Should have nodes for each exon-frame combination
        self.assertIsInstance(graph, dict)
        self.assertGreater(len(graph), 0)

    def test_build_graph_with_junctions(self):
        """Test building graph with junctions."""
        exon_positions = {
            0: (0, 100),
            1: (200, 300),
            2: (400, 500),
        }
        observed_junctions = {
            Junction(chrom="chr1", start=100, end=200, strand="+"),
            Junction(chrom="chr1", start=300, end=400, strand="+"),
        }
        genome_fasta = {"chr1": "A" * 600}

        graph = build_translation_graph(
            exon_positions, observed_junctions, genome_fasta
        )

        self.assertIsInstance(graph, dict)
        # Graph should have nodes for all exons and frames
        self.assertEqual(len(graph), 9)  # 3 exons * 3 frames


class TestNMDEdgeCases(unittest.TestCase):
    """Test edge cases for NMD classification."""

    def test_different_strands(self):
        """Test classification on different strands."""
        for strand in ["+", "-"]:
            junction = Junction(chrom="chr1", start=100, end=200, strand=strand)

            exon_positions = {0: (0, 100), 1: (200, 300)}
            genome_fasta = {"chr1": "A" * 500}

            classification = classify_junction_nmd(
                junction, exon_positions, genome_fasta
            )

            self.assertIsInstance(classification, NMDClassification)

    def test_missing_chromosome(self):
        """Test classification with missing chromosome in genome."""
        junction = Junction(chrom="chrX", start=100, end=200, strand="+")

        exon_positions = {0: (0, 100), 1: (200, 300)}
        genome_fasta = {"chr1": "A" * 500}  # Missing chrX

        classification = classify_junction_nmd(
            junction, exon_positions, genome_fasta
        )

        # Should still return a classification
        self.assertIsInstance(classification, NMDClassification)

    def test_junction_coordinates(self):
        """Test that junction coordinates are preserved."""
        junction = Junction(chrom="chr1", start=1000, end=2000, strand="+")

        exon_positions = {
            0: (0, 1000),
            1: (2000, 3000),
        }
        genome_fasta = {"chr1": "A" * 4000}

        classification = classify_junction_nmd(
            junction, exon_positions, genome_fasta
        )

        self.assertEqual(classification.junction, junction)
        self.assertEqual(classification.junction.start, 1000)
        self.assertEqual(classification.junction.end, 2000)


class TestNMDIntegration(unittest.TestCase):
    """Integration tests for NMD classification."""

    def test_full_pipeline(self):
        """Test complete NMD classification pipeline."""
        # Create a set of junctions
        junctions = [
            Junction(chrom="chr1", start=500, end=1000, strand="+"),
            Junction(chrom="chr1", start=1500, end=2000, strand="+"),
            Junction(chrom="chr1", start=2500, end=3000, strand="+"),
        ]

        # Create exon structure
        exon_positions = {
            0: (0, 500),
            1: (1000, 1500),
            2: (2000, 2500),
            3: (3000, 4000),
        }

        # Create genome sequence
        genome_fasta = {"chr1": "A" * 4100}

        # Classify all junctions
        classifications = classify_all_junctions_nmd(
            junctions, exon_positions, genome_fasta, strand="+"
        )

        # Verify results
        self.assertEqual(len(classifications), 3)

        for i, classification in enumerate(classifications):
            self.assertEqual(classification.junction, junctions[i])
            self.assertIn(classification.classification, ["PR", "UP", "NE", "IN"])
            # Productive + unproductive should be at least 1
            self.assertGreater(
                classification.n_productive_paths + classification.n_unproductive_paths,
                0,
            )


if __name__ == "__main__":
    unittest.main()
