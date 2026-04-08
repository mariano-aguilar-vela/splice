"""
Test suite for Module 9: core/confidence_scorer.py

Tests junction confidence scoring including:
- Component score calculation
- Composite score calculation
- Filtering by confidence threshold
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from splicekit.core.confidence_scorer import (
    JunctionConfidence,
    filter_junctions_by_confidence,
    score_all_junctions,
    score_junction,
)
from splicekit.utils.genomic import Junction


class TestJunctionConfidence(unittest.TestCase):
    """Test JunctionConfidence dataclass."""

    def test_confidence_creation(self):
        """Test creating a JunctionConfidence object."""
        conf = JunctionConfidence(
            junction_id="chr1:1000-2000:+",
            annotation_score=1.0,
            motif_score=0.8,
            recurrence_score=0.75,
            anchor_score=0.9,
            mapq_score=0.85,
            composite_score=0.85,
        )

        self.assertEqual(conf.junction_id, "chr1:1000-2000:+")
        self.assertEqual(conf.annotation_score, 1.0)
        self.assertEqual(conf.motif_score, 0.8)
        self.assertEqual(conf.recurrence_score, 0.75)
        self.assertEqual(conf.anchor_score, 0.9)
        self.assertEqual(conf.mapq_score, 0.85)
        self.assertEqual(conf.composite_score, 0.85)

    def test_all_scores_zero(self):
        """Test confidence with all zero scores."""
        conf = JunctionConfidence(
            junction_id="chr1:1000-2000:+",
            annotation_score=0.0,
            motif_score=0.0,
            recurrence_score=0.0,
            anchor_score=0.0,
            mapq_score=0.0,
            composite_score=0.0,
        )

        self.assertEqual(conf.composite_score, 0.0)

    def test_all_scores_one(self):
        """Test confidence with all maximum scores."""
        conf = JunctionConfidence(
            junction_id="chr1:1000-2000:+",
            annotation_score=1.0,
            motif_score=1.0,
            recurrence_score=1.0,
            anchor_score=1.0,
            mapq_score=1.0,
            composite_score=1.0,
        )

        self.assertEqual(conf.composite_score, 1.0)


class TestScoreJunction(unittest.TestCase):
    """Test the score_junction function."""

    def _create_mock_evidence(
        self,
        is_annotated=True,
        motif_score=1.0,
        cross_sample_recurrence=1.0,
        max_anchor=20,
        sample_mapq_mean=None,
    ):
        """Helper to create mock JunctionEvidence."""
        if sample_mapq_mean is None:
            sample_mapq_mean = [60.0, 60.0]

        evidence = MagicMock()
        evidence.junction = MagicMock()
        evidence.junction.to_string.return_value = "chr1:1000-2000:+"
        evidence.is_annotated = is_annotated
        evidence.motif_score = motif_score
        evidence.cross_sample_recurrence = cross_sample_recurrence
        evidence.max_anchor = max_anchor
        evidence.sample_mapq_mean = np.array(sample_mapq_mean)

        return evidence

    def test_annotated_high_quality_junction(self):
        """Test scoring a high-quality annotated junction."""
        evidence = self._create_mock_evidence(
            is_annotated=True,
            motif_score=1.0,  # GT/AG
            cross_sample_recurrence=1.0,
            max_anchor=30,
            sample_mapq_mean=[60.0, 60.0],
        )

        conf = score_junction(evidence)

        self.assertEqual(conf.annotation_score, 1.0)
        self.assertEqual(conf.motif_score, 1.0)
        self.assertEqual(conf.recurrence_score, 1.0)
        self.assertAlmostEqual(conf.anchor_score, 1.0)
        self.assertAlmostEqual(conf.mapq_score, 1.0)
        self.assertGreater(conf.composite_score, 0.95)

    def test_novel_low_quality_junction(self):
        """Test scoring a low-quality novel junction."""
        evidence = self._create_mock_evidence(
            is_annotated=False,
            motif_score=0.2,  # non-canonical
            cross_sample_recurrence=0.1,
            max_anchor=5,
            sample_mapq_mean=[20.0, 20.0],
        )

        conf = score_junction(evidence)

        self.assertEqual(conf.annotation_score, 0.0)
        self.assertAlmostEqual(conf.motif_score, 0.0)
        self.assertAlmostEqual(conf.recurrence_score, 0.1)
        self.assertAlmostEqual(conf.anchor_score, 0.25)
        self.assertAlmostEqual(conf.mapq_score, 0.333, places=2)
        self.assertLess(conf.composite_score, 0.2)

    def test_annotation_score(self):
        """Test annotation score component."""
        evidence_annotated = self._create_mock_evidence(is_annotated=True)
        evidence_novel = self._create_mock_evidence(is_annotated=False)

        conf_annotated = score_junction(evidence_annotated)
        conf_novel = score_junction(evidence_novel)

        self.assertEqual(conf_annotated.annotation_score, 1.0)
        self.assertEqual(conf_novel.annotation_score, 0.0)

    def test_motif_score_normalization(self):
        """Test motif score normalization to [0, 1]."""
        # GT/AG: 1.0 -> (1.0 - 0.2) / 0.8 = 1.0
        evidence_gt_ag = self._create_mock_evidence(motif_score=1.0)
        conf_gt_ag = score_junction(evidence_gt_ag)
        self.assertAlmostEqual(conf_gt_ag.motif_score, 1.0)

        # non-canonical: 0.2 -> (0.2 - 0.2) / 0.8 = 0.0
        evidence_non_canon = self._create_mock_evidence(motif_score=0.2)
        conf_non_canon = score_junction(evidence_non_canon)
        self.assertAlmostEqual(conf_non_canon.motif_score, 0.0)

        # GC/AG: 0.8 -> (0.8 - 0.2) / 0.8 = 0.75
        evidence_gc_ag = self._create_mock_evidence(motif_score=0.8)
        conf_gc_ag = score_junction(evidence_gc_ag)
        self.assertAlmostEqual(conf_gc_ag.motif_score, 0.75)

    def test_anchor_score(self):
        """Test anchor score calculation."""
        # 20 bp anchor -> 1.0
        evidence_good = self._create_mock_evidence(max_anchor=20)
        conf_good = score_junction(evidence_good)
        self.assertAlmostEqual(conf_good.anchor_score, 1.0)

        # 10 bp anchor -> 0.5
        evidence_medium = self._create_mock_evidence(max_anchor=10)
        conf_medium = score_junction(evidence_medium)
        self.assertAlmostEqual(conf_medium.anchor_score, 0.5)

        # 5 bp anchor -> 0.25
        evidence_poor = self._create_mock_evidence(max_anchor=5)
        conf_poor = score_junction(evidence_poor)
        self.assertAlmostEqual(conf_poor.anchor_score, 0.25)

        # 40 bp anchor -> capped at 1.0
        evidence_high = self._create_mock_evidence(max_anchor=40)
        conf_high = score_junction(evidence_high)
        self.assertAlmostEqual(conf_high.anchor_score, 1.0)

    def test_mapq_score(self):
        """Test MAPQ score calculation."""
        # MAPQ 60 -> 1.0
        evidence_good = self._create_mock_evidence(sample_mapq_mean=[60.0, 60.0])
        conf_good = score_junction(evidence_good)
        self.assertAlmostEqual(conf_good.mapq_score, 1.0)

        # MAPQ 30 -> 0.5
        evidence_medium = self._create_mock_evidence(sample_mapq_mean=[30.0, 30.0])
        conf_medium = score_junction(evidence_medium)
        self.assertAlmostEqual(conf_medium.mapq_score, 0.5)

        # MAPQ 120 -> capped at 1.0
        evidence_high = self._create_mock_evidence(sample_mapq_mean=[120.0, 120.0])
        conf_high = score_junction(evidence_high)
        self.assertAlmostEqual(conf_high.mapq_score, 1.0)

    def test_custom_weights(self):
        """Test scoring with custom weights."""
        evidence = self._create_mock_evidence(
            is_annotated=True,
            motif_score=0.5,
            cross_sample_recurrence=0.5,
            max_anchor=20,
            sample_mapq_mean=[60.0],
        )

        # All weights equal
        conf = score_junction(
            evidence,
            annotation_weight=0.2,
            motif_weight=0.2,
            recurrence_weight=0.2,
            anchor_weight=0.2,
            mapq_weight=0.2,
        )

        # Composite should be average of all scores
        expected_composite = (1.0 + 0.375 + 0.5 + 1.0 + 1.0) / 5
        self.assertAlmostEqual(conf.composite_score, expected_composite, places=4)

    def test_invalid_weights(self):
        """Test that invalid weights raise error."""
        evidence = self._create_mock_evidence()

        with self.assertRaises(ValueError):
            score_junction(
                evidence,
                annotation_weight=0.5,
                motif_weight=0.5,  # Sum = 1.0 but other weights not included
            )


class TestScoreAllJunctions(unittest.TestCase):
    """Test the score_all_junctions function."""

    def test_empty_junctions(self):
        """Test scoring empty junction dict."""
        result = score_all_junctions({})
        self.assertEqual(len(result), 0)

    def test_multiple_junctions(self):
        """Test scoring multiple junctions."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=2000, end=3000, strand="+")

        evidence1 = MagicMock()
        evidence1.junction = j1
        evidence1.is_annotated = True
        evidence1.motif_score = 1.0
        evidence1.cross_sample_recurrence = 1.0
        evidence1.max_anchor = 20
        evidence1.sample_mapq_mean = np.array([60.0])

        evidence2 = MagicMock()
        evidence2.junction = j2
        evidence2.is_annotated = False
        evidence2.motif_score = 0.2
        evidence2.cross_sample_recurrence = 0.1
        evidence2.max_anchor = 5
        evidence2.sample_mapq_mean = np.array([20.0])

        junction_evidence = {j1: evidence1, j2: evidence2}

        scores = score_all_junctions(junction_evidence)

        self.assertEqual(len(scores), 2)
        self.assertIn(j1, scores)
        self.assertIn(j2, scores)
        self.assertGreater(scores[j1].composite_score, scores[j2].composite_score)


class TestFilterJunctionsByConfidence(unittest.TestCase):
    """Test the filter_junctions_by_confidence function."""

    def test_empty_junctions(self):
        """Test filtering empty junction dict."""
        result = filter_junctions_by_confidence({})
        self.assertEqual(len(result), 0)

    def test_filter_by_threshold(self):
        """Test filtering by confidence threshold."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")
        j2 = Junction(chrom="chr1", start=2000, end=3000, strand="+")

        conf1 = JunctionConfidence(
            junction_id="j1",
            annotation_score=1.0,
            motif_score=1.0,
            recurrence_score=1.0,
            anchor_score=1.0,
            mapq_score=1.0,
            composite_score=0.9,
        )

        conf2 = JunctionConfidence(
            junction_id="j2",
            annotation_score=0.0,
            motif_score=0.0,
            recurrence_score=0.0,
            anchor_score=0.0,
            mapq_score=0.0,
            composite_score=0.1,
        )

        scores = {j1: conf1, j2: conf2}

        # Filter with threshold 0.5
        result = filter_junctions_by_confidence(scores, min_score=0.5)

        self.assertEqual(len(result), 1)
        self.assertIn(j1, result)
        self.assertNotIn(j2, result)

    def test_filter_keeps_equal_threshold(self):
        """Test that junctions exactly at threshold are kept."""
        j1 = Junction(chrom="chr1", start=1000, end=2000, strand="+")

        conf = JunctionConfidence(
            junction_id="j1",
            annotation_score=0.5,
            motif_score=0.5,
            recurrence_score=0.5,
            anchor_score=0.5,
            mapq_score=0.5,
            composite_score=0.5,
        )

        scores = {j1: conf}

        result = filter_junctions_by_confidence(scores, min_score=0.5)

        self.assertEqual(len(result), 1)
        self.assertIn(j1, result)


if __name__ == "__main__":
    unittest.main()
