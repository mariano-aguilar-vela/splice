"""
Module 9: core/confidence_scorer.py

Assign continuous confidence scores to junctions based on multiple evidence types.
Combines annotation status, motif strength, cross-sample recurrence, anchor quality,
and mapping quality into a composite confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from splicekit.core.junction_extractor import JunctionEvidence
from splicekit.utils.genomic import Junction


@dataclass
class JunctionConfidence:
    """Confidence scores for a junction.

    Attributes:
        junction_id: Identifier for the junction (from to_string()).
        annotation_score: 1.0 if annotated, 0.0 if novel.
        motif_score: Splice site motif strength (0.2 to 1.0).
        recurrence_score: Cross-sample recurrence (0.0 to 1.0).
        anchor_score: Anchor quality (min(1.0, max_anchor / 20)).
        mapq_score: Mapping quality score (mean MAPQ / 60, clamped to [0, 1]).
        composite_score: Weighted combination of all scores (0.0 to 1.0).
    """

    junction_id: str
    annotation_score: float
    motif_score: float
    recurrence_score: float
    anchor_score: float
    mapq_score: float
    composite_score: float


def score_junction(
    evidence: JunctionEvidence,
    annotation_weight: float = 0.25,
    motif_weight: float = 0.25,
    recurrence_weight: float = 0.20,
    anchor_weight: float = 0.15,
    mapq_weight: float = 0.15,
) -> JunctionConfidence:
    """Compute confidence scores for a junction.

    Computes individual component scores and a weighted composite score.
    All component scores are normalized to [0, 1].

    Args:
        evidence: JunctionEvidence object containing junction statistics.
        annotation_weight: Weight for annotation score (default 0.25).
        motif_weight: Weight for motif score (default 0.25).
        recurrence_weight: Weight for recurrence score (default 0.20).
        anchor_weight: Weight for anchor score (default 0.15).
        mapq_weight: Weight for MAPQ score (default 0.15).

    Returns:
        JunctionConfidence object with all component and composite scores.
    """
    # Annotation score: 1.0 if annotated, 0.0 if novel
    annotation_score = 1.0 if evidence.is_annotated else 0.0

    # Motif score: already in [0.2, 1.0], normalize to [0, 1]
    # Score of 0.2 (non-canonical) -> 0.0, score of 1.0 (GT/AG) -> 1.0
    motif_score = (evidence.motif_score - 0.2) / (1.0 - 0.2)
    motif_score = max(0.0, min(1.0, motif_score))

    # Recurrence score: already in [0, 1]
    recurrence_score = evidence.cross_sample_recurrence

    # Anchor score: min(1.0, max_anchor / 20)
    # Minimum anchor of 20 bp gives score 1.0
    anchor_score = min(1.0, evidence.max_anchor / 20.0)

    # MAPQ score: mean of sample_mapq_mean / 60, clamped to [0, 1]
    # MAPQ of 60 gives score 1.0
    mean_mapq = np.mean(evidence.sample_mapq_mean)
    mapq_score = min(1.0, mean_mapq / 60.0)

    # Verify weights sum to 1.0
    total_weight = (
        annotation_weight
        + motif_weight
        + recurrence_weight
        + anchor_weight
        + mapq_weight
    )
    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            f"Weights must sum to 1.0, got {total_weight}"
        )

    # Composite score: weighted average
    composite_score = (
        annotation_weight * annotation_score
        + motif_weight * motif_score
        + recurrence_weight * recurrence_score
        + anchor_weight * anchor_score
        + mapq_weight * mapq_score
    )

    return JunctionConfidence(
        junction_id=evidence.junction.to_string(),
        annotation_score=annotation_score,
        motif_score=motif_score,
        recurrence_score=recurrence_score,
        anchor_score=anchor_score,
        mapq_score=mapq_score,
        composite_score=composite_score,
    )


def score_all_junctions(
    junction_evidence: Dict[Junction, JunctionEvidence],
    annotation_weight: float = 0.25,
    motif_weight: float = 0.25,
    recurrence_weight: float = 0.20,
    anchor_weight: float = 0.15,
    mapq_weight: float = 0.15,
) -> Dict[Junction, JunctionConfidence]:
    """Score all junctions.

    Args:
        junction_evidence: Dict mapping Junction -> JunctionEvidence.
        annotation_weight: Weight for annotation score.
        motif_weight: Weight for motif score.
        recurrence_weight: Weight for recurrence score.
        anchor_weight: Weight for anchor score.
        mapq_weight: Weight for MAPQ score.

    Returns:
        Dict mapping Junction -> JunctionConfidence.
    """
    confidence_scores: Dict[Junction, JunctionConfidence] = {}

    for junction, evidence in junction_evidence.items():
        score = score_junction(
            evidence,
            annotation_weight=annotation_weight,
            motif_weight=motif_weight,
            recurrence_weight=recurrence_weight,
            anchor_weight=anchor_weight,
            mapq_weight=mapq_weight,
        )
        confidence_scores[junction] = score

    return confidence_scores


def filter_junctions_by_confidence(
    confidence_scores: Dict[Junction, JunctionConfidence],
    min_score: float = 0.5,
) -> Dict[Junction, JunctionConfidence]:
    """Filter junctions by minimum composite confidence score.

    Args:
        confidence_scores: Dict mapping Junction -> JunctionConfidence.
        min_score: Minimum composite score (default 0.5).

    Returns:
        Filtered dict with only junctions >= min_score.
    """
    return {
        j: c
        for j, c in confidence_scores.items()
        if c.composite_score >= min_score
    }
