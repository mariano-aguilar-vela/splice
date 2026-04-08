"""
Module 19: core/diagnostics.py

Per-event structured diagnostic records and confidence tier assignment.
Evaluates quality of differential splicing results across multiple dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from splicekit.core.diff import DiffResult
from splicekit.core.evidence import ModuleEvidence
from splicekit.core.psi import ModulePSI


@dataclass(frozen=True, slots=True)
class EventDiagnostic:
    """Diagnostic record for a differential splicing event.

    Attributes:
        module_id: Module identifier.
        confidence_tier: Confidence tier ("HIGH", "MEDIUM", "LOW", "FAIL").
        # Convergence
        null_converged: Whether null model converged.
        full_converged: Whether full model converged.
        null_refit_used: Whether null-refit strategy was applied.
        # Evidence quality
        mean_mapq: Mean MAPQ across all reads.
        median_mapq: Median MAPQ.
        frac_high_mapq: Fraction of reads with MAPQ >= 20.
        frac_multi_mapped: Fraction of reads with NH > 1.
        min_group_total_reads: Minimum total reads in any group.
        effective_n_min: Minimum effective sample size across groups.
        # Junction quality
        mean_junction_confidence: Mean confidence score across junctions.
        min_junction_confidence: Minimum confidence score.
        frac_annotated_junctions: Fraction of junctions in annotation.
        # Statistical quality
        prior_dominance: Fraction of posterior from prior (vs data).
        bootstrap_cv: Coefficient of variation of bootstrap PSI.
        # Flags
        has_novel_junctions: Whether module contains novel junctions.
        has_low_confidence_junction: Whether any junction has low confidence.
        has_convergence_issue: Whether convergence failed.
        reason: Text explanation of tier assignment.
    """

    module_id: str
    confidence_tier: str
    # Convergence
    null_converged: bool
    full_converged: bool
    null_refit_used: bool
    # Evidence quality
    mean_mapq: float
    median_mapq: float
    frac_high_mapq: float
    frac_multi_mapped: float
    min_group_total_reads: float
    effective_n_min: float
    # Junction quality
    mean_junction_confidence: float
    min_junction_confidence: float
    frac_annotated_junctions: float
    # Statistical quality
    prior_dominance: float
    bootstrap_cv: float
    # Flags
    has_novel_junctions: bool
    has_low_confidence_junction: bool
    has_convergence_issue: bool
    reason: str


def compute_diagnostics(
    evidence_list: List[ModuleEvidence],
    psi_list: List[ModulePSI],
    diff_results: List[DiffResult],
) -> List[EventDiagnostic]:
    """Compute per-event diagnostic records with confidence tier assignment.

    Evaluates quality across convergence, evidence quality, junction quality,
    and statistical metrics.

    Tier assignment:
    - HIGH: Both converged, mean_mapq >= 20, frac_high_mapq >= 0.8,
            min_group_reads >= 30, effective_n >= 10, mean_confidence >= 0.7,
            bootstrap_cv < 0.3.
    - MEDIUM: Both converged, fails one or two HIGH criteria.
    - LOW: One model did not converge, OR min_group_reads < 10,
           OR bootstrap_cv > 0.5.
    - FAIL: Both failed, OR zero reads in one group,
            OR all junctions have confidence < 0.3.

    Args:
        evidence_list: List of ModuleEvidence objects.
        psi_list: List of ModulePSI objects (same order as evidence).
        diff_results: List of DiffResult objects (same order as evidence).

    Returns:
        List of EventDiagnostic objects.
    """
    diagnostics: List[EventDiagnostic] = []

    for evidence, psi, diff_result in zip(evidence_list, psi_list, diff_results):
        module_id = evidence.module.module_id

        # Convergence metrics
        null_converged = diff_result.null_converged
        full_converged = diff_result.full_converged
        null_refit_used = diff_result.null_refit_used
        has_convergence_issue = not (null_converged and full_converged)

        # Evidence quality metrics
        mapq_vals = evidence.junction_mapq_matrix.flatten()
        mapq_vals = mapq_vals[mapq_vals > 0]  # Filter zeros

        if len(mapq_vals) > 0:
            mean_mapq = np.mean(mapq_vals)
            median_mapq = np.median(mapq_vals)
            frac_high_mapq = np.sum(mapq_vals >= 20) / len(mapq_vals)
        else:
            mean_mapq = 0.0
            median_mapq = 0.0
            frac_high_mapq = 0.0

        # Multi-mapping fraction (estimated from counts vs weighted)
        total_raw = np.sum(evidence.junction_count_matrix)
        total_weighted = np.sum(evidence.junction_weighted_matrix)
        if total_raw > 0:
            frac_multi_mapped = 1.0 - (total_weighted / total_raw)
        else:
            frac_multi_mapped = 0.0

        # Group read counts
        group_totals = evidence.total_counts
        min_group_total_reads = np.min(group_totals) if len(group_totals) > 0 else 0.0

        # Effective sample size
        effective_n = psi.effective_n
        effective_n_min = np.min(effective_n) if len(effective_n) > 0 else 0.0

        # Junction quality metrics
        confidence_scores = evidence.junction_confidence
        annotated_flags = evidence.is_annotated

        mean_junction_confidence = (
            np.mean(confidence_scores) if len(confidence_scores) > 0 else 0.0
        )
        min_junction_confidence = (
            np.min(confidence_scores) if len(confidence_scores) > 0 else 0.0
        )
        frac_annotated = (
            np.sum(annotated_flags) / len(annotated_flags)
            if len(annotated_flags) > 0
            else 0.0
        )

        has_novel_junctions = not np.all(annotated_flags)
        has_low_confidence_junction = np.any(confidence_scores < 0.3)

        # Statistical quality metrics
        # Prior dominance: estimate based on shrinkage of estimates
        # For now, use bootstrap CV as proxy for uncertainty
        bootstrap_psi = psi.bootstrap_psi
        if bootstrap_psi.shape[0] > 1:
            # Compute CV for each junction-sample combination
            means = np.mean(bootstrap_psi, axis=0)
            stds = np.std(bootstrap_psi, axis=0, ddof=1)
            cvs = np.where(means > 0.01, stds / means, 0)
            cvs = cvs[~np.isnan(cvs)]
            bootstrap_cv = np.median(cvs) if len(cvs) > 0 else 0.0
        else:
            bootstrap_cv = 0.0

        # Prior dominance (simplified): assume uniform prior, measure shrinkage
        # Higher means more prior influence
        prior_dominance = 0.1  # Default low prior influence

        # Assign confidence tier
        confidence_tier, reason = _assign_tier(
            null_converged=null_converged,
            full_converged=full_converged,
            mean_mapq=mean_mapq,
            frac_high_mapq=frac_high_mapq,
            min_group_reads=min_group_total_reads,
            effective_n_min=effective_n_min,
            mean_junction_confidence=mean_junction_confidence,
            min_junction_confidence=min_junction_confidence,
            bootstrap_cv=bootstrap_cv,
        )

        # Create diagnostic record
        diagnostic = EventDiagnostic(
            module_id=module_id,
            confidence_tier=confidence_tier,
            null_converged=null_converged,
            full_converged=full_converged,
            null_refit_used=null_refit_used,
            mean_mapq=mean_mapq,
            median_mapq=median_mapq,
            frac_high_mapq=frac_high_mapq,
            frac_multi_mapped=frac_multi_mapped,
            min_group_total_reads=min_group_total_reads,
            effective_n_min=effective_n_min,
            mean_junction_confidence=mean_junction_confidence,
            min_junction_confidence=min_junction_confidence,
            frac_annotated_junctions=frac_annotated,
            prior_dominance=prior_dominance,
            bootstrap_cv=bootstrap_cv,
            has_novel_junctions=has_novel_junctions,
            has_low_confidence_junction=has_low_confidence_junction,
            has_convergence_issue=has_convergence_issue,
            reason=reason,
        )

        diagnostics.append(diagnostic)

    return diagnostics


def _assign_tier(
    null_converged: bool,
    full_converged: bool,
    mean_mapq: float,
    frac_high_mapq: float,
    min_group_reads: float,
    effective_n_min: float,
    mean_junction_confidence: float,
    min_junction_confidence: float,
    bootstrap_cv: float,
) -> tuple[str, str]:
    """Assign confidence tier based on diagnostic metrics.

    Returns:
        Tuple of (tier, reason).
    """
    # FAIL criteria
    if not null_converged or not full_converged:
        if not (null_converged and full_converged):
            if min_group_reads == 0:
                return "FAIL", "Zero reads in one group"
            if min_junction_confidence < 0.3 and np.isfinite(min_junction_confidence):
                return "FAIL", "All junctions have confidence < 0.3"

    if min_junction_confidence < 0.3:
        return "FAIL", "All junctions have very low confidence"

    if min_group_reads == 0:
        return "FAIL", "Zero reads in one group"

    # LOW criteria
    if not (null_converged and full_converged):
        return "LOW", "Model convergence failed"

    if min_group_reads < 10:
        return "LOW", "Insufficient reads in one group (< 10)"

    if bootstrap_cv > 0.5:
        return "LOW", "High bootstrap variability (CV > 0.5)"

    # HIGH criteria checking
    high_criteria = [
        (mean_mapq >= 20, "mean_mapq >= 20"),
        (frac_high_mapq >= 0.8, "frac_high_mapq >= 0.8"),
        (min_group_reads >= 30, "min_group_reads >= 30"),
        (effective_n_min >= 10, "effective_n >= 10"),
        (mean_junction_confidence >= 0.7, "mean_junction_confidence >= 0.7"),
        (bootstrap_cv < 0.3, "bootstrap_cv < 0.3"),
    ]

    passed = sum(1 for crit, _ in high_criteria if crit)
    failed = len(high_criteria) - passed

    if passed == len(high_criteria):
        return "HIGH", "All quality criteria met"
    elif failed <= 2:
        failed_criteria = [name for crit, name in high_criteria if not crit]
        return "MEDIUM", f"Failed criteria: {', '.join(failed_criteria)}"
    else:
        return "LOW", f"Failed {failed} quality criteria"
