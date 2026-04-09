"""
Module 21: io/output_writer.py

Write differential splicing results to TSV files.
Includes per-event results, per-junction details, and summary statistics.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from splice.core.diff import DiffResult
from splice.core.diagnostics import EventDiagnostic
from splice.core.nmd_classifier import NMDClassification
from splice.utils.genomic import Junction


def write_results_tsv(
    diff_results: List[DiffResult],
    diagnostics: List[EventDiagnostic],
    output_path: str,
) -> None:
    """Write per-event differential splicing results to TSV.

    Combines DiffResult and EventDiagnostic into a single results table.

    Args:
        diff_results: List of DiffResult objects.
        diagnostics: List of EventDiagnostic objects (same order as diff_results).
        output_path: Path to output TSV file.
    """
    with open(output_path, "w") as f:
        # Write header
        header = [
            "module_id",
            "gene_id",
            "gene_name",
            "chrom",
            "strand",
            "event_type",
            "n_junctions",
            "max_abs_delta_psi",
            "psi_group1_mean",
            "psi_group2_mean",
            "log_likelihood_null",
            "log_likelihood_full",
            "degrees_of_freedom",
            "p_value",
            "fdr",
            "confidence_tier",
            "null_converged",
            "full_converged",
            "mean_mapq",
            "frac_high_mapq",
            "min_group_total_reads",
            "effective_n_min",
            "mean_junction_confidence",
            "bootstrap_cv",
            "has_novel_junctions",
            "has_convergence_issue",
            "reason",
        ]
        f.write("\t".join(header) + "\n")

        # Write rows
        for diff_result, diagnostic in zip(diff_results, diagnostics):
            psi_group1_mean = np.mean(diff_result.psi_group1)
            psi_group2_mean = np.mean(diff_result.psi_group2)

            row = [
                diff_result.module_id,
                diff_result.gene_id,
                diff_result.gene_name,
                diff_result.chrom,
                diff_result.strand,
                diff_result.event_type,
                str(diff_result.n_junctions),
                f"{diff_result.max_abs_delta_psi:.6f}",
                f"{psi_group1_mean:.6f}",
                f"{psi_group2_mean:.6f}",
                f"{diff_result.log_likelihood_null:.6f}",
                f"{diff_result.log_likelihood_full:.6f}",
                str(diff_result.degrees_of_freedom),
                f"{diff_result.p_value:.6e}",
                f"{diff_result.fdr:.6e}",
                diagnostic.confidence_tier,
                str(diagnostic.null_converged),
                str(diagnostic.full_converged),
                f"{diagnostic.mean_mapq:.2f}",
                f"{diagnostic.frac_high_mapq:.4f}",
                f"{diagnostic.min_group_total_reads:.1f}",
                f"{diagnostic.effective_n_min:.2f}",
                f"{diagnostic.mean_junction_confidence:.4f}",
                f"{diagnostic.bootstrap_cv:.4f}",
                str(diagnostic.has_novel_junctions),
                str(diagnostic.has_convergence_issue),
                diagnostic.reason,
            ]
            f.write("\t".join(row) + "\n")


def write_junction_details_tsv(
    junction_evidence: Dict[str, dict],
    junction_confidence: Dict[str, float],
    nmd_classifications: Dict[str, NMDClassification],
    output_path: str,
) -> None:
    """Write per-junction detail file.

    Args:
        junction_evidence: Dict mapping junction_id to evidence dict with keys:
            'junction' (Junction object), 'gene_id', 'gene_name', 'is_annotated',
            'motif', 'motif_score', 'total_reads', 'mean_mapq', 'sample_counts' (list).
        junction_confidence: Dict mapping junction_id to confidence score.
        nmd_classifications: Dict mapping junction_id to NMDClassification.
        output_path: Path to output TSV file.
    """
    with open(output_path, "w") as f:
        # Write header
        header = [
            "junction_id",
            "chrom",
            "start",
            "end",
            "strand",
            "gene_id",
            "gene_name",
            "is_annotated",
            "motif",
            "motif_score",
            "confidence_score",
            "nmd_class",
            "nmd_confidence",
            "n_productive_paths",
            "n_unproductive_paths",
            "total_reads",
            "mean_mapq",
            "cross_sample_recurrence",
            "sample_counts",
        ]
        f.write("\t".join(header) + "\n")

        # Write rows
        for junction_id, evidence in junction_evidence.items():
            junction: Junction = evidence["junction"]
            sample_counts: List[int] = evidence.get("sample_counts", [])
            confidence = junction_confidence.get(junction_id, np.nan)
            nmd_class = nmd_classifications.get(junction_id)

            # Compute cross-sample recurrence (count of non-zero sample counts)
            cross_sample_recurrence = sum(1 for count in sample_counts if count > 0)

            # Format sample counts as comma-separated string
            sample_counts_str = ",".join(str(c) for c in sample_counts)

            nmd_class_str = (
                nmd_class.classification if nmd_class else "NA"
            )
            nmd_confidence_str = (
                f"{nmd_class.confidence:.4f}" if nmd_class else "NA"
            )
            n_productive_str = (
                str(nmd_class.n_productive_paths) if nmd_class else "NA"
            )
            n_unproductive_str = (
                str(nmd_class.n_unproductive_paths) if nmd_class else "NA"
            )

            row = [
                junction_id,
                junction.chrom,
                str(junction.start),
                str(junction.end),
                junction.strand,
                evidence.get("gene_id", "NA"),
                evidence.get("gene_name", "NA"),
                str(evidence.get("is_annotated", False)),
                evidence.get("motif", "NA"),
                f"{evidence.get('motif_score', np.nan):.4f}",
                f"{confidence:.4f}",
                nmd_class_str,
                nmd_confidence_str,
                n_productive_str,
                n_unproductive_str,
                f"{evidence.get('total_reads', 0):.1f}",
                f"{evidence.get('mean_mapq', np.nan):.2f}",
                str(cross_sample_recurrence),
                sample_counts_str,
            ]
            f.write("\t".join(row) + "\n")


def write_summary_tsv(
    diff_results: List[DiffResult],
    diagnostics: List[EventDiagnostic],
    event_types: Dict[str, int],
    output_path: str,
) -> None:
    """Write summary statistics to TSV.

    Args:
        diff_results: List of DiffResult objects.
        diagnostics: List of EventDiagnostic objects.
        event_types: Dict mapping event type to count.
        output_path: Path to output TSV file.
    """
    with open(output_path, "w") as f:
        f.write("Metric\tValue\n")

        # Overall statistics
        n_total = len(diff_results)
        f.write(f"Total events\t{n_total}\n")

        # Event type breakdown
        for event_type, count in sorted(event_types.items()):
            f.write(f"{event_type}\t{count}\n")

        # Significance by FDR threshold
        fdr_05 = sum(1 for dr in diff_results if dr.fdr < 0.05)
        fdr_01 = sum(1 for dr in diff_results if dr.fdr < 0.01)
        f.write(f"Significant (FDR < 0.05)\t{fdr_05}\n")
        f.write(f"Significant (FDR < 0.01)\t{fdr_01}\n")

        # Confidence tier breakdown
        tier_counts = {}
        for diag in diagnostics:
            tier = diag.confidence_tier
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        for tier in sorted(tier_counts.keys()):
            f.write(f"Confidence {tier}\t{tier_counts[tier]}\n")

        # Quality metrics (averages)
        if diagnostics:
            mean_mapq = np.mean([d.mean_mapq for d in diagnostics])
            mean_mapq_annotated = np.mean(
                [d.mean_junction_confidence for d in diagnostics]
            )
            mean_bootstrap_cv = np.mean([d.bootstrap_cv for d in diagnostics])

            f.write(f"Mean MAPQ (average)\t{mean_mapq:.2f}\n")
            f.write(
                f"Mean junction confidence (average)\t{mean_mapq_annotated:.4f}\n"
            )
            f.write(f"Mean bootstrap CV (average)\t{mean_bootstrap_cv:.4f}\n")

        # Convergence statistics
        null_converged = sum(1 for d in diagnostics if d.null_converged)
        full_converged = sum(1 for d in diagnostics if d.full_converged)
        f.write(f"Null model converged\t{null_converged}/{len(diagnostics)}\n")
        f.write(f"Full model converged\t{full_converged}/{len(diagnostics)}\n")
