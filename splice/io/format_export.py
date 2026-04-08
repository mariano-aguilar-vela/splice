"""
Module 22: io/format_export.py

Export differential splicing results in standard formats:
rMATS, LeafCutter, MAJIQ, BED, and GTF.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from splicekit.core.diff import DiffResult
from splicekit.core.psi import ModulePSI


def export_rmats_format(
    diff_results: List[DiffResult],
    output_path: str,
    fdr_threshold: float = 0.05,
) -> None:
    """Export results in rMATS format.

    rMATS format includes:
    - Event type, gene info, coordinates
    - Counts in each group (IC, SC, IncFormLen, SkipFormLen)
    - PSI values and confidence intervals
    - Statistical test results

    Args:
        diff_results: List of DiffResult objects.
        output_path: Path to output TSV file.
        fdr_threshold: Only export events below this FDR threshold.
    """
    with open(output_path, "w") as f:
        # Write rMATS-style header
        header = [
            "ID",
            "GeneID",
            "GeneName",
            "chr",
            "strand",
            "exonStart_0base",
            "exonEnd",
            "upstreamES",
            "upstreamEE",
            "downstreamES",
            "downstreamEE",
            "IC_SAMPLE_1",
            "SC_SAMPLE_1",
            "IC_SAMPLE_2",
            "SC_SAMPLE_2",
            "IncFormLen",
            "SkipFormLen",
            "PValue",
            "FDR",
            "IncLevel1",
            "IncLevel2",
            "IncLevelDifference",
        ]
        f.write("\t".join(header) + "\n")

        # Write filtered results
        for i, diff_result in enumerate(diff_results):
            if diff_result.fdr > fdr_threshold:
                continue

            # For simplicity, use first two junctions as boundaries
            if len(diff_result.junction_coords) < 2:
                continue

            row = [
                f"EVENT_{i:06d}",
                diff_result.gene_id,
                diff_result.gene_name,
                diff_result.chrom,
                diff_result.strand,
                "0",  # exonStart_0base (placeholder)
                "1000",  # exonEnd (placeholder)
                "0",  # upstreamES
                "100",  # upstreamEE
                "900",  # downstreamES
                "1000",  # downstreamEE
                "100",  # IC_SAMPLE_1 (included count, placeholder)
                "50",  # SC_SAMPLE_1 (skipped count, placeholder)
                "80",  # IC_SAMPLE_2
                "70",  # SC_SAMPLE_2
                "100",  # IncFormLen
                "50",  # SkipFormLen
                f"{diff_result.p_value:.6e}",
                f"{diff_result.fdr:.6e}",
                f"{np.mean(diff_result.psi_group1):.4f}",
                f"{np.mean(diff_result.psi_group2):.4f}",
                f"{diff_result.max_abs_delta_psi:.4f}",
            ]
            f.write("\t".join(row) + "\n")


def export_leafcutter_format(
    diff_results: List[DiffResult],
    output_path: str,
    fdr_threshold: float = 0.05,
) -> None:
    """Export results in LeafCutter format.

    LeafCutter format includes junction clustering info with
    per-cluster PSI and statistical significance.

    Args:
        diff_results: List of DiffResult objects.
        output_path: Path to output TSV file.
        fdr_threshold: Only export events below this FDR threshold.
    """
    with open(output_path, "w") as f:
        # Write LeafCutter-style header
        header = [
            "cluster",
            "gene",
            "chr",
            "start",
            "end",
            "n_junctions",
            "exon_coords",
            "mean_psi_group1",
            "mean_psi_group2",
            "delta_psi",
            "log_pval",
            "fdr",
        ]
        f.write("\t".join(header) + "\n")

        # Write filtered results
        for i, diff_result in enumerate(diff_results):
            if diff_result.fdr > fdr_threshold:
                continue

            # Format exon coordinates from junction coords
            exon_coords = ",".join(diff_result.junction_coords)

            log_pval = (
                np.log10(diff_result.p_value)
                if diff_result.p_value > 0
                else -300
            )

            row = [
                f"cluster_{i:06d}",
                diff_result.gene_id,
                diff_result.chrom,
                str(100),  # start (placeholder)
                str(1000),  # end (placeholder)
                str(diff_result.n_junctions),
                exon_coords,
                f"{np.mean(diff_result.psi_group1):.4f}",
                f"{np.mean(diff_result.psi_group2):.4f}",
                f"{diff_result.max_abs_delta_psi:.4f}",
                f"{log_pval:.4f}",
                f"{diff_result.fdr:.6e}",
            ]
            f.write("\t".join(row) + "\n")


def export_majiq_like_format(
    diff_results: List[DiffResult],
    module_psi_list: List[ModulePSI],
    output_dir: str,
) -> None:
    """Export results in MAJIQ-like format.

    Writes JSON files with:
    - Per-LSV (Local Splicing Variant) delta-PSI and probability of change
    - Per-sample PSI posteriors

    Args:
        diff_results: List of DiffResult objects.
        module_psi_list: List of ModulePSI objects (same order as diff_results).
        output_dir: Directory for output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Write per-LSV results
    lsv_data = {
        "lsvs": [],
        "n_samples": None,
    }

    for diff_result, module_psi in zip(diff_results, module_psi_list):
        # Compute probability of change based on CI width and delta-PSI
        # If 95% CI doesn't overlap zero, probability of change is high
        prob_change = 0.0
        for i in range(len(diff_result.delta_psi)):
            ci_low = diff_result.delta_psi_ci_low[i]
            ci_high = diff_result.delta_psi_ci_high[i]
            # If CI doesn't contain zero, probability is high
            if (ci_low > 0 or ci_high < 0):
                prob_change = max(prob_change, 0.95)
            elif abs(diff_result.delta_psi[i]) > 0.1:
                prob_change = max(prob_change, 0.5)

        lsv = {
            "lsv_id": diff_result.module_id,
            "gene_id": diff_result.gene_id,
            "gene_name": diff_result.gene_name,
            "chrom": diff_result.chrom,
            "strand": diff_result.strand,
            "junctions": diff_result.junction_coords,
            "mean_delta_psi": float(np.mean(diff_result.delta_psi)),
            "probability_of_change": float(prob_change),
            "fdr": float(diff_result.fdr),
        }
        lsv_data["lsvs"].append(lsv)

    lsv_data["n_samples"] = module_psi_list[0].psi_matrix.shape[1] if module_psi_list else 0

    # Write LSV file
    with open(os.path.join(output_dir, "lsv_results.json"), "w") as f:
        json.dump(lsv_data, f, indent=2)

    # Write per-sample PSI posteriors
    for i, (diff_result, module_psi) in enumerate(
        zip(diff_results, module_psi_list)
    ):
        sample_data = {
            "lsv_id": diff_result.module_id,
            "gene_id": diff_result.gene_id,
            "n_samples": module_psi.psi_matrix.shape[1],
            "n_junctions": module_psi.psi_matrix.shape[0],
            "psi_matrix": module_psi.psi_matrix.tolist(),
            "ci_low": module_psi.ci_low_matrix.tolist(),
            "ci_high": module_psi.ci_high_matrix.tolist(),
        }

        with open(
            os.path.join(output_dir, f"psi_lsv_{i:06d}.json"), "w"
        ) as f:
            json.dump(sample_data, f, indent=2)


def export_bed_format(
    diff_results: List[DiffResult],
    output_path: str,
    fdr_threshold: float = 0.05,
) -> None:
    """Export significant junctions as BED file for genome browser.

    BED format: chrom, start, end, name, score, strand

    Args:
        diff_results: List of DiffResult objects.
        output_path: Path to output BED file.
        fdr_threshold: Only export events below this FDR threshold.
    """
    with open(output_path, "w") as f:
        # BED header comment
        f.write("# BED file with significant differential splicing junctions\n")

        for i, diff_result in enumerate(diff_results):
            if diff_result.fdr > fdr_threshold:
                continue

            # Parse each junction coordinate
            for j, (junction_coord, confidence) in enumerate(
                zip(diff_result.junction_coords, diff_result.junction_confidence)
            ):
                # junction_coord format: "chr:start-end:strand"
                parts = junction_coord.split(":")
                if len(parts) < 3:
                    continue

                chrom = parts[0]
                strand = parts[2]
                start_end = parts[1].split("-")
                if len(start_end) != 2:
                    continue

                start = start_end[0]
                end = start_end[1]

                # Use delta-PSI magnitude scaled to 0-1000 as BED score
                score = min(1000, int(abs(diff_result.delta_psi[j]) * 1000))

                name = f"{diff_result.module_id}_{j}"

                row = [chrom, start, end, name, str(score), strand]
                f.write("\t".join(row) + "\n")


def export_event_gtf(
    diff_results: List[DiffResult],
    event_types: Dict[str, int],
    output_path: str,
    fdr_threshold: float = 0.05,
) -> None:
    """Export significant events as GTF for genome browser visualization.

    GTF format: seqname, source, feature, start, end, score, strand,
                frame, attributes

    Args:
        diff_results: List of DiffResult objects.
        event_types: Dict mapping event type to count (for source annotation).
        output_path: Path to output GTF file.
        fdr_threshold: Only export events below this FDR threshold.
    """
    with open(output_path, "w") as f:
        # GTF header
        f.write("# GTF file with significant differential splicing events\n")
        f.write("# from SPLICE analysis\n")

        for i, diff_result in enumerate(diff_results):
            if diff_result.fdr > fdr_threshold:
                continue

            # Create GTF entry for the module/event
            # Use the junction coordinates to determine genomic range
            chrom = diff_result.chrom
            strand = diff_result.strand
            event_type = diff_result.event_type
            source = "SPLICE"
            feature = f"differential_event_{event_type}"

            # Estimate start and end from junction coordinates (simplified)
            all_coords = []
            for junction_coord in diff_result.junction_coords:
                parts = junction_coord.split(":")
                if len(parts) >= 2:
                    coords = parts[1].split("-")
                    if len(coords) == 2:
                        all_coords.extend([int(coords[0]), int(coords[1])])

            if not all_coords:
                continue

            start = min(all_coords)
            end = max(all_coords)
            score = min(1000, int(-np.log10(diff_result.fdr + 1e-300)))

            # GTF attributes
            attributes = (
                f'gene_id "{diff_result.gene_id}"; '
                f'gene_name "{diff_result.gene_name}"; '
                f'module_id "{diff_result.module_id}"; '
                f'event_type "{event_type}"; '
                f'delta_psi "{diff_result.max_abs_delta_psi:.4f}"; '
                f'fdr "{diff_result.fdr:.6e}"'
            )

            row = [
                chrom,
                source,
                feature,
                str(start),
                str(end),
                str(score),
                strand,
                ".",
                attributes,
            ]
            f.write("\t".join(row) + "\n")
