"""
Sashimi plot generation for differential splicing visualization.

Produces publication-quality sashimi plots showing read coverage,
gene structure, and junction usage between two sample groups.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

try:
    import pysam
except ImportError:
    pysam = None


def get_coverage_for_region(
    bam_path: str, chrom: str, start: int, end: int,
) -> np.ndarray:
    """Compute per-base read coverage across a region.

    Args:
        bam_path: Path to indexed BAM file.
        chrom: Chromosome name.
        start: Start position (0-based).
        end: End position (0-based, exclusive).

    Returns:
        Coverage array of shape (end - start,).
    """
    if pysam is None:
        raise ImportError("pysam is required for coverage extraction")

    length = end - start
    coverage = np.zeros(length, dtype=int)

    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for col in bam.pileup(chrom, start, end, truncate=True):
                pos = col.reference_pos - start
                if 0 <= pos < length:
                    coverage[pos] = col.nsegments
    except (ValueError, OSError):
        return coverage

    return coverage


def get_junction_reads(
    bam_path: str, chrom: str, start: int, end: int, min_anchor: int = 6,
) -> List[Tuple[int, int, int]]:
    """Extract junction-spanning reads in a region.

    Args:
        bam_path: Path to indexed BAM file.
        chrom: Chromosome name.
        start: Region start (0-based).
        end: Region end (0-based, exclusive).
        min_anchor: Minimum anchor length on each side of junction.

    Returns:
        List of (junction_start, junction_end, read_count) tuples.
    """
    if pysam is None:
        raise ImportError("pysam is required for junction extraction")

    junction_counts: Dict[Tuple[int, int], int] = {}

    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for read in bam.fetch(chrom, start, end):
                if read.is_unmapped or read.is_secondary or read.is_duplicate:
                    continue

                blocks = read.get_blocks()
                for i in range(len(blocks) - 1):
                    intron_start = blocks[i][1]
                    intron_end = blocks[i + 1][0]
                    if intron_end <= intron_start:
                        continue

                    left_anchor = blocks[i][1] - blocks[i][0]
                    right_anchor = blocks[i + 1][1] - blocks[i + 1][0]
                    if left_anchor < min_anchor or right_anchor < min_anchor:
                        continue

                    # Only count junctions overlapping the region
                    if intron_end < start or intron_start > end:
                        continue

                    key = (intron_start, intron_end)
                    junction_counts[key] = junction_counts.get(key, 0) + 1
    except (ValueError, OSError):
        return []

    return [(s, e, c) for (s, e), c in junction_counts.items()]


def draw_gene_model(
    ax, exons: List[Tuple[int, int]], strand: str, y_position: float = 0.0,
    color: str = "#34495E",
):
    """Draw exon rectangles and intron lines.

    Args:
        ax: Matplotlib axis.
        exons: List of (start, end) exon tuples.
        strand: '+' or '-' for strand direction.
        y_position: Vertical center for the gene model.
        color: Color for exons and introns.
    """
    if not exons:
        return

    sorted_exons = sorted(exons, key=lambda e: e[0])
    exon_height = 0.3

    for start, end in sorted_exons:
        rect = Rectangle(
            (start, y_position - exon_height / 2),
            end - start, exon_height,
            facecolor=color, edgecolor=color, linewidth=0.5,
        )
        ax.add_patch(rect)

    # Draw intron lines between consecutive exons
    for i in range(len(sorted_exons) - 1):
        intron_start = sorted_exons[i][1]
        intron_end = sorted_exons[i + 1][0]
        if intron_end > intron_start:
            ax.plot(
                [intron_start, intron_end], [y_position, y_position],
                color=color, linewidth=1, zorder=0,
            )

            # Strand arrow
            mid = (intron_start + intron_end) / 2
            arrow_dx = (intron_end - intron_start) * 0.05
            if strand == "+":
                ax.annotate(
                    "", xy=(mid + arrow_dx, y_position),
                    xytext=(mid - arrow_dx, y_position),
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                )
            elif strand == "-":
                ax.annotate(
                    "", xy=(mid - arrow_dx, y_position),
                    xytext=(mid + arrow_dx, y_position),
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                )


def draw_coverage_track(
    ax, coverage: np.ndarray, start: int,
    color: str = "#3498DB", label: str = "",
):
    """Draw filled coverage plot.

    Args:
        ax: Matplotlib axis.
        coverage: Coverage array.
        start: Genomic start position.
        color: Fill color.
        label: Label for the track.
    """
    if len(coverage) == 0:
        return

    x = np.arange(start, start + len(coverage))
    ax.fill_between(x, 0, coverage, color=color, alpha=0.7, linewidth=0)
    ax.plot(x, coverage, color=color, linewidth=0.5)

    if label:
        ax.text(
            0.01, 0.95, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", color=color,
        )

    ax.set_ylim(0, max(coverage.max() * 1.1, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Coverage", fontsize=9)


def draw_junction_arcs(
    ax, junctions: List[Tuple[int, int, int]],
    y_base: float, color: str = "#2C3E50",
):
    """Draw curved arcs connecting junction donors to acceptors.

    Arc height is proportional to log(read_count). Thicker arcs for
    more reads.

    Args:
        ax: Matplotlib axis.
        junctions: List of (junction_start, junction_end, read_count).
        y_base: Y-coordinate where the arcs originate (top of coverage).
        color: Arc color.
    """
    if not junctions:
        return

    max_count = max(c for _, _, c in junctions) if junctions else 1
    y_range = ax.get_ylim()
    y_max = y_range[1]

    for j_start, j_end, count in junctions:
        if count < 1:
            continue

        # Arc height proportional to log
        height_frac = np.log1p(count) / np.log1p(max_count)
        arc_height = y_base + (y_max - y_base) * 0.6 * height_frac

        # Bezier curve
        verts = [
            (j_start, y_base),
            (j_start, arc_height),
            (j_end, arc_height),
            (j_end, y_base),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        path = MplPath(verts, codes)

        # Line width proportional to count
        lw = 0.5 + 2.5 * height_frac
        patch = PathPatch(
            path, facecolor="none", edgecolor=color,
            linewidth=lw, alpha=0.7,
        )
        ax.add_patch(patch)

        # Read count label at arc apex
        mid_x = (j_start + j_end) / 2
        label_y = y_base + (arc_height - y_base) * 0.85
        ax.text(
            mid_x, label_y, str(count),
            ha="center", va="bottom", fontsize=8, color=color,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=color, linewidth=0.5),
        )


def _format_title(gene_name, event_type, delta_psi, fdr):
    """Format the sashimi plot title."""
    return (
        f"{gene_name}  |  {event_type}  |  "
        f"$\\Delta\\Psi$ = {delta_psi:.3f}  |  FDR = {fdr:.2e}"
    )


def generate_sashimi_plot(
    bam_paths_group1: List[str],
    bam_paths_group2: List[str],
    chrom: str,
    start: int,
    end: int,
    exons: List[Tuple[int, int]],
    strand: str,
    gene_name: str,
    event_type: str,
    delta_psi: float,
    fdr: float,
    output_path: str,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
) -> None:
    """Create a multi-panel sashimi plot.

    Top panel: Group 1 mean coverage with junction arcs (blue)
    Middle panel: Gene model
    Bottom panel: Group 2 mean coverage with junction arcs (red)

    Args:
        bam_paths_group1: BAM files for group 1.
        bam_paths_group2: BAM files for group 2.
        chrom: Chromosome.
        start: Region start.
        end: Region end.
        exons: List of (start, end) exon tuples for the gene.
        strand: '+' or '-'.
        gene_name: Gene symbol for the title.
        event_type: Event type (SE, A3SS, etc.).
        delta_psi: Effect size.
        fdr: FDR-corrected p-value.
        output_path: Output path (without extension; saves SVG and PNG).
        group1_name: Label for group 1.
        group2_name: Label for group 2.
    """
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[3, 1, 3], hspace=0.05,
    )
    ax_top = fig.add_subplot(gs[0])
    ax_gene = fig.add_subplot(gs[1], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[2], sharex=ax_top)

    # Compute mean coverage and junction counts per group
    def _mean_coverage(bam_paths):
        if not bam_paths:
            return np.zeros(end - start)
        cov_arrays = [
            get_coverage_for_region(bp, chrom, start, end)
            for bp in bam_paths
        ]
        return np.mean(cov_arrays, axis=0)

    def _summed_junctions(bam_paths):
        merged = {}
        for bp in bam_paths:
            for j_start, j_end, count in get_junction_reads(bp, chrom, start, end):
                key = (j_start, j_end)
                merged[key] = merged.get(key, 0) + count
        # Average per sample
        n = max(len(bam_paths), 1)
        return [(s, e, c // n) for (s, e), c in merged.items() if c // n > 0]

    cov1 = _mean_coverage(bam_paths_group1)
    cov2 = _mean_coverage(bam_paths_group2)
    junc1 = _summed_junctions(bam_paths_group1)
    junc2 = _summed_junctions(bam_paths_group2)

    # Top: Group 1
    color1 = "#3498DB"
    draw_coverage_track(ax_top, cov1, start, color=color1, label=group1_name)
    draw_junction_arcs(ax_top, junc1, y_base=cov1.max() if len(cov1) else 0, color=color1)
    ax_top.set_xticks([])

    # Middle: Gene model
    ax_gene.set_ylim(-0.5, 0.5)
    draw_gene_model(ax_gene, exons, strand, y_position=0.0)
    ax_gene.set_yticks([])
    ax_gene.spines["top"].set_visible(False)
    ax_gene.spines["right"].set_visible(False)
    ax_gene.spines["left"].set_visible(False)
    ax_gene.set_xticks([])

    # Bottom: Group 2 (inverted)
    color2 = "#E74C3C"
    draw_coverage_track(ax_bot, cov2, start, color=color2, label=group2_name)
    draw_junction_arcs(ax_bot, junc2, y_base=cov2.max() if len(cov2) else 0, color=color2)
    ax_bot.set_xlabel(f"{chrom}:{start:,}-{end:,}", fontsize=10)
    ax_bot.invert_yaxis()  # Mirror group 2 below

    # Title
    fig.suptitle(
        _format_title(gene_name, event_type, delta_psi, fdr),
        fontsize=12, fontweight="bold", y=0.98,
    )

    # Save SVG and PNG
    base, _ = os.path.splitext(output_path)
    fig.savefig(f"{base}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{base}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _gene_exons_from_gtf(genes_dict, gene_id: str) -> List[Tuple[int, int]]:
    """Extract a deduplicated exon list from a parsed GTF gene."""
    if gene_id not in genes_dict:
        # Try without version suffix
        for gid, gene in genes_dict.items():
            if gid.split(".")[0] == gene_id.split(".")[0]:
                return _gene_exons_from_gene(gene)
        return []
    return _gene_exons_from_gene(genes_dict[gene_id])


def _gene_exons_from_gene(gene) -> List[Tuple[int, int]]:
    """Get unique exons from a Gene object."""
    seen = set()
    exons = []
    for tx_id, tx_exons in gene.transcripts.items():
        for ex in tx_exons:
            if ex not in seen:
                seen.add(ex)
                exons.append(ex)
    return sorted(exons)


def generate_top_sashimi_plots(
    splice_results_path: str,
    bam_paths_group1: List[str],
    bam_paths_group2: List[str],
    gtf_path: str,
    output_dir: str,
    n_top: int = 20,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
) -> None:
    """Generate sashimi plots for top significant events.

    Loads results, sorts by FDR, and creates one sashimi plot per gene
    for the top n_top events.

    Args:
        splice_results_path: Path to splice_results.tsv.
        bam_paths_group1: BAM files for group 1.
        bam_paths_group2: BAM files for group 2.
        gtf_path: Path to GTF for exon structure.
        output_dir: Output directory for sashimi plots.
        n_top: Number of top events to plot.
        group1_name: Label for group 1.
        group2_name: Label for group 2.
    """
    import pandas as pd
    from splice.core.gtf_parser import parse_gtf

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(splice_results_path, sep="\t")
    df = df.sort_values("fdr").head(n_top)

    genes = parse_gtf(gtf_path, gene_type_filter="protein_coding")

    for _, row in df.iterrows():
        gene_id = row["gene_id"]
        gene_name = row.get("gene_name", gene_id)

        gene = genes.get(gene_id)
        if gene is None:
            for gid, g in genes.items():
                if gid.split(".")[0] == str(gene_id).split(".")[0]:
                    gene = g
                    break
        if gene is None:
            continue

        exons = _gene_exons_from_gene(gene)
        if not exons:
            continue

        chrom = gene.chrom
        start = max(0, min(e[0] for e in exons) - 200)
        end = max(e[1] for e in exons) + 200

        delta_psi = row.get("max_abs_delta_psi", row.get("delta_psi", 0.0))
        fdr = row.get("fdr", 1.0)
        event_type = row.get("event_type", "Unknown")

        safe_name = "".join(c if c.isalnum() else "_" for c in str(gene_name))
        output_path = os.path.join(output_dir, f"sashimi_{safe_name}_{event_type}")

        try:
            generate_sashimi_plot(
                bam_paths_group1=bam_paths_group1,
                bam_paths_group2=bam_paths_group2,
                chrom=chrom, start=start, end=end,
                exons=exons, strand=gene.strand,
                gene_name=gene_name, event_type=event_type,
                delta_psi=float(delta_psi), fdr=float(fdr),
                output_path=output_path,
                group1_name=group1_name, group2_name=group2_name,
            )
        except Exception as e:
            print(f"  Warning: failed to plot {gene_name}: {e}")
