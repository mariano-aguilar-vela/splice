"""
Publication-quality PDF report for SPLICE differential splicing results.

Generates a multi-page PDF with volcano plots, event type distributions,
confidence tier charts, and top significant events table.
Individual figures are also saved as SVG files.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from splice.core.diff import DiffResult
from splice.core.diagnostics import EventDiagnostic


# Color palette for event types
EVENT_COLORS = {
    "SE": "#E74C3C",
    "A3SS": "#3498DB",
    "A5SS": "#2ECC71",
    "MXE": "#9B59B6",
    "RI": "#F39C12",
    "TandemCassette": "#1ABC9C",
    "Complex": "#95A5A6",
}


def _save_svg(fig, output_dir, filename):
    """Save figure as SVG in the figures subdirectory."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(
        os.path.join(figures_dir, filename),
        format="svg", bbox_inches="tight", dpi=150,
    )


def _make_title_page(fig, sample_info, parameters):
    """Create the title page."""
    fig.clear()
    ax = fig.add_subplot(111)
    ax.axis("off")

    title_text = "SPLICE\nDifferential Splicing Analysis Report"
    ax.text(0.5, 0.75, title_text, transform=ax.transAxes,
            fontsize=24, fontweight="bold", ha="center", va="center",
            color="#2C3E50")

    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    info_lines = [f"Generated: {date_str}", ""]
    if sample_info:
        for key, val in sample_info.items():
            info_lines.append(f"{key}: {val}")
    if parameters:
        info_lines.append("")
        for key, val in parameters.items():
            info_lines.append(f"{key}: {val}")

    info_text = "\n".join(info_lines)
    ax.text(0.5, 0.35, info_text, transform=ax.transAxes,
            fontsize=12, ha="center", va="center", color="#34495E",
            family="monospace")


def _make_volcano_plot(fig, diff_results, output_dir):
    """Create volcano plot: delta-PSI vs -log10(FDR)."""
    fig.clear()
    ax = fig.add_subplot(111)

    if not diff_results:
        ax.text(0.5, 0.5, "No differential results", ha="center", va="center")
        return

    delta_psi = np.array([dr.max_abs_delta_psi for dr in diff_results])
    fdr = np.array([dr.fdr for dr in diff_results])
    neg_log_fdr = -np.log10(np.maximum(fdr, 1e-300))
    event_types = [dr.event_type for dr in diff_results]

    # Determine sign of delta_psi from first junction
    signed_delta = []
    for dr in diff_results:
        if len(dr.delta_psi) > 0:
            max_idx = np.argmax(np.abs(dr.delta_psi))
            signed_delta.append(dr.delta_psi[max_idx])
        else:
            signed_delta.append(0.0)
    signed_delta = np.array(signed_delta)

    # Plot by event type
    for evt in sorted(set(event_types)):
        mask = np.array([e == evt for e in event_types])
        color = EVENT_COLORS.get(evt, "#95A5A6")
        ax.scatter(
            signed_delta[mask], neg_log_fdr[mask],
            c=color, label=evt, alpha=0.6, s=15, edgecolors="none",
        )

    # Significance thresholds
    ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(-np.log10(0.01), color="darkred", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(0.1, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.axvline(-0.1, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Delta PSI", fontsize=12)
    ax.set_ylabel("-log10(FDR)", fontsize=12)
    ax.set_title("Volcano Plot: Differential Splicing", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_svg(fig, output_dir, "volcano_plot.svg")


def _make_event_type_charts(fig, event_type_counts, diff_results, output_dir):
    """Create event type pie chart and significant events bar chart."""
    fig.clear()

    # Pie chart
    ax1 = fig.add_subplot(121)
    types = sorted(event_type_counts.keys())
    counts = [event_type_counts[t] for t in types]
    colors = [EVENT_COLORS.get(t, "#95A5A6") for t in types]

    if sum(counts) > 0:
        ax1.pie(counts, labels=types, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 9})
        ax1.set_title("Event Type Distribution", fontsize=12, fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "No events", ha="center", va="center")

    # Bar chart of significant events per type
    ax2 = fig.add_subplot(122)
    sig_by_type = {}
    for dr in diff_results:
        if dr.fdr < 0.05:
            sig_by_type[dr.event_type] = sig_by_type.get(dr.event_type, 0) + 1

    if sig_by_type:
        sig_types = sorted(sig_by_type.keys())
        sig_counts = [sig_by_type[t] for t in sig_types]
        sig_colors = [EVENT_COLORS.get(t, "#95A5A6") for t in sig_types]
        bars = ax2.bar(sig_types, sig_counts, color=sig_colors, edgecolor="white")
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("Significant Events (FDR < 0.05)", fontsize=12, fontweight="bold")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for bar, count in zip(bars, sig_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(count), ha="center", va="bottom", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No significant events", ha="center", va="center",
                 transform=ax2.transAxes)
        ax2.set_title("Significant Events (FDR < 0.05)", fontsize=12, fontweight="bold")

    fig.tight_layout()
    _save_svg(fig, output_dir, "event_types.svg")


def _make_diagnostics_chart(fig, diagnostics, diff_results, output_dir):
    """Create confidence tier distribution and convergence chart."""
    fig.clear()

    # Confidence tier bar chart
    ax1 = fig.add_subplot(121)
    if diagnostics:
        tier_counts = {}
        for d in diagnostics:
            tier_counts[d.confidence_tier] = tier_counts.get(d.confidence_tier, 0) + 1
        tiers = ["HIGH", "MEDIUM", "LOW", "FAIL"]
        tier_colors = {"HIGH": "#27AE60", "MEDIUM": "#F39C12", "LOW": "#E74C3C", "FAIL": "#95A5A6"}
        present_tiers = [t for t in tiers if t in tier_counts]
        counts = [tier_counts[t] for t in present_tiers]
        colors = [tier_colors.get(t, "#95A5A6") for t in present_tiers]

        if present_tiers:
            bars = ax1.bar(present_tiers, counts, color=colors, edgecolor="white")
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         str(count), ha="center", va="bottom", fontsize=9)
    ax1.set_title("Confidence Tier Distribution", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Convergence success rate
    ax2 = fig.add_subplot(122)
    if diff_results:
        n_total = len(diff_results)
        n_null = sum(1 for dr in diff_results if dr.null_converged)
        n_full = sum(1 for dr in diff_results if dr.full_converged)
        n_both = sum(1 for dr in diff_results if dr.null_converged and dr.full_converged)

        labels = ["Null\nConverged", "Full\nConverged", "Both\nConverged"]
        values = [n_null/n_total*100, n_full/n_total*100, n_both/n_total*100]
        colors = ["#3498DB", "#2ECC71", "#9B59B6"]

        bars = ax2.bar(labels, values, color=colors, edgecolor="white")
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
        ax2.set_ylim(0, 110)
    ax2.set_title("Model Convergence", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Percentage", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_svg(fig, output_dir, "diagnostics.svg")


def _make_top_events_table(fig, diff_results, diagnostics, output_dir):
    """Create a table of top 20 significant events."""
    fig.clear()
    ax = fig.add_subplot(111)
    ax.axis("off")

    sig = sorted(
        [dr for dr in diff_results if dr.fdr < 0.05],
        key=lambda dr: dr.fdr,
    )[:20]

    if not sig:
        ax.text(0.5, 0.5, "No significant events (FDR < 0.05)",
                ha="center", va="center", fontsize=14)
        return

    diag_map = {}
    if diagnostics:
        for d in diagnostics:
            diag_map[d.module_id] = d

    headers = ["Gene", "Type", "Delta PSI", "FDR", "Tier"]
    table_data = []
    for dr in sig:
        diag = diag_map.get(dr.module_id)
        tier = diag.confidence_tier if diag else "NA"
        table_data.append([
            dr.gene_name or dr.gene_id,
            dr.event_type,
            f"{dr.max_abs_delta_psi:.3f}",
            f"{dr.fdr:.2e}",
            tier,
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j, header in enumerate(headers):
        cell = table[0, j]
        cell.set_facecolor("#8B0000")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(len(table_data)):
        color = "#F2F2F2" if i % 2 == 0 else "#FFFFFF"
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    ax.set_title("Top 20 Significant Events", fontsize=14, fontweight="bold", pad=20)
    _save_svg(fig, output_dir, "top_events.svg")


def generate_pdf_report(
    diff_results: List[DiffResult],
    diagnostics: List[EventDiagnostic],
    event_type_counts: Dict[str, int],
    output_path: str,
    output_dir: str,
    sample_info: Optional[Dict[str, str]] = None,
    parameters: Optional[Dict[str, str]] = None,
) -> None:
    """Generate a publication-quality PDF report.

    Creates a multi-page PDF with figures and tables, and saves individual
    figures as SVG files in output_dir/figures/.

    Args:
        diff_results: List of DiffResult objects.
        diagnostics: List of EventDiagnostic objects.
        event_type_counts: Dict mapping event type to count.
        output_path: Path to output PDF file.
        output_dir: Directory for SVG figure files.
        sample_info: Optional dict with sample information for title page.
        parameters: Optional dict with analysis parameters for title page.
    """
    fig = plt.figure(figsize=(11, 8.5))

    with PdfPages(output_path) as pdf:
        # Page 1: Title
        _make_title_page(fig, sample_info, parameters)
        pdf.savefig(fig, bbox_inches="tight")

        # Page 2: Volcano plot
        _make_volcano_plot(fig, diff_results, output_dir)
        pdf.savefig(fig, bbox_inches="tight")

        # Page 3: Event type charts
        _make_event_type_charts(fig, event_type_counts, diff_results, output_dir)
        pdf.savefig(fig, bbox_inches="tight")

        # Page 4: Diagnostics
        _make_diagnostics_chart(fig, diagnostics, diff_results, output_dir)
        pdf.savefig(fig, bbox_inches="tight")

        # Page 5: Top events table
        _make_top_events_table(fig, diff_results, diagnostics, output_dir)
        pdf.savefig(fig, bbox_inches="tight")

    plt.close(fig)
