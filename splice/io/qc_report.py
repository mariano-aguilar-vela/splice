"""
Module 24: io/qc_report.py

Generate comprehensive HTML QC report with embedded matplotlib figures.
"""

from __future__ import annotations

import base64
import io
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use as mpl_use

# Use non-interactive backend
mpl_use("Agg")


def _figure_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def _html_image(base64_str: str) -> str:
    """Create HTML img tag from base64 string."""
    return f'<img src="data:image/png;base64,{base64_str}" style="max-width: 100%; height: auto;">'


def generate_qc_report(
    diff_results: List,
    het_results: List,
    diagnostics: List,
    event_types: Dict[str, int],
    junction_evidence: Dict[str, dict],
    nmd_classifications: Dict[str, object],
    output_path: str,
) -> None:
    """Generate comprehensive HTML QC report with embedded figures.

    Report includes 7 sections with statistics and visualizations:
    1. Data Summary - mapping statistics
    2. Junction Discovery - motif distribution, recurrence
    3. Clustering - module size, event types
    4. Differential Splicing - volcano plot, distributions
    5. Diagnostics - confidence tiers, MAPQ
    6. Heterogeneity - index distribution
    7. Functional Annotation - NMD classifications

    Args:
        diff_results: List of DiffResult objects.
        het_results: List of HetResult objects.
        diagnostics: List of EventDiagnostic objects.
        event_types: Dict mapping event type to count.
        junction_evidence: Dict mapping junction_id to evidence dict.
        nmd_classifications: Dict mapping junction_id to NMDClassification.
        output_path: Path to output HTML file.
    """
    html_sections = []

    # Header
    html_sections.append(_generate_header())

    # Section 1: Data Summary
    html_sections.append(_generate_data_summary(diff_results, diagnostics))

    # Section 2: Junction Discovery
    html_sections.append(
        _generate_junction_discovery(junction_evidence, nmd_classifications)
    )

    # Section 3: Clustering
    html_sections.append(_generate_clustering(event_types, diff_results))

    # Section 4: Differential Splicing
    html_sections.append(_generate_diff_splicing(diff_results))

    # Section 5: Diagnostics
    html_sections.append(_generate_diagnostics(diagnostics, diff_results))

    # Section 6: Heterogeneity
    if het_results:
        html_sections.append(_generate_heterogeneity(het_results))

    # Section 7: Functional Annotation
    if nmd_classifications:
        html_sections.append(
            _generate_functional_annotation(diff_results, nmd_classifications)
        )

    # Footer
    html_sections.append(_generate_footer())

    # Write HTML
    html_content = "\n".join(html_sections)
    with open(output_path, "w") as f:
        f.write(html_content)


def _generate_header() -> str:
    """Generate HTML header."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPLICE QC Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        h3 {
            color: #7f8c8d;
        }
        .stats-box {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .stat-item {
            display: inline-block;
            margin: 10px 20px 10px 0;
        }
        .stat-label {
            font-weight: bold;
            color: #2c3e50;
        }
        .stat-value {
            font-size: 1.2em;
            color: #3498db;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        th, td {
            border: 1px solid #bdc3c7;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #34495e;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #ecf0f1;
        }
        img {
            margin: 15px 0;
            border: 1px solid #bdc3c7;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>SPLICE Quality Control Report</h1>
    <p>Comprehensive analysis of differential splicing results.</p>
"""


def _generate_footer() -> str:
    """Generate HTML footer."""
    return """
</div>
</body>
</html>
"""


def _generate_data_summary(diff_results: List, diagnostics: List) -> str:
    """Generate Data Summary section."""
    html = '<h2>Section 1: Data Summary</h2>\n'

    if diff_results:
        n_samples_estimate = max(
            (
                len(dr.psi_group1) + len(dr.psi_group2)
                if hasattr(dr, "psi_group1")
                else 0
            )
            for dr in diff_results[:1]
        )
    else:
        n_samples_estimate = 0

    html += '<div class="stats-box">\n'
    html += f'<div class="stat-item"><span class="stat-label">Total events tested:</span> <span class="stat-value">{len(diff_results)}</span></div>\n'
    html += f'<div class="stat-item"><span class="stat-label">Estimated samples:</span> <span class="stat-value">{n_samples_estimate}</span></div>\n'

    if diagnostics:
        mean_mapq = np.mean([d.mean_mapq for d in diagnostics])
        html += f'<div class="stat-item"><span class="stat-label">Mean MAPQ:</span> <span class="stat-value">{mean_mapq:.2f}</span></div>\n'

    html += "</div>\n"

    return html


def _generate_junction_discovery(
    junction_evidence: Dict[str, dict], nmd_classifications: Dict[str, object]
) -> str:
    """Generate Junction Discovery section."""
    html = '<h2>Section 2: Junction Discovery</h2>\n'

    n_total = len(junction_evidence)
    n_annotated = sum(
        1 for ev in junction_evidence.values() if ev.get("is_annotated", False)
    )
    n_novel = n_total - n_annotated

    html += '<div class="stats-box">\n'
    html += f'<div class="stat-item"><span class="stat-label">Total junctions:</span> <span class="stat-value">{n_total}</span></div>\n'
    html += f'<div class="stat-item"><span class="stat-label">Annotated:</span> <span class="stat-value">{n_annotated}</span></div>\n'
    html += f'<div class="stat-item"><span class="stat-label">Novel:</span> <span class="stat-value">{n_novel}</span></div>\n'
    html += "</div>\n"

    # Motif distribution chart
    motifs = {}
    for ev in junction_evidence.values():
        motif = ev.get("motif", "NA")
        motifs[motif] = motifs.get(motif, 0) + 1

    if motifs:
        fig, ax = plt.subplots(figsize=(8, 5))
        motif_names = list(motifs.keys())
        motif_counts = list(motifs.values())
        ax.bar(motif_names, motif_counts, color="#3498db", edgecolor="black")
        ax.set_ylabel("Count")
        ax.set_title("Splice Site Motif Distribution")
        plt.xticks(rotation=45)
        img_base64 = _figure_to_base64(fig)
        html += f"<h3>Motif Distribution</h3>\n{_html_image(img_base64)}\n"

    return html


def _generate_clustering(event_types: Dict[str, int], diff_results: List) -> str:
    """Generate Clustering section."""
    html = '<h2>Section 3: Clustering</h2>\n'

    n_modules = len(diff_results)
    html += '<div class="stats-box">\n'
    html += f'<div class="stat-item"><span class="stat-label">Total modules:</span> <span class="stat-value">{n_modules}</span></div>\n'
    html += "</div>\n"

    # Event type distribution pie chart
    if event_types:
        fig, ax = plt.subplots(figsize=(8, 5))
        event_names = list(event_types.keys())
        event_counts = list(event_types.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(event_names)))
        ax.pie(
            event_counts,
            labels=event_names,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax.set_title("Event Type Distribution")
        img_base64 = _figure_to_base64(fig)
        html += f"<h3>Event Type Distribution</h3>\n{_html_image(img_base64)}\n"

    return html


def _generate_diff_splicing(diff_results: List) -> str:
    """Generate Differential Splicing section."""
    html = '<h2>Section 4: Differential Splicing</h2>\n'

    n_sig = sum(1 for dr in diff_results if dr.fdr < 0.05)

    html += '<div class="stats-box">\n'
    html += f'<div class="stat-item"><span class="stat-label">Significant (FDR < 0.05):</span> <span class="stat-value">{n_sig}</span></div>\n'
    if len(diff_results) > 0:
        html += f'<div class="stat-item"><span class="stat-label">Percentage:</span> <span class="stat-value">{100*n_sig/len(diff_results):.1f}%</span></div>\n'
    html += "</div>\n"

    if diff_results:
        # Volcano plot
        log10_pval = np.array(
            [
                -np.log10(dr.p_value + 1e-300) if dr.p_value > 0 else 300
                for dr in diff_results
            ]
        )
        delta_psi = np.array([dr.max_abs_delta_psi for dr in diff_results])
        is_sig = np.array([dr.fdr < 0.05 for dr in diff_results])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            delta_psi[~is_sig],
            log10_pval[~is_sig],
            alpha=0.5,
            s=30,
            color="gray",
            label="Not significant",
        )
        ax.scatter(
            delta_psi[is_sig],
            log10_pval[is_sig],
            alpha=0.7,
            s=50,
            color="#e74c3c",
            label="Significant (FDR < 0.05)",
        )
        ax.set_xlabel("Delta-PSI")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title("Volcano Plot")
        ax.legend()
        ax.grid(True, alpha=0.3)
        img_base64 = _figure_to_base64(fig)
        html += f"<h3>Volcano Plot</h3>\n{_html_image(img_base64)}\n"

        # Delta-PSI distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(delta_psi, bins=30, color="#3498db", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Delta-PSI")
        ax.set_ylabel("Count")
        ax.set_title("Delta-PSI Distribution")
        img_base64 = _figure_to_base64(fig)
        html += f"<h3>Delta-PSI Distribution</h3>\n{_html_image(img_base64)}\n"

        # FDR distribution
        fdr_vals = np.array([dr.fdr for dr in diff_results])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            fdr_vals, bins=30, color="#2ecc71", edgecolor="black", alpha=0.7
        )
        ax.set_xlabel("FDR")
        ax.set_ylabel("Count")
        ax.set_title("FDR Distribution")
        ax.axvline(0.05, color="red", linestyle="--", linewidth=2, label="FDR=0.05")
        ax.legend()
        img_base64 = _figure_to_base64(fig)
        html += f"<h3>FDR Distribution</h3>\n{_html_image(img_base64)}\n"

    return html


def _generate_diagnostics(diagnostics: List, diff_results: List) -> str:
    """Generate Diagnostics section."""
    html = '<h2>Section 5: Diagnostics</h2>\n'

    # Confidence tier distribution
    tier_counts = {}
    for diag in diagnostics:
        tier = diag.confidence_tier
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    html += '<div class="stats-box">\n'
    n_converged = sum(
        1 for diag in diagnostics if diag.null_converged and diag.full_converged
    )
    html += f'<div class="stat-item"><span class="stat-label">Models converged:</span> <span class="stat-value">{n_converged}/{len(diagnostics)}</span></div>\n'
    html += "</div>\n"

    if tier_counts:
        fig, ax = plt.subplots(figsize=(8, 5))
        tiers = sorted(tier_counts.keys())
        counts = [tier_counts[t] for t in tiers]
        colors = {
            "HIGH": "#2ecc71",
            "MEDIUM": "#f39c12",
            "LOW": "#e67e22",
            "FAIL": "#e74c3c",
        }
        bar_colors = [colors.get(t, "#95a5a6") for t in tiers]
        ax.bar(tiers, counts, color=bar_colors, edgecolor="black")
        ax.set_ylabel("Count")
        ax.set_title("Confidence Tier Distribution")
        img_base64 = _figure_to_base64(fig)
        html += f"<h3>Confidence Tier Distribution</h3>\n{_html_image(img_base64)}\n"

    # MAPQ distribution
    if diagnostics:
        mapq_vals = np.array([d.mean_mapq for d in diagnostics])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(mapq_vals, bins=20, color="#9b59b6", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Mean MAPQ")
        ax.set_ylabel("Count")
        ax.set_title("MAPQ Distribution Across Events")
        img_base64 = _figure_to_base64(fig)
        html += f"<h3>MAPQ Distribution</h3>\n{_html_image(img_base64)}\n"

    return html


def _generate_heterogeneity(het_results: List) -> str:
    """Generate Heterogeneity section."""
    html = '<h2>Section 6: Heterogeneity</h2>\n'

    n_het = len(het_results)
    html += '<div class="stats-box">\n'
    html += f'<div class="stat-item"><span class="stat-label">Heterogeneous events:</span> <span class="stat-value">{n_het}</span></div>\n'
    html += "</div>\n"

    if het_results:
        het_indices = np.array(
            [
                hr.heterogeneity_index if hasattr(hr, "heterogeneity_index") else 0
                for hr in het_results
            ]
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            het_indices, bins=20, color="#1abc9c", edgecolor="black", alpha=0.7
        )
        ax.set_xlabel("Heterogeneity Index")
        ax.set_ylabel("Count")
        ax.set_title("Heterogeneity Index Distribution")
        img_base64 = _figure_to_base64(fig)
        html += f"<h3>Heterogeneity Index Distribution</h3>\n{_html_image(img_base64)}\n"

    return html


def _generate_functional_annotation(
    diff_results: List, nmd_classifications: Dict[str, object]
) -> str:
    """Generate Functional Annotation section."""
    html = '<h2>Section 7: Functional Annotation (NMD Classification)</h2>\n'

    if not nmd_classifications:
        html += "<p>No NMD classifications available.</p>\n"
        return html

    # Count classifications
    class_counts = {}
    for nmd_class in nmd_classifications.values():
        cls = getattr(nmd_class, "classification", "NA")
        class_counts[cls] = class_counts.get(cls, 0) + 1

    html += '<div class="stats-box">\n'
    n_total = len(nmd_classifications)
    html += f'<div class="stat-item"><span class="stat-label">Total junctions classified:</span> <span class="stat-value">{n_total}</span></div>\n'
    html += "</div>\n"

    # NMD classification distribution
    if class_counts:
        fig, ax = plt.subplots(figsize=(8, 5))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = {"PR": "#2ecc71", "UP": "#e74c3c", "NE": "#f39c12", "IN": "#95a5a6"}
        bar_colors = [colors.get(c, "#3498db") for c in classes]
        ax.bar(classes, counts, color=bar_colors, edgecolor="black")
        ax.set_ylabel("Count")
        ax.set_title("NMD Classification Distribution")
        img_base64 = _figure_to_base64(fig)
        html += f"<h3>NMD Classification Distribution</h3>\n{_html_image(img_base64)}\n"

    return html
