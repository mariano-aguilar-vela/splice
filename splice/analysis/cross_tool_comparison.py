"""
Cross-tool comparison for differential splicing analysis.

Compares SPLICE results against rMATS, MAJIQ, and SUPPA2 output files.
Generates concordance statistics and visualization plots.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# rMATS event type files
RMATS_EVENT_FILES = ["SE", "A3SS", "A5SS", "MXE", "RI"]


def load_splice_results(path: str) -> pd.DataFrame:
    """Load SPLICE results TSV.

    Returns DataFrame with columns: gene_id, gene_name, chrom, strand,
    event_type, delta_psi, fdr, p_value, significant.
    """
    df = pd.read_csv(path, sep="\t")
    # Standardize columns
    if "max_abs_delta_psi" in df.columns:
        df["delta_psi"] = df["max_abs_delta_psi"]
    df["significant"] = df["fdr"] < 0.05
    df["tool"] = "SPLICE"
    return df[[
        "gene_id", "gene_name", "chrom", "strand", "event_type",
        "delta_psi", "p_value", "fdr", "significant", "tool",
    ]]


def load_rmats_results(directory: str) -> pd.DataFrame:
    """Load rMATS output files (.MATS.JC.txt for SE, A3SS, A5SS, MXE, RI).

    Args:
        directory: Path to rMATS output directory.

    Returns:
        DataFrame with combined events from all 5 event types.
    """
    rows = []
    for event_type in RMATS_EVENT_FILES:
        path = os.path.join(directory, f"{event_type}.MATS.JC.txt")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep="\t")
        if df.empty:
            continue
        for _, r in df.iterrows():
            gene_id = str(r.get("GeneID", "")).strip('"')
            gene_name = str(r.get("geneSymbol", "")).strip('"')
            chrom = str(r.get("chr", ""))
            strand = str(r.get("strand", "+"))
            fdr = float(r.get("FDR", 1.0))
            p_value = float(r.get("PValue", 1.0))
            delta_psi = float(r.get("IncLevelDifference", 0.0))
            rows.append({
                "gene_id": gene_id,
                "gene_name": gene_name,
                "chrom": chrom,
                "strand": strand,
                "event_type": event_type,
                "delta_psi": delta_psi,
                "p_value": p_value,
                "fdr": fdr,
                "significant": fdr < 0.05,
                "tool": "rMATS",
            })

    if not rows:
        return pd.DataFrame(columns=[
            "gene_id", "gene_name", "chrom", "strand", "event_type",
            "delta_psi", "p_value", "fdr", "significant", "tool",
        ])
    return pd.DataFrame(rows)


def load_majiq_results(tsv_path: str) -> pd.DataFrame:
    """Load MAJIQ deltapsi.tsv file.

    Args:
        tsv_path: Path to MAJIQ deltapsi TSV.

    Returns:
        DataFrame with one row per LSV junction.
    """
    df = pd.read_csv(tsv_path, sep="\t", comment="#")
    rows = []
    for _, r in df.iterrows():
        gene_id = str(r.get("Gene ID", r.get("gene_id", ""))).strip()
        gene_name = str(r.get("Gene Name", r.get("gene_name", ""))).strip()
        lsv_id = str(r.get("LSV ID", r.get("lsv_id", "")))

        # Mean dPSI per junction (semicolon-separated string)
        dpsi_str = str(r.get(
            "mean_dpsi_per_lsv_junction",
            r.get("E(dPSI) per LSV junction", "0"),
        ))
        try:
            dpsi_vals = [float(x) for x in dpsi_str.split(";") if x.strip()]
            max_dpsi = max(dpsi_vals, key=abs) if dpsi_vals else 0.0
        except (ValueError, AttributeError):
            max_dpsi = 0.0

        # Probability of change
        prob_str = str(r.get(
            "P(|dPSI|>=0.20) per LSV junction",
            r.get("probability_changing", "0"),
        ))
        try:
            prob_vals = [float(x) for x in prob_str.split(";") if x.strip()]
            max_prob = max(prob_vals) if prob_vals else 0.0
        except (ValueError, AttributeError):
            max_prob = 0.0

        chrom = str(r.get("chr", r.get("seqid", "")))
        strand = str(r.get("strand", "+"))

        rows.append({
            "gene_id": gene_id,
            "gene_name": gene_name,
            "chrom": chrom,
            "strand": strand,
            "event_type": "LSV",
            "delta_psi": max_dpsi,
            "p_value": 1.0 - max_prob,
            "fdr": 1.0 - max_prob,
            "significant": max_prob > 0.95,
            "tool": "MAJIQ",
        })

    if not rows:
        return pd.DataFrame(columns=[
            "gene_id", "gene_name", "chrom", "strand", "event_type",
            "delta_psi", "p_value", "fdr", "significant", "tool",
        ])
    return pd.DataFrame(rows)


def load_suppa2_results(dpsi_path: str, psivec_path: Optional[str] = None) -> pd.DataFrame:
    """Load SUPPA2 .dpsi file (and optionally .psivec).

    Args:
        dpsi_path: Path to SUPPA2 .dpsi file.
        psivec_path: Optional path to .psivec file.

    Returns:
        DataFrame with one row per event.
    """
    df = pd.read_csv(dpsi_path, sep="\t", index_col=0)

    rows = []
    for event_id, r in df.iterrows():
        # Event ID format: GENE_ID;EVENT_TYPE:CHR:COORDS:STRAND
        gene_id = str(event_id).split(";")[0] if ";" in str(event_id) else str(event_id)
        event_parts = str(event_id).split(";")
        event_type = "Unknown"
        chrom = ""
        strand = "+"
        if len(event_parts) > 1:
            type_field = event_parts[1].split(":")
            if type_field:
                event_type = type_field[0]
            for part in type_field:
                if part.startswith("chr"):
                    chrom = part
                if part in ("+", "-"):
                    strand = part

        # The dPSI column has variable name; take first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        delta_psi = 0.0
        p_value = 1.0
        if len(numeric_cols) >= 1:
            delta_psi = float(r[numeric_cols[0]])
        if len(numeric_cols) >= 2:
            p_value = float(r[numeric_cols[1]])

        if pd.isna(delta_psi):
            delta_psi = 0.0
        if pd.isna(p_value):
            p_value = 1.0

        rows.append({
            "gene_id": gene_id,
            "gene_name": gene_id,
            "chrom": chrom,
            "strand": strand,
            "event_type": event_type,
            "delta_psi": delta_psi,
            "p_value": p_value,
            "fdr": p_value,
            "significant": p_value < 0.05,
            "tool": "SUPPA2",
        })

    if not rows:
        return pd.DataFrame(columns=[
            "gene_id", "gene_name", "chrom", "strand", "event_type",
            "delta_psi", "p_value", "fdr", "significant", "tool",
        ])
    return pd.DataFrame(rows)


def _normalize_gene_id(gene_id: str) -> str:
    """Strip ENSEMBL version suffixes and quotes."""
    if not gene_id:
        return ""
    g = str(gene_id).strip().strip('"')
    if "." in g and g.startswith("ENSG"):
        g = g.split(".")[0]
    return g


def _significant_genes(df: pd.DataFrame) -> Set[str]:
    """Get set of significant gene IDs from a tool's DataFrame."""
    if df is None or df.empty:
        return set()
    sig = df[df["significant"]]
    return {_normalize_gene_id(g) for g in sig["gene_id"] if g}


def match_events_by_gene(
    splice_df: pd.DataFrame,
    other_df: pd.DataFrame,
    tool_name: str,
) -> pd.DataFrame:
    """Match events between SPLICE and another tool at the gene level.

    Args:
        splice_df: SPLICE results DataFrame.
        other_df: Other tool's results DataFrame.
        tool_name: Name of the other tool (for column labeling).

    Returns:
        DataFrame with columns: gene_id, splice_significant, other_significant,
        splice_delta_psi, other_delta_psi, splice_fdr, other_fdr, concordant.
    """
    if splice_df is None or splice_df.empty or other_df is None or other_df.empty:
        return pd.DataFrame(columns=[
            "gene_id", "splice_significant", f"{tool_name}_significant",
            "splice_delta_psi", f"{tool_name}_delta_psi",
            "splice_fdr", f"{tool_name}_fdr", "concordant",
        ])

    # Get most significant event per gene from each tool
    splice_norm = splice_df.copy()
    splice_norm["gene_id_norm"] = splice_norm["gene_id"].apply(_normalize_gene_id)
    splice_top = (
        splice_norm.sort_values("fdr")
        .groupby("gene_id_norm")
        .first()
        .reset_index()
    )

    other_norm = other_df.copy()
    other_norm["gene_id_norm"] = other_norm["gene_id"].apply(_normalize_gene_id)
    other_top = (
        other_norm.sort_values("fdr")
        .groupby("gene_id_norm")
        .first()
        .reset_index()
    )

    # Merge by normalized gene_id
    merged = pd.merge(
        splice_top[["gene_id_norm", "significant", "delta_psi", "fdr"]],
        other_top[["gene_id_norm", "significant", "delta_psi", "fdr"]],
        on="gene_id_norm", how="outer", suffixes=("_splice", f"_{tool_name}"),
    )

    merged = merged.rename(columns={
        "gene_id_norm": "gene_id",
        "significant_splice": "splice_significant",
        f"significant_{tool_name}": f"{tool_name}_significant",
        "delta_psi_splice": "splice_delta_psi",
        f"delta_psi_{tool_name}": f"{tool_name}_delta_psi",
        "fdr_splice": "splice_fdr",
        f"fdr_{tool_name}": f"{tool_name}_fdr",
    })

    # Fill NaN: gene was missing from one tool
    merged["splice_significant"] = merged["splice_significant"].fillna(False).astype(bool)
    merged[f"{tool_name}_significant"] = (
        merged[f"{tool_name}_significant"].fillna(False).astype(bool)
    )

    merged["concordant"] = merged["splice_significant"] & merged[f"{tool_name}_significant"]
    return merged


def compute_concordance_stats(
    splice_df: pd.DataFrame,
    rmats_df: Optional[pd.DataFrame] = None,
    majiq_df: Optional[pd.DataFrame] = None,
    suppa2_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """Compute pairwise concordance statistics between SPLICE and other tools.

    Returns dict with per-tool stats: shared_genes, jaccard_index,
    splice_unique, other_unique.
    """
    splice_genes = _significant_genes(splice_df)

    stats = {"splice_n_significant": len(splice_genes)}

    tool_dfs = {"rMATS": rmats_df, "MAJIQ": majiq_df, "SUPPA2": suppa2_df}

    for tool_name, df in tool_dfs.items():
        if df is None or df.empty:
            stats[tool_name] = {
                "n_significant": 0,
                "shared_with_splice": 0,
                "splice_only": len(splice_genes),
                f"{tool_name.lower()}_only": 0,
                "jaccard": 0.0,
            }
            continue

        other_genes = _significant_genes(df)
        shared = splice_genes & other_genes
        union = splice_genes | other_genes
        jaccard = len(shared) / len(union) if union else 0.0

        stats[tool_name] = {
            "n_significant": len(other_genes),
            "shared_with_splice": len(shared),
            "splice_only": len(splice_genes - other_genes),
            f"{tool_name.lower()}_only": len(other_genes - splice_genes),
            "jaccard": jaccard,
        }

    return stats


def _write_concordance_summary(stats: Dict, output_path: str):
    """Write concordance summary as TSV."""
    rows = []
    rows.append(["Tool", "n_significant", "shared_with_SPLICE",
                 "SPLICE_only", "tool_only", "Jaccard"])
    rows.append(["SPLICE", stats["splice_n_significant"], "-", "-", "-", "-"])
    for tool in ("rMATS", "MAJIQ", "SUPPA2"):
        s = stats.get(tool, {})
        rows.append([
            tool,
            s.get("n_significant", 0),
            s.get("shared_with_splice", 0),
            s.get("splice_only", 0),
            s.get(f"{tool.lower()}_only", 0),
            f"{s.get('jaccard', 0.0):.4f}",
        ])

    with open(output_path, "w") as f:
        for row in rows:
            f.write("\t".join(str(c) for c in row) + "\n")


def _venn_diagram(gene_sets: Dict[str, Set[str]], output_path: str):
    """Create a 4-way Venn-like diagram showing gene set overlaps.

    Uses matplotlib_venn for 2-way and 3-way Venns; for 4-way uses
    a custom layout with overlapping circles.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    tools = list(gene_sets.keys())
    n_tools = len(tools)
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]

    if n_tools <= 3:
        try:
            from matplotlib_venn import venn2, venn3
            if n_tools == 2:
                venn2(
                    [gene_sets[tools[0]], gene_sets[tools[1]]],
                    set_labels=tools, ax=ax,
                )
            elif n_tools == 3:
                venn3(
                    [gene_sets[t] for t in tools],
                    set_labels=tools, ax=ax,
                )
        except ImportError:
            pass
    else:
        # 4-way: draw 4 overlapping ellipses
        from matplotlib.patches import Ellipse

        positions = [
            (0.35, 0.55, 30),
            (0.45, 0.55, -30),
            (0.55, 0.45, 30),
            (0.65, 0.45, -30),
        ]
        for (x, y, angle), tool, color in zip(positions, tools, colors):
            ell = Ellipse(
                (x, y), 0.5, 0.3, angle=angle,
                facecolor=color, alpha=0.4, edgecolor=color, linewidth=2,
            )
            ax.add_patch(ell)

        # Tool labels
        label_positions = [(0.05, 0.85), (0.95, 0.85), (0.05, 0.15), (0.95, 0.15)]
        for (x, y), tool, color, gene_set in zip(label_positions, tools, colors, gene_sets.values()):
            ax.text(x, y, f"{tool}\n(n={len(gene_set)})",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color))

        # All four intersection
        all_shared = set.intersection(*gene_sets.values()) if gene_sets else set()
        ax.text(0.5, 0.5, f"All 4:\n{len(all_shared)}",
                ha="center", va="center", fontsize=11, fontweight="bold")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

    ax.set_title("Significant Genes by Tool", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _upset_plot(gene_sets: Dict[str, Set[str]], output_path: str):
    """Create an UpSet-style bar chart of intersection sizes."""
    from itertools import combinations

    tools = list(gene_sets.keys())
    intersections = []

    # Compute all non-empty intersections
    for r in range(1, len(tools) + 1):
        for combo in combinations(tools, r):
            included = set.intersection(*[gene_sets[t] for t in combo])
            excluded = set()
            for t in tools:
                if t not in combo:
                    excluded |= gene_sets[t]
            unique_to_combo = included - excluded
            if len(unique_to_combo) > 0:
                intersections.append((combo, len(unique_to_combo)))

    intersections.sort(key=lambda x: -x[1])
    intersections = intersections[:15]  # Top 15

    if not intersections:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No significant intersections",
                ha="center", va="center")
        fig.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        return

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        sharex=True,
    )

    x = np.arange(len(intersections))
    counts = [c for _, c in intersections]

    ax_top.bar(x, counts, color="#34495E", edgecolor="white")
    for i, c in enumerate(counts):
        ax_top.text(i, c + max(counts) * 0.01, str(c),
                    ha="center", va="bottom", fontsize=9)
    ax_top.set_ylabel("Intersection Size", fontsize=11)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.set_title("Tool Intersections (Significant Genes)",
                     fontsize=13, fontweight="bold")

    # Bottom panel: dot matrix
    for i, (combo, _) in enumerate(intersections):
        for j, tool in enumerate(tools):
            color = "#34495E" if tool in combo else "#BDC3C7"
            ax_bot.scatter(i, j, c=color, s=120, edgecolors="white")
        # Connect dots in same intersection
        in_combo_y = [j for j, t in enumerate(tools) if t in combo]
        if len(in_combo_y) > 1:
            ax_bot.plot([i, i], [min(in_combo_y), max(in_combo_y)],
                        color="#34495E", linewidth=2, zorder=0)

    ax_bot.set_yticks(range(len(tools)))
    ax_bot.set_yticklabels(tools)
    ax_bot.set_xticks([])
    ax_bot.set_xlim(-0.5, len(intersections) - 0.5)
    ax_bot.set_ylim(-0.5, len(tools) - 0.5)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.spines["bottom"].set_visible(False)

    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _delta_psi_correlation(
    splice_df: pd.DataFrame,
    other_dfs: Dict[str, pd.DataFrame],
    output_path: str,
):
    """Scatter plots of SPLICE delta-PSI vs each other tool."""
    valid_tools = [(name, df) for name, df in other_dfs.items()
                   if df is not None and not df.empty]
    n_tools = len(valid_tools)

    if n_tools == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No comparison data", ha="center", va="center")
        fig.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, n_tools, figsize=(5 * n_tools, 5))
    if n_tools == 1:
        axes = [axes]

    for ax, (tool_name, df) in zip(axes, valid_tools):
        merged = match_events_by_gene(splice_df, df, tool_name)
        concordant = merged[merged["concordant"]]

        if not concordant.empty:
            x = concordant["splice_delta_psi"].astype(float)
            y = concordant[f"{tool_name}_delta_psi"].astype(float)
            ax.scatter(x, y, alpha=0.6, s=20, color="#3498DB", edgecolors="white")

            if len(x) > 1:
                corr = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else 0.0
                ax.text(0.05, 0.95, f"r = {corr:.3f}\nn = {len(x)}",
                        transform=ax.transAxes, va="top", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor="gray"))
        else:
            ax.text(0.5, 0.5, "No concordant\nevents",
                    transform=ax.transAxes, ha="center", va="center")

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("SPLICE delta-PSI", fontsize=11)
        ax.set_ylabel(f"{tool_name} delta-PSI", fontsize=11)
        ax.set_title(f"SPLICE vs {tool_name}", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _concordance_heatmap(gene_sets: Dict[str, Set[str]], output_path: str):
    """Pairwise Jaccard similarity matrix as heatmap."""
    tools = list(gene_sets.keys())
    n = len(tools)
    matrix = np.zeros((n, n))

    for i, t1 in enumerate(tools):
        for j, t2 in enumerate(tools):
            s1, s2 = gene_sets[t1], gene_sets[t2]
            union = s1 | s2
            matrix[i, j] = len(s1 & s2) / len(union) if union else 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tools, rotation=45, ha="right")
    ax.set_yticklabels(tools)
    ax.set_title("Pairwise Jaccard Similarity", fontsize=13, fontweight="bold")

    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] > 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, label="Jaccard index")
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_comparison_report(
    splice_dir: str,
    rmats_dir: Optional[str] = None,
    majiq_dir: Optional[str] = None,
    suppa2_dir: Optional[str] = None,
    output_dir: str = "./comparison",
) -> Dict:
    """Generate a complete cross-tool comparison report.

    Args:
        splice_dir: Directory containing splice_results.tsv.
        rmats_dir: Optional rMATS output directory.
        majiq_dir: Optional MAJIQ output directory.
        suppa2_dir: Optional SUPPA2 output directory.
        output_dir: Output directory for report files.

    Returns:
        Dict of concordance statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all results
    splice_path = os.path.join(splice_dir, "splice_results.tsv")
    splice_df = load_splice_results(splice_path)

    rmats_df = None
    if rmats_dir and os.path.isdir(rmats_dir):
        rmats_df = load_rmats_results(rmats_dir)

    majiq_df = None
    if majiq_dir and os.path.isdir(majiq_dir):
        for fname in os.listdir(majiq_dir):
            if fname.endswith("deltapsi.tsv") or fname.endswith(".deltapsi.tsv"):
                majiq_df = load_majiq_results(os.path.join(majiq_dir, fname))
                break

    suppa2_df = None
    if suppa2_dir and os.path.isdir(suppa2_dir):
        dpsi_files = [f for f in os.listdir(suppa2_dir) if f.endswith(".dpsi")]
        if dpsi_files:
            dpsi_path = os.path.join(suppa2_dir, dpsi_files[0])
            psivec_files = [f for f in os.listdir(suppa2_dir) if f.endswith(".psivec")]
            psivec_path = os.path.join(suppa2_dir, psivec_files[0]) if psivec_files else None
            suppa2_df = load_suppa2_results(dpsi_path, psivec_path)

    # Compute stats
    stats = compute_concordance_stats(splice_df, rmats_df, majiq_df, suppa2_df)

    # Write summary
    _write_concordance_summary(stats, os.path.join(output_dir, "concordance_summary.tsv"))

    # Build gene sets for visualization
    gene_sets = {"SPLICE": _significant_genes(splice_df)}
    if rmats_df is not None:
        gene_sets["rMATS"] = _significant_genes(rmats_df)
    if majiq_df is not None:
        gene_sets["MAJIQ"] = _significant_genes(majiq_df)
    if suppa2_df is not None:
        gene_sets["SUPPA2"] = _significant_genes(suppa2_df)

    # Generate visualizations
    _venn_diagram(gene_sets, os.path.join(output_dir, "venn_diagram.svg"))
    _upset_plot(gene_sets, os.path.join(output_dir, "upset_plot.svg"))
    _concordance_heatmap(gene_sets, os.path.join(output_dir, "concordance_heatmap.svg"))

    other_dfs = {"rMATS": rmats_df, "MAJIQ": majiq_df, "SUPPA2": suppa2_df}
    _delta_psi_correlation(
        splice_df, other_dfs,
        os.path.join(output_dir, "delta_psi_correlation.svg"),
    )

    return stats
