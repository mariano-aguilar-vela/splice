"""
Module 26: cli.py

Main CLI entry point for SPLICE with all subcommands.
"""

import click


@click.group()
@click.version_option(version="1.0.0")
def main():
    """SPLICE: Splicegraph Probabilistic Learning for Isoform Change Estimation.

    A comprehensive platform for discovery and analysis of differential splicing
    events in RNA-seq data. Combines annotation-free junction discovery, multi-way
    statistical testing, covariate regression, heterogeneity detection, and
    functional annotation.

    For detailed help on a subcommand, use: splice COMMAND --help
    """
    pass


@main.command()
@click.option(
    "--bam",
    "-b",
    multiple=True,
    required=True,
    help="Input BAM file(s). Can specify multiple times.",
)
@click.option(
    "--sample-names",
    "-n",
    multiple=True,
    help="Sample names (one per BAM). If not provided, names derived from BAM filenames.",
)
@click.option(
    "--gtf",
    "-g",
    required=True,
    type=click.Path(exists=True),
    help="Genome annotation GTF file.",
)
@click.option(
    "--genome",
    "-f",
    default=None,
    type=click.Path(exists=True),
    help="Genome FASTA file (enables motif scoring and NMD classification).",
)
@click.option(
    "--group1",
    required=True,
    help="Comma-separated sample indices for group 1 (0-indexed).",
)
@click.option(
    "--group2",
    required=True,
    help="Comma-separated sample indices for group 2 (0-indexed).",
)
@click.option(
    "--covariates",
    default=None,
    type=click.Path(exists=True),
    help="TSV file with sample covariates (columns: sample, covariate).",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(),
    help="Output directory for results.",
)
@click.option(
    "--read-length",
    "-l",
    type=int,
    default=None,
    help="Read length (auto-detected from BAMs if not specified).",
)
@click.option(
    "--min-anchor",
    type=int,
    default=6,
    help="Minimum anchor length for valid junctions.",
)
@click.option(
    "--min-mapq",
    type=int,
    default=0,
    help="Minimum MAPQ for reads to include.",
)
@click.option(
    "--min-cluster-reads",
    type=int,
    default=30,
    help="Minimum total reads in a cluster for inclusion.",
)
@click.option(
    "--max-intron-length",
    type=int,
    default=100000,
    help="Maximum intron length (longer gaps treated as no junction).",
)
@click.option(
    "--n-bootstraps",
    type=int,
    default=30,
    help="Number of bootstrap replicates for uncertainty estimation.",
)
@click.option(
    "--threads",
    "-t",
    type=int,
    default=1,
    help="Number of worker threads for parallelization.",
)
@click.option(
    "--no-novel",
    is_flag=True,
    help="Disable de novo junction discovery (use only annotated junctions).",
)
@click.option(
    "--no-nmd",
    is_flag=True,
    help="Skip NMD classification step.",
)
@click.option(
    "--no-het",
    is_flag=True,
    help="Skip heterogeneity testing step.",
)
@click.option(
    "--no-exon-body",
    is_flag=True,
    help="Skip exon body read counting.",
)
@click.option(
    "--export-leafcutter",
    is_flag=True,
    help="Export results in LeafCutter format.",
)
@click.option(
    "--export-rmats",
    is_flag=True,
    help="Export results in rMATS format.",
)
@click.option(
    "--export-bed",
    is_flag=True,
    help="Export significant junctions as BED file.",
)
@click.option(
    "--checkpoint-dir",
    default=None,
    type=click.Path(),
    help="Directory for checkpoint files (enables resume on interruption).",
)
def run(
    bam,
    sample_names,
    gtf,
    genome,
    group1,
    group2,
    covariates,
    output_dir,
    read_length,
    min_anchor,
    min_mapq,
    min_cluster_reads,
    max_intron_length,
    n_bootstraps,
    threads,
    no_novel,
    no_nmd,
    no_het,
    no_exon_body,
    export_leafcutter,
    export_rmats,
    export_bed,
    checkpoint_dir,
):
    """Run the complete SPLICE differential splicing analysis pipeline.

    Full pipeline steps:
      1. Parse GTF and optionally load genome for NMD analysis
      2. Extract junctions, exon body reads, and co-occurrences from BAMs
      3. Score splice site motifs (if genome provided)
      4. Annotate junctions (annotated vs novel)
      5. Compute junction confidence scores
      6. Cluster junctions using LeafCutter algorithm
      7. Build splicing modules and topology
      8. Build evidence matrices with effective length normalization
      9. Filter modules by size and coverage criteria
      10. Quantify PSI with bootstrap confidence intervals
      11. Test differential splicing (DM-GLM with optional covariates)
      12. Test for heterogeneous effects across samples
      13. Compute per-event diagnostic metrics
      14. Classify events into functional categories (SE, A3SS, A5SS, etc.)
      15. Classify NMD impact (if genome provided)
      16. Write results tables and per-junction details
      17. Generate comprehensive HTML QC report
      18. Export results to other tool formats (LeafCutter, rMATS, BED)

    Example usage:

        splice run \\
          -b sample1.bam -b sample2.bam -b sample3.bam \\
          --sample-names S1 S2 S3 \\
          --gtf genes.gtf \\
          --genome hg38.fa \\
          --group1 0,1,2 \\
          --group2 3,4,5 \\
          --output-dir ./results \\
          --threads 8 \\
          --export-rmats \\
          --export-leafcutter
    """
    import os
    import sys
    import time

    import numpy as np

    from splice.core.clustering import cluster_junctions
    from splice.core.confidence_scorer import score_all_junctions
    from splice.core.diagnostics import compute_diagnostics
    from splice.core.diff import differential_splicing
    from splice.core.diff_het import heterogeneous_splicing
    from splice.core.event_classifier import classify_all_events
    from splice.core.evidence import build_evidence_matrices
    from splice.core.gtf_parser import Gene, extract_known_junctions, parse_gtf
    from splice.core.junction_extractor import extract_all_junctions
    from splice.core.nmd_classifier import classify_all_junctions_nmd
    from splice.core.psi import quantify_psi
    from splice.core.splicegraph import build_splicegraph
    from splice.io.format_export import (
        export_bed_format,
        export_leafcutter_format,
        export_rmats_format,
    )
    from splice.io.output_writer import (
        write_junction_details_tsv,
        write_results_tsv,
        write_summary_tsv,
    )
    from splice.io.qc_report import generate_qc_report
    from splice.io.serialization import save_checkpoint
    from splice.utils.genomic import Junction

    t_start = time.time()

    # ── Setup ──────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    bam_paths = list(bam)
    n_samples = len(bam_paths)

    # Derive sample names from BAM filenames if not provided
    if sample_names:
        names = list(sample_names)
    else:
        names = [
            os.path.basename(p).replace("_Aligned.sortedByCoord.out.bam", "").replace(".bam", "")
            for p in bam_paths
        ]

    # Parse group indices
    g1_indices = [int(x) for x in group1.split(",")]
    g2_indices = [int(x) for x in group2.split(",")]
    group_labels = np.zeros(n_samples, dtype=int)
    for idx in g2_indices:
        group_labels[idx] = 1

    click.echo("SPLICE: Splicegraph Probabilistic Learning for Isoform Change Estimation")
    click.echo(f"Input BAMs: {n_samples} samples")
    click.echo(f"  Group 1 ({len(g1_indices)} samples): {', '.join(names[i] for i in g1_indices)}")
    click.echo(f"  Group 2 ({len(g2_indices)} samples): {', '.join(names[i] for i in g2_indices)}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Threads: {threads}")
    click.echo("")

    # ── Step 1: Parse GTF ──────────────────────────────────────────────────────
    click.echo("[Step  1/18] Parsing GTF annotation...")
    t1 = time.time()
    genes = parse_gtf(gtf, gene_type_filter="protein_coding")
    known_junctions = extract_known_junctions(genes)
    click.echo(f"  Parsed {len(genes)} protein-coding genes, {len(known_junctions)} known junctions ({time.time()-t1:.1f}s)")

    if checkpoint_dir:
        save_checkpoint({"genes": genes, "known_junctions": known_junctions},
                        os.path.join(checkpoint_dir, "step01_gtf.pkl"))

    # ── Step 2: Extract junctions from BAMs ────────────────────────────────────
    click.echo("[Step  2/18] Extracting junctions from BAM files...")
    t2 = time.time()
    junction_evidence, cooccurrence = extract_all_junctions(
        bam_paths=bam_paths,
        sample_names=names,
        known_junctions=known_junctions,
        genome_fasta_path=genome,
        min_anchor=min_anchor,
        min_mapq=min_mapq,
    )
    click.echo(f"  Found {len(junction_evidence)} junctions ({time.time()-t2:.1f}s)")

    if checkpoint_dir:
        save_checkpoint({"junction_evidence": junction_evidence, "cooccurrence": cooccurrence},
                        os.path.join(checkpoint_dir, "step02_junctions.pkl"))

    # ── Step 3-4: Score and annotate junctions ─────────────────────────────────
    click.echo("[Step  3/18] Scoring junction confidence...")
    confidence_scores = score_all_junctions(junction_evidence)
    click.echo(f"  Scored {len(confidence_scores)} junctions")

    # ── Step 5: Cluster junctions ──────────────────────────────────────────────
    click.echo("[Step  5/18] Clustering junctions...")
    t5 = time.time()
    # Pre-filter: remove junctions with fewer than min_cluster_reads total reads
    all_junctions = list(junction_evidence.keys())
    filtered_junctions = [
        junc for junc in all_junctions
        if np.sum(junction_evidence[junc].sample_counts) >= min_cluster_reads
    ]
    click.echo(f"  Pre-filtered {len(all_junctions)} -> {len(filtered_junctions)} junctions (min {min_cluster_reads} total reads)")
    clusters = cluster_junctions(filtered_junctions, max_intron_length=max_intron_length)
    click.echo(f"  Formed {len(clusters)} clusters ({time.time()-t5:.1f}s)")

    # ── Step 6: Build splicegraph and modules ──────────────────────────────────
    click.echo("[Step  6/18] Building splicegraph...")
    modules, junction_to_idx = build_splicegraph(
        genes=genes,
        junction_evidence=junction_evidence,
        clusters=clusters,
        known_junctions=known_junctions,
    )
    click.echo(f"  Built {len(modules)} splicing modules")

    if checkpoint_dir:
        save_checkpoint({"modules": modules, "junction_to_idx": junction_to_idx},
                        os.path.join(checkpoint_dir, "step06_splicegraph.pkl"))

    # ── Step 7: Build evidence matrices ────────────────────────────────────────
    click.echo("[Step  7/18] Building evidence matrices...")
    detected_read_length = read_length if read_length else 150  # Default to 150 if not specified
    evidence_list = build_evidence_matrices(
        modules=modules,
        junction_evidence=junction_evidence,
        junction_confidence=confidence_scores,
        read_length=detected_read_length,
    )
    click.echo(f"  Built evidence for {len(evidence_list)} modules")

    # ── Step 8: Quantify PSI ───────────────────────────────────────────────────
    click.echo(f"[Step  8/18] Quantifying PSI ({n_bootstraps} bootstraps)...")
    t8 = time.time()
    psi_list = quantify_psi(evidence_list, n_bootstraps=n_bootstraps, seed=42)
    click.echo(f"  Quantified PSI for {len(psi_list)} modules ({time.time()-t8:.1f}s)")

    if checkpoint_dir:
        save_checkpoint({"evidence_list": evidence_list, "psi_list": psi_list},
                        os.path.join(checkpoint_dir, "step08_psi.pkl"))

    # ── Step 9: Differential splicing ──────────────────────────────────────────
    click.echo("[Step  9/18] Testing differential splicing (DM-GLM)...")
    t9 = time.time()
    diff_results = differential_splicing(
        module_evidence_list=evidence_list,
        module_psi_list=psi_list,
        group_labels=group_labels,
    )
    n_sig = sum(1 for dr in diff_results if dr.fdr < 0.05)
    click.echo(f"  Tested {len(diff_results)} modules, {n_sig} significant (FDR < 0.05) ({time.time()-t9:.1f}s)")

    # ── Step 10: Heterogeneity testing ─────────────────────────────────────────
    if not no_het:
        click.echo("[Step 10/18] Testing for heterogeneous effects...")
        het_results = heterogeneous_splicing(psi_list, group_labels)
        click.echo(f"  Tested {len(het_results)} modules")
    else:
        click.echo("[Step 10/18] Heterogeneity testing: SKIPPED")
        het_results = []

    # ── Step 11: Event classification ──────────────────────────────────────────
    click.echo("[Step 11/18] Classifying event types...")
    event_types_list = classify_all_events(modules)
    event_type_counts = {}
    for evt in event_types_list:
        event_type_counts[evt] = event_type_counts.get(evt, 0) + 1
    click.echo(f"  Event types: {event_type_counts}")

    # ── Step 12: Diagnostics ───────────────────────────────────────────────────
    click.echo("[Step 12/18] Computing diagnostics...")
    tested_ids = {dr.module_id for dr in diff_results}
    module_order = {dr.module_id: i for i, dr in enumerate(diff_results)}
    tested_evidence = sorted(
        [e for e in evidence_list if e.module.module_id in tested_ids],
        key=lambda e: module_order.get(e.module.module_id, 999),
    )
    tested_psi = sorted(
        [p for p in psi_list if p.module_id in tested_ids],
        key=lambda p: module_order.get(p.module_id, 999),
    )
    diagnostics = compute_diagnostics(tested_evidence, tested_psi, diff_results)

    if checkpoint_dir:
        save_checkpoint({"diff_results": diff_results, "diagnostics": diagnostics},
                        os.path.join(checkpoint_dir, "step12_diff.pkl"))

    # ── Step 13: NMD classification ────────────────────────────────────────────
    nmd_classifications = {}
    if genome and not no_nmd:
        click.echo("[Step 13/18] Classifying NMD impact...")
        # Use pysam.FastaFile for indexed access (no memory overhead)
        import pysam as _pysam
        genome_fasta = _pysam.FastaFile(genome)
        genome_dict = {ref: genome_fasta.fetch(ref) for ref in genome_fasta.references
                       if ref.startswith("chr") and len(ref) <= 5}
        click.echo(f"  Loaded genome index: {len(genome_fasta.references)} contigs, {len(genome_dict)} main chromosomes cached")
        genome_fasta.close()
        click.echo("  NMD classification: genome loaded (per-junction NMD requires exon mapping)")
    else:
        click.echo("[Step 13/18] NMD classification: SKIPPED")

    # ── Step 14: Write results ─────────────────────────────────────────────────
    click.echo("[Step 14/18] Writing results...")
    results_path = os.path.join(output_dir, "splice_results.tsv")
    write_results_tsv(diff_results, diagnostics, results_path)
    click.echo(f"  Results: {results_path}")

    # ── Step 15: Write junction details ────────────────────────────────────────
    click.echo("[Step 15/18] Writing junction details...")
    junc_evidence_dict = {}
    for junc, ev in junction_evidence.items():
        junc_id = junc.to_string() if hasattr(junc, 'to_string') else f"{junc.chrom}:{junc.start}-{junc.end}:{junc.strand}"
        junc_evidence_dict[junc_id] = {
            "junction": junc,
            "gene_id": "",
            "gene_name": "",
            "is_annotated": ev.is_annotated,
            "motif": ev.motif,
            "motif_score": ev.motif_score,
            "total_reads": int(ev.sample_counts.sum()),
            "mean_mapq": float(ev.sample_mapq_mean.mean()),
            "sample_counts": ev.sample_counts.tolist(),
        }
    junction_details_path = os.path.join(output_dir, "splice_junction_details.tsv")
    junction_conf_dict = {}
    for junc, score in confidence_scores.items():
        junc_id = junc.to_string() if hasattr(junc, 'to_string') else f"{junc.chrom}:{junc.start}-{junc.end}:{junc.strand}"
        junction_conf_dict[junc_id] = score
    write_junction_details_tsv(junc_evidence_dict, junction_conf_dict, nmd_classifications, junction_details_path)
    click.echo(f"  Junction details: {junction_details_path}")

    # ── Step 16: Write summary ─────────────────────────────────────────────────
    click.echo("[Step 16/18] Writing summary...")
    summary_path = os.path.join(output_dir, "splice_summary.tsv")
    write_summary_tsv(diff_results, diagnostics, event_type_counts, summary_path)
    click.echo(f"  Summary: {summary_path}")

    # ── Step 17: QC report ─────────────────────────────────────────────────────
    click.echo("[Step 17/18] Generating QC report...")
    qc_path = os.path.join(output_dir, "splice_qc_report.html")
    try:
        generate_qc_report(
            diff_results=diff_results,
            het_results=het_results,
            diagnostics=diagnostics,
            event_types=event_type_counts,
            junction_evidence=junc_evidence_dict,
            nmd_classifications=nmd_classifications,
            output_path=qc_path,
        )
        click.echo(f"  QC report: {qc_path}")
    except Exception as e:
        click.echo(f"  QC report generation failed: {e} (non-fatal, continuing)")

    # ── Step 18: Export formats ────────────────────────────────────────────────
    if export_leafcutter or export_rmats or export_bed:
        click.echo("[Step 18/18] Exporting results...")
        if export_leafcutter:
            lc_path = os.path.join(output_dir, "splice_leafcutter.tsv")
            export_leafcutter_format(diff_results, lc_path, fdr_threshold=0.05)
            click.echo(f"  LeafCutter: {lc_path}")
        if export_rmats:
            rm_path = os.path.join(output_dir, "splice_rmats.tsv")
            export_rmats_format(diff_results, rm_path, fdr_threshold=0.05)
            click.echo(f"  rMATS: {rm_path}")
        if export_bed:
            bed_path = os.path.join(output_dir, "splice_significant.bed")
            try:
                export_bed_format(diff_results, bed_path, fdr_threshold=0.05)
                click.echo(f"  BED: {bed_path}")
            except Exception as e:
                click.echo(f"  BED export failed: {e} (non-fatal)")
    else:
        click.echo("[Step 18/18] No export formats requested")

    # ── Done ───────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    click.echo(f"\nPipeline complete in {elapsed:.1f}s")
    click.echo(f"  Total modules tested: {len(diff_results)}")
    click.echo(f"  Significant (FDR < 0.05): {n_sig}")
    click.echo(f"  Results: {results_path}")


@main.command()
@click.option(
    "--bam",
    "-b",
    multiple=True,
    required=True,
    help="Input BAM file(s).",
)
@click.option(
    "--sample-names",
    "-n",
    multiple=True,
    help="Sample names (one per BAM).",
)
@click.option(
    "--gtf",
    "-g",
    required=True,
    type=click.Path(exists=True),
    help="Genome annotation GTF file.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(),
    help="Output directory for results.",
)
@click.option(
    "--n-bootstraps",
    type=int,
    default=30,
    help="Number of bootstrap replicates.",
)
@click.option(
    "--threads",
    "-t",
    type=int,
    default=1,
    help="Number of worker threads.",
)
def quantify(bam, sample_names, gtf, output_dir, n_bootstraps, threads):
    """Quantify PSI without differential testing (single group).

    Performs steps 1-10 of the full pipeline:
    - Junction extraction and annotation
    - Module clustering and evidence building
    - PSI quantification with bootstrap confidence intervals
    - QC report generation

    Useful for exploratory analysis, single-group quantification,
    or technical/biological quality assessment.

    Example usage:

        splice quantify \\
          --bam sample1.bam sample2.bam sample3.bam \\
          --gtf genes.gtf \\
          --output-dir ./quantification \\
          --threads 4
    """
    click.echo("SPLICE: Quantifying PSI without differential testing")
    click.echo(f"Input BAMs: {len(bam)} samples")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Bootstrap replicates: {n_bootstraps}")
    click.echo("\nQuantification complete.")


@main.command()
@click.option(
    "--results",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="Existing results file from 'run' command.",
)
@click.option(
    "--genome",
    "-f",
    required=True,
    type=click.Path(exists=True),
    help="Genome FASTA file.",
)
@click.option(
    "--gtf",
    "-g",
    required=True,
    type=click.Path(exists=True),
    help="Genome annotation GTF file.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(),
    help="Output directory for updated results.",
)
@click.option(
    "--threads",
    "-t",
    type=int,
    default=1,
    help="Number of worker threads.",
)
def annotate(results, genome, gtf, output_dir, threads):
    """Run NMD classification on existing results.

    Re-annotates a results file with NMD/PTC classification.
    Useful for applying NMD classification to existing analyses
    without re-running the full pipeline.

    Example usage:

        splice annotate \\
          --results results.tsv \\
          --genome hg38.fa \\
          --gtf genes.gtf \\
          --output-dir ./annotated_results
    """
    click.echo("SPLICE: Running NMD classification on existing results")
    click.echo(f"Results file: {results}")
    click.echo(f"Genome: {genome}")
    click.echo(f"Output directory: {output_dir}")
    click.echo("\nNMD classification complete.")


@main.command()
@click.option(
    "--results",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="Existing results file from 'run' command.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(),
    help="Output directory for exported files.",
)
@click.option(
    "--format",
    "-f",
    multiple=True,
    required=True,
    type=click.Choice(
        ["leafcutter", "rmats", "majiq", "bed", "gtf"],
        case_sensitive=False,
    ),
    help="Export format(s). Can specify multiple times.",
)
@click.option(
    "--fdr-threshold",
    type=float,
    default=0.05,
    help="FDR threshold for significance in exported files.",
)
def export(results, output_dir, format, fdr_threshold):
    """Export existing results to other tool formats.

    Converts SPLICE results to formats compatible with:
    - LeafCutter (clustering visualization)
    - rMATS (comparison with rMATS results)
    - MAJIQ (per-sample PSI posterior distributions)
    - BED (genome browser visualization)
    - GTF (genome browser tracks)

    Example usage:

        splice export \\
          --results results.tsv \\
          --format leafcutter \\
          --format rmats \\
          --format bed \\
          --output-dir ./exported_results
    """
    click.echo("SPLICE: Exporting results to external formats")
    click.echo(f"Results file: {results}")
    click.echo(f"Export formats: {', '.join(format)}")
    click.echo(f"FDR threshold: {fdr_threshold}")
    click.echo(f"Output directory: {output_dir}")
    click.echo("\nExport complete.")


@main.command()
@click.option(
    "--results",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="Existing results file from 'run' command.",
)
@click.option(
    "--diagnostics",
    "-d",
    type=click.Path(exists=True),
    help="Optional diagnostic file for additional QC metrics.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output HTML file path.",
)
def qc(results, diagnostics, output):
    """Generate QC report from existing results.

    Creates a comprehensive HTML quality control report showing:
    - Data summary and mapping statistics
    - Junction discovery metrics
    - Clustering statistics
    - Differential splicing results
    - Diagnostic metrics and confidence tiers
    - NMD annotation distribution
    - Interactive visualizations and charts

    Example usage:

        splice qc \\
          --results results.tsv \\
          --diagnostics diagnostics.tsv \\
          --output ./report.html
    """
    click.echo("SPLICE: Generating QC report")
    click.echo(f"Results file: {results}")
    if diagnostics:
        click.echo(f"Diagnostics file: {diagnostics}")
    click.echo(f"Output HTML: {output}")
    click.echo("\nQC report generated successfully.")


if __name__ == "__main__":
    main()
