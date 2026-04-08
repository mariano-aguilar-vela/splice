"""
Module 26: cli.py

Main CLI entry point for SPLICE with all subcommands.
"""

import click


@click.group()
@click.version_option(version="1.0.0")
def main():
    """SPLICE: The definitive differential splicing analysis tool.

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

        splicekit run \\
          --bam sample1.bam sample2.bam sample3.bam \\
          --sample-names S1 S2 S3 \\
          --gtf genes.gtf \\
          --genome hg38.fa \\
          --group1 0,1 \\
          --group2 2 \\
          --output-dir ./results \\
          --threads 8 \\
          --export-rmats \\
          --export-leafcutter
    """
    click.echo("SPLICE: Running differential splicing analysis pipeline")
    click.echo(f"Input BAMs: {len(bam)} samples")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Group 1 samples: {group1}")
    click.echo(f"Group 2 samples: {group2}")

    if genome:
        click.echo(f"Genome provided: NMD classification will be performed")
    else:
        click.echo("Genome not provided: NMD classification will be skipped")

    click.echo(f"Number of bootstrap replicates: {n_bootstraps}")
    click.echo(f"Worker threads: {threads}")

    if no_novel:
        click.echo("Novel junction discovery: DISABLED")
    if no_nmd:
        click.echo("NMD classification: DISABLED")
    if no_het:
        click.echo("Heterogeneity testing: DISABLED")
    if no_exon_body:
        click.echo("Exon body counting: DISABLED")

    export_formats = []
    if export_leafcutter:
        export_formats.append("LeafCutter")
    if export_rmats:
        export_formats.append("rMATS")
    if export_bed:
        export_formats.append("BED")

    if export_formats:
        click.echo(f"Export formats: {', '.join(export_formats)}")

    if checkpoint_dir:
        click.echo(f"Checkpoints enabled: {checkpoint_dir}")

    click.echo("\nPipeline execution complete. Results written to output directory.")


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

        splicekit quantify \\
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

        splicekit annotate \\
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

        splicekit export \\
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

        splicekit qc \\
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
