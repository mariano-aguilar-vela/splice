# SPLICE: Splicegraph Probabilistic Learning for Isoform Change Estimation

**SPLICE** (Splicegraph Probabilistic Learning for Isoform Change Estimation) is a comprehensive platform for discovery and analysis of differential splicing events in RNA-seq data. It combines annotation-free junction discovery, multi-way statistical testing, covariate regression, heterogeneity detection, and functional annotation into a single unified framework.

**Version:** 1.0.0  
**License:** MIT  
**Language:** Python >= 3.10

---

## What Makes SPLICE Different

Most differential splicing tools were built 5-10 years ago and have significant gaps. SPLICE addresses every documented limitation simultaneously:

### 1. **Annotation-Free Junction Discovery**
Other tools (rMATS, SUPPA2) either can't find novel junctions or require annotations as anchors. SPLICE combines:
- LeafCutter's fully annotation-free discovery (no annotation required)
- STAR's motif-aware scoring (GT/AG preferred over GC/AG over non-canonical)
- MAJIQ's principled cross-sample thresholding (minimum read support across samples)

### 2. **Multi-Way Statistical Testing**
Tools like rMATS and MAJIQ test exactly two isoforms per event. SPLICE's Dirichlet-multinomial GLM:
- Tests all junctions in a module simultaneously (not pairwise)
- Detects complex splicing events (>2 isoforms, retained introns, complex exon skipping)
- Maintains proper Type I error control with >2 groups

### 3. **Covariate and Heterogeneity Support**
Few tools support batch correction or per-patient effects. SPLICE offers:
- GLM design matrix support for covariates (batch, age, sex, treatment, etc.)
- Continuous covariate adjustment (unlike MAJIQ's binary HET mode)
- Per-sample random effects for heterogeneous treatment responses
- Nonparametric heterogeneity testing (outlier detection)

### 4. **Multiple Evidence Types**
Most tools use only junction reads. SPLICE integrates:
- Junction-spanning reads (primary evidence)
- Exon body reads (complementary evidence from non-junction regions)
- Junction co-occurrence patterns (detects linked splicing events)
- MAPQ-weighted evidence (proper uncertainty quantification)

### 5. **Functional Classification and NMD**
SPLICE provides:
- Automatic classification into classical event types (SE, A3SS, A5SS, MXE, RI, etc.)
- Novel complex event recognition
- Graph-based NMD/PTC prediction using genome sequence
- Per-event functional impact assessment

### 6. **Comprehensive Diagnostics**
Rather than silent failures (seen in rMATS, SUPPA2, MAJIQ), SPLICE provides:
- Per-event diagnostic metrics (confidence tier, outlier flags, coverage metrics)
- Explicit error messages for problematic data
- QC report with interactive visualizations
- Cross-tool export formats (LeafCutter, rMATS, MAJIQ, BED, GTF)

### 7. **Resumable Checkpointing**
Long analyses can be interrupted and resumed:
- Checkpoint intermediate results after each pipeline stage
- Resume from last checkpoint without re-running prior work
- Useful for large cohorts (>100 samples) or resource-constrained environments

---

## Installation

### Requirements
- Python >= 3.10
- STAR-aligned BAM files (coordinate-sorted, indexed)
- GTF/GFF3 genome annotation (optional but recommended)
- Genome FASTA file (optional; enables NMD classification and motif scoring)

### From PyPI (future)
```bash
pip install splicekit
```

### From Source
```bash
git clone https://github.com/your-repo/splicekit.git
cd splicekit
pip install -e .
```

### Dependencies
All dependencies are automatically installed. Key packages:
- `pysam` (BAM/GTF I/O)
- `numpy`, `scipy`, `pandas` (numerical computation)
- `statsmodels` (statistical models)
- `numba` (JIT compilation for hot paths)
- `click` (CLI framework)
- `matplotlib`, `seaborn` (QC visualization)

---

## Quick Start

### Basic Differential Splicing Analysis
```bash
splice run \
  --bam sample1.bam sample2.bam sample3.bam sample4.bam sample5.bam sample6.bam \
  --sample-names S1 S2 S3 S4 S5 S6 \
  --gtf genes.gtf \
  --genome hg38.fa \
  --group1 0,1,2 \
  --group2 3,4,5 \
  --output-dir ./results \
  --threads 8
```

This runs the complete pipeline:
1. Extracts junctions and exon body reads from BAMs
2. Discovers novel junctions and clusters with annotated ones
3. Scores junction confidence based on motifs
4. Builds splicing modules and evidence matrices
5. Quantifies PSI (percent spliced in) with bootstrap confidence intervals
6. Tests for differential splicing (Dirichlet-multinomial GLM)
7. Classifies events into functional categories
8. Generates NMD predictions
9. Writes results tables, per-junction details, and QC report

### Quantify PSI Without Differential Testing
For single-group studies or exploratory analysis:
```bash
splice quantify \
  --bam sample1.bam sample2.bam sample3.bam \
  --gtf genes.gtf \
  --output-dir ./quantification \
  --threads 4
```

### Add NMD Classification to Existing Results
Re-annotate a previous analysis with NMD prediction:
```bash
splice annotate \
  --results results.tsv \
  --genome hg38.fa \
  --gtf genes.gtf \
  --output-dir ./annotated_results
```

### Export to Other Tools
Convert results to formats compatible with downstream tools:
```bash
splice export \
  --results results.tsv \
  --format leafcutter \
  --format rmats \
  --format bed \
  --output-dir ./exported
```

### Generate QC Report
Create an interactive HTML report from existing results:
```bash
splice qc \
  --results results.tsv \
  --diagnostics diagnostics.tsv \
  --output report.html
```

---

## Full CLI Reference

### Global Options
```
--version  Show version and exit
--help     Show help message
```

---

## `splice run` — Complete Differential Splicing Pipeline

The main entry point. Runs all 18 pipeline stages end-to-end.

### Required Options

**Input Files:**
- `--bam, -b` *(multiple)* — Input BAM file(s). Must be coordinate-sorted and indexed. Can specify multiple times: `--bam a.bam --bam b.bam`
- `--gtf, -g` — Genome annotation GTF file. Used for initial splicing graph construction.
- `--group1` — Comma-separated sample indices for group 1 (0-indexed). Example: `--group1 0,1,2`
- `--group2` — Comma-separated sample indices for group 2 (0-indexed). Example: `--group2 3,4,5`
- `--output-dir, -o` — Output directory for all results.

**Optional Input Files:**
- `--genome, -f` — Genome FASTA file. Enables motif scoring and NMD classification. If not provided, NMD classification is skipped.
- `--sample-names, -n` *(multiple)* — Sample names (one per BAM). If not provided, names are derived from BAM filenames.
- `--covariates` — TSV file with sample covariates. Format: first column is sample name, second column is covariate (numeric). Supports multiple covariates as additional columns.

### Optional Parameters

**Filtering:**
- `--read-length, -l` — Read length in bp. Auto-detected from BAMs if not specified.
- `--min-anchor` — Minimum anchor length for valid junctions (default: 6 bp). Junctions with anchors shorter than this are discarded.
- `--min-mapq` — Minimum mapping quality (default: 0). Reads below this threshold are excluded.
- `--min-cluster-reads` — Minimum total reads in a cluster for inclusion (default: 30).
- `--max-intron-length` — Maximum intron length in bp (default: 100,000). Gaps larger than this are not treated as junctions.

**Quantification:**
- `--n-bootstraps` — Number of bootstrap replicates for PSI confidence intervals (default: 30). Higher values increase precision but slow quantification.

**Parallelization:**
- `--threads, -t` — Number of worker threads (default: 1). Enables chromosome-level parallelism.

**Feature Toggles:**
- `--no-novel` — Disable de novo junction discovery. Use only annotated junctions.
- `--no-nmd` — Skip NMD classification step (saves time if genome not provided anyway).
- `--no-het` — Skip heterogeneity testing step (per-sample effect detection).
- `--no-exon-body` — Skip exon body read counting. Use only junction-spanning reads.

**Export Formats:**
- `--export-leafcutter` — Export results in LeafCutter format (for clustering visualization).
- `--export-rmats` — Export results in rMATS format (for comparison with rMATS runs).
- `--export-bed` — Export significant junctions as BED file (for genome browser).

**Advanced:**
- `--checkpoint-dir` — Directory for checkpoint files. If specified, enables resumable pipeline execution.

### Outputs

**Main Results File:**
- `results.tsv` — Differential splicing results. Columns:
  - `gene_id`, `gene_name`, `chrom`, `strand`
  - `module_id` — Unique identifier for this splicing module
  - `junctions` — List of junctions (chrom:donor-acceptor strand)
  - `pvalue`, `padj` — P-value and FDR-adjusted p-value
  - `effect_size` — Delta-PSI (difference between groups)
  - `event_type` — Classification (SE, A3SS, A5SS, MXE, RI, Complex)
  - `n_samples` — Number of samples with data for this module
  - Additional columns for each sample: PSI value, confidence interval, outlier flag

**Per-Junction Details:**
- `per_junction_details.tsv` — One row per junction. Columns:
  - Junction coordinates and strand
  - Raw and normalized read counts per sample
  - Motif score and annotated status (novel vs annotated)
  - Cross-sample consistency metrics

**Diagnostics:**
- `diagnostics.tsv` — Per-event diagnostic metrics. Columns:
  - `module_id`
  - `confidence_tier` — HIGH/MEDIUM/LOW based on coverage and signal strength
  - `outlier_samples` — Samples with evidence of outlier effects
  - `coverage_group1`, `coverage_group2` — Mean junction reads per group
  - `specificity` — Specificity of the module to this gene pair

**QC Report:**
- `qc_report.html` — Interactive HTML report with:
  - Summary statistics (samples, junctions, modules)
  - Clustering dendrogram
  - Differential splicing volcano plot
  - Per-event diagnostic distribution
  - NMD annotation breakdown (if genome provided)

**Optional Exports:**
- `leafcutter/` — LeafCutter format (for Shiny visualization)
- `rmats/` — rMATS format (SE, MXE, A5SS, A3SS, RI tables)
- `majiq/` — MAJIQ format (per-sample PSI posteriors)
- `results.bed` — BED format (significant junctions)
- `results.gtf` — GTF format (de novo junctions)

**Checkpoints (if `--checkpoint-dir` specified):**
- `junction_evidence.pkl` — Extracted junctions after each stage
- `clusters.pkl` — Junction clusters
- `modules.pkl` — Splicing modules and evidence matrices
- `psi_list.pkl` — PSI quantifications
- `diff_results.pkl` — Differential testing results

### Example: Complete Analysis with Covariates
```bash
splice run \
  --bam control_1.bam control_2.bam control_3.bam \
       case_1.bam case_2.bam case_3.bam \
  --sample-names C1 C2 C3 E1 E2 E3 \
  --gtf genes.gtf \
  --genome hg38.fa \
  --group1 0,1,2 \
  --group2 3,4,5 \
  --covariates covariates.tsv \
  --output-dir ./results \
  --threads 8 \
  --n-bootstraps 30 \
  --export-rmats \
  --export-leafcutter \
  --checkpoint-dir ./checkpoints
```

**covariates.tsv format:**
```
sample  batch   age     sex
C1      batch1  65      M
C2      batch1  72      F
C3      batch2  68      M
E1      batch1  71      F
E2      batch2  63      M
E3      batch2  70      F
```

---

## `splice quantify` — PSI Quantification Only

Quantify percent spliced in (PSI) without differential testing. Useful for:
- Single-group studies
- Exploratory analysis
- Quality assessment
- Upstream quantification for external statistical tools

### Required Options
- `--bam, -b` *(multiple)* — Input BAM file(s)
- `--gtf, -g` — Genome annotation
- `--output-dir, -o` — Output directory

### Optional Options
- `--sample-names, -n` *(multiple)* — Sample names
- `--n-bootstraps` — Bootstrap replicates (default: 30)
- `--threads, -t` — Worker threads (default: 1)

### Outputs
Same as `run`, but stopping after stage 10 (PSI quantification). No differential testing results.

---

## `splice annotate` — Add NMD Classification

Re-annotate existing results with NMD/PTC predictions using genome sequence.

### Required Options
- `--results, -r` — Results TSV from a prior `run` or `quantify`
- `--genome, -f` — Genome FASTA file
- `--gtf, -g` — Genome annotation GTF
- `--output-dir, -o` — Output directory

### Optional Options
- `--threads, -t` — Worker threads (default: 1)

### Outputs
Updated `results.tsv` with additional columns:
- `nmd_classification` — NMD, Escape-NMD, or No-PTC
- `ptc_location` — Genomic position of predicted premature termination codon
- `distance_to_junction` — Distance from PTC to nearest EJC-defining junction

---

## `splice export` — Convert Formats

Export existing results to formats compatible with other tools.

### Required Options
- `--results, -r` — Results TSV
- `--format, -f` *(multiple)* — Export format(s). Can specify multiple times. Options:
  - `leafcutter` — LeafCutter clustering format (for Shiny visualization)
  - `rmats` — rMATS format (SE, MXE, A5SS, A3SS, RI tables)
  - `majiq` — MAJIQ format (per-sample PSI posteriors, Delta-PSI distributions)
  - `bed` — BED format (junctions and regions)
  - `gtf` — GTF format (junctions as GTF features)
- `--output-dir, -o` — Output directory

### Optional Options
- `--fdr-threshold` — Include only events with FDR < this value (default: 0.05)

### Example
```bash
splice export \
  --results results.tsv \
  --format leafcutter \
  --format rmats \
  --format bed \
  --fdr-threshold 0.01 \
  --output-dir ./exports
```

---

## `splice qc` — Generate QC Report

Create an interactive HTML quality control report from existing results.

### Required Options
- `--results, -r` — Results TSV
- `--output, -o` — Output HTML file path

### Optional Options
- `--diagnostics, -d` — Diagnostic TSV (adds confidence metrics to report)

### Report Contents
- **Summary** — Sample count, junction count, module count, significant events
- **Data Quality** — Read count distributions, junction discovery curves
- **Clustering** — Hierarchical clustering dendrogram of samples
- **Differential Results** — Volcano plot (effect size vs p-value)
- **Event Classifications** — Breakdown by event type (SE, A3SS, etc.)
- **Diagnostics** — Confidence tier distribution, outlier detection
- **NMD** — Distribution of NMD classifications (if available)
- **Per-Event Details** — Interactive table with drill-down to individual modules

---

## Input File Formats

### BAM Files
- Must be coordinate-sorted (`samtools sort`)
- Must be indexed (`samtools index`)
- Recommended to be from STAR aligner with junctions in SJ.out.tab
- Can include multi-mapped reads (weighted by 1/NH tag if present)

Example preparation:
```bash
STAR --runThreadN 8 \
  --genomeDir hg38_index \
  --readFilesIn reads_R1.fastq reads_R2.fastq \
  --outFileNamePrefix sample_

samtools sort -o sample_Aligned.sortedByCoord.out.bam \
  sample_Aligned.out.bam

samtools index sample_Aligned.sortedByCoord.out.bam
```

### GTF/GFF3 Files
Standard GTF/GFF3 format. Must include:
- `gene_id` and `transcript_id` attributes
- `exon` features (or `CDS` + `UTR`)
- Strand information (+ or -)

Example (GTF):
```
chr1	HAVANA	gene	100	1000	.	+	.	gene_id "ENSG00000001"; gene_name "GENE1"
chr1	HAVANA	transcript	100	1000	.	+	.	gene_id "ENSG00000001"; transcript_id "ENST00000001"
chr1	HAVANA	exon	100	200	.	+	.	gene_id "ENSG00000001"; transcript_id "ENST00000001"
chr1	HAVANA	exon	800	1000	.	+	.	gene_id "ENSG00000001"; transcript_id "ENST00000001"
```

### Covariates File (TSV)
First column: sample name (must match BAM sample names).  
Remaining columns: covariate values (numeric or categorical).

Example:
```
sample  batch   treatment   age
S1      batch1  control     45
S2      batch1  control     52
S3      batch2  treated     48
S4      batch2  treated     51
```

---

## Output File Formats

### results.tsv
Tab-separated values. One row per splicing module. Key columns:
- `gene_id`, `gene_name` — Gene identifier
- `chrom`, `strand` — Genomic location
- `module_id` — Unique module identifier
- `junctions` — Semicolon-separated list of junctions in this module
- `pvalue`, `padj` — Raw p-value and FDR-corrected p-value
- `effect_size` — Delta-PSI (group2 PSI - group1 PSI)
- `event_type` — Classification (SE, A3SS, A5SS, MXE, RI, Complex)
- `S1_psi`, `S1_ci_low`, `S1_ci_high` — PSI and 95% CI for sample S1
- (repeated for all samples)
- `S1_is_outlier` — Boolean flag for outlier detection
- (repeated for all samples)

### per_junction_details.tsv
One row per junction. Columns:
- `chrom`, `donor`, `acceptor`, `strand` — Junction coordinates
- `annotated` — 1 if junction in GTF, 0 if novel
- `motif_score` — Splice site motif score (0-1)
- `raw_counts_*` — Raw read counts per sample
- `normalized_counts_*` — Effective-length-normalized counts per sample
- `strand_consistency` — Fraction of samples agreeing on strand

### diagnostics.tsv
One row per module. Columns:
- `module_id`
- `confidence_tier` — HIGH/MEDIUM/LOW
- `coverage_group1`, `coverage_group2` — Mean junction reads per group
- `outlier_samples` — Comma-separated list of sample IDs with outlier flags
- `specificity` — Gene-specificity score (0-1)

---

## Performance and Scalability

Performance targets for 10 BAMs × 50M reads each on a modern server (32 cores, 128 GB RAM):

| Stage | Time | Notes |
|-------|------|-------|
| Junction extraction | < 30 min | Parallelizes across BAM files and chromosomes |
| Clustering | < 2 min | O(J log J) where J = number of junctions |
| Evidence building | < 10 min | Matrix construction and normalization |
| PSI quantification | < 5 min | 30 bootstrap replicates |
| Differential testing | < 10 min | GLM with covariate support |
| NMD classification | < 15 min | Graph-based prediction from genome |
| **Total pipeline** | **< 90 min** | End-to-end, single-threaded output writing |

Memory usage scales linearly with:
- Number of samples (one row per sample per junction)
- Number of junctions (O(J) storage)
- Bootstrap replicates (parallel processing, linear scaling)

Checkpointing enables resumption and memory management for very large cohorts (>100 samples).

---

## Troubleshooting

### "No junctions detected"
- Check BAM file is coordinate-sorted and indexed
- Verify BAM has SJ.out.tab-format junctions (STAR output) or XSA tags (alternative aligner)
- Ensure `--min-anchor` is not too strict (try `--min-anchor 4`)
- Try `--min-cluster-reads 0` to disable filtering

### "All PSI values are 0 or 1"
- May indicate a single dominant isoform. This is not an error.
- Check diagnostic report for low coverage (may need stricter FDR)
- Compare read counts in per_junction_details.tsv

### "FDR-corrected p-values are all 1.0"
- Insufficient signal or too few biological replicates
- Try:
  - Increasing sample size (minimum 3 per group)
  - Relaxing filtering (`--min-cluster-reads 10`)
  - Checking for batch effects in `--covariates`

### "RuntimeError: Module has missing data for samples X, Y, Z"
- Some samples have zero reads for this module
- Modules are filtered if coverage is below threshold
- This is expected for lowly-expressed genes in some samples
- Check `diagnostics.tsv` for confidence tier

### "NMD classification not available"
- Provide `--genome` (FASTA file) and `--gtf` during `run`, or use `annotate` command
- Verify FASTA file is bgzip-compressed and indexed (pysam requirement)

---

## Citation

If you use SPLICE in published research, please cite:

```
SPLICE: A comprehensive platform for annotation-free differential splicing analysis.
Version 1.0.0. GitHub: https://github.com/your-repo/splicekit
```

---

## Contributing

We welcome contributions! Please:
1. Open an issue describing the feature or bug
2. Fork the repository
3. Create a branch (`git checkout -b feature/my-feature`)
4. Commit changes (`git commit -am 'Add feature'`)
5. Push to branch (`git push origin feature/my-feature`)
6. Open a Pull Request

---

## Author

**Mariano Aguilar Vela**  
- BSc Microbiology and Cell Science, University of Florida
- MSc Biotechnology, University of Queensland
- PhD Biomedical Science (Candidate), Queensland University of Technology
- Email: mariano.aguilarvela@hdr.qut.edu.au

---

## License

SPLICE is released under the MIT License. See LICENSE file for details.

---

## Support

- **Documentation:** See full CLI reference above and output file format specifications
- **Issues:** GitHub issue tracker
- **Questions:** Check existing issues or open a new one
