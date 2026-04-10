# SPLICE: Splicegraph Probabilistic Learning for Isoform Change Estimation

**SPLICE** is a comprehensive platform for discovery and analysis of differential splicing events in RNA-seq data. It combines annotation-free junction discovery, multi-way statistical testing, covariate regression, heterogeneity detection, and functional annotation into a single unified framework.

**Version:** 1.0.0
**License:** MIT
**Language:** Python >= 3.10, with optional Rust acceleration

---

## Key Features

- **Annotation-free junction discovery** using bipartite-graph union-find clustering (O(N) time)
- **Chromosome-level parallelism** -- processes all chromosomes simultaneously for ~3-4x speedup
- **Rust-accelerated BAM reader** -- optional compiled extension for 10-50x faster junction extraction
- **Dirichlet-multinomial GLM** with null-refit strategy for proper FDR calibration
- **Vectorized bootstrap** PSI quantification using numpy broadcasting
- **Covariate and heterogeneity support** (batch correction, per-patient effects)
- **NMD/PTC classification** using transcript-based analysis
- **Cross-tool export** (rMATS, LeafCutter, MAJIQ, BED, GTF formats)

---

## Installation

### Basic Install (Python only)

```bash
git clone https://github.com/mariano-aguilar-vela/splice.git
cd splice
pip install -e .
```

This installs SPLICE with the pure Python BAM reader. All features work, but junction extraction from large BAM files will be slower.

### Enable Rust Acceleration (recommended)

The Rust BAM reader provides 10-50x faster junction extraction. To enable it:

```bash
# Option 1: Automatic (handles Rust installation if needed)
splice build-rust

# Option 2: Manual
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
pip install maturin
cd /path/to/splice
maturin develop --release
```

Verify:
```bash
python -c "from splice._rust_bam import RUST_AVAILABLE; print(f'Rust: {RUST_AVAILABLE}')"
```

If Rust is not compiled, SPLICE uses the Python BAM reader automatically with identical results. No configuration needed.

### From PyPI (future)

```bash
pip install splice
```

Pre-compiled wheels will include the Rust extension for supported platforms.

---

## Performance

Benchmarks on 6 BAM files (~170M reads each, human RNA-seq):

| Configuration | Junction Extraction | Full Pipeline | Notes |
|---------------|-------------------|---------------|-------|
| Python only, sequential | ~4 hours | ~5 hours | Default if Rust not compiled |
| Python + Rust | ~30 min | ~1.5 hours | `splice build-rust` |
| Python + Rust + 8 threads | ~5 min | ~30 min | `-t 8` (recommended) |

Memory usage: ~2-4 GB per worker thread. With 8 threads: ~16-32 GB total.

---

## Quick Start

### Basic Differential Splicing Analysis

```bash
splice run \
  -b sample1.bam -b sample2.bam -b sample3.bam \
  -b sample4.bam -b sample5.bam -b sample6.bam \
  --sample-names S1 S2 S3 S4 S5 S6 \
  --gtf genes.gtf \
  --genome hg38.fa \
  --group1 0,1,2 \
  --group2 3,4,5 \
  --output-dir ./results \
  --threads 8 \
  --n-bootstraps 100
```

### Pipeline Architecture (10 steps)

```
Step 1:  Parse GTF annotation (sequential)
Step 2:  Detect chromosomes in BAM files
Step 3:  Process chromosomes in parallel:
           - Extract junctions (Rust or Python)
           - Score junction confidence
           - Pre-filter low-count junctions
           - Cluster with union-find (O(N) time)
           - Build splicegraph and modules
           - Build evidence matrices
           - Quantify PSI with bootstrap CIs
           - Test differential splicing (DM-GLM + null-refit)
           - Heterogeneity testing
           - Event classification and diagnostics
Step 4:  Merge chromosome results + global FDR correction
Step 5:  NMD/PTC classification (if genome provided)
Step 6:  Write results table
Step 7:  Write junction details
Step 8:  Write summary
Step 9:  Generate QC report (HTML)
Step 10: Export formats (rMATS, LeafCutter, BED)
```

Steps 3 processes all chromosomes simultaneously using `--threads` workers. Each chromosome runs the full analysis pipeline independently, then results are merged with genome-wide FDR correction.

### Analysis with Covariates

```bash
splice run \
  -b control_1.bam -b control_2.bam -b control_3.bam \
  -b case_1.bam -b case_2.bam -b case_3.bam \
  --sample-names C1 C2 C3 E1 E2 E3 \
  --gtf genes.gtf \
  --genome hg38.fa \
  --group1 0,1,2 \
  --group2 3,4,5 \
  --covariates covariates.tsv \
  --output-dir ./results \
  --threads 8 \
  --n-bootstraps 100 \
  --export-rmats \
  --export-leafcutter \
  --checkpoint-dir ./checkpoints
```

### Other Commands

```bash
# Quantify PSI without differential testing
splice quantify -b s1.bam -b s2.bam --gtf genes.gtf -o ./quant

# Add NMD classification to existing results
splice annotate --results results.tsv --genome hg38.fa --gtf genes.gtf -o ./annotated

# Export to other tool formats
splice export --results results.tsv --format rmats --format bed -o ./exports

# Generate QC report
splice qc --results results.tsv --output report.html

# Build Rust extension for faster BAM reading
splice build-rust
```

---

## CLI Reference

### `splice run` -- Full Differential Splicing Pipeline

**Required:**
- `-b, --bam` *(multiple)* -- Input BAM file(s), specify once per file: `-b a.bam -b b.bam`
- `-g, --gtf` -- Genome annotation GTF file
- `--group1` -- Comma-separated sample indices for group 1 (0-indexed)
- `--group2` -- Comma-separated sample indices for group 2 (0-indexed)
- `-o, --output-dir` -- Output directory

**Optional:**
- `-f, --genome` -- Genome FASTA (enables NMD classification and motif scoring)
- `-n, --sample-names` *(multiple)* -- Sample names (derived from BAM filenames if omitted)
- `--covariates` -- TSV file with sample covariates
- `-l, --read-length` -- Read length (auto-detected if omitted)
- `--min-anchor` -- Minimum anchor length (default: 6)
- `--min-mapq` -- Minimum mapping quality (default: 0)
- `--min-cluster-reads` -- Minimum total reads per cluster (default: 30)
- `--max-intron-length` -- Maximum intron length (default: 100,000)
- `--n-bootstraps` -- Bootstrap replicates for PSI CIs (default: 30)
- `-t, --threads` -- Worker threads (default: 1)
- `--no-novel` -- Disable de novo junction discovery
- `--no-nmd` -- Skip NMD classification
- `--no-het` -- Skip heterogeneity testing
- `--no-exon-body` -- Skip exon body counting
- `--export-leafcutter` -- Export LeafCutter format
- `--export-rmats` -- Export rMATS format
- `--export-bed` -- Export BED format
- `--checkpoint-dir` -- Enable resumable checkpointing

---

## Output Files

### `splice_results.tsv`

One row per splicing module:
- `gene_id`, `gene_name`, `chrom`, `strand`
- `module_id` -- Unique module identifier
- `event_type` -- SE, A3SS, A5SS, MXE, RI, Complex
- `p_value`, `fdr` -- Raw and FDR-corrected p-values
- `delta_psi` -- PSI difference between groups
- Per-sample PSI values and confidence intervals

### `splice_junction_details.tsv`

One row per junction: coordinates, annotation status, motif score, per-sample counts, confidence.

### `splice_summary.tsv`

Summary statistics: event type counts, significance thresholds, coverage metrics.

### `splice_qc_report.html`

Interactive HTML report with volcano plots, clustering dendrograms, event type distributions, and diagnostic metrics.

---

## Input File Requirements

### BAM Files
- Coordinate-sorted (`samtools sort`)
- Indexed (`samtools index`)
- STAR aligner recommended (junctions from CIGAR N operations)
- Multi-mapped reads weighted by 1/NH tag

### GTF Files
- Standard GTF format with `gene_id` and `transcript_id` attributes
- `exon` features required
- Strand information required

### Covariates File (TSV)
```
sample  batch   age     sex
S1      batch1  65      M
S2      batch1  72      F
S3      batch2  68      M
```

---

## Troubleshooting

### "No junctions detected"
- Verify BAM is sorted and indexed
- Try `--min-anchor 4` or `--min-cluster-reads 0`

### "All PSI values are 0 or 1"
- Normal for single-isoform genes
- Check per_junction_details.tsv for coverage

### "FDR-corrected p-values are all 1.0"
- Insufficient signal or too few replicates (minimum 3 per group)
- Try relaxing `--min-cluster-reads 10`

### Rust build fails
- Run `splice build-rust` and check the error output
- Common fix: `conda install -c conda-forge zlib bzip2 xz libcurl`
- SPLICE works identically without Rust, just slower

---

## Citation

If you use SPLICE in published research, please cite:

```
SPLICE: Splicegraph Probabilistic Learning for Isoform Change Estimation.
Version 1.0.0. https://github.com/mariano-aguilar-vela/splice
```

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
