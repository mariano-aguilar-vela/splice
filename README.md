# SPLICE: Splicegraph Probabilistic Learning for Isoform Change Estimation

**SPLICE** is a comprehensive platform for discovery and analysis of differential splicing events in RNA-seq data. It combines annotation-free junction discovery, multi-way statistical testing, covariate regression, heterogeneity detection, and functional annotation into a single unified framework.

**Version:** 1.0.0
**License:** MIT
**Language:** Python >= 3.10 + Rust (required for production)

---

## Key Features

- **Chromosome-level parallelism** processes all chromosomes in parallel for major speedup
- **Rust-accelerated BAM reader** provides 10-50x faster junction extraction
- **Annotation-free junction discovery** using bipartite-graph union-find clustering (O(N) time)
- **Dirichlet-multinomial GLM** with null-refit strategy for proper FDR calibration
- **Vectorized bootstrap** PSI quantification
- **Covariate and heterogeneity support** (batch correction, per-patient effects)
- **NMD/PTC classification** using transcript-based analysis
- **Cross-tool comparison** against rMATS, MAJIQ, SUPPA2
- **Publication-quality outputs**: Excel workbooks, PDF reports, sashimi plots

---

## Installation

### Step 1: Python package

```bash
git clone https://github.com/mariano-aguilar-vela/splice.git
cd splice
pip install -e .
```

### Step 2: Rust acceleration (required for production)

```bash
splice build-rust
```

This automatically installs the Rust toolchain (if not present), installs maturin, and compiles the native BAM reader. It requires a C compiler, which is already present in any environment where pysam is installed.

Verify:
```bash
python -c "from splice._rust_bam import RUST_AVAILABLE; print(f'Rust: {RUST_AVAILABLE}')"
```

If Rust is not compiled, SPLICE falls back to a pure Python BAM reader automatically. Results are identical but ~10x slower. **Production runs should always use Rust.**

### Manual Rust install (alternative)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
pip install maturin
cd /path/to/splice
maturin develop --release
```

---

## Performance

Benchmarks on 6 human RNA-seq BAMs (~170M reads each):

| Configuration | Junction Extraction | Full Pipeline |
|---------------|---------------------|---------------|
| Python, sequential | ~4 hours | ~5 hours |
| Python, 8-thread chromosome parallel | ~45 min | ~1.5 hours |
| **Rust, 8-thread chromosome parallel** | **~5 min** | **~25 min** |

### Why it's fast

**Chromosome-level parallelism**: SPLICE processes each chromosome as an independent pipeline stage. Splicing events on different chromosomes never interact, so the results are mathematically identical to sequential processing while running ~8x faster on 8 cores. Global FDR correction is applied after merging per-chromosome results.

**Rust BAM reader**: Junction extraction is the bottleneck step. The Rust implementation reads BAMs via rust-htslib with zero Python interpreter overhead per read, producing ~10-50x speedup over pure Python.

---

## Quick Start

### Complete differential splicing analysis

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
  --n-bootstraps 100 \
  --export-rmats --export-bed
```

On startup you will see:
```
SPLICE: Splicegraph Probabilistic Learning for Isoform Change Estimation
Input BAMs: 6 samples
  Group 1 (3 samples): S1, S2, S3
  Group 2 (3 samples): S4, S5, S6
Output directory: ./results
BAM reader: Rust-accelerated
Parallelism: 8 chromosome workers
```

### Pipeline stages

```
Step 1:  Parse GTF annotation
Step 2:  Detect chromosomes in BAM files
Step 3:  Process chromosomes in parallel:
           - Extract junctions (Rust)
           - Score junction confidence
           - Pre-filter low-count junctions
           - Cluster junctions (union-find)
           - Build splicegraph and modules
           - Build evidence matrices
           - Quantify PSI with bootstrap CIs
           - Test differential splicing (DM-GLM + null-refit)
           - Heterogeneity testing
           - Event classification
Step 4:  Merge chromosome results + global FDR correction
Step 5:  NMD/PTC classification
Step 6:  Write results table
Step 7:  Write junction details
Step 8:  Write summary
Step 9:  Generate QC report (HTML)
Step 10: Export formats (rMATS, BED, XLSX, PDF)
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `splice run` | Full differential splicing pipeline |
| `splice quantify` | PSI quantification only (single group) |
| `splice annotate` | Add NMD classification to existing results |
| `splice export` | Convert results to other tool formats |
| `splice qc` | Generate QC report from existing results |
| `splice compare` | Cross-tool comparison (rMATS/MAJIQ/SUPPA2) |
| `splice sashimi` | Generate sashimi plots for top events |
| `splice build-rust` | Build/install the Rust BAM reader |

### `splice run` -- Full pipeline

**Required:**
- `-b, --bam` *(multiple)* -- Input BAM file(s). Specify once per file: `-b a.bam -b b.bam`
- `-g, --gtf` -- Genome annotation GTF file
- `--group1` -- Comma-separated sample indices for group 1 (0-indexed)
- `--group2` -- Comma-separated sample indices for group 2 (0-indexed)
- `-o, --output-dir` -- Output directory

**Key parameters:**
| Flag | Default | Description |
|------|---------|-------------|
| `-t, --threads` | 1 | Number of chromosome workers (set to 8 for full speed) |
| `--n-bootstraps` | 30 | Bootstrap replicates for PSI confidence intervals (use 100 for publication) |
| `--min-cluster-reads` | 30 | Minimum total reads per junction before clustering |
| `--max-intron-length` | 100,000 | Maximum intron length for clustering |
| `--min-anchor` | 6 | Minimum anchor length on each side of junction |
| `--min-mapq` | 0 | Minimum mapping quality |

**Feature toggles:**
- `--no-novel` -- Disable de novo junction discovery
- `--no-nmd` -- Skip NMD classification
- `--no-het` -- Skip heterogeneity testing
- `--no-exon-body` -- Skip exon body counting

**Export formats:**
- `--export-rmats` -- rMATS-compatible TSV
- `--export-leafcutter` -- LeafCutter format
- `--export-bed` -- BED for genome browser
- `--export-xlsx` -- Multi-sheet Excel workbook
- `--export-pdf` -- PDF report with figures

**Advanced:**
- `-f, --genome` -- Genome FASTA (enables NMD and motif scoring)
- `-n, --sample-names` *(multiple)* -- Sample names
- `--covariates` -- TSV with sample covariates for batch correction
- `--checkpoint-dir` -- Enable resumable checkpointing

### `splice compare` -- Cross-tool comparison

```bash
splice compare \
  --splice-dir ./results \
  --rmats-dir ./rmats_out \
  --majiq-dir ./majiq_out \
  --suppa2-dir ./suppa2_out \
  --output-dir ./comparison
```

Produces concordance statistics, Venn diagram, UpSet plot, delta-PSI correlation plots, and pairwise Jaccard heatmap.

### `splice sashimi` -- Sashimi plots

```bash
splice sashimi \
  --results ./results/splice_results.tsv \
  --bam-group1 c1.bam -bam-group1 c2.bam -bam-group1 c3.bam \
  --bam-group2 t1.bam -bam-group2 t2.bam -bam-group2 t3.bam \
  --gtf genes.gtf \
  --output-dir ./sashimi \
  --n-top 20
```

Generates multi-panel sashimi plots for the top 20 significant events, saved as both SVG and 300 DPI PNG.

### Other commands

```bash
# Quantify PSI without differential testing
splice quantify -b s1.bam -b s2.bam --gtf genes.gtf -o ./quant

# Add NMD classification to existing results
splice annotate --results results.tsv --genome hg38.fa --gtf genes.gtf -o ./annotated

# Export to other tool formats
splice export --results results.tsv --format rmats --format bed -o ./exports

# Generate QC report
splice qc --results results.tsv --output report.html
```

---

## Output Files

After `splice run`, the output directory contains:

| File | Description |
|------|-------------|
| **`splice_results.tsv`** | Main results: one row per splicing module with gene info, event type, p-value, FDR, delta-PSI, PSI per group for the event-defining junction, confidence tier, convergence diagnostics |
| **`splice_junction_details.tsv`** | Per-junction details: coordinates, annotation status, motif score, confidence, per-sample read counts |
| **`splice_summary.tsv`** | Summary statistics: event type counts, significance counts, convergence success rates |
| **`splice_qc_report.html`** | Interactive HTML QC report with volcano plot, event distributions, diagnostic charts |
| `splice_significant.bed` | BED file with significant junctions (if `--export-bed`) |
| `splice_rmats.tsv` | rMATS-compatible format (if `--export-rmats`) |
| `splice_leafcutter.tsv` | LeafCutter format (if `--export-leafcutter`) |
| `splice_results.xlsx` | Multi-sheet Excel workbook (if `--export-xlsx`) |
| `splice_report.pdf` | PDF report with figures (if `--export-pdf`) |
| `figures/` | Individual SVG figures from PDF report |
| `checkpoints/` | Resumable pipeline state (if `--checkpoint-dir`) |

### `splice_results.tsv` columns

- `module_id`, `gene_id`, `gene_name`, `chrom`, `strand`
- `event_type` -- SE, A3SS, A5SS, MXE, RI, Complex
- `n_junctions` -- junctions in this module
- `max_abs_delta_psi` -- max absolute effect size
- `psi_group1_max_effect_junction`, `psi_group2_max_effect_junction` -- PSI of the event-defining junction
- `p_value`, `fdr` -- raw and FDR-corrected p-values (global correction across all chromosomes)
- `confidence_tier` -- HIGH/MEDIUM/LOW based on coverage, convergence, and signal
- `null_converged`, `full_converged`, `null_refit_used` -- DM-GLM diagnostics
- `mean_mapq`, `frac_high_mapq`, `mean_junction_confidence`, `bootstrap_cv`
- `has_novel_junctions`, `has_convergence_issue`, `reason`

---

## Input Requirements

### BAM files
- Coordinate-sorted (`samtools sort`)
- Indexed (`samtools index`)
- STAR aligner recommended (junctions from CIGAR N operations)
- Multi-mapped reads weighted by 1/NH tag if present

### GTF file
- Standard GTF format
- Must include `gene_id`, `transcript_id` attributes
- `exon` features required
- Strand information required

### Covariates file (optional TSV)
```
sample  batch   age     sex
S1      batch1  65      M
S2      batch1  72      F
S3      batch2  68      M
```

---

## Troubleshooting

### "BAM reader: Python (slow mode)" warning
- Run `splice build-rust` to enable the Rust reader
- If that fails, check you have a C compiler (try `conda install gxx_linux-64`)
- SPLICE still works in Python mode, just slower

### "No junctions detected"
- Verify BAMs are coordinate-sorted and indexed
- Try `--min-anchor 4` and `--min-cluster-reads 10`

### "All FDR = 1.0"
- Insufficient signal or too few replicates (minimum 3 per group)
- Check `splice_summary.tsv` for coverage metrics

### "Rust build fails: cannot find stddef.h"
- Load a newer GCC: `module load GCC/14.2.0` (HPC systems)
- Or install: `conda install -c conda-forge gcc gxx`

---

## Citation

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
