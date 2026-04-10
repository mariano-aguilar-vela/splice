#!/bin/bash
#
# Jiang et al. 2023 differential splicing benchmark for SPLICE
#
# Usage:
#   bash benchmark/run_jiang_benchmark.sh [BENCHMARK_DIR] [OUTPUT_DIR]
#
# Default paths:
#   BENCHMARK_DIR=./benchmark/jiang2023_data
#   OUTPUT_DIR=./benchmark/jiang2023_results

set -e

BENCHMARK_DIR="${1:-./benchmark/jiang2023_data}"
OUTPUT_DIR="${2:-./benchmark/jiang2023_results}"
SPLICE_OUT="${OUTPUT_DIR}/splice_run"

mkdir -p "${BENCHMARK_DIR}" "${OUTPUT_DIR}"

echo "=== Step 1: Download benchmark data ==="
python -c "
from benchmark.jiang2023_benchmark import download_benchmark_data
download_benchmark_data('${BENCHMARK_DIR}')
"

REPO_DIR="${BENCHMARK_DIR}/Benchmarking_DS"

echo ""
echo "=== Step 2: Prepare SPLICE input ==="
python -c "
from benchmark.jiang2023_benchmark import prepare_splice_input
result = prepare_splice_input('${REPO_DIR}', '${SPLICE_OUT}')
print(f'  Group 1 BAMs: {len(result[\"bams_group1\"])}')
print(f'  Group 2 BAMs: {len(result[\"bams_group2\"])}')
print(f'  GTF: {result[\"gtf_path\"]}')
print(f'  Genome: {result[\"genome_path\"]}')
print()
print('SPLICE command:')
print(f'  {result[\"splice_command\"]}')
"

echo ""
echo "=== Step 3: Run SPLICE ==="
echo "  This step is intentionally left as a manual step."
echo "  Copy the splice command above and run it."
echo "  When complete, splice_results.tsv should be in ${SPLICE_OUT}/"
echo ""

if [ ! -f "${SPLICE_OUT}/splice_results.tsv" ]; then
    echo "  splice_results.tsv not found. Skipping evaluation."
    echo "  Re-run this script after SPLICE completes."
    exit 0
fi

echo ""
echo "=== Step 4: Evaluate against ground truth ==="
python -c "
from benchmark.jiang2023_benchmark import (
    load_ground_truth, evaluate_splice_results,
    compare_with_published_results, generate_benchmark_report,
)

truth = load_ground_truth('${REPO_DIR}')
print(f'  Loaded {len(truth)} ground truth events ({truth[\"is_differential\"].sum()} differential)')

metrics = evaluate_splice_results(
    '${SPLICE_OUT}/splice_results.tsv', truth, fdr_threshold=0.05,
)
print(f'  Gene-level: TPR={metrics[\"gene_level\"][\"tpr\"]:.3f}, '
      f'FDR={metrics[\"gene_level\"][\"fdr\"]:.3f}, '
      f'F-score={metrics[\"gene_level\"][\"f_score\"]:.3f}')
print(f'  Event-level: TPR={metrics[\"event_level\"][\"tpr\"]:.3f}, '
      f'FDR={metrics[\"event_level\"][\"fdr\"]:.3f}, '
      f'F-score={metrics[\"event_level\"][\"f_score\"]:.3f}')

comparison = compare_with_published_results(metrics, '${REPO_DIR}')
print()
print('Top 10 tools by F-score:')
print(comparison.head(10).to_string(index=False))

generate_benchmark_report(metrics, comparison, '${OUTPUT_DIR}/report')
print()
print(f'  Report written to ${OUTPUT_DIR}/report/')
"
