"""
Module 25: utils/parallel.py

Chromosome-level parallelism via multiprocessing.
"""

from __future__ import annotations

from multiprocessing import Pool
from typing import Any, Callable, List


def parallel_by_chromosome(
    func: Callable,
    chromosomes: List[str],
    n_workers: int = 1,
    **kwargs,
) -> List[Any]:
    """Execute function for each chromosome, optionally in parallel.

    Distributes work across chromosomes using multiprocessing Pool.
    Each worker process runs func(chrom, **kwargs) for a chromosome.

    Args:
        func: Function to execute. Signature: func(chrom: str, **kwargs) -> Any
        chromosomes: List of chromosome names to process.
        n_workers: Number of worker processes. If 1, run sequentially (for debugging).
                  If > 1, use multiprocessing.Pool with n_workers processes.
        **kwargs: Additional keyword arguments passed to func.

    Returns:
        List of results, one per chromosome (same order as input chromosomes).

    Example:
        def analyze_chrom(chrom, output_dir):
            # Do work for this chromosome
            return summary_data

        results = parallel_by_chromosome(
            analyze_chrom,
            ['chr1', 'chr2', 'chr3'],
            n_workers=4,
            output_dir='/path/to/output'
        )
    """
    if n_workers == 1:
        # Run sequentially (useful for debugging)
        results = [func(chrom, **kwargs) for chrom in chromosomes]
    else:
        # Run in parallel using multiprocessing
        with Pool(processes=n_workers) as pool:
            # Create tasks for each chromosome
            tasks = [
                pool.apply_async(func, (chrom,), kwargs)
                for chrom in chromosomes
            ]
            # Collect results in order
            results = [task.get() for task in tasks]

    return results


def get_default_chromosomes(include_sex: bool = True) -> List[str]:
    """Get default list of human chromosomes.

    Returns autosomes 1-22, optionally including sex chromosomes.

    Args:
        include_sex: If True, include 'chrX' and 'chrY' in the list.
                    If False, return only autosomes chr1-chr22.

    Returns:
        List of chromosome names in order: ['chr1', 'chr2', ..., 'chr22']
        optionally followed by ['chrX', 'chrY'].

    Example:
        all_chroms = get_default_chromosomes(include_sex=True)
        # Returns: ['chr1', ..., 'chr22', 'chrX', 'chrY']

        autosomes = get_default_chromosomes(include_sex=False)
        # Returns: ['chr1', ..., 'chr22']
    """
    chromosomes = [f"chr{i}" for i in range(1, 23)]

    if include_sex:
        chromosomes.extend(["chrX", "chrY"])

    return chromosomes
