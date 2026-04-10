"""
Chromosome-level pipeline worker.

Runs the complete SPLICE analysis pipeline for a single chromosome.
Each chromosome is processed independently, exploiting the biological fact
that splice junctions on different chromosomes never interact.

This module enables chromosome-level parallelism: the main pipeline dispatches
one worker per chromosome, and results are merged at the end with global FDR
correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from splice.core.clustering import cluster_junctions
from splice.core.confidence_scorer import score_all_junctions
from splice.core.diagnostics import EventDiagnostic, compute_diagnostics
from splice.core.diff import DiffResult, differential_splicing
from splice.core.diff_het import HetResult, heterogeneous_splicing
from splice.core.event_classifier import classify_all_events
from splice.core.evidence import ModuleEvidence, build_evidence_matrices
from splice.core.junction_extractor import (
    JunctionEvidence,
    CooccurrenceEvidence,
    extract_junctions_for_chromosome,
)
from splice.core.psi import ModulePSI, quantify_psi
from splice.core.splicegraph import SplicingModule, build_splicegraph
from splice.utils.genomic import Junction, JunctionPair
from splice.utils.stats import benjamini_hochberg


@dataclass
class ChromosomeResult:
    """All results from processing a single chromosome.

    Attributes:
        chrom: Chromosome name.
        junction_evidence: Junction evidence for this chromosome.
        cooccurrence: Co-occurrence evidence for this chromosome.
        modules: Splicing modules on this chromosome.
        evidence_list: Module evidence matrices.
        psi_list: PSI quantification results.
        diff_results: Differential splicing results (FDR not yet corrected globally).
        het_results: Heterogeneity testing results.
        event_types: Event type classification per module.
        diagnostics: Per-event diagnostic records.
        n_junctions_raw: Total junctions before filtering.
        n_junctions_filtered: Junctions after pre-filtering.
        n_clusters: Number of junction clusters.
        n_modules: Number of splicing modules.
        n_tested: Number of modules tested for differential splicing.
        elapsed_seconds: Wall time for this chromosome.
    """
    chrom: str
    junction_evidence: Dict[Junction, JunctionEvidence]
    cooccurrence: Dict[JunctionPair, CooccurrenceEvidence]
    modules: List[SplicingModule]
    evidence_list: List[ModuleEvidence]
    psi_list: List[ModulePSI]
    diff_results: List[DiffResult]
    het_results: List[HetResult]
    event_types: List[str]
    diagnostics: List[EventDiagnostic]
    n_junctions_raw: int
    n_junctions_filtered: int
    n_clusters: int
    n_modules: int
    n_tested: int
    elapsed_seconds: float


def _empty_result(chrom, t_start, junction_evidence=None, cooccurrence=None,
                  n_junctions_raw=0, n_junctions_filtered=0, n_clusters=0):
    """Return an empty ChromosomeResult."""
    import time
    return ChromosomeResult(
        chrom=chrom,
        junction_evidence=junction_evidence or {},
        cooccurrence=cooccurrence or {},
        modules=[],
        evidence_list=[],
        psi_list=[],
        diff_results=[],
        het_results=[],
        event_types=[],
        diagnostics=[],
        n_junctions_raw=n_junctions_raw,
        n_junctions_filtered=n_junctions_filtered,
        n_clusters=n_clusters,
        n_modules=0,
        n_tested=0,
        elapsed_seconds=time.time() - t_start,
    )


def process_chromosome(
    chrom: str,
    bam_paths: List[str],
    sample_names: List[str],
    genes: Dict,
    known_junctions: Set[Junction],
    group_labels: np.ndarray,
    genome_fasta_path: Optional[str] = None,
    min_anchor: int = 6,
    min_mapq: int = 0,
    min_cluster_reads: int = 30,
    max_intron_length: int = 100000,
    n_bootstraps: int = 30,
    read_length: int = 150,
    run_het: bool = True,
) -> ChromosomeResult:
    """Run the complete SPLICE pipeline for a single chromosome.

    This function is designed to be called in parallel via multiprocessing.
    It is self-contained and does not write to any shared state.

    Pipeline steps for this chromosome:
    1. Extract junctions from all BAMs (region-restricted to this chrom)
    2. Score junction confidence
    3. Pre-filter low-count junctions
    4. Cluster junctions (union-find)
    5. Build splicegraph and modules
    6. Build evidence matrices
    7. Quantify PSI with bootstrap
    8. Differential splicing testing (FDR set to 1.0, corrected globally later)
    9. Heterogeneity testing (optional)
    10. Event classification
    11. Compute diagnostics

    Args:
        chrom: Chromosome name (e.g., "chr1").
        bam_paths: List of BAM file paths.
        sample_names: List of sample names.
        genes: Dict mapping gene_id to Gene objects (full genome, filtered to chrom).
        known_junctions: Set of all known junctions (filtered to chrom internally).
        group_labels: Array of group memberships (0 or 1).
        genome_fasta_path: Optional genome FASTA path for motif scoring.
        min_anchor: Minimum anchor length.
        min_mapq: Minimum mapping quality.
        min_cluster_reads: Minimum total reads for pre-filter.
        max_intron_length: Maximum intron length for clustering.
        n_bootstraps: Number of bootstrap replicates.
        read_length: Sequencing read length.
        run_het: Whether to run heterogeneity testing.

    Returns:
        ChromosomeResult with all results for this chromosome.
    """
    import time
    t_start = time.time()

    # Step 1: Extract junctions for this chromosome
    junction_evidence, cooccurrence = extract_junctions_for_chromosome(
        chrom=chrom,
        bam_paths=bam_paths,
        sample_names=sample_names,
        known_junctions=known_junctions,
        genome_fasta_path=genome_fasta_path,
        min_anchor=min_anchor,
        min_mapq=min_mapq,
    )

    n_junctions_raw = len(junction_evidence)

    if n_junctions_raw == 0:
        return _empty_result(chrom, t_start)

    # Step 2: Score junction confidence
    confidence_scores = score_all_junctions(junction_evidence)

    # Step 3: Pre-filter low-count junctions
    filtered_junctions = [
        junc for junc, ev in junction_evidence.items()
        if np.sum(ev.sample_counts) >= min_cluster_reads
    ]
    n_junctions_filtered = len(filtered_junctions)

    if n_junctions_filtered < 2:
        return _empty_result(chrom, t_start, junction_evidence, cooccurrence,
                             n_junctions_raw, n_junctions_filtered)

    # Step 4: Cluster junctions
    clusters = cluster_junctions(filtered_junctions, max_intron_length=max_intron_length)
    n_clusters = len(clusters)

    # Step 5: Build splicegraph -- filter genes to this chromosome
    chrom_genes = {
        gid: g for gid, g in genes.items()
        if g.chrom == chrom
    }
    modules, junction_to_idx = build_splicegraph(
        genes=chrom_genes,
        junction_evidence=junction_evidence,
        clusters=clusters,
        known_junctions=known_junctions,
    )
    n_modules = len(modules)

    if n_modules == 0:
        return _empty_result(chrom, t_start, junction_evidence, cooccurrence,
                             n_junctions_raw, n_junctions_filtered, n_clusters)

    # Step 6: Build evidence matrices
    evidence_list = build_evidence_matrices(
        modules=modules,
        junction_evidence=junction_evidence,
        junction_confidence=confidence_scores,
        read_length=read_length,
    )

    # Step 7: Quantify PSI
    seed = 42 + hash(chrom) % (2**31)
    psi_list = quantify_psi(evidence_list, n_bootstraps=n_bootstraps, seed=seed)

    # Step 8: Differential splicing (per-chromosome, FDR corrected globally later)
    diff_results = differential_splicing(
        module_evidence_list=evidence_list,
        module_psi_list=psi_list,
        group_labels=group_labels,
    )
    n_tested = len(diff_results)

    # Step 9: Heterogeneity testing
    het_results = []
    if run_het:
        het_results = heterogeneous_splicing(psi_list, group_labels)

    # Step 10: Event classification
    event_types = classify_all_events(modules)

    # Step 11: Diagnostics
    tested_ids = {dr.module_id for dr in diff_results}
    module_order = {dr.module_id: i for i, dr in enumerate(diff_results)}
    tested_evidence = sorted(
        [e for e in evidence_list if e.module.module_id in tested_ids],
        key=lambda e: module_order.get(e.module.module_id, 999999),
    )
    tested_psi = sorted(
        [p for p in psi_list if p.module_id in tested_ids],
        key=lambda p: module_order.get(p.module_id, 999999),
    )
    diagnostics = []
    if tested_evidence and tested_psi and diff_results:
        diagnostics = compute_diagnostics(tested_evidence, tested_psi, diff_results)

    elapsed = time.time() - t_start

    return ChromosomeResult(
        chrom=chrom,
        junction_evidence=junction_evidence,
        cooccurrence=cooccurrence,
        modules=modules,
        evidence_list=evidence_list,
        psi_list=psi_list,
        diff_results=diff_results,
        het_results=het_results,
        event_types=event_types,
        diagnostics=diagnostics,
        n_junctions_raw=n_junctions_raw,
        n_junctions_filtered=n_junctions_filtered,
        n_clusters=n_clusters,
        n_modules=n_modules,
        n_tested=n_tested,
        elapsed_seconds=elapsed,
    )


def merge_chromosome_results(
    chrom_results: List[ChromosomeResult],
) -> Tuple[
    Dict[Junction, JunctionEvidence],
    Dict[JunctionPair, CooccurrenceEvidence],
    List[SplicingModule],
    List[ModuleEvidence],
    List[ModulePSI],
    List[DiffResult],
    List[HetResult],
    List[str],
    List[EventDiagnostic],
]:
    """Merge results from all chromosomes and apply global FDR correction.

    Concatenates all per-chromosome results and recomputes FDR across the
    entire genome. This is critical: FDR must be computed globally, not
    per-chromosome, to correctly account for the total number of tests.

    Args:
        chrom_results: List of ChromosomeResult objects, one per chromosome.

    Returns:
        Tuple of merged results:
        (junction_evidence, cooccurrence, modules, evidence_list, psi_list,
         diff_results, het_results, event_types, diagnostics)
        where diff_results have globally corrected FDR values.
    """
    all_junction_evidence = {}
    all_cooccurrence = {}
    all_modules = []
    all_evidence = []
    all_psi = []
    all_diff = []
    all_het = []
    all_event_types = []
    all_diagnostics = []

    for cr in chrom_results:
        all_junction_evidence.update(cr.junction_evidence)
        all_cooccurrence.update(cr.cooccurrence)
        all_modules.extend(cr.modules)
        all_evidence.extend(cr.evidence_list)
        all_psi.extend(cr.psi_list)
        all_diff.extend(cr.diff_results)
        all_het.extend(cr.het_results)
        all_event_types.extend(cr.event_types)
        all_diagnostics.extend(cr.diagnostics)

    # Global FDR correction across ALL chromosomes
    if all_diff:
        p_values = np.array([dr.p_value for dr in all_diff])
        global_fdr = benjamini_hochberg(p_values)

        corrected_diff = []
        for dr, fdr in zip(all_diff, global_fdr):
            corrected_diff.append(DiffResult(
                module_id=dr.module_id,
                gene_id=dr.gene_id,
                gene_name=dr.gene_name,
                chrom=dr.chrom,
                strand=dr.strand,
                event_type=dr.event_type,
                n_junctions=dr.n_junctions,
                junction_coords=dr.junction_coords,
                junction_confidence=dr.junction_confidence,
                is_annotated=dr.is_annotated,
                psi_group1=dr.psi_group1,
                psi_group2=dr.psi_group2,
                delta_psi=dr.delta_psi,
                max_abs_delta_psi=dr.max_abs_delta_psi,
                delta_psi_ci_low=dr.delta_psi_ci_low,
                delta_psi_ci_high=dr.delta_psi_ci_high,
                log_likelihood_null=dr.log_likelihood_null,
                log_likelihood_full=dr.log_likelihood_full,
                degrees_of_freedom=dr.degrees_of_freedom,
                p_value=dr.p_value,
                fdr=float(fdr),
                null_converged=dr.null_converged,
                full_converged=dr.full_converged,
                null_refit_used=dr.null_refit_used,
                null_iterations=dr.null_iterations,
                full_iterations=dr.full_iterations,
                null_gradient_norm=dr.null_gradient_norm,
                full_gradient_norm=dr.full_gradient_norm,
            ))
        all_diff = corrected_diff

    # Also correct het FDR globally
    if all_het:
        het_p_values = np.array([hr.ttest_pvalue for hr in all_het])
        het_global_fdr = benjamini_hochberg(het_p_values)

        corrected_het = []
        for hr, fdr in zip(all_het, het_global_fdr):
            corrected_het.append(HetResult(
                module_id=hr.module_id,
                gene_id=hr.gene_id,
                gene_name=hr.gene_name,
                event_type=hr.event_type,
                n_junctions=hr.n_junctions,
                sample_psi=hr.sample_psi,
                group_labels=hr.group_labels,
                ttest_pvalue=hr.ttest_pvalue,
                mannwhitney_pvalue=hr.mannwhitney_pvalue,
                within_group_variance=hr.within_group_variance,
                between_group_variance=hr.between_group_variance,
                heterogeneity_index=hr.heterogeneity_index,
                bimodal_pvalue=hr.bimodal_pvalue,
                n_outlier_samples=hr.n_outlier_samples,
                fdr=float(fdr),
            ))
        all_het = corrected_het

    return (
        all_junction_evidence,
        all_cooccurrence,
        all_modules,
        all_evidence,
        all_psi,
        all_diff,
        all_het,
        all_event_types,
        all_diagnostics,
    )
