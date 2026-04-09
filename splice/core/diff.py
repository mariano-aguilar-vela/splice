"""
Module 17: core/diff.py

Differential splicing testing using Dirichlet-multinomial GLM.
Supports multi-group comparisons with optional covariates and covariate adjustment.
Implements null-refit strategy for improved calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from splice.core.evidence import ModuleEvidence
from splice.core.psi import ModulePSI
from splice.utils.dm_glm import (
    build_design_matrix,
    dm_lrt,
    fit_dm_full,
    fit_dm_null,
)
from splice.utils.stats import benjamini_hochberg


@dataclass(frozen=True, slots=True)
class DiffResult:
    """Result from differential splicing test for one module.

    Attributes:
        module_id: Module identifier.
        gene_id: Associated gene ID.
        gene_name: Associated gene name.
        chrom: Chromosome.
        strand: Strand (+ or -).
        event_type: Classification (SE, A3SS, A5SS, MXE, RI, Complex).
        n_junctions: Number of junctions in module.
        junction_coords: List of junction coordinate strings.
        junction_confidence: List of confidence scores per junction.
        is_annotated: List of annotation flags per junction.
        # PSI
        psi_group1: (n_junctions,) PSI in group 0.
        psi_group2: (n_junctions,) PSI in group 1.
        delta_psi: (n_junctions,) PSI difference (group1 - group2).
        max_abs_delta_psi: Maximum absolute delta-PSI.
        # Bootstrap uncertainty
        delta_psi_ci_low: (n_junctions,) lower CI bound.
        delta_psi_ci_high: (n_junctions,) upper CI bound.
        # Test results
        log_likelihood_null: Log-likelihood under null model.
        log_likelihood_full: Log-likelihood under full model.
        degrees_of_freedom: Degrees of freedom for chi-squared test.
        p_value: Likelihood ratio test p-value.
        fdr: False discovery rate adjusted p-value.
        # Convergence diagnostics
        null_converged: Whether null model converged.
        full_converged: Whether full model converged.
        null_refit_used: Whether null-refit strategy was applied.
        null_iterations: Iterations for null model fitting.
        full_iterations: Iterations for full model fitting.
        null_gradient_norm: Gradient norm at null convergence.
        full_gradient_norm: Gradient norm at full convergence.
    """

    module_id: str
    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    event_type: str
    n_junctions: int
    junction_coords: List[str]
    junction_confidence: List[float]
    is_annotated: List[bool]
    # PSI
    psi_group1: np.ndarray
    psi_group2: np.ndarray
    delta_psi: np.ndarray
    max_abs_delta_psi: float
    # Bootstrap uncertainty
    delta_psi_ci_low: np.ndarray
    delta_psi_ci_high: np.ndarray
    # Test results
    log_likelihood_null: float
    log_likelihood_full: float
    degrees_of_freedom: int
    p_value: float
    fdr: float
    # Convergence diagnostics
    null_converged: bool
    full_converged: bool
    null_refit_used: bool
    null_iterations: int
    full_iterations: int
    null_gradient_norm: float
    full_gradient_norm: float


def differential_splicing(
    module_evidence_list: List[ModuleEvidence],
    module_psi_list: List[ModulePSI],
    group_labels: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    covariate_names: Optional[List[str]] = None,
    min_total_reads_per_group: int = 10,
    min_samples_per_group: int = 3,
) -> List[DiffResult]:
    """Test differential splicing with DM-GLM.

    Full differential splicing pipeline with covariates:
    1. Build design matrices (full with group, null without).
    2. For each module, fit null and full DM-GLM.
    3. LRT with null-refit strategy.
    4. BH FDR correction.
    5. Compute bootstrap delta-PSI CIs from ModulePSI bootstrap data.

    Args:
        module_evidence_list: List of ModuleEvidence objects.
        module_psi_list: List of ModulePSI objects (same order as evidence).
        group_labels: Array of shape (n_samples,) with group membership (0, 1, ...).
        covariates: Optional array of shape (n_samples, n_covariates).
        covariate_names: Optional names of covariates.
        min_total_reads_per_group: Minimum total reads required per group.
        min_samples_per_group: Minimum samples per group with reads.

    Returns:
        List of DiffResult objects, one per module.
    """
    n_samples = len(group_labels)
    assert (
        len(module_evidence_list) == len(module_psi_list)
    ), "Evidence and PSI lists must have same length"

    # Build design matrices (shared for all modules)
    design_full, design_null, df = build_design_matrix(
        group_labels, covariates=covariates, covariate_names=covariate_names
    )

    # Fit models and compute test statistics for each module
    diff_results: List[DiffResult] = []
    p_values: List[float] = []

    for evidence, psi in zip(module_evidence_list, module_psi_list):
        module = evidence.module
        n_junctions = evidence.junction_count_matrix.shape[0]

        # Check minimum coverage per group
        group_totals = np.zeros(len(np.unique(group_labels)))
        group_samples = np.zeros(len(np.unique(group_labels)))

        for g in np.unique(group_labels):
            mask = group_labels == g
            group_totals[int(g)] = np.sum(evidence.total_counts[mask])
            group_samples[int(g)] = np.sum(evidence.total_counts[mask] > 0)

        # Skip if insufficient coverage
        if np.any(group_totals < min_total_reads_per_group) or np.any(
            group_samples < min_samples_per_group
        ):
            continue

        # Fit null and full models
        # Note: fit_dm_glm expects (n_samples, n_junctions), so transpose
        count_matrix_t = evidence.junction_count_matrix.T

        result_null = fit_dm_null(
            count_matrix_t, design_null, max_iter=500
        )
        result_full = fit_dm_full(
            count_matrix_t, design_full, max_iter=500
        )

        # LRT
        p_value = dm_lrt(result_null, result_full, df)

        # Null-refit strategy (following LeafCutter dm_glm_multi_conc.R lines 79-92)
        null_refit_used = False
        if p_value < 0.001:
            result_null_refit = fit_dm_null(
                count_matrix_t, design_null, max_iter=500,
                init_from=result_full,
            )
            if result_null_refit.log_likelihood > result_null.log_likelihood:
                result_null = result_null_refit
                p_value = dm_lrt(result_null, result_full, df)
                null_refit_used = True

        # Compute PSI by group
        group_0_mask = group_labels == 0
        group_1_mask = group_labels == 1

        psi_group0 = np.mean(psi.psi_matrix[:, group_0_mask], axis=1)
        psi_group1 = np.mean(psi.psi_matrix[:, group_1_mask], axis=1)

        # Delta-PSI
        delta_psi = psi.psi_matrix[:, group_1_mask].mean(axis=1) - psi.psi_matrix[
            :, group_0_mask
        ].mean(axis=1)
        max_abs_delta_psi = np.max(np.abs(delta_psi))

        # Bootstrap delta-PSI CIs
        boot_delta_psi = (
            psi.bootstrap_psi[:, :, group_1_mask].mean(axis=2)
            - psi.bootstrap_psi[:, :, group_0_mask].mean(axis=2)
        )
        delta_psi_ci_low = np.percentile(boot_delta_psi, 2.5, axis=0)
        delta_psi_ci_high = np.percentile(boot_delta_psi, 97.5, axis=0)

        # Junction information
        junction_coords = [
            f"{j.chrom}:{j.start}-{j.end}:{j.strand}" for j in module.junctions
        ]
        junction_confidence = [
            float(c) for c in evidence.junction_confidence
        ]
        is_annotated = [bool(a) for a in evidence.is_annotated]

        # Create result
        result = DiffResult(
            module_id=module.module_id,
            gene_id=module.gene_id,
            gene_name=module.gene_name,
            chrom=module.chrom,
            strand=module.strand,
            event_type="Complex",  # Will be set by event_classifier later
            n_junctions=n_junctions,
            junction_coords=junction_coords,
            junction_confidence=junction_confidence,
            is_annotated=is_annotated,
            psi_group1=psi_group0,
            psi_group2=psi_group1,
            delta_psi=delta_psi,
            max_abs_delta_psi=max_abs_delta_psi,
            delta_psi_ci_low=delta_psi_ci_low,
            delta_psi_ci_high=delta_psi_ci_high,
            log_likelihood_null=result_null.log_likelihood,
            log_likelihood_full=result_full.log_likelihood,
            degrees_of_freedom=df,
            p_value=p_value,
            fdr=1.0,  # Will be set after BH correction
            null_converged=result_null.converged,
            full_converged=result_full.converged,
            null_refit_used=null_refit_used,
            null_iterations=result_null.n_iterations,
            full_iterations=result_full.n_iterations,
            null_gradient_norm=result_null.gradient_norm,
            full_gradient_norm=result_full.gradient_norm,
        )

        diff_results.append(result)
        p_values.append(p_value)

    # BH FDR correction
    if p_values:
        p_values_array = np.array(p_values)
        fdr_values = benjamini_hochberg(p_values_array)

        # Update FDR in results
        updated_results = []
        for result, fdr in zip(diff_results, fdr_values):
            updated_result = DiffResult(
                module_id=result.module_id,
                gene_id=result.gene_id,
                gene_name=result.gene_name,
                chrom=result.chrom,
                strand=result.strand,
                event_type=result.event_type,
                n_junctions=result.n_junctions,
                junction_coords=result.junction_coords,
                junction_confidence=result.junction_confidence,
                is_annotated=result.is_annotated,
                psi_group1=result.psi_group1,
                psi_group2=result.psi_group2,
                delta_psi=result.delta_psi,
                max_abs_delta_psi=result.max_abs_delta_psi,
                delta_psi_ci_low=result.delta_psi_ci_low,
                delta_psi_ci_high=result.delta_psi_ci_high,
                log_likelihood_null=result.log_likelihood_null,
                log_likelihood_full=result.log_likelihood_full,
                degrees_of_freedom=result.degrees_of_freedom,
                p_value=result.p_value,
                fdr=float(fdr),
                null_converged=result.null_converged,
                full_converged=result.full_converged,
                null_refit_used=result.null_refit_used,
                null_iterations=result.null_iterations,
                full_iterations=result.full_iterations,
                null_gradient_norm=result.null_gradient_norm,
                full_gradient_norm=result.full_gradient_norm,
            )
            updated_results.append(updated_result)

        return updated_results

    return diff_results
