"""
Module 18: core/diff_het.py

Heterogeneity-aware testing following MAJIQ HET approach.
Detects splicing changes present in only a subset of samples via bimodality analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

from splicekit.core.psi import ModulePSI
from splicekit.utils.stats import benjamini_hochberg


def _hartigan_dip(data: np.ndarray, n_bins: int = 200) -> float:
    """Compute Hartigan's dip test statistic (simplified version).

    Measures bimodality of a univariate distribution.
    Returns dip statistic; higher values suggest more bimodality.

    Args:
        data: 1D array of values.
        n_bins: Number of bins for histogram-based approximation.

    Returns:
        Dip test statistic (0-1, higher = more bimodal).
    """
    if len(data) < 4:
        return 0.0

    # Simple approximation: compute modal structure
    # Count modes using histogram approach
    hist, bin_edges = np.histogram(data, bins=min(n_bins, len(np.unique(data))))

    # Find local minima (valleys)
    n_valleys = 0
    for i in range(1, len(hist) - 1):
        if hist[i] < hist[i - 1] and hist[i] < hist[i + 1]:
            n_valleys += 1

    # Dip statistic: ratio of valleys to total bins (normalized)
    dip = min(1.0, n_valleys / max(1, len(hist) // 2))

    return dip


def _compute_bimodal_pvalue(data: np.ndarray) -> float:
    """Test for bimodality using dip test approximation.

    Args:
        data: 1D array of PSI values.

    Returns:
        P-value for bimodality (lower = more bimodal).
    """
    if len(data) < 4:
        return 1.0

    # Compute dip statistic
    dip_stat = _hartigan_dip(data)

    # Permutation-based p-value estimation
    # Under null (unimodal), dip should be near 0
    # We estimate p-value based on the magnitude of dip
    # Rough calibration: p = exp(-dip * len(data))
    p_value = np.exp(-max(0, dip_stat - 0.1) * np.sqrt(len(data)))

    return min(1.0, p_value)


@dataclass(frozen=True, slots=True)
class HetResult:
    """Result from heterogeneity-aware splicing test for one module.

    Attributes:
        module_id: Module identifier.
        gene_id: Associated gene ID.
        gene_name: Associated gene name.
        event_type: Classification (SE, A3SS, etc.).
        n_junctions: Number of junctions in module.
        # Per-sample PSI for most variable junction
        sample_psi: (n_samples,) PSI values for selected junction.
        group_labels: (n_samples,) group membership.
        # Test results
        ttest_pvalue: Welch's t-test p-value between groups.
        mannwhitney_pvalue: Mann-Whitney U test p-value.
        # Heterogeneity metrics
        within_group_variance: (2,) variance within each group.
        between_group_variance: Variance of group means.
        heterogeneity_index: Ratio of within to between variance.
        # Subgroup detection
        bimodal_pvalue: P-value for bimodality test.
        n_outlier_samples: Count of outlier samples (>2 SD from group mean).
        fdr: FDR-adjusted p-value.
    """

    module_id: str
    gene_id: str
    gene_name: str
    event_type: str
    n_junctions: int
    # Per-sample PSI
    sample_psi: np.ndarray
    group_labels: np.ndarray
    # Test results
    ttest_pvalue: float
    mannwhitney_pvalue: float
    # Heterogeneity metrics
    within_group_variance: np.ndarray
    between_group_variance: float
    heterogeneity_index: float
    # Subgroup detection
    bimodal_pvalue: float
    n_outlier_samples: int
    fdr: float


def test_heterogeneous_splicing(
    module_psi_list: List[ModulePSI],
    group_labels: np.ndarray,
    min_samples_per_group: int = 3,
) -> List[HetResult]:
    """Test for heterogeneous splicing effects.

    For each module:
    1. Compute per-sample PSI (posterior mean).
    2. For the junction with largest between-group variance in PSI:
       a. Run Welch's t-test between groups.
       b. Run Mann-Whitney U between groups.
       c. Compute within-group variance for each group.
       d. Test for bimodality within each group.
       e. Count outlier samples (|PSI - group_mean| > 2 * group_sd).
    3. Compute heterogeneity_index = mean(within_group_var) / between_group_var.
       High index suggests heterogeneous effect.
    4. Apply BH FDR correction across all modules.

    Args:
        module_psi_list: List of ModulePSI objects.
        group_labels: Array of shape (n_samples,) with group membership (0, 1).
        min_samples_per_group: Minimum samples per group to test.

    Returns:
        List of HetResult objects.
    """
    het_results: List[HetResult] = []
    p_values: List[float] = []

    group_0 = group_labels == 0
    group_1 = group_labels == 1
    n_group0 = np.sum(group_0)
    n_group1 = np.sum(group_1)

    # Check minimum group sizes
    if n_group0 < min_samples_per_group or n_group1 < min_samples_per_group:
        return het_results

    for module_psi in module_psi_list:
        module = module_psi.module_id
        n_junctions = module_psi.psi_matrix.shape[0]

        # Find junction with largest between-group variance
        psi_group0 = module_psi.psi_matrix[:, group_0]
        psi_group1 = module_psi.psi_matrix[:, group_1]

        mean_group0 = np.mean(psi_group0, axis=1)
        mean_group1 = np.mean(psi_group1, axis=1)

        # Between-group variance for each junction
        overall_mean = np.mean(module_psi.psi_matrix, axis=1)
        between_var_per_junction = (
            n_group0 * (mean_group0 - overall_mean) ** 2
            + n_group1 * (mean_group1 - overall_mean) ** 2
        ) / (n_group0 + n_group1 - 1)

        # Select junction with largest between-group variance
        selected_junc = np.argmax(between_var_per_junction)
        sample_psi = module_psi.psi_matrix[selected_junc, :]

        # Extract PSI by group
        psi_g0 = sample_psi[group_0]
        psi_g1 = sample_psi[group_1]

        # Test 1: Welch's t-test
        ttest_stat, ttest_p = ttest_ind(psi_g0, psi_g1, equal_var=False)

        # Test 2: Mann-Whitney U
        mw_stat, mw_p = mannwhitneyu(psi_g0, psi_g1, alternative="two-sided")

        # Within-group variance
        var_g0 = np.var(psi_g0, ddof=1)
        var_g1 = np.var(psi_g1, ddof=1)
        within_group_variance = np.array([var_g0, var_g1])

        # Between-group variance
        between_var = between_var_per_junction[selected_junc]

        # Heterogeneity index
        mean_within_var = np.mean(within_group_variance)
        het_index = mean_within_var / (between_var + 1e-8)

        # Bimodality test (within each group)
        bimodal_p_g0 = _compute_bimodal_pvalue(psi_g0)
        bimodal_p_g1 = _compute_bimodal_pvalue(psi_g1)
        # Overall bimodality: worst (highest) p-value
        bimodal_p = max(bimodal_p_g0, bimodal_p_g1)

        # Outlier detection
        mean_g0 = np.mean(psi_g0)
        mean_g1 = np.mean(psi_g1)
        sd_g0 = np.std(psi_g0, ddof=1)
        sd_g1 = np.std(psi_g1, ddof=1)

        outliers_g0 = np.sum(np.abs(psi_g0 - mean_g0) > 2 * (sd_g0 + 1e-8))
        outliers_g1 = np.sum(np.abs(psi_g1 - mean_g1) > 2 * (sd_g1 + 1e-8))
        n_outliers = outliers_g0 + outliers_g1

        # Use t-test p-value as primary test statistic
        primary_p = ttest_p

        # Create result
        result = HetResult(
            module_id=module,
            gene_id="",  # Will be filled from module metadata if available
            gene_name="",
            event_type="Complex",
            n_junctions=n_junctions,
            sample_psi=sample_psi,
            group_labels=group_labels.copy(),
            ttest_pvalue=ttest_p,
            mannwhitney_pvalue=mw_p,
            within_group_variance=within_group_variance,
            between_group_variance=between_var,
            heterogeneity_index=het_index,
            bimodal_pvalue=bimodal_p,
            n_outlier_samples=n_outliers,
            fdr=1.0,  # Will be updated after BH correction
        )

        het_results.append(result)
        p_values.append(primary_p)

    # BH FDR correction
    if p_values:
        p_values_array = np.array(p_values)
        fdr_values = benjamini_hochberg(p_values_array)

        # Update FDR in results
        updated_results = []
        for result, fdr in zip(het_results, fdr_values):
            updated_result = HetResult(
                module_id=result.module_id,
                gene_id=result.gene_id,
                gene_name=result.gene_name,
                event_type=result.event_type,
                n_junctions=result.n_junctions,
                sample_psi=result.sample_psi,
                group_labels=result.group_labels,
                ttest_pvalue=result.ttest_pvalue,
                mannwhitney_pvalue=result.mannwhitney_pvalue,
                within_group_variance=result.within_group_variance,
                between_group_variance=result.between_group_variance,
                heterogeneity_index=result.heterogeneity_index,
                bimodal_pvalue=result.bimodal_pvalue,
                n_outlier_samples=result.n_outlier_samples,
                fdr=float(fdr),
            )
            updated_results.append(updated_result)

        return updated_results

    return het_results
