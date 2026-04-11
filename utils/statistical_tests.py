"""
statistical_tests.py
=====================

Shared statistical test implementations to eliminate code duplication.

This module provides validated, non-degenerate statistical tests that properly
handle sample arrays (not single-value lists) and include input validation.

Usage::

    from utils.statistical_tests import (
        safe_ttest_1samp,
        safe_pearsonr,
        safe_binomtest,
        safe_mannwhitneyu,
        compute_cohens_d,
        compute_power_analysis,
    )
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy import stats

# =============================================================================
# INPUT VALIDATION
# =============================================================================


def validate_sample_array(
    data: Union[np.ndarray, list], min_n: int = 2, name: str = "data"
) -> np.ndarray:
    """
    Validate that data is a proper sample array with sufficient samples.

    Args:
        data: Input data (array or list)
        min_n: Minimum required sample size
        name: Name of the variable for error messages

    Returns:
        Validated numpy array

    Raises:
        ValueError: If data is invalid or insufficient
    """
    if not isinstance(data, (np.ndarray, list)):
        raise ValueError(f"{name} must be an array or list, got {type(data)}")

    arr = np.asarray(data)

    if len(arr) < min_n:
        raise ValueError(
            f"{name} requires N≥{min_n}, got {len(arr)}. "
            f"Degenerate statistical tests with single-value arrays produce "
            f"mathematically meaningless results."
        )

    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")

    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains Inf values")

    return arr


def validate_paired_arrays(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    min_n: int = 2,
    name_x: str = "x",
    name_y: str = "y",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate that two arrays are paired and have sufficient samples.

    Args:
        x: First array
        y: Second array
        min_n: Minimum required sample size
        name_x: Name of first variable
        name_y: Name of second variable

    Returns:
        Tuple of validated numpy arrays

    Raises:
        ValueError: If arrays are invalid or mismatched
    """
    x_arr = validate_sample_array(x, min_n, name_x)
    y_arr = validate_sample_array(y, min_n, name_y)

    if len(x_arr) != len(y_arr):
        raise ValueError(
            f"{name_x} and {name_y} must have same length for paired tests: "
            f"got {len(x_arr)} vs {len(y_arr)}"
        )

    return x_arr, y_arr


# =============================================================================
# SAFE STATISTICAL TESTS
# =============================================================================


def safe_ttest_1samp(
    data: Union[np.ndarray, list],
    popmean: float = 0.0,
    alpha: float = 0.05,
    min_n: int = 30,
) -> Tuple[float, float, bool]:
    """
    Safe one-sample t-test with proper input validation.

    Args:
        data: Sample data (must have N≥30 for meaningful results)
        popmean: Population mean under null hypothesis
        alpha: Significance level
        min_n: Minimum required sample size

    Returns:
        Tuple of (t_statistic, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    data_arr = validate_sample_array(data, min_n, "data")

    t_stat, p_value = stats.ttest_1samp(data_arr, popmean)
    significant = p_value < alpha

    return t_stat, p_value, significant


def safe_ttest_ind(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    alpha: float = 0.05,
    min_n: int = 30,
) -> Tuple[float, float, bool]:
    """
    Safe independent two-sample t-test with proper input validation.

    Args:
        x: First sample data
        y: Second sample data
        alpha: Significance level
        min_n: Minimum required sample size per group

    Returns:
        Tuple of (t_statistic, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    x_arr = validate_sample_array(x, min_n, "x")
    y_arr = validate_sample_array(y, min_n, "y")

    t_stat, p_value = stats.ttest_ind(x_arr, y_arr)
    significant = p_value < alpha

    return t_stat, p_value, significant


def safe_ttest_rel(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    alpha: float = 0.05,
    min_n: int = 30,
) -> Tuple[float, float, bool]:
    """
    Safe paired two-sample t-test with proper input validation.

    Args:
        x: First sample data
        y: Second sample data (must be same length as x)
        alpha: Significance level
        min_n: Minimum required sample size

    Returns:
        Tuple of (t_statistic, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    x_arr, y_arr = validate_paired_arrays(x, y, min_n, "x", "y")

    t_stat, p_value = stats.ttest_rel(x_arr, y_arr)
    significant = p_value < alpha

    return t_stat, p_value, significant


def safe_pearsonr(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    alpha: float = 0.05,
    min_n: int = 30,
) -> Tuple[float, float, bool]:
    """
    Safe Pearson correlation with proper input validation.

    Args:
        x: First variable data
        y: Second variable data (must be same length as x)
        alpha: Significance level
        min_n: Minimum required sample size

    Returns:
        Tuple of (correlation_coefficient, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    x_arr, y_arr = validate_paired_arrays(x, y, min_n, "x", "y")

    # Check for zero variance
    if np.var(x_arr) == 0 or np.var(y_arr) == 0:
        raise ValueError(
            "Cannot compute correlation with zero variance in one or both arrays"
        )

    corr, p_value = stats.pearsonr(x_arr, y_arr)
    significant = p_value < alpha

    return corr, p_value, significant


def safe_spearmanr(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    alpha: float = 0.05,
    min_n: int = 10,
) -> Tuple[float, float, bool]:
    """
    Safe Spearman rank correlation with proper input validation.

    Args:
        x: First variable data
        y: Second variable data (must be same length as x)
        alpha: Significance level
        min_n: Minimum required sample size

    Returns:
        Tuple of (correlation_coefficient, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    x_arr, y_arr = validate_paired_arrays(x, y, min_n, "x", "y")

    corr, p_value = stats.spearmanr(x_arr, y_arr)
    significant = p_value < alpha

    return corr, p_value, significant


def safe_binomtest(
    k: int, n: int, p: float = 0.5, alpha: float = 0.05
) -> Tuple[float, bool]:
    """
    Safe binomial test (modern scipy.stats.binomtest, not deprecated binom_test).

    Args:
        k: Number of successes
        n: Number of trials
        p: Expected probability under null hypothesis
        alpha: Significance level

    Returns:
        Tuple of (p_value, significant)

    Raises:
        ValueError: If parameters are invalid
    """
    if n < 1:
        raise ValueError(f"Number of trials n must be ≥1, got {n}")

    if k < 0 or k > n:
        raise ValueError(f"Number of successes k must be in [0, n], got {k} for n={n}")

    if p <= 0 or p >= 1:
        raise ValueError(f"Probability p must be in (0, 1), got {p}")

    # Use modern binomtest (not deprecated binom_test)
    result = stats.binomtest(k, n, p, alternative="two-sided")
    p_value = result.pvalue
    significant = p_value < alpha

    return p_value, significant


def safe_mannwhitneyu(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    alpha: float = 0.05,
    min_n: int = 20,
) -> Tuple[float, float, bool]:
    """
    Safe Mann-Whitney U test with proper input validation.

    Args:
        x: First sample data
        y: Second sample data
        alpha: Significance level
        min_n: Minimum required sample size per group

    Returns:
        Tuple of (u_statistic, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    x_arr = validate_sample_array(x, min_n, "x")
    y_arr = validate_sample_array(y, min_n, "y")

    u_stat, p_value = stats.mannwhitneyu(x_arr, y_arr, alternative="two-sided")
    significant = p_value < alpha

    return u_stat, p_value, significant


def safe_wilcoxon(
    x: Union[np.ndarray, list],
    y: Optional[Union[np.ndarray, list]] = None,
    alpha: float = 0.05,
    min_n: int = 20,
) -> Tuple[float, float, bool]:
    """
    Safe Wilcoxon signed-rank test with proper input validation.

    Args:
        x: First sample data (or differences if y is None)
        y: Second sample data (optional)
        alpha: Significance level
        min_n: Minimum required sample size

    Returns:
        Tuple of (statistic, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    if y is None:
        x_arr = validate_sample_array(x, min_n, "x")
        stat, p_value = stats.wilcoxon(x_arr)
    else:
        x_arr, y_arr = validate_paired_arrays(x, y, min_n, "x", "y")
        stat, p_value = stats.wilcoxon(x_arr, y_arr)

    significant = p_value < alpha

    return stat, p_value, significant


def safe_anova_oneway(
    *samples: Union[np.ndarray, list], alpha: float = 0.05, min_n: int = 20
) -> Tuple[float, float, bool]:
    """
    Safe one-way ANOVA with proper input validation.

    Args:
        *samples: Variable number of sample arrays
        alpha: Significance level
        min_n: Minimum required sample size per group

    Returns:
        Tuple of (f_statistic, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    if len(samples) < 2:
        raise ValueError("ANOVA requires at least 2 groups")

    validated_samples = []
    for i, sample in enumerate(samples):
        validated_samples.append(validate_sample_array(sample, min_n, f"sample_{i}"))

    f_stat, p_value = stats.f_oneway(*validated_samples)
    significant = p_value < alpha

    return f_stat, p_value, significant


# =============================================================================
# EFFECT SIZE CALCULATIONS
# =============================================================================


def compute_cohens_d(
    x: Union[np.ndarray, list], y: Union[np.ndarray, list], min_n: int = 30
) -> float:
    """
    Compute Cohen's d effect size for two samples.

    Args:
        x: First sample data
        y: Second sample data
        min_n: Minimum required sample size per group

    Returns:
        Cohen's d effect size

    Raises:
        ValueError: If data is invalid or insufficient
    """
    x_arr = validate_sample_array(x, min_n, "x")
    y_arr = validate_sample_array(y, min_n, "y")

    # Pooled standard deviation
    n1, n2 = len(x_arr), len(y_arr)
    var1, var2 = np.var(x_arr, ddof=1), np.var(y_arr, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    cohens_d = (np.mean(x_arr) - np.mean(y_arr)) / pooled_std
    return cohens_d


def compute_cliffs_delta(
    x: Union[np.ndarray, list], y: Union[np.ndarray, list]
) -> float:
    """
    Compute Cliff's delta effect size for non-parametric data.

    Args:
        x: First sample data
        y: Second sample data

    Returns:
        Cliff's delta (range: -1 to 1)

    Raises:
        ValueError: If data is invalid
    """
    x_arr = validate_sample_array(x, min_n=2, name="x")
    y_arr = validate_sample_array(y, min_n=2, name="y")

    # Count number of x > y, x < y, x = y
    n_greater = 0
    n_less = 0
    n_equal = 0

    for xi in x_arr:
        for yi in y_arr:
            if xi > yi:
                n_greater += 1
            elif xi < yi:
                n_less += 1
            else:
                n_equal += 1

    n_total = len(x_arr) * len(y_arr)

    if n_total == 0:
        return 0.0

    # Cliff's delta formula
    cliffs_delta = (n_greater - n_less) / n_total
    return cliffs_delta


def compute_eta_squared(f_statistic: float, df_between: int, df_within: int) -> float:
    """
    Compute eta-squared from ANOVA F statistic.

    Args:
        f_statistic: F statistic from ANOVA
        df_between: Degrees of freedom between groups
        df_within: Degrees of freedom within groups

    Returns:
        Eta-squared effect size
    """
    eta_squared = (f_statistic * df_between) / (f_statistic * df_between + df_within)
    return eta_squared


# =============================================================================
# POWER ANALYSIS
# =============================================================================


def compute_power_analysis(
    effect_size: float,
    n_per_group: int,
    alpha: float = 0.05,
    test_type: str = "ttest_ind",
) -> float:
    """
    Compute statistical power for a given effect size and sample size.

    Args:
        effect_size: Cohen's d or other effect size measure
        n_per_group: Sample size per group
        alpha: Significance level
        test_type: Type of test ("ttest_ind", "ttest_rel", "pearsonr", etc.)

    Returns:
        Statistical power (0 to 1)
    """
    try:
        from statsmodels.stats.power import TTestIndPower, TTestPower

        if test_type == "ttest_ind":
            power_analysis = TTestIndPower()
            power = power_analysis.power(
                effect_size=effect_size,
                nobs1=n_per_group,
                alpha=alpha,
                ratio=1.0,
                alternative="two-sided",
            )
        elif test_type == "ttest_rel":
            power_analysis = TTestPower()
            power = power_analysis.power(
                effect_size=effect_size,
                nobs=n_per_group,
                alpha=alpha,
                alternative="two-sided",
            )
        else:
            # Approximation for other tests
            power = 0.8  # Default placeholder

        return power
    except ImportError:
        # If statsmodels not available, use approximation
        # This is a rough approximation based on Cohen's power tables
        if effect_size >= 0.8 and n_per_group >= 26:
            return 0.80
        elif effect_size >= 0.5 and n_per_group >= 64:
            return 0.80
        elif effect_size >= 0.2 and n_per_group >= 394:
            return 0.80
        else:
            return 0.5


def compute_required_n(
    effect_size: float,
    desired_power: float = 0.80,
    alpha: float = 0.05,
    test_type: str = "ttest_ind",
) -> int:
    """
    Compute required sample size for desired power.

    Args:
        effect_size: Cohen's d or other effect size measure
        desired_power: Desired statistical power (0 to 1)
        alpha: Significance level
        test_type: Type of test ("ttest_ind", "ttest_rel", etc.)

    Returns:
        Required sample size per group
    """
    try:
        from statsmodels.stats.power import TTestIndPower, TTestPower

        if test_type == "ttest_ind":
            power_analysis = TTestIndPower()
            n = power_analysis.solve_power(
                effect_size=effect_size,
                power=desired_power,
                alpha=alpha,
                ratio=1.0,
                alternative="two-sided",
            )
        elif test_type == "ttest_rel":
            power_analysis = TTestPower()
            n = power_analysis.solve_power(
                effect_size=effect_size,
                power=desired_power,
                alpha=alpha,
                alternative="two-sided",
            )
        else:
            # Approximation for other tests
            n = 50  # Default placeholder

        return int(np.ceil(n))
    except ImportError:
        # If statsmodels not available, use Cohen's conventions
        if effect_size >= 0.8:
            return 26
        elif effect_size >= 0.5:
            return 64
        elif effect_size >= 0.2:
            return 394
        else:
            return 100


# =============================================================================
# BOOTSTRAP METHODS
# =============================================================================


def bootstrap_ci(
    data: Union[np.ndarray, list],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: str = "mean",
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level (e.g., 0.95 for 95% CI)
        statistic: Statistic to compute ("mean", "median", "std")

    Returns:
        Tuple of (statistic_value, lower_bound, upper_bound)
    """
    data_arr = validate_sample_array(data, min_n=2, name="data")

    bootstrap_stats_list = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_arr, size=len(data_arr), replace=True)

        if statistic == "mean":
            bootstrap_stats_list.append(np.mean(sample))
        elif statistic == "median":
            bootstrap_stats_list.append(np.median(sample))
        elif statistic == "std":
            bootstrap_stats_list.append(np.std(sample, ddof=1))
        else:
            bootstrap_stats_list.append(np.mean(sample))

    bootstrap_stats = np.array(bootstrap_stats_list)

    if statistic == "mean":
        stat_value = np.mean(data_arr)
    elif statistic == "median":
        stat_value = np.median(data_arr)
    elif statistic == "std":
        stat_value = np.std(data_arr, ddof=1)
    else:
        stat_value = np.mean(data_arr)

    lower = np.percentile(bootstrap_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 + ci) / 2 * 100)

    return stat_value, lower, upper


def bootstrap_one_sample_test(
    data: Union[np.ndarray, list],
    null_value: float = 0.0,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Perform one-sample test using bootstrap.

    Args:
        data: Sample data
        null_value: Null hypothesis value
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level

    Returns:
        Tuple of (test_statistic, p_value)
    """
    data_arr = validate_sample_array(data, min_n=2, name="data")

    if len(data_arr) < 2:
        return 0.0, 1.0

    observed_mean = np.mean(data_arr)
    bootstrap_means_list = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data_arr, size=len(data_arr), replace=True)
        bootstrap_means_list.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means_list)

    # Two-sided p-value
    if observed_mean >= null_value:
        p_value = np.mean(bootstrap_means >= 2 * null_value - observed_mean)
    else:
        p_value = np.mean(bootstrap_means <= 2 * null_value - observed_mean)

    # Test statistic
    test_stat = (
        (observed_mean - null_value) / (np.std(data_arr) / np.sqrt(len(data_arr)))
        if np.std(data_arr) > 0
        else 0.0
    )

    return test_stat, p_value


# =============================================================================
# PERMUTATION TESTS
# =============================================================================


def permutation_test(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    n_permutations: int = 10000,
    statistic: str = "mean_diff",
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Perform permutation test for two samples.

    Args:
        x: First sample data
        y: Second sample data
        n_permutations: Number of permutations
        statistic: Statistic to test ("mean_diff", "median_diff", "corr")
        alpha: Significance level

    Returns:
        Tuple of (observed_statistic, p_value, significant)
    """
    x_arr = validate_sample_array(x, min_n=2, name="x")
    y_arr = validate_sample_array(y, min_n=2, name="y")

    combined = np.concatenate([x_arr, y_arr])
    n_x = len(x_arr)

    # Compute observed statistic
    if statistic == "mean_diff":
        observed = np.mean(x_arr) - np.mean(y_arr)
    elif statistic == "median_diff":
        observed = np.median(x_arr) - np.median(y_arr)
    elif statistic == "corr":
        observed = np.corrcoef(x_arr, y_arr)[0, 1]
    else:
        observed = np.mean(x_arr) - np.mean(y_arr)

    # Permutation distribution
    perm_stats_list = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_x = combined[:n_x]
        perm_y = combined[n_x:]

        if statistic == "mean_diff":
            perm_stat = np.mean(perm_x) - np.mean(perm_y)
        elif statistic == "median_diff":
            perm_stat = np.median(perm_x) - np.median(perm_y)
        elif statistic == "corr":
            perm_stat = np.corrcoef(perm_x, perm_y)[0, 1]
        else:
            perm_stat = np.mean(perm_x) - np.mean(perm_y)

        perm_stats_list.append(perm_stat)

    perm_stats = np.array(perm_stats_list)

    # Two-sided p-value
    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
    significant = p_value < alpha

    return observed, p_value, significant


# =============================================================================
# MULTIPLE COMPARISON CORRECTION
# =============================================================================


def apply_multiple_comparison_correction(
    p_values: Union[np.ndarray, list],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> dict:
    """
    Apply multiple comparison correction to control FWER or FDR.

    Args:
        p_values: Array of p-values from multiple hypothesis tests
        method: Correction method - "bonferroni", "holm", "fdr_bh" (Benjamini-Hochberg),
                or "fdr_by" (Benjamini-Yekutieli)
        alpha: Family-wise significance level (default 0.05)

    Returns:
        Dictionary containing:
        - corrected_p_values: Array of corrected p-values
        - significant: Boolean array indicating which tests are significant
        - reject: Boolean array (same as significant, for compatibility with statsmodels)
        - method: The correction method used
        - alpha: The significance level used
        - n_tests: Number of tests performed

    Raises:
        ValueError: If method is not supported or p_values is invalid

    Example:
        >>> p_values = [0.01, 0.04, 0.02, 0.06, 0.001, 0.03]
        >>> result = apply_multiple_comparison_correction(p_values, method="bonferroni")
        >>> print(result['corrected_p_values'])
        >>> print(result['significant'])
    """
    p_arr = np.asarray(p_values)

    if len(p_arr) == 0:
        raise ValueError("p_values cannot be empty")

    if np.any((p_arr < 0) | (p_arr > 1)):
        raise ValueError("All p-values must be between 0 and 1")

    n_tests = len(p_arr)
    original_shape = p_arr.shape
    p_flat = p_arr.flatten()

    # Sort p-values for methods that require ordered testing
    sorted_indices = np.argsort(p_flat)
    sorted_p = p_flat[sorted_indices]

    if method == "bonferroni":
        # Bonferroni correction: multiply each p-value by number of tests
        corrected = np.minimum(sorted_p * n_tests, 1.0)

    elif method == "holm":
        # Holm-Bonferroni step-down procedure
        # More powerful than simple Bonferroni while controlling FWER
        corrected = np.zeros_like(sorted_p)
        for i, p in enumerate(sorted_p):
            corrected[i] = min(p * (n_tests - i), 1.0)
        # Ensure monotonicity (corrected p-values should be non-decreasing)
        corrected = np.maximum.accumulate(corrected)

    elif method in ["fdr_bh", "benjamini-hochberg"]:
        # Benjamini-Hochberg FDR control (independent/positive dependent tests)
        corrected = np.zeros_like(sorted_p)
        for i, p in enumerate(sorted_p):
            corrected[i] = p * n_tests / (i + 1)
        # Ensure monotonicity
        corrected = np.minimum.accumulate(corrected[::-1])[::-1]
        corrected = np.minimum(corrected, 1.0)

    elif method in ["fdr_by", "benjamini-yekutieli"]:
        # Benjamini-Yekutieli FDR control (arbitrary dependence)
        # More conservative than BH, works under any dependence structure
        corrected = np.zeros_like(sorted_p)
        harmonic_sum = np.sum(1.0 / np.arange(1, n_tests + 1))
        for i, p in enumerate(sorted_p):
            corrected[i] = p * n_tests * harmonic_sum / (i + 1)
        # Ensure monotonicity
        corrected = np.minimum.accumulate(corrected[::-1])[::-1]
        corrected = np.minimum(corrected, 1.0)

    else:
        raise ValueError(
            f"Unknown correction method: {method}. "
            f"Supported methods: 'bonferroni', 'holm', 'fdr_bh', 'fdr_by'"
        )

    # Restore original order
    corrected_restored = np.zeros_like(corrected)
    corrected_restored[sorted_indices] = corrected
    corrected_final = corrected_restored.reshape(original_shape)

    # Determine significance
    significant = corrected_final < alpha

    return {
        "corrected_p_values": corrected_final,
        "significant": significant,
        "reject": significant,  # For compatibility with statsmodels
        "method": method,
        "alpha": alpha,
        "n_tests": n_tests,
    }
