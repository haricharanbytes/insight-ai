"""Column summaries and classical statistical tests (Welch t, Mann-Whitney, ANOVA, Kruskal, chi-square)."""
import numpy as np
import pandas as pd
from pandas.api import types as pdt
from scipy import stats

from src.data.profiler import _numeric_details, _profile_column

from .common import (
    AnalysisError,
    df_to_table,
    make_result,
    require_columns,
    require_numeric,
    to_py,
)
from .filtering import apply_filters

TESTS = ("ttest", "mannwhitney", "anova", "kruskal", "chi_square")
_TWO_GROUP = ("ttest", "mannwhitney")
MAX_GROUPS = 20
MAX_CATEGORIES = 50
MIN_GROUP_N = 2
SMALL_GROUP_N = 5

NOTE_GROUP = (
    "A small p-value means the difference is unlikely to be due to chance alone. It does not show "
    "why the groups differ, and with very large samples even trivial differences become "
    "significant, so check the effect size. Observations are assumed independent, and no "
    "correction was made for running multiple tests."
)
NOTE_CHI = (
    "Shows association between the two columns, not causation. Observations are assumed independent "
    "(repeat customers or repeated orders can break this). Expected counts below 5 make the result unreliable."
)


# ------------------------------------------------------------ column summary

def get_column_summary(df: pd.DataFrame, column: str, filters=None) -> dict:
    """Descriptive summary of one column, optionally on a filtered subset."""
    require_columns(df, [column])
    work, applied, fw = apply_filters(df, filters)
    if len(work) == 0:
        raise AnalysisError("No rows remain after filtering.", "no_rows", {"filters": applied, "warnings": fw})

    s = work[column]
    profile = _profile_column(column, s)
    # An explicitly requested numeric column (e.g. an ID-like one) still gets numeric statistics.
    if profile["inferred_type"] != "numeric" and pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s):
        profile.update(_numeric_details(s))

    top = profile.get("top_values")
    return make_result(
        "get_column_summary",
        method="Descriptive statistics from the column's values. Missing values are counted separately, never treated as zero.",
        inputs={"column": column, "filters": applied},
        rows_used=len(work),
        result={"column": column, "profile": profile},
        table=df_to_table(pd.DataFrame(top)) if top else None,
        warnings=fw,
    )


# ------------------------------------------------------------ helpers

def _effect(name: str, value, cuts: tuple) -> dict:
    if value is None or not np.isfinite(value):
        return {"name": name, "value": None, "magnitude": None}
    a = abs(value)
    mag = "negligible" if a < cuts[0] else "small" if a < cuts[1] else "medium" if a < cuts[2] else "large"
    return {"name": name, "value": float(value), "magnitude": mag}


def _collect_groups(work: pd.DataFrame, value_column: str, group_column: str, groups, two_only: bool):
    sub = work[[group_column, value_column]].dropna()
    vals = sub[value_column].to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(vals)
    labels = sub[group_column].astype(str).to_numpy()[finite]
    vals = vals[finite]
    available = sorted(set(labels.tolist()))

    if groups is None:
        if len(available) > MAX_GROUPS:
            raise AnalysisError(
                f"'{group_column}' has {len(available)} groups (limit {MAX_GROUPS}). "
                "Pass groups=[...] to choose which to compare.",
                "too_many_groups",
            )
        chosen = available
    else:
        if isinstance(groups, str) or not isinstance(groups, (list, tuple)):
            raise AnalysisError("'groups' must be a list of group values.", "invalid_parameter")
        chosen = [str(g) for g in groups]
        if len(set(chosen)) != len(chosen):
            raise AnalysisError("'groups' must not contain duplicates.", "invalid_parameter")
        missing = [g for g in chosen if g not in available]
        if missing:
            raise AnalysisError(
                f"Group(s) not found in '{group_column}': {missing}. Available: {available[:20]}.",
                "group_not_found", {"missing": missing, "available": available[:50]},
            )

    if len(chosen) < 2:
        raise AnalysisError(
            f"Need at least two groups in '{group_column}' to compare; found {len(chosen)}.",
            "insufficient_data",
        )
    if two_only and len(chosen) != 2:
        raise AnalysisError(
            f"This test compares exactly two groups, but '{group_column}' has {len(chosen)}. "
            "Pass groups=[first, second], or use anova / kruskal.",
            "invalid_parameter", {"available": available[:20]},
        )

    arrays = {g: vals[labels == g] for g in chosen}
    for g, a in arrays.items():
        if len(a) < MIN_GROUP_N:
            raise AnalysisError(
                f"Group '{g}' has only {len(a)} usable value(s); at least {MIN_GROUP_N} are needed.",
                "insufficient_data", {"group": g},
            )
    return arrays, chosen


# ------------------------------------------------------------ the tests

def _ttest(arrays, labels, alpha):
    ga, gb = labels
    a, b = arrays[ga], arrays[gb]
    r = stats.ttest_ind(a, b, equal_var=False)
    n1, n2 = len(a), len(b)
    diff = float(a.mean() - b.mean())
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = diff / pooled if pooled > 0 else None
    ci = None
    try:
        c = r.confidence_interval(confidence_level=1 - alpha)
        ci = {"level": 1 - alpha, "low": to_py(c.low), "high": to_py(c.high)}
    except (AttributeError, TypeError, ValueError):
        pass
    extra = {
        "group_a": ga, "group_b": gb,
        "mean_a": to_py(a.mean()), "mean_b": to_py(b.mean()),
        "mean_difference": to_py(diff),
        "degrees_of_freedom": to_py(getattr(r, "df", None)),
        "confidence_interval": ci,
        "effect_size": _effect("Cohen's d", d, (0.2, 0.5, 0.8)),
    }
    method = ("Welch's t-test (does not assume equal variances) comparing the means of the two groups; "
              "mean_difference = mean_a - mean_b.")
    return "Welch's t-test", method, r.statistic, r.pvalue, extra


def _mannwhitney(arrays, labels, alpha):
    ga, gb = labels
    a, b = arrays[ga], arrays[gb]
    r = stats.mannwhitneyu(a, b, alternative="two-sided")
    rank_biserial = 2 * float(r.statistic) / (len(a) * len(b)) - 1
    extra = {
        "group_a": ga, "group_b": gb,
        "median_a": to_py(np.median(a)), "median_b": to_py(np.median(b)),
        "effect_size": _effect("Rank-biserial correlation", rank_biserial, (0.1, 0.3, 0.5)),
    }
    method = ("Mann-Whitney U test (rank-based, no normality assumption). The statistic is U for group_a; "
              "a positive rank-biserial value means group_a tends to have larger values.")
    return "Mann-Whitney U test", method, r.statistic, r.pvalue, extra


def _anova(arrays, labels, alpha):
    groups = list(arrays.values())
    r = stats.f_oneway(*groups)
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((allv - grand) ** 2).sum())
    eta = ss_between / ss_total if ss_total > 0 else None
    extra = {
        "df_between": len(groups) - 1,
        "df_within": int(len(allv) - len(groups)),
        "effect_size": _effect("Eta squared", eta, (0.01, 0.06, 0.14)),
    }
    method = ("One-way ANOVA comparing group means. It shows that at least one group differs, "
              "not which one. It assumes roughly equal variances and approximately normal data.")
    return "One-way ANOVA", method, r.statistic, r.pvalue, extra


def _kruskal(arrays, labels, alpha):
    groups = list(arrays.values())
    try:
        r = stats.kruskal(*groups)
    except ValueError as exc:
        raise AnalysisError(f"The Kruskal-Wallis test could not be computed: {exc}", "test_failed") from None
    n = int(sum(len(g) for g in groups))
    eps = float(r.statistic) / (n - 1) if n > 1 else None
    extra = {
        "degrees_of_freedom": len(groups) - 1,
        "effect_size": _effect("Epsilon squared", eps, (0.01, 0.06, 0.14)),
    }
    method = ("Kruskal-Wallis H test (rank-based alternative to ANOVA). It shows that at least one group "
              "differs, not which one.")
    return "Kruskal-Wallis H test", method, r.statistic, r.pvalue, extra


_RUNNERS = {"ttest": _ttest, "mannwhitney": _mannwhitney, "anova": _anova, "kruskal": _kruskal}


def _group_test(work, test, value_column, group_column, groups, applied, alpha, warnings):
    arrays, chosen = _collect_groups(work, value_column, group_column, groups, test in _TWO_GROUP)
    sizes = {g: int(len(a)) for g, a in arrays.items()}
    n_total = sum(sizes.values())

    excluded = len(work) - n_total
    if excluded > 0:
        warnings.append(
            f"{excluded:,} rows were not used (missing or non-finite values, or groups not selected)."
        )
    small = [g for g, n in sizes.items() if n < SMALL_GROUP_N]
    if small:
        warnings.append(f"Very small group(s) {small}: results are unreliable with fewer than {SMALL_GROUP_N} values.")

    allv = np.concatenate(list(arrays.values()))
    if np.ptp(allv) == 0:
        raise AnalysisError("All values are identical, so the test cannot be computed.", "no_variation")

    name, method, stat, p, extra = _RUNNERS[test](arrays, chosen, alpha)
    stat, p = to_py(stat), to_py(p)
    if p is None:
        raise AnalysisError("The test could not be computed for this data.", "test_failed")

    group_table = pd.DataFrame([
        {"group": g, "n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
         "std": float(a.std(ddof=1))}
        for g, a in arrays.items()
    ])
    result = {
        "test": name, "statistic": stat, "p_value": p, "alpha": float(alpha),
        "significant": bool(p < alpha),
        "value_column": value_column, "group_column": group_column, "n_groups": len(chosen),
        **extra, "note": NOTE_GROUP,
    }
    return make_result(
        "run_statistical_test", method=method,
        inputs={"test": test, "value_column": value_column, "group_column": group_column,
                "groups": chosen, "filters": applied, "alpha": float(alpha)},
        rows_used=n_total, result=result, table=df_to_table(group_table), warnings=warnings,
    )


def _chi_square(work, column_a, column_b, applied, alpha, warnings):
    sub = work[[column_a, column_b]].dropna()
    dropped = len(work) - len(sub)
    if dropped:
        warnings.append(f"{dropped:,} rows with a missing value in either column were excluded.")
    if sub.empty:
        raise AnalysisError("No rows have values in both columns.", "no_rows")

    a, b = sub[column_a].astype(str), sub[column_b].astype(str)
    for name, s in ((column_a, a), (column_b, b)):
        k = int(s.nunique())
        if k > MAX_CATEGORIES:
            raise AnalysisError(
                f"'{name}' has {k} distinct values (limit {MAX_CATEGORIES}). Use a column with fewer categories.",
                "too_many_categories",
            )
        if k < 2:
            raise AnalysisError(f"'{name}' has only one distinct value, so there is nothing to compare.", "insufficient_data")

    ct = pd.crosstab(a, b)
    chi2, p, dof, expected = stats.chi2_contingency(ct.to_numpy(), correction=False)
    n = int(ct.to_numpy().sum())
    k = min(ct.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else None
    low_share = float((expected < 5).mean())
    if low_share > 0.2:
        warnings.append(
            f"{low_share:.0%} of cells have an expected count below 5, so the chi-square approximation may be unreliable."
        )

    stat, p = to_py(chi2), to_py(p)
    if p is None:
        raise AnalysisError("The test could not be computed for this data.", "test_failed")
    return make_result(
        "run_statistical_test",
        method="Chi-square test of independence on the contingency table (no continuity correction); effect size is Cramér's V.",
        inputs={"test": "chi_square", "column_a": column_a, "column_b": column_b,
                "filters": applied, "alpha": float(alpha)},
        rows_used=n,
        result={
            "test": "Chi-square test of independence", "statistic": stat, "p_value": p,
            "dof": int(dof), "alpha": float(alpha), "significant": bool(p < alpha),
            "column_a": column_a, "column_b": column_b,
            "effect_size": _effect("Cramér's V", cramers_v, (0.1, 0.3, 0.5)),
            "cells_expected_below_5_pct": round(100 * low_share, 1),
            "note": NOTE_CHI,
        },
        table=df_to_table(ct.reset_index()),
        warnings=warnings,
    )


def run_statistical_test(df: pd.DataFrame, test: str, value_column: str | None = None,
                         group_column: str | None = None, groups=None,
                         column_a: str | None = None, column_b: str | None = None,
                         filters=None, alpha: float = 0.05) -> dict:
    """Run a hypothesis test.

    ttest / mannwhitney: numeric value_column by group_column (exactly two groups)
    anova / kruskal:     numeric value_column by group_column (two or more groups)
    chi_square:          association between categorical column_a and column_b
    """
    test = str(test).lower()
    if test not in TESTS:
        raise AnalysisError(
            f"Unsupported test '{test}'. Supported: {', '.join(TESTS)}.",
            "unsupported_operation", {"supported": list(TESTS)},
        )
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("'alpha' must be a number between 0 and 1.", "invalid_parameter")

    if test == "chi_square":
        if not column_a or not column_b:
            raise AnalysisError("chi_square needs column_a and column_b.", "invalid_parameter")
        if column_a == column_b:
            raise AnalysisError("column_a and column_b must be different columns.", "invalid_parameter")
        require_columns(df, [column_a, column_b])
    else:
        if not value_column or not group_column:
            raise AnalysisError(
                f"'{test}' needs value_column (numeric) and group_column (categories).", "invalid_parameter"
            )
        if value_column == group_column:
            raise AnalysisError("value_column and group_column must be different columns.", "invalid_parameter")
        require_columns(df, [value_column, group_column])
        require_numeric(df, value_column)

    work, applied, fw = apply_filters(df, filters)
    warnings = list(fw)
    if len(work) == 0:
        raise AnalysisError("No rows remain after filtering.", "no_rows", {"filters": applied, "warnings": fw})

    if test == "chi_square":
        return _chi_square(work, column_a, column_b, applied, alpha, warnings)
    return _group_test(work, test, value_column, group_column, groups, applied, alpha, warnings)