"""Assumption checks and post-hoc pairwise tests with multiple-comparison correction."""
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .common import AnalysisError, df_to_table, make_result, require_columns, require_numeric, to_py
from .filtering import apply_filters

MAX_GROUPS = 12
MIN_GROUP_N = 3
SHAPIRO_MAX = 5000
LARGE_N = 30
SEED = 0
CORRECTIONS = {"holm": "holm", "bonferroni": "bonferroni", "fdr_bh": "fdr_bh"}
PAIR_METHODS = ("welch", "mannwhitney")

NOTE_CHECK = ("Normality tests are sensitive: with many rows they flag tiny departures, with few rows they miss "
              "large ones. Welch's t-test and ANOVA tolerate mild non-normality when groups are large. Treat this "
              "as guidance, not proof.")
NOTE_POSTHOC = ("Run pairwise comparisons after a significant overall test (anova or kruskal), otherwise the "
                "chance of a false positive grows. Adjusted p-values control for the number of pairs compared. "
                "Significance does not show why groups differ.")


def _groups(df, value_column, group_column, groups, filters, warnings):
    require_columns(df, [value_column, group_column])
    require_numeric(df, value_column)
    if value_column == group_column:
        raise AnalysisError("value_column and group_column must be different.", "invalid_parameter")
    work, applied, fw = apply_filters(df, filters)
    warnings += fw
    if len(work) == 0:
        raise AnalysisError("No rows remain after filtering.", "no_rows", {"filters": applied})
    sub = work[[group_column, value_column]].dropna()
    vals = sub[value_column].to_numpy(dtype=float)
    ok = np.isfinite(vals)
    labels = sub[group_column].astype(str).to_numpy()[ok]
    vals = vals[ok]
    available = sorted(set(labels.tolist()))
    if groups is None:
        chosen = available
    else:
        if isinstance(groups, str) or not isinstance(groups, (list, tuple)):
            raise AnalysisError("'groups' must be a list.", "invalid_parameter")
        chosen = [str(g) for g in groups]
        if len(set(chosen)) != len(chosen):
            raise AnalysisError("'groups' must not contain duplicates.", "invalid_parameter")
        missing = [g for g in chosen if g not in available]
        if missing:
            raise AnalysisError(f"Group(s) not found in '{group_column}': {missing}. Available: {available[:20]}.",
                                "group_not_found", {"missing": missing, "available": available[:50]})
    if len(chosen) < 2:
        raise AnalysisError("Need at least two groups.", "insufficient_data")
    if len(chosen) > MAX_GROUPS:
        raise AnalysisError(f"{len(chosen)} groups is too many (limit {MAX_GROUPS}). Pass groups=[...] to choose.",
                            "too_many_groups")
    arrays = {g: vals[labels == g] for g in chosen}
    for g, a in arrays.items():
        if len(a) < MIN_GROUP_N:
            raise AnalysisError(f"Group '{g}' has only {len(a)} usable value(s); at least {MIN_GROUP_N} are needed.",
                                "insufficient_data", {"group": g})
    return arrays, applied, int(len(vals[np.isin(labels, chosen)]))


def check_test_assumptions(df: pd.DataFrame, value_column: str, group_column: str, groups=None, filters=None) -> dict:
    warnings: list[str] = []
    arrays, applied, n = _groups(df, value_column, group_column, groups, filters, warnings)
    rows = []
    non_normal_small = []
    for g, a in arrays.items():
        sample = a
        sampled = False
        if len(a) > SHAPIRO_MAX:
            sample = np.random.default_rng(SEED).choice(a, SHAPIRO_MAX, replace=False)
            sampled = True
        if np.ptp(sample) == 0:
            rows.append({"group": g, "n": len(a), "shapiro_w": None, "shapiro_p": None, "looks_normal": None,
                         "note": "all values identical"})
            non_normal_small.append(g)
            continue
        w, p = stats.shapiro(sample)
        normal = bool(p >= 0.05)
        if not normal and len(a) < LARGE_N:
            non_normal_small.append(g)
        rows.append({"group": g, "n": len(a), "shapiro_w": float(w), "shapiro_p": float(p), "looks_normal": normal,
                     "note": f"tested on a random {SHAPIRO_MAX:,} of {len(a):,}" if sampled else ""})

    all_vals = np.concatenate(list(arrays.values()))
    if np.ptp(all_vals) == 0:
        raise AnalysisError("All values are identical, so the checks cannot be computed.", "no_variation")
    lev = stats.levene(*arrays.values(), center="median")
    equal_var = bool(lev.pvalue >= 0.05)

    two = len(arrays) == 2
    if non_normal_small:
        rec = "mannwhitney" if two else "kruskal"
        why = (f"Group(s) {non_normal_small} are small and not clearly normal, so a rank-based test is safer.")
    else:
        rec = "ttest" if two else "anova"
        why = ("Groups look roughly normal or are large enough (30 or more values) for the mean-based test; "
               "Welch's version does not need equal variances." if two else
               "Groups look roughly normal or are large enough (30 or more values) for ANOVA.")
        if not equal_var and not two:
            why += " Variances differ, so interpret ANOVA cautiously or prefer kruskal."
    if len(all_vals) > 20000:
        warnings.append("With very many rows, normality tests flag trivial departures; look at the distribution too.")

    table = pd.DataFrame(rows)
    return make_result(
        "check_test_assumptions",
        method=("Shapiro-Wilk normality test per group (p < 0.05 suggests non-normal) and Levene's test with the "
                "median (Brown-Forsythe) for equal variances (p < 0.05 suggests unequal)."),
        inputs={"value_column": value_column, "group_column": group_column, "groups": list(arrays), "filters": applied},
        rows_used=n,
        result={"levene_statistic": float(lev.statistic), "levene_p": float(lev.pvalue), "equal_variances": equal_var,
                "all_groups_look_normal": all(r["looks_normal"] for r in rows if r["looks_normal"] is not None)
                and not any(r["looks_normal"] is None for r in rows),
                "recommended_test": rec, "reason": why, "note": NOTE_CHECK},
        table=df_to_table(table), warnings=warnings)


def _pair(method, a, b):
    if method == "welch":
        r = stats.ttest_ind(a, b, equal_var=False)
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = float((a.mean() - b.mean()) / pooled) if pooled > 0 else None
        return float(r.pvalue), d, float(a.mean() - b.mean())
    r = stats.mannwhitneyu(a, b, alternative="two-sided")
    rb = 2 * float(r.statistic) / (len(a) * len(b)) - 1
    return float(r.pvalue), rb, float(np.median(a) - np.median(b))


def run_posthoc_tests(df: pd.DataFrame, value_column: str, group_column: str, method: str = "welch",
                      correction: str = "holm", groups=None, filters=None, alpha: float = 0.05) -> dict:
    warnings: list[str] = []
    method = str(method).lower()
    if method not in PAIR_METHODS:
        raise AnalysisError(f"Unsupported method '{method}'. Supported: {', '.join(PAIR_METHODS)}.",
                            "unsupported_operation", {"supported": list(PAIR_METHODS)})
    correction = str(correction).lower()
    if correction not in CORRECTIONS:
        raise AnalysisError(f"Unsupported correction '{correction}'. Supported: {', '.join(CORRECTIONS)}.",
                            "unsupported_operation", {"supported": list(CORRECTIONS)})
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("'alpha' must be between 0 and 1.", "invalid_parameter")

    arrays, applied, n = _groups(df, value_column, group_column, groups, filters, warnings)
    if np.ptp(np.concatenate(list(arrays.values()))) == 0:
        raise AnalysisError("All values are identical, so the tests cannot be computed.", "no_variation")
    small = [g for g, a in arrays.items() if len(a) < 5]
    if small:
        warnings.append(f"Very small group(s) {small}: pairwise results are unreliable with fewer than 5 values.")

    pairs = list(itertools.combinations(arrays, 2))
    raw, rows = [], []
    for ga, gb in pairs:
        a, b = arrays[ga], arrays[gb]
        if np.ptp(np.concatenate([a, b])) == 0:
            p, eff, diff = 1.0, None, 0.0
        else:
            p, eff, diff = _pair(method, a, b)
        raw.append(p)
        rows.append({"group_a": ga, "group_b": gb, "n_a": len(a), "n_b": len(b),
                     "difference": diff, "effect_size": eff, "p_raw": p})
    reject, adj, _, _ = multipletests(raw, alpha=alpha, method=CORRECTIONS[correction])
    for r, a_p, rej in zip(rows, adj, reject):
        r["p_adjusted"] = float(a_p)
        r["significant"] = bool(rej)
    table = pd.DataFrame(rows).sort_values("p_adjusted", kind="mergesort").reset_index(drop=True)

    n_sig = int(sum(r["significant"] for r in rows))
    raw_sig = int(sum(p < alpha for p in raw))
    return make_result(
        "run_posthoc_tests",
        method=(f"{'Pairwise Welch t-tests (difference = mean_a - mean_b; effect = Cohen d)' if method == 'welch' else 'Pairwise Mann-Whitney U tests (difference = median_a - median_b; effect = rank-biserial)'} "
                f"across {len(arrays)} groups ({len(pairs)} pairs), p-values adjusted by {correction}."),
        inputs={"value_column": value_column, "group_column": group_column, "groups": list(arrays),
                "method": method, "correction": correction, "alpha": float(alpha), "filters": applied},
        rows_used=n,
        result={"n_groups": len(arrays), "n_pairs": len(pairs), "alpha": float(alpha), "correction": correction,
                "n_significant_adjusted": n_sig, "n_significant_unadjusted": raw_sig, "note": NOTE_POSTHOC},
        table=df_to_table(table), warnings=warnings)