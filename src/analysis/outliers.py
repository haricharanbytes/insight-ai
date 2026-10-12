"""Outlier detection. iqr_bounds is used by the profiler; detect_outliers is an agent tool."""
import numpy as np
import pandas as pd
from pandas.api import types as pdt

from .common import AnalysisError, check_limit, df_to_table, make_result, require_columns, require_numeric, to_py
from .filtering import apply_filters

MIN_POINTS = 8
METHODS = ("iqr", "zscore")
DEFAULT_THRESHOLD = {"iqr": 1.5, "zscore": 3.0}


def iqr_bounds(series: pd.Series, k: float = 1.5) -> dict | None:
    """IQR (Tukey fence) outlier indicator for the profiler.

    Returns None when there are too few values or the IQR is zero
    (the method is not meaningful in those cases).
    These are *indicators*, not proof that a value is wrong.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[np.isfinite(s)]
    if len(s) < MIN_POINTS:
        return None

    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return None

    lower, upper = q1 - k * iqr, q3 + k * iqr
    low_count = int((s < lower).sum())
    high_count = int((s > upper).sum())
    count = low_count + high_count
    return {
        "method": "iqr", "k": k, "q1": q1, "q3": q3, "lower": lower, "upper": upper,
        "count": count, "low_count": low_count, "high_count": high_count,
        "pct": round(100 * count / len(s), 2),
    }


def detect_outliers(df: pd.DataFrame, column: str, method: str = "iqr",
                    threshold: float | None = None, filters=None, limit: int = 10) -> dict:
    """Flag statistically unusual values. Nothing is removed or changed."""
    warnings: list[str] = []
    limit = check_limit(limit, warnings, maximum=50)
    method = str(method).lower()
    if method not in METHODS:
        raise AnalysisError(
            f"Unsupported method '{method}'. Supported: {', '.join(METHODS)}.",
            "unsupported_operation", {"supported": list(METHODS)},
        )
    require_columns(df, [column])
    require_numeric(df, column)

    if threshold is None:
        threshold = DEFAULT_THRESHOLD[method]
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold) or threshold <= 0):
        raise AnalysisError("'threshold' must be a positive number.", "invalid_parameter")
    threshold = float(threshold)

    work, applied, fw = apply_filters(df, filters)
    warnings += fw
    if len(work) == 0:
        raise AnalysisError("No rows remain after filtering.", "no_rows", {"filters": applied, "warnings": fw})

    raw = work[column]
    values = pd.Series(raw.to_numpy(dtype=float, na_value=np.nan), index=raw.index)
    n_missing = int(values.isna().sum())
    n_inf = int(np.isinf(values).sum())
    if n_missing:
        warnings.append(f"{n_missing:,} missing values were ignored.")
    if n_inf:
        warnings.append(f"{n_inf:,} infinite values were ignored.")
    finite = values[np.isfinite(values)]
    if len(finite) < MIN_POINTS:
        raise AnalysisError(
            f"Need at least {MIN_POINTS} values to detect outliers; found {len(finite)}.",
            "insufficient_data",
        )

    if method == "iqr":
        q1, q3 = float(finite.quantile(0.25)), float(finite.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            raise AnalysisError(
                "The IQR is zero (over half the values are identical), so the IQR method "
                "cannot flag outliers. Try method='zscore'.",
                "method_not_applicable",
            )
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        bounds = {"lower": lower, "upper": upper, "q1": q1, "q3": q3}
        method_text = f"IQR method: values outside Q1 - {threshold:g}xIQR and Q3 + {threshold:g}xIQR are flagged."
        note = ("Flagged values are statistical indicators, not errors; skewed data (such as sales "
                "amounts) naturally produces many. Nothing was removed or changed.")
    else:
        mean, std = float(finite.mean()), float(finite.std(ddof=1))
        if not np.isfinite(std) or std == 0:
            raise AnalysisError(
                "The standard deviation is zero, so z-scores cannot be computed.",
                "method_not_applicable",
            )
        lower, upper = mean - threshold * std, mean + threshold * std
        bounds = {"lower": lower, "upper": upper, "mean": mean, "std": std}
        method_text = f"Z-score method: values more than {threshold:g} standard deviations from the mean are flagged."
        note = ("Z-scores use the mean and standard deviation, which extreme values themselves inflate, "
                "so real outliers can be masked in small or skewed data. Nothing was removed or changed.")

    mask = (finite < lower) | (finite > upper)
    flagged = finite[mask]
    below = (flagged < lower).to_numpy()
    distance = np.where(below, lower - flagged.to_numpy(), flagged.to_numpy() - upper)
    row_numbers = (flagged.index + 1) if pdt.is_integer_dtype(flagged.index) else flagged.index.astype(str)

    tbl = pd.DataFrame({
        "row_number": np.asarray(row_numbers),
        "value": flagged.to_numpy(),
        "direction": np.where(below, "low", "high"),
        "distance_beyond_bound": distance,
    }).sort_values("distance_beyond_bound", ascending=False, kind="mergesort").reset_index(drop=True)

    n_out = int(mask.sum())
    return make_result(
        "detect_outliers",
        method=method_text + " row_number is the 1-based position in the loaded data (header excluded).",
        inputs={"column": column, "method": method, "threshold": threshold, "filters": applied, "limit": limit},
        rows_used=len(finite),
        result={
            "column": column,
            "method": method,
            "threshold": threshold,
            "n_values": int(len(finite)),
            "n_outliers": n_out,
            "pct_outliers": round(100 * n_out / len(finite), 2),
            "low_count": int(below.sum()),
            "high_count": int(n_out - below.sum()),
            "bounds": {k: to_py(v) for k, v in bounds.items()},
            "min_value": to_py(finite.min()),
            "max_value": to_py(finite.max()),
            "note": note,
        },
        table=df_to_table(tbl, limit),
        warnings=warnings,
    )