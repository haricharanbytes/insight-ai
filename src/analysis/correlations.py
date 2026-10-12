"""Pairwise correlation between numeric columns."""
import numpy as np
import pandas as pd
from pandas.api import types as pdt
from scipy import stats

from .common import (
    AnalysisError,
    check_limit,
    df_to_table,
    make_result,
    require_columns,
    require_numeric,
    to_py,
)
from .filtering import apply_filters

METHODS = ("pearson", "spearman")
MAX_COLUMNS = 15
MIN_PAIR_N = 5

NOTE = (
    "Correlation measures association, not causation. It can be driven by outliers, "
    "shared underlying causes, or chance. P-values are not adjusted for testing many "
    "pairs, and with many rows even tiny correlations look 'significant', so judge "
    "importance by the size of r."
)


def _strength(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    if a < 0.7:
        return "strong"
    return "very strong"


def _pair(x: pd.Series, y: pd.Series, method: str) -> dict:
    xv = x.to_numpy(dtype=float, na_value=np.nan)
    yv = y.to_numpy(dtype=float, na_value=np.nan)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[ok], yv[ok]
    n = int(len(xv))
    if n < MIN_PAIR_N:
        return {"n": n, "reason": f"fewer than {MIN_PAIR_N} rows with both values"}
    if np.ptp(xv) == 0 or np.ptp(yv) == 0:
        return {"n": n, "reason": "one of the columns is constant"}
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    r, p = fn(xv, yv)
    if not np.isfinite(r):
        return {"n": n, "reason": "correlation could not be computed"}
    return {"n": n, "r": float(r), "p": float(p)}


def _select_columns(df: pd.DataFrame, columns, warnings: list) -> list[str]:
    if columns is None:
        from src.data.profiler import _name_looks_like_id  # lazy: avoids circular import

        cols, skipped = [], []
        for c in df.columns:
            s = df[c]
            if pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s):
                (skipped if _name_looks_like_id(str(c)) else cols).append(str(c))
        if skipped:
            warnings.append(
                "Identifier-like numeric columns were excluded automatically: "
                + ", ".join(skipped) + "."
            )
        if len(cols) > MAX_COLUMNS:
            warnings.append(
                f"Using the first {MAX_COLUMNS} of {len(cols)} numeric columns. "
                "Name the columns you care about to choose others."
            )
            cols = cols[:MAX_COLUMNS]
        if len(cols) < 2:
            raise AnalysisError("Need at least two numeric columns for a correlation.", "invalid_parameter")
        return cols

    if isinstance(columns, str):
        columns = [columns]
    if not isinstance(columns, (list, tuple)):
        raise AnalysisError("'columns' must be a list of column names.", "invalid_parameter")
    cols = list(columns)
    if not 2 <= len(cols) <= MAX_COLUMNS:
        raise AnalysisError(f"Provide between 2 and {MAX_COLUMNS} numeric columns.", "invalid_parameter")
    require_columns(df, cols)
    if len(set(cols)) != len(cols):
        raise AnalysisError("Columns must be unique.", "invalid_parameter")
    for c in cols:
        require_numeric(df, c)
    return cols


def calculate_correlation(df: pd.DataFrame, columns=None, method: str = "pearson",
                          filters=None, limit: int = 10) -> dict:
    """Pairwise correlations (rows with both values only), strongest first.

    With no 'columns', all numeric non-identifier columns are used.
    """
    warnings: list[str] = []
    limit = check_limit(limit, warnings)
    method = str(method).lower()
    if method not in METHODS:
        raise AnalysisError(
            f"Unsupported method '{method}'. Supported: {', '.join(METHODS)}.",
            "unsupported_operation", {"supported": list(METHODS)},
        )

    cols = _select_columns(df, columns, warnings)
    work, applied, fw = apply_filters(df, filters)
    warnings += fw
    if len(work) == 0:
        raise AnalysisError("No rows remain after filtering.", "no_rows", {"filters": applied, "warnings": fw})

    n = len(cols)
    matrix: list[list] = [[None] * n for _ in range(n)]
    pairs, skipped = [], []
    for i in range(n):
        for j in range(i + 1, n):
            res = _pair(work[cols[i]], work[cols[j]], method)
            if "r" not in res:
                skipped.append(f"{cols[i]} / {cols[j]}: {res['reason']}")
                continue
            r = res["r"]
            matrix[i][j] = matrix[j][i] = r
            pairs.append({
                "column_a": cols[i], "column_b": cols[j], "r": r, "p_value": res["p"],
                "n": res["n"], "strength": _strength(r),
                "direction": "positive" if r > 0 else "negative" if r < 0 else "none",
            })

    if not pairs:
        raise AnalysisError(
            "No column pair had enough usable data. " + "; ".join(skipped[:3]),
            "insufficient_data",
        )
    if skipped:
        warnings.append(f"{len(skipped)} pair(s) skipped: " + "; ".join(skipped[:3]) + ("; ..." if len(skipped) > 3 else ""))

    used = {p["column_a"] for p in pairs} | {p["column_b"] for p in pairs}
    for i, c in enumerate(cols):
        if c in used:
            matrix[i][i] = 1.0

    tbl = pd.DataFrame(pairs)
    tbl = (tbl.assign(_abs=tbl["r"].abs())
              .sort_values("_abs", ascending=False, kind="mergesort")
              .drop(columns="_abs").reset_index(drop=True))

    label = "Pearson (linear)" if method == "pearson" else "Spearman (rank-based, monotonic)"
    return make_result(
        "calculate_correlation",
        method=f"{label} correlation, using only rows where both columns have a value.",
        inputs={"columns": cols, "method": method, "filters": applied, "limit": limit},
        rows_used=len(work),
        result={
            "method": method,
            "n_columns": n,
            "columns": cols,
            "pairs_computed": len(pairs),
            "pairs_skipped": len(skipped),
            "strongest": {k: to_py(v) for k, v in tbl.iloc[0].items()},
            "matrix": {"columns": cols, "values": matrix},
            "note": NOTE,
        },
        table=df_to_table(tbl, limit),
        warnings=warnings,
    )