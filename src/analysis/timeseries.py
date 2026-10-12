"""Time-series aggregation, period-over-period change, and a simple trend test."""
import numpy as np
import pandas as pd
from scipy import stats

from .common import (
    AnalysisError,
    df_to_table,
    get_datetime_series,
    make_result,
    require_columns,
    require_numeric,
    to_py,
)
from .filtering import apply_filters

FREQS = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}
TS_OPS = ("sum", "mean", "median", "count", "min", "max")
MAX_PERIODS = 2000
MIN_TREND_POINTS = 4
PARTIAL_THRESHOLD = 0.95


def _coverage_warnings(full: pd.PeriodIndex, lo: pd.Timestamp, hi: pd.Timestamp, freq: str) -> list[str]:
    """Warn when the first/last period is only partly covered by the data."""
    if freq == "D":
        return []
    checks = [("first", full[0])] + ([("last", full[-1])] if len(full) > 1 else [])
    msgs = []
    for label, period in checks:
        start, end = period.start_time.normalize(), period.end_time.normalize()
        total = (end - start).days + 1
        covered = (min(end, hi) - max(start, lo)).days + 1
        share = covered / total
        if share < PARTIAL_THRESHOLD:
            msgs.append(
                f"The {label} period ({period}) is partial: the data covers about {share:.0%} of it, "
                "so its value is not comparable with complete periods."
            )
    return msgs


def _trend(values: np.ndarray) -> dict:
    ok = ~np.isnan(values)
    y = values[ok]
    x = np.flatnonzero(ok).astype(float)
    if len(y) < MIN_TREND_POINTS:
        return {"available": False, "reason": f"Need at least {MIN_TREND_POINTS} periods with values for a trend test."}
    if np.ptp(y) == 0:
        return {"available": True, "direction": "flat", "slope_per_period": 0.0,
                "r_squared": None, "p_value": None, "n_points": int(len(y))}
    res = stats.linregress(x, y)
    p = float(res.pvalue)
    if p < 0.05:
        direction = "increasing" if res.slope > 0 else "decreasing"
    else:
        direction = "no clear trend"
    return {
        "available": True,
        "direction": direction,
        "slope_per_period": float(res.slope),
        "r_squared": float(res.rvalue ** 2),
        "p_value": p,
        "n_points": int(len(y)),
        "note": ("Linear trend over the period index. The p-value assumes independent points, which "
                 "time series often violate, so treat it as indicative only."),
    }


def _point(row: pd.Series) -> dict:
    return {"period": str(row["period"]), "value": to_py(row["value"])}


def analyze_time_series(df: pd.DataFrame, date_column: str, value_column: str | None = None,
                        op: str = "sum", freq: str = "M", filters=None) -> dict:
    """Aggregate a value (or row count) per period and describe how it changed.

    With no value_column, rows per period are counted.
    """
    warnings: list[str] = []
    freq = str(freq).upper()
    if freq not in FREQS:
        raise AnalysisError(
            f"Unsupported frequency '{freq}'. Use one of: {', '.join(FREQS)} (day, week, month, quarter, year).",
            "invalid_parameter",
        )
    op = str(op).lower()
    if op not in TS_OPS:
        raise AnalysisError(
            f"Unsupported aggregation '{op}'. Supported: {', '.join(TS_OPS)}.",
            "unsupported_operation", {"supported": list(TS_OPS)},
        )

    require_columns(df, [date_column] + ([value_column] if value_column else []))
    if value_column:
        require_numeric(df, value_column)
    else:
        op = "count"

    work, applied, fw = apply_filters(df, filters)
    warnings += fw
    if len(work) == 0:
        raise AnalysisError("No rows remain after filtering.", "no_rows", {"filters": applied, "warnings": fw})

    dates = get_datetime_series(work, date_column)
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)

    valid = dates.notna().to_numpy()
    n_bad = int((~valid).sum())
    if n_bad:
        warnings.append(f"{n_bad:,} rows with a missing or unreadable date were excluded.")
    if not valid.any():
        raise AnalysisError("No rows with a usable date.", "no_rows")

    d = dates[valid]
    frame = pd.DataFrame({"period": d.dt.to_period(freq)})
    if value_column:
        frame["value"] = work[value_column].to_numpy()[valid]

    unit = FREQS[freq]
    if value_column is None:
        series = frame.groupby("period").size()
    else:
        grouped = frame.groupby("period")["value"]
        series = grouped.agg(op)
        if op == "sum":
            series = series.where(grouped.count() > 0)  # all-missing period -> missing, not 0
    series = series.astype(float)

    present = series.index
    full = pd.period_range(present.min(), present.max())
    if len(full) > MAX_PERIODS:
        raise AnalysisError(
            f"This would create {len(full):,} {unit} periods (limit {MAX_PERIODS:,}). "
            "Use a coarser frequency or filter the date range.",
            "too_many_periods",
        )

    series = series.reindex(full)
    absent = ~full.isin(present)
    n_absent = int(absent.sum())
    if n_absent:
        if op in ("sum", "count"):
            series = series.mask(absent, 0.0)
            warnings.append(f"{n_absent} {unit}(s) had no records and are shown as 0 for '{op}'.")
        else:
            warnings.append(f"{n_absent} {unit}(s) had no records and are left empty (not zero).")

    lo, hi = d.min().normalize(), d.max().normalize()
    warnings += _coverage_warnings(full, lo, hi, freq)

    table = pd.DataFrame({"period": [str(p) for p in full], "value": series.to_numpy(dtype=float)})
    prev = table["value"].shift(1)
    table["change"] = table["value"] - prev
    table["change_pct"] = (100 * (table["value"] - prev) / prev).where(prev > 0)

    known = table.dropna(subset=["value"])
    if known.empty:
        raise AnalysisError("No values could be computed for any period.", "no_rows")
    first_row, last_row = known.iloc[0], known.iloc[-1]
    peak_row = known.loc[known["value"].idxmax()]
    low_row = known.loc[known["value"].idxmin()]
    change_abs = last_row["value"] - first_row["value"]
    change_pct = 100 * change_abs / first_row["value"] if first_row["value"] > 0 else None

    return make_result(
        "analyze_time_series",
        method=(f"Rows grouped by {unit} of '{date_column}'; "
                f"{op} of {repr(value_column) if value_column else 'rows'} per {unit}. "
                "Linear regression over the period index for the trend."),
        inputs={"date_column": date_column, "value_column": value_column, "operation": op,
                "frequency": unit, "filters": applied},
        rows_used=int(valid.sum()),
        result={
            "frequency": unit,
            "operation": op,
            "n_periods": int(len(table)),
            "date_range": {"start": to_py(d.min()), "end": to_py(d.max())},
            "first": _point(first_row),
            "last": _point(last_row),
            "peak": _point(peak_row),
            "lowest": _point(low_row),
            "overall_change": {"absolute": to_py(change_abs), "pct": to_py(change_pct)},
            "trend": _trend(table["value"].to_numpy(dtype=float)),
        },
        table=df_to_table(table, MAX_PERIODS),
        warnings=warnings,
    )