"""Shared helpers for the deterministic analysis functions.

Every analysis function returns a plain, JSON-serializable dict built by
make_result(), so the same structure can feed the API, charts, reports and
(later) the LLM, which only ever explains these verified values.
"""
import datetime
import difflib
import math

import numpy as np
import pandas as pd
from pandas.api import types as pdt

MAX_TABLE_ROWS = 200
DATE_PARSE_MIN_RATIO = 0.9


class AnalysisError(Exception):
    """User-facing analysis error with a machine-readable code."""

    def __init__(self, message: str, code: str = "analysis_error", details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"error": True, "code": self.code, "message": self.message, "details": self.details}


def to_py(value):
    """Convert numpy/pandas scalars to JSON-safe Python values (NaN/inf -> None)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (pd.Timestamp, datetime.date)):
        return value.isoformat()
    return str(value)


def require_columns(df: pd.DataFrame, columns) -> None:
    """Raise column_not_found (with 'did you mean' hints) if any column is absent."""
    for c in columns:
        if not isinstance(c, str):
            raise AnalysisError("Column names must be text.", "invalid_parameter")
    missing = [c for c in columns if c not in df.columns]
    if not missing:
        return

    available = [str(x) for x in df.columns]
    hints = {}
    for c in missing:
        close = difflib.get_close_matches(c, available, n=3, cutoff=0.6)
        if close:
            hints[c] = close

    msg = f"Column(s) not found: {', '.join(repr(c) for c in missing)}."
    if hints:
        msg += " Did you mean: " + "; ".join(f"{k} -> {', '.join(v)}" for k, v in hints.items()) + "?"
    raise AnalysisError(
        msg, "column_not_found",
        {"missing": missing, "suggestions": hints, "available_columns": available[:100]},
    )


def require_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    s = df[column]
    if pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s):
        return s
    hint = ""
    if s.dtype == object:
        hint = " If it holds numbers stored as text, it needs a (user-approved) conversion first."
    raise AnalysisError(
        f"Column '{column}' is not numeric (type: {s.dtype}).{hint}",
        "invalid_column_type", {"column": column, "dtype": str(s.dtype)},
    )


def get_datetime_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return the column as datetimes (parsing text dates). Never modifies df."""
    s = df[column]
    if pdt.is_datetime64_any_dtype(s):
        return s
    if pdt.is_numeric_dtype(s) or pdt.is_bool_dtype(s):
        raise AnalysisError(
            f"Column '{column}' is numeric, not a date column.",
            "invalid_column_type", {"column": column},
        )
    from src.data.profiler import parse_text_dates  # lazy import: avoids a circular import

    non_null = int(s.notna().sum())
    parsed = parse_text_dates(s)
    if non_null == 0 or parsed.notna().sum() / non_null < DATE_PARSE_MIN_RATIO:
        raise AnalysisError(
            f"Column '{column}' could not be read as dates.",
            "invalid_column_type", {"column": column},
        )
    return parsed


def check_limit(limit, warnings: list, maximum: int = MAX_TABLE_ROWS) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise AnalysisError("'limit' must be a positive integer.", "invalid_parameter")
    if limit > maximum:
        warnings.append(f"'limit' was reduced to the maximum of {maximum}.")
        return maximum
    return limit


def df_to_table(df: pd.DataFrame, limit: int = MAX_TABLE_ROWS) -> dict:
    shown = df.head(limit)
    return {
        "columns": [str(c) for c in shown.columns],
        "rows": [[to_py(v) for v in row] for row in shown.itertuples(index=False, name=None)],
        "total_rows": int(len(df)),
        "truncated": bool(len(df) > limit),
    }


def make_result(tool: str, *, method: str, inputs: dict, rows_used: int,
                result: dict | None = None, table: dict | None = None,
                warnings: list | None = None) -> dict:
    return {
        "tool": tool,
        "method": method,
        "inputs": inputs,
        "rows_used": int(rows_used),
        "result": result or {},
        "table": table,
        "warnings": warnings or [],
    }