"""Row filtering with strict validation. Never modifies the input DataFrame."""
import math

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from .common import (
    AnalysisError,
    check_limit,
    df_to_table,
    get_datetime_series,
    make_result,
    require_columns,
)

FILTER_OPS = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "between", "contains", "is_null", "not_null")
_ORDERED = {"gt", "gte", "lt", "lte", "between"}
_NO_VALUE = {"is_null", "not_null"}
MAX_FILTERS = 10
MAX_IN_VALUES = 500


def _kind(df: pd.DataFrame, column: str, op: str):
    s = df[column]
    if pdt.is_bool_dtype(s):
        return "bool", s
    if pdt.is_numeric_dtype(s):
        return "numeric", s
    if pdt.is_datetime64_any_dtype(s):
        return "datetime", s
    if op in _ORDERED:  # text column: only OK if it is really a date column
        try:
            return "datetime", get_datetime_series(df, column)
        except AnalysisError:
            raise AnalysisError(
                f"Operator '{op}' needs a numeric or date column, but '{column}' is text.",
                "invalid_column_type", {"column": column},
            ) from None
    return "text", s


def _coerce(value, kind: str, column: str):
    try:
        if kind == "numeric":
            if isinstance(value, bool):
                raise ValueError
            v = float(value)
            if not math.isfinite(v):
                raise ValueError
            return v
        if kind == "datetime":
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                raise ValueError
            return ts
        if kind == "bool":
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in ("true", "1", "yes"):
                return True
            if text in ("false", "0", "no"):
                return False
            raise ValueError
        return str(value)
    except (ValueError, TypeError):
        raise AnalysisError(
            f"Value {value!r} cannot be used to filter column '{column}' ({kind}).",
            "invalid_filter_value", {"column": column, "value": str(value)},
        ) from None


def _as_bool(mask) -> np.ndarray:
    return pd.Series(mask).fillna(False).astype(bool).to_numpy()


def _build_mask(df: pd.DataFrame, f: dict):
    column, op, value = f["column"], f["op"], f["value"]

    if op in _NO_VALUE:
        s = df[column]
        return s.isna() if op == "is_null" else s.notna()

    kind, s = _kind(df, column, op)

    if op in _ORDERED and kind not in ("numeric", "datetime"):
        raise AnalysisError(
            f"Operator '{op}' needs a numeric or date column.", "invalid_column_type", {"column": column}
        )

    if op == "contains":
        if kind != "text":
            raise AnalysisError(
                f"'contains' only works on text columns, but '{column}' is {kind}.",
                "invalid_column_type", {"column": column},
            )
        text = _coerce(value, "text", column)
        return s.astype("string").str.contains(text, case=False, regex=False, na=False)

    if op in ("eq", "ne"):
        v = _coerce(value, kind, column)
        return (s == v) if op == "eq" else ((s != v) & s.notna())

    if op in ("in", "not_in"):
        if not isinstance(value, (list, tuple)) or not value:
            raise AnalysisError(f"'{op}' needs a non-empty list of values.", "invalid_filter_value")
        if len(value) > MAX_IN_VALUES:
            raise AnalysisError(f"'{op}' accepts at most {MAX_IN_VALUES} values.", "invalid_filter_value")
        vals = [_coerce(v, kind, column) for v in value]
        hit = s.isin(vals)
        return hit if op == "in" else (~hit & s.notna())

    if op == "between":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise AnalysisError("'between' needs a [low, high] pair.", "invalid_filter_value")
        lo, hi = _coerce(value[0], kind, column), _coerce(value[1], kind, column)
        if lo > hi:
            raise AnalysisError("'between' low value is greater than the high value.", "invalid_filter_value")
        return (s >= lo) & (s <= hi)  # inclusive on both ends

    v = _coerce(value, kind, column)
    return {"gt": s > v, "gte": s >= v, "lt": s < v, "lte": s <= v}[op]


def _validate_filters(df: pd.DataFrame, filters) -> list[dict]:
    if filters is None:
        return []
    if not isinstance(filters, (list, tuple)):
        raise AnalysisError("'filters' must be a list of {column, op, value} objects.", "invalid_parameter")
    if len(filters) > MAX_FILTERS:
        raise AnalysisError(f"At most {MAX_FILTERS} filters are allowed.", "invalid_parameter")

    clean = []
    for f in filters:
        if not isinstance(f, dict):
            raise AnalysisError("Each filter must be an object with 'column' and 'op'.", "invalid_parameter")
        column = f.get("column")
        op = str(f.get("op", "")).lower()
        if op not in FILTER_OPS:
            raise AnalysisError(
                f"Unsupported filter operator '{op}'. Supported: {', '.join(FILTER_OPS)}.",
                "unsupported_operation", {"supported": list(FILTER_OPS)},
            )
        require_columns(df, [column])
        if op not in _NO_VALUE and "value" not in f:
            raise AnalysisError(f"Filter '{op}' on '{column}' needs a 'value'.", "invalid_filter_value")
        clean.append({"column": column, "op": op, "value": f.get("value")})
    return clean


def _no_match_warning(df: pd.DataFrame, f: dict) -> str:
    msg = f"Filter {f['column']} {f['op']} {f['value']!r} matched no rows."
    s = df[f["column"]]
    if not (pdt.is_numeric_dtype(s) or pdt.is_datetime64_any_dtype(s)):
        examples = [str(v)[:40] for v in s.dropna().value_counts().head(5).index]
        if examples:
            msg += f" Text matching is exact; example values: {examples}."
    return msg


def apply_filters(df: pd.DataFrame, filters):
    """Apply filters (combined with AND). Returns (filtered_df, clean_filters, warnings)."""
    clean = _validate_filters(df, filters)
    warnings: list[str] = []
    if not clean:
        return df, clean, warnings

    keep = np.ones(len(df), dtype=bool)
    for f in clean:
        try:
            mask = _as_bool(_build_mask(df, f))
        except TypeError:
            raise AnalysisError(
                f"Filter on '{f['column']}' with value {f['value']!r} is not compatible with the column.",
                "invalid_filter_value", {"column": f["column"]},
            ) from None
        if not mask.any():
            warnings.append(_no_match_warning(df, f))
        keep &= mask

    if not keep.any() and len(clean) > 1:
        warnings.append("The combined filters match no rows.")
    return df[keep], clean, warnings


def filter_data(df: pd.DataFrame, filters, columns=None, limit: int = 10) -> dict:
    """Filter rows and return counts plus a small preview."""
    warnings: list[str] = []
    limit = check_limit(limit, warnings, maximum=50)
    if not filters:
        raise AnalysisError("filter_data needs at least one filter.", "invalid_parameter")

    show_cols = list(df.columns)
    if columns:
        require_columns(df, list(columns))
        show_cols = list(columns)

    filtered, clean, fw = apply_filters(df, filters)
    warnings += fw
    n_before, n_after = len(df), len(filtered)

    return make_result(
        "filter_data",
        method="Rows matching ALL filters (AND). Missing values never match a comparison.",
        inputs={"filters": clean, "columns": show_cols if columns else None, "limit": limit},
        rows_used=n_before,
        result={
            "rows_before": n_before,
            "rows_after": n_after,
            "pct_kept": round(100 * n_after / n_before, 2) if n_before else 0.0,
        },
        table=df_to_table(filtered[show_cols], limit),
        warnings=warnings,
    )