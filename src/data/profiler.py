"""Dataset profiler: structure, types, missing values, outliers, quality issues.

Read-only: never modifies the DataFrame. Output is JSON-serializable.
"""
import datetime
import math
import re

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from pandas.tseries.api import guess_datetime_format

from src.analysis.outliers import iqr_bounds

MAX_TOP_VALUES = 10
DATE_SAMPLE_SIZE = 200
DATE_PARSE_THRESHOLD = 0.9
PLACEHOLDERS = {"-", "--", "?", "nil", "unknown", "missing", "n.a.", "not available"}

_ID_NAME = re.compile(
    r"(^|[^a-z])(id|uuid|guid|code|zip|postal|pincode|phone)([^a-z]|$)", re.IGNORECASE
)
_NUMERIC_STR = r"[+-]?\d+(\.\d+)?"


# helpers

def _py(value):
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


def _pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 2) if whole else 0.0


def _name_looks_like_id(name: str) -> bool:
    return bool(_ID_NAME.search(name)) or bool(re.search(r"[a-z]ID$", name))


# date parsing

def parse_text_dates(series: pd.Series) -> pd.Series:
    """Parse a text column to datetimes. Unparseable/blank -> NaT.

    Returns a new Series aligned to the input index; input is not modified.
    """
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    fmt = guess_datetime_format(s.iloc[0])
    parsed = None
    try:
        if fmt:
            parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed is None or parsed.notna().sum() < len(s):
            mixed = pd.to_datetime(s, format="mixed", errors="coerce")
            if parsed is None or mixed.notna().sum() > parsed.notna().sum():
                parsed = mixed
    except (ValueError, TypeError):
        parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    return parsed.reindex(series.index)


def _is_date_text_column(series: pd.Series) -> bool:
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if len(s) < 3:
        return False
    sample = s.sample(DATE_SAMPLE_SIZE, random_state=0) if len(s) > DATE_SAMPLE_SIZE else s

    # Plain numbers (e.g. 20200101, 2020) are not treated as dates.
    if sample.str.fullmatch(_NUMERIC_STR).mean() > 0.5:
        return False
    if sample.str.contains(r"\d").mean() < 0.9:
        return False
    if sample.str.len().median() < 6:
        return False
    return parse_text_dates(sample).notna().mean() >= DATE_PARSE_THRESHOLD


# type inference

def _infer_type(name: str, s: pd.Series) -> tuple[str, bool]:
    """Return (inferred_type, dates_are_text)."""
    non_null = s.dropna()
    if len(non_null) == 0:
        return "empty", False
    if pdt.is_bool_dtype(s):
        return "boolean", False
    if pdt.is_datetime64_any_dtype(s):
        return "datetime", False
    if pdt.is_numeric_dtype(s):
        return ("identifier" if _name_looks_like_id(name) else "numeric"), False

    as_str = non_null.astype(str)
    n_unique = as_str.nunique()

    # Numbers stored as text (e.g. "1,200")
    cleaned = as_str.str.replace(",", "", regex=False).str.strip()
    if n_unique > 10 and pd.to_numeric(cleaned, errors="coerce").notna().mean() >= 0.95:
        return "numeric_text", False

    if _is_date_text_column(s):
        return "datetime", True

    unique_ratio = n_unique / len(non_null)
    if as_str.str.len().mean() > 50:
        return "text", False
    if _name_looks_like_id(name) and unique_ratio > 0.3:
        return "identifier", False
    if len(non_null) > 20 and unique_ratio > 0.9:
        return "identifier", False
    return "categorical", False


# per-type details

def _numeric_details(s: pd.Series) -> dict:
    vals = s.dropna().astype(float)
    inf_count = int(np.isinf(vals).sum())
    vals = vals[np.isfinite(vals)]
    return {
        "mean": _py(vals.mean()),
        "median": _py(vals.median()),
        "std": _py(vals.std()),
        "min": _py(vals.min()),
        "max": _py(vals.max()),
        "q1": _py(vals.quantile(0.25)),
        "q3": _py(vals.quantile(0.75)),
        "zero_count": int((vals == 0).sum()),
        "negative_count": int((vals < 0).sum()),
        "inf_count": inf_count,
        "outliers": iqr_bounds(vals),
    }


def _datetime_details(s: pd.Series, from_text: bool) -> dict:
    parsed = parse_text_dates(s) if from_text else s
    valid = parsed.dropna()
    unparsed = 0
    if from_text:
        non_blank = int(s.dropna().astype(str).str.strip().ne("").sum())
        unparsed = non_blank - len(valid)
    details = {
        "parsed_from_text": from_text,
        "unparsed_count": int(unparsed),
        "min": None,
        "max": None,
        "span_days": None,
    }
    if len(valid):
        details["min"] = _py(valid.min())
        details["max"] = _py(valid.max())
        details["span_days"] = int((valid.max() - valid.min()).days)
    return details


def _top_values(as_str: pd.Series) -> list[dict]:
    total = len(as_str)
    vc = as_str.value_counts().head(MAX_TOP_VALUES)
    return [
        {"value": str(v)[:60], "count": int(c), "pct": _pct(int(c), total)}
        for v, c in vc.items()
    ]


def _text_details(non_null: pd.Series, inferred: str) -> dict:
    as_str = non_null.astype(str)
    stripped = as_str.str.strip()
    details = {
        "blank_count": int(stripped.eq("").sum()),
        "placeholder_count": int(stripped.str.lower().isin(PLACEHOLDERS).sum()),
    }
    if inferred in ("categorical", "boolean"):
        n_unique = int(as_str.nunique())
        details["top_values"] = _top_values(as_str)
        details["top_values_truncated"] = n_unique > MAX_TOP_VALUES
        details["label_variants"] = int(n_unique - stripped.str.lower().nunique())
    return details


def _profile_column(name: str, s: pd.Series) -> dict:
    n = len(s)
    non_null = s.dropna()
    inferred, text_dates = _infer_type(name, s)
    missing = n - len(non_null)

    col = {
        "name": name,
        "dtype": str(s.dtype),
        "inferred_type": inferred,
        "missing_count": int(missing),
        "missing_pct": _pct(missing, n),
        "unique_count": int(non_null.nunique()),
        "sample_values": [str(v)[:50] for v in non_null.head(3)],
    }
    if inferred == "numeric":
        col.update(_numeric_details(s))
    elif inferred == "datetime":
        col.update(_datetime_details(s, text_dates))
    elif inferred in ("categorical", "boolean", "identifier", "text", "numeric_text"):
        col.update(_text_details(non_null, inferred))
    return col


# quality issues

def _issue(severity, code, message, action, column=None) -> dict:
    return {
        "severity": severity,
        "code": code,
        "column": column,
        "message": message,
        "suggested_action": action,
    }


def _build_issues(columns: list[dict], dup_count: int, n_rows: int) -> list[dict]:
    issues: list[dict] = []
    review = "Nothing was changed. Review before deciding how to handle this."

    if dup_count > 0:
        issues.append(_issue(
            "warning", "duplicate_rows",
            f"{dup_count:,} exact duplicate rows ({_pct(dup_count, n_rows)}%).",
            "Remove duplicates only after confirming they are not legitimate repeated "
            "records. Requires your approval.",
        ))

    for c in columns:
        name, t = c["name"], c["inferred_type"]

        if t == "empty":
            issues.append(_issue(
                "critical", "empty_column", "Column contains no values.",
                "Consider excluding this column from analysis.", name))
            continue

        mp = c["missing_pct"]
        if mp > 0:
            sev = "critical" if mp > 50 else "warning" if mp > 20 else "info"
            issues.append(_issue(
                sev, "missing_values",
                f"{c['missing_count']:,} missing values ({mp}%).", review, name))

        if c["unique_count"] == 1 and n_rows > 1 and c["missing_count"] < n_rows - 1:
            issues.append(_issue(
                "info", "constant_column",
                "Column has a single unique value, so it cannot explain variation.",
                "Usually safe to ignore in analysis.", name))

        if c.get("blank_count", 0) > 0:
            issues.append(_issue(
                "warning", "blank_strings",
                f"{c['blank_count']:,} blank (whitespace-only) text values. "
                "These are not counted as missing.", review, name))
        if c.get("placeholder_count", 0) > 0:
            issues.append(_issue(
                "info", "placeholder_values",
                f"{c['placeholder_count']:,} placeholder-like values (e.g. '-', '?'). "
                "These are not counted as missing.", review, name))

        if t == "numeric":
            if c["negative_count"] > 0:
                issues.append(_issue(
                    "info", "negative_values",
                    f"{c['negative_count']:,} negative values. Check whether negatives are valid "
                    "for this column or are data-entry errors.",
                    "Confirm what negative values mean in this column before using it in totals "
                    "or averages. Nothing was changed.", name))
            if c["inf_count"] > 0:
                issues.append(_issue(
                    "warning", "infinite_values",
                    f"{c['inf_count']:,} infinite values (excluded from statistics).",
                    review, name))
            o = c.get("outliers")
            if o and o["pct"] >= 1:
                issues.append(_issue(
                    "info", "outliers",
                    f"{o['count']:,} potential outliers ({o['pct']}%) by the IQR method. "
                    "They may be genuine extreme values.",
                    "Inspect before excluding anything.", name))

        if t == "numeric_text":
            issues.append(_issue(
                "warning", "numeric_stored_as_text",
                "Values look numeric but are stored as text, so numeric analysis "
                "is not applied.", "Convert the column to numbers (requires approval).", name))

        if t == "datetime" and c.get("unparsed_count", 0) > 0:
            issues.append(_issue(
                "warning", "unparsed_dates",
                f"{c['unparsed_count']:,} values could not be read as dates.",
                review, name))

        if t == "categorical" and c.get("label_variants", 0) > 0:
            issues.append(_issue(
                "info", "inconsistent_labels",
                f"{c['label_variants']:,} label(s) differ only by case or spacing.",
                "Standardize labels (requires approval).", name))

    return issues


# public API

def profile_dataset(df: pd.DataFrame) -> dict:
    """Build a JSON-serializable profile of a DataFrame without modifying it."""
    n_rows, n_cols = df.shape
    columns = [_profile_column(str(name), df[name]) for name in df.columns]

    groups: dict[str, list[str]] = {}
    for c in columns:
        groups.setdefault(c["inferred_type"], []).append(c["name"])

    total_cells = n_rows * n_cols
    missing_cells = int(df.isna().sum().sum())
    dup_count = int(df.duplicated().sum())

    issues = _build_issues(columns, dup_count, n_rows)
    severities = [i["severity"] for i in issues]
    if "critical" in severities:
        status = "poor"
    elif "warning" in severities:
        status = "fair"
    else:
        status = "good"

    n_categorical = len(groups.get("categorical", [])) + len(groups.get("boolean", []))

    return {
        "shape": {"rows": int(n_rows), "columns": int(n_cols)},
        "summary": {
            "rows": int(n_rows),
            "columns": int(n_cols),
            "numeric_columns": len(groups.get("numeric", [])),
            "categorical_columns": n_categorical,
            "datetime_columns": len(groups.get("datetime", [])),
            "identifier_columns": len(groups.get("identifier", [])),
            "missing_cells": missing_cells,
            "missing_pct": _pct(missing_cells, total_cells),
            "duplicate_rows": dup_count,
            "quality_status": status,
        },
        "type_groups": groups,
        "columns": columns,
        "duplicates": {"count": dup_count, "pct": _pct(dup_count, n_rows)},
        "missing": {
            "total_cells_missing": missing_cells,
            "pct": _pct(missing_cells, total_cells),
            "columns_with_missing": [
                {"name": c["name"], "count": c["missing_count"], "pct": c["missing_pct"]}
                for c in columns if c["missing_count"] > 0
            ],
        },
        "quality": {
            "status": status,
            "issue_counts": {
                s: severities.count(s) for s in ("critical", "warning", "info")
            },
            "issues": issues,
        },
    }