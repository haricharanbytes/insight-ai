"""Group-by aggregation with strict validation."""
import numpy as np
import pandas as pd

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

AGG_OPS = ("sum", "mean", "median", "count", "min", "max", "nunique")
NUMERIC_ONLY = {"sum", "mean", "median", "min", "max"}
MAX_GROUP_COLUMNS = 3
MAX_METRICS = 6


def _normalize_metrics(df: pd.DataFrame, metrics) -> list[dict]:
    if not metrics:
        return [{"column": "*", "op": "count", "name": "row_count"}]
    if not isinstance(metrics, (list, tuple)):
        raise AnalysisError("'metrics' must be a list of {column, op} objects.", "invalid_parameter")
    if len(metrics) > MAX_METRICS:
        raise AnalysisError(f"At most {MAX_METRICS} metrics are allowed.", "invalid_parameter")

    out, seen = [], set()
    for m in metrics:
        if not isinstance(m, dict):
            raise AnalysisError("Each metric must be an object with 'column' and 'op'.", "invalid_parameter")
        column = m.get("column")
        op = str(m.get("op", "")).lower()
        if op not in AGG_OPS:
            raise AnalysisError(
                f"Unsupported aggregation '{op}'. Supported: {', '.join(AGG_OPS)}.",
                "unsupported_operation", {"supported": list(AGG_OPS)},
            )
        if column == "*":
            if op != "count":
                raise AnalysisError("'*' can only be used with 'count' (row count).", "invalid_parameter")
            name = "row_count"
        else:
            require_columns(df, [column])
            if op in NUMERIC_ONLY:
                require_numeric(df, column)
            name = f"{op}_{column}"
        if name not in seen:
            seen.add(name)
            out.append({"column": column, "op": op, "name": name})
    return out


def _row_dict(row: pd.Series) -> dict:
    return {str(k): to_py(v) for k, v in row.items()}


def group_by_aggregate(df: pd.DataFrame, group_by=None, metrics=None, filters=None,
                       sort_by: str | None = None, ascending: bool = False,
                       limit: int = 20) -> dict:
    """Aggregate metrics, optionally per group.

    metrics: [{"column": "Sales", "op": "sum"}, {"column": "*", "op": "count"}]
    With no group_by, returns one overall row.
    """
    warnings: list[str] = []
    limit = check_limit(limit, warnings)

    if isinstance(group_by, str):
        group_by = [group_by]
    keys = list(group_by or [])
    if len(keys) > MAX_GROUP_COLUMNS:
        raise AnalysisError(f"At most {MAX_GROUP_COLUMNS} grouping columns are allowed.", "invalid_parameter")
    if len(set(keys)) != len(keys):
        raise AnalysisError("Grouping columns must be unique.", "invalid_parameter")
    require_columns(df, keys)

    specs = _normalize_metrics(df, metrics)
    for s in specs:
        if s["column"] in keys:
            raise AnalysisError(
                f"Cannot aggregate '{s['column']}' because it is also a grouping column.", "invalid_parameter"
            )
        if s["name"] in keys:
            raise AnalysisError(f"Metric name '{s['name']}' collides with a grouping column.", "invalid_parameter")

    work, applied, fw = apply_filters(df, filters)
    warnings += fw
    if len(work) == 0:
        raise AnalysisError(
            "No rows remain after filtering.", "no_rows", {"filters": applied, "warnings": fw}
        )

    inputs = {
        "group_by": keys,
        "metrics": [{"column": s["column"], "op": s["op"]} for s in specs],
        "filters": applied,
        "limit": limit,
    }

    # ---- overall (no grouping) ----
    if not keys:
        row = {}
        for s in specs:
            if s["column"] == "*":
                row[s["name"]] = len(work)
            elif s["op"] == "sum":
                row[s["name"]] = work[s["column"]].sum(min_count=1)  # all-missing -> NaN, not 0
            else:
                row[s["name"]] = getattr(work[s["column"]], s["op"])()
        values = {k: to_py(v) for k, v in row.items()}
        return make_result(
            "group_by_aggregate",
            method="Aggregation over all rows; missing values are ignored by each aggregation.",
            inputs=inputs, rows_used=len(work),
            result={"values": values},
            table=df_to_table(pd.DataFrame([row])),
            warnings=warnings,
        )

    # ---- grouped ----
    needed = list(dict.fromkeys(keys + [s["column"] for s in specs if s["column"] != "*"]))
    sub = work[needed].assign(__row__=1)

    named = {}
    for s in specs:
        col = "__row__" if s["column"] == "*" else s["column"]
        named[s["name"]] = pd.NamedAgg(column=col, aggfunc=s["op"])
        if s["op"] == "sum":  # hidden non-null count, used to keep all-missing sums as missing
            named[f"__n__{s['name']}"] = pd.NamedAgg(column=col, aggfunc="count")

    res = sub.groupby(keys, dropna=False, observed=True).agg(**named)
    for s in specs:
        if s["op"] == "sum":
            empty = res[f"__n__{s['name']}"] == 0
            if empty.any():
                res[s["name"]] = res[s["name"]].astype(float).where(~empty)
    res = res.drop(columns=[c for c in res.columns if c.startswith("__n__")]).reset_index()

    for k in keys:
        if res[k].isna().any():
            warnings.append(f"Rows with a missing '{k}' are grouped under '(missing)'.")
            res[k] = res[k].astype(object).where(res[k].notna(), "(missing)")

    # Share of total: only meaningful for non-negative sums/counts
    first = specs[0]
    share_basis = None
    if first["op"] in ("sum", "count"):
        vals = res[first["name"]]
        total = vals.sum()
        if total > 0 and (vals.dropna() >= 0).all():
            res["share_pct"] = (100 * vals / total).round(2)
            share_basis = first["name"]

    sort_col = sort_by or first["name"]
    if sort_col not in res.columns:
        raise AnalysisError(
            f"Cannot sort by '{sort_col}'. Available: {list(res.columns)}.",
            "column_not_found", {"available_columns": list(res.columns)},
        )
    res = res.sort_values(sort_col, ascending=bool(ascending), kind="mergesort",
                          na_position="last").reset_index(drop=True)
    inputs["sort_by"], inputs["ascending"] = sort_col, bool(ascending)

    return make_result(
        "group_by_aggregate",
        method=("pandas groupby aggregation; missing values are ignored by each aggregation; "
                "share_pct is the group's share of the total over ALL groups."),
        inputs=inputs, rows_used=len(work),
        result={
            "total_groups": int(len(res)),
            "returned_groups": int(min(len(res), limit)),
            "sorted_by": sort_col,
            "share_basis": share_basis,
            "top_group": _row_dict(res.iloc[0]),
            "bottom_group": _row_dict(res.iloc[-1]),
        },
        table=df_to_table(res, limit),
        warnings=warnings,
    )