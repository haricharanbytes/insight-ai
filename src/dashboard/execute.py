"""Run dashboard widgets through the deterministic tools. A widget failure never breaks the dashboard."""
import logging
import time

import pandas as pd

from src.agent.tools import run_tool
from src.analysis.common import AnalysisError
from src.visualization.charts import chart_for_output

from .spec import Dashboard, widget_filters

logger = logging.getLogger("insightai.dashboard")


def _with_filters(params: dict, filters: list[dict]) -> dict:
    return {**params, "filters": filters} if filters else params


def widget_call(w, filters: list[dict]) -> tuple[str, dict]:
    """The (tool name, params) that produce this widget."""
    t = w.type
    if t == "kpi":
        p = {"metrics": [{"column": w.metric.column, "op": w.metric.op}]}
        return "group_by_aggregate", _with_filters(p, filters)
    if t in ("bar", "donut"):
        p = {"group_by": [w.group_by], "metrics": [{"column": w.metric.column, "op": w.metric.op}],
             "ascending": w.ascending, "limit": w.limit}
        return "group_by_aggregate", _with_filters(p, filters)
    if t == "line":
        p = {"date_column": w.date_column, "op": w.op, "freq": w.freq}
        if w.value_column:
            p["value_column"] = w.value_column
        return "analyze_time_series", _with_filters(p, filters)
    if t == "histogram":
        return "generate_chart", _with_filters({"chart_type": "histogram", "x": w.column, "bins": w.bins}, filters)
    if t == "scatter":
        return "generate_chart", _with_filters({"chart_type": "scatter", "x": w.x, "y": w.y}, filters)
    if t == "box":
        p = {"chart_type": "box", "x": w.column}
        if w.group_by:
            p["group_by"] = w.group_by
        return "generate_chart", _with_filters(p, filters)
    if t == "heatmap":
        p = {"method": w.method}
        if w.columns:
            p["columns"] = list(w.columns)
        return "calculate_correlation", _with_filters(p, filters)
    # table
    p = {"limit": w.limit}
    if w.columns:
        p["columns"] = list(w.columns)
    
    if not filters:
        # filter_data requires at least one filter; "not_null" on any column keeps all rows with a value,
        # so use a cheap always-true approach handled in execute_widget instead.
        return "filter_data", p
    return "filter_data", _with_filters(p, filters)


def _default_title(w) -> str:
    t = w.type
    if t == "kpi":
        return f"{w.metric.op} of {w.metric.column}" if w.metric.column != "*" else "Row count"
    if t in ("bar", "donut"):
        return f"{w.metric.op} of {w.metric.column if w.metric.column != '*' else 'rows'} by {w.group_by}"
    if t == "line":
        return f"{w.value_column or 'Rows'} per {w.freq}"
    if t == "histogram":
        return f"Distribution of {w.column}"
    if t == "scatter":
        return f"{w.y} vs {w.x}"
    if t == "box":
        return f"{w.column}" + (f" by {w.group_by}" if w.group_by else "")
    if t == "heatmap":
        return "Correlation matrix"
    return "Data table"


def execute_widget(df: pd.DataFrame, dash: Dashboard, w) -> dict:
    """Compute one widget. Always returns a dict with ok=True/False and never raises."""
    base = {"id": w.id, "type": w.type, "title": w.title or _default_title(w), "size": w.size}
    started = time.monotonic()
    try:
        filters = widget_filters(dash, w)
        tool, params = widget_call(w, filters)

        if w.type == "table" and not filters:
            first = (w.columns or [str(df.columns[0])])[0]
            params["filters"] = [{"column": first, "op": "not_null"}] if False else []
            out = _table_all_rows(df, params)
        else:
            out = run_tool(df, tool, params)

        result = {**base, "ok": True, "tool": tool, "rows_used": out["rows_used"], "warnings": out["warnings"],
                  "method": out["method"]}
        if w.type == "kpi":
            values = out["result"]["values"]
            result["value"] = next(iter(values.values()))
            result["label"] = next(iter(values.keys()))
        elif w.type == "table":
            result["table"] = out["table"]
            result["rows_before"] = out["result"].get("rows_before")
            result["rows_after"] = out["result"].get("rows_after")
        else:
            chart = chart_for_output(out)
            if chart is None:
                result.update(ok=False, error={"code": "no_chart", "message": "There is nothing to plot for this widget."})
            else:
                result["chart"] = chart
        result["latency_s"] = round(time.monotonic() - started, 3)
        return result
    except AnalysisError as exc:
        msg = exc.message
        if exc.code == "no_rows":
            msg = "No rows match the current filters."
        return {**base, "ok": False, "error": {"code": exc.code, "message": msg}}
    except Exception:
        logger.exception("Widget %s (%s) crashed", w.id, w.type)
        return {**base, "ok": False, "error": {"code": "widget_failed", "message": "This widget could not be computed."}}


def _table_all_rows(df: pd.DataFrame, params: dict) -> dict:
    """A table with no filters: preview of the first rows without going through filter_data."""
    from src.analysis.common import check_limit, df_to_table, make_result, require_columns

    warnings: list[str] = []
    limit = check_limit(params.get("limit", 10), warnings, maximum=50)
    cols = params.get("columns") or [str(c) for c in df.columns[:12]]
    require_columns(df, cols)
    return make_result(
        "filter_data", method="First rows of the dataset (no filters).",
        inputs={"columns": cols, "limit": limit, "filters": []}, rows_used=len(df),
        result={"rows_before": len(df), "rows_after": len(df), "pct_kept": 100.0},
        table=df_to_table(df[cols], limit), warnings=warnings)


def execute_dashboard(df: pd.DataFrame, dash: Dashboard, only_ids: set[str] | None = None) -> list[dict]:
    """Compute widgets in order. With only_ids, just those; used when a filter changes."""
    return [execute_widget(df, dash, w) for w in dash.widgets if only_ids is None or w.id in only_ids]


def affected_widget_ids(dash: Dashboard) -> set[str]:
    """Widgets that react to global filters (the ones to recompute when a filter changes)."""
    return {w.id for w in dash.widgets if w.use_global_filters}