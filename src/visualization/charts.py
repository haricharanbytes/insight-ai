"""Build Plotly figure specs (plain dicts) from tool outputs. Never plots data the tool did not return."""
import math

NAVY, BLUE, GREY = "#1e3a5f", "#2f6fdb", "#8a94a6"
PALETTE = ["#2f6fdb", "#1e3a5f", "#5aa0f2", "#8a94a6", "#0f8b8d", "#c47f17"]
MAX_BARS = 25


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def _layout(title: str, x: str = "", y: str = "", **extra) -> dict:
    layout = {
        "title": {"text": title, "x": 0.02, "font": {"size": 16, "color": NAVY}},
        "xaxis": {"title": {"text": x}, "automargin": True},
        "yaxis": {"title": {"text": y}, "automargin": True},
        "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, system-ui, sans-serif", "color": "#2b3445"},
        "showlegend": False,
    }
    layout.update(extra)
    return layout


def _chart(kind: str, title: str, caption: str, data: list, layout: dict) -> dict:
    return {"kind": kind, "title": title, "caption": caption, "figure": {"data": data, "layout": layout}}


def _cols(table: dict) -> dict:
    return {c: i for i, c in enumerate(table["columns"])}


def _truncation_note(table: dict, shown: int) -> str:
    total = table.get("total_rows", shown)
    return f" Showing {shown} of {total:,} groups." if total > shown else ""


# ------------------------------------------------------------ per tool

def _bar_or_donut(out: dict) -> dict | None:
    table, inp = out["table"], out["inputs"]
    if not table or not table["rows"] or not inp.get("group_by"):
        return None
    idx = _cols(table)
    keys = inp["group_by"]
    metric = out["result"].get("sorted_by") or next(c for c in table["columns"] if c not in keys)
    if metric not in idx:
        return None
    rows = table["rows"][:MAX_BARS]
    labels = [" / ".join(str(r[idx[k]]) for k in keys) for r in rows]
    values = [r[idx[metric]] for r in rows]
    if all(v is None for v in values):
        return None

    op = next((m["op"] for m in inp["metrics"] if f"{m['op']}_{m['column']}" == metric), "count")
    title = f"{metric.replace('_', ' ', 1)} by {', '.join(keys)}"
    clean = [v for v in values if v is not None]
    note = _truncation_note(table, len(rows))

    if (len(rows) == len(table["rows"]) == table.get("total_rows", 0) and 2 <= len(rows) <= 6
            and op in ("sum", "count") and all(v >= 0 for v in clean) and len(clean) == len(values)):
        data = [{"type": "pie", "labels": labels, "values": values, "hole": 0.5, "sort": False,
                 "marker": {"colors": PALETTE[: len(labels)]},
                 "hovertemplate": "%{label}<br>%{value:,.2f} (%{percent})<extra></extra>"}]
        return _chart("donut", title, f"Share of each group in the total {metric}.", data,
                      _layout(title, showlegend=True))

    ordered = list(zip(labels, values))[::-1]  # horizontal bars read top-down
    data = [{"type": "bar", "orientation": "h", "x": [v for _, v in ordered], "y": [l for l, _ in ordered],
             "marker": {"color": BLUE}, "hovertemplate": "%{y}<br>" + metric + ": %{x:,.2f}<extra></extra>"}]
    height = max(260, 28 * len(rows) + 110)
    return _chart("bar", title, f"{metric} per group, sorted by value.{note}", data,
                  _layout(title, metric, ", ".join(keys), height=height, margin={"l": 140, "r": 20, "t": 50, "b": 50}))


def _time_series(out: dict) -> dict | None:
    table, res, inp = out["table"], out["result"], out["inputs"]
    if not table or not table["rows"]:
        return None
    idx = _cols(table)
    periods = [r[idx["period"]] for r in table["rows"]]
    values = [r[idx["value"]] for r in table["rows"]]  # None stays a gap, never zero
    unit, op = res["frequency"], res["operation"]
    what = inp.get("value_column") or "rows"
    ylabel = f"{op} of {what}" if inp.get("value_column") else "Number of rows"
    title = f"{ylabel[0].upper() + ylabel[1:]} per {unit}"
    hover = "%{x}<br>%{y:,.2f}<extra></extra>"

    if op == "count":
        data = [{"type": "bar", "x": periods, "y": values, "marker": {"color": BLUE}, "hovertemplate": hover}]
        kind = "bar"
    else:
        data = [{"type": "scatter", "mode": "lines+markers", "x": periods, "y": values, "connectgaps": False,
                 "line": {"color": BLUE, "width": 2}, "marker": {"size": 6, "color": NAVY},
                 "hovertemplate": hover}]
        kind = "line"
    caption = f"{ylabel} for each {unit}, {periods[0]} to {periods[-1]}."
    if any(v is None for v in values):
        caption += " Gaps are periods with no value (not zero)."
    if any("partial" in w for w in out.get("warnings", [])):
        caption += " The first or last period is partial and not comparable."
    return _chart(kind, title, caption, data, _layout(title, unit.capitalize(), ylabel))


def _correlation(out: dict) -> dict | None:
    m = out["result"].get("matrix")
    if not m:
        return None
    cols, z = m["columns"], m["values"]
    text = [[("" if v is None else f"{v:.2f}") for v in row] for row in z]
    method = out["result"]["method"].capitalize()
    title = f"{method} correlation matrix"
    data = [{"type": "heatmap", "x": cols, "y": cols, "z": z, "zmin": -1, "zmax": 1, "text": text,
             "texttemplate": "%{text}", "colorscale": "RdBu", "reversescale": True, "hoverongaps": False,
             "hovertemplate": "%{y} vs %{x}: %{z:.2f}<extra></extra>"}]
    n = len(cols)
    return _chart("heatmap", title,
                  "Each cell is the correlation between two columns (-1 to +1). Blank cells could not be computed. "
                  "Association only, not causation.",
                  data, _layout(title, height=max(320, 60 * n + 120), yaxis={"autorange": "reversed", "automargin": True}))


def _outliers(out: dict) -> dict | None:
    res, table = out["result"], out["table"]
    col, b = res["column"], res["bounds"]
    shown = [r[1] for r in table["rows"]] if table else []
    data = []
    inliers_note = ""
    if shown:
        data.append({"type": "box", "y": shown, "name": "flagged values", "boxpoints": "all", "jitter": 0.4,
                     "marker": {"color": "#c47f17"}, "hovertemplate": col + ": %{y:,.2f}<extra></extra>"})
        if res["n_outliers"] > len(shown):
            inliers_note = f" Showing the {len(shown)} most extreme of {res['n_outliers']:,}."
    shapes = [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": b[k], "y1": b[k],
               "line": {"color": GREY, "dash": "dash", "width": 1}} for k in ("lower", "upper")]
    title = f"Outlier bounds for {col}"
    caption = (f"{res['n_outliers']:,} of {res['n_values']:,} values ({res['pct_outliers']}%) lie outside the dashed "
               f"bounds ({_fmt(b['lower'])} to {_fmt(b['upper'])}).{inliers_note} Indicators only; nothing was removed.")
    if not data:
        data = [{"type": "scatter", "x": [0], "y": [res["min_value"]], "mode": "markers", "marker": {"opacity": 0}}]
        caption = f"No values lie outside the bounds ({_fmt(b['lower'])} to {_fmt(b['upper'])})."
    return _chart("box", title, caption, data,
                  _layout(title, "", col, shapes=shapes, yaxis={"title": {"text": col}, "range": None, "automargin": True}))


def _stat_test(out: dict) -> dict | None:
    res, table, inp = out["result"], out["table"], out["inputs"]
    if not table or not table["rows"]:
        return None
    idx = _cols(table)
    if inp["test"] == "chi_square":
        label_col = table["columns"][0]
        rows = table["rows"][:MAX_BARS]
        data = [{"type": "bar", "name": str(c), "x": [str(r[0]) for r in rows], "y": [r[i] for r, i in
                 ((r, idx[c]) for r in rows)]} for c in table["columns"][1:]]
        title = f"{res['column_a']} by {res['column_b']} (counts)"
        for i, d in enumerate(data):
            d["marker"] = {"color": PALETTE[i % len(PALETTE)]}
        return _chart("bar", title, "Row counts for each combination. Used for the chi-square test of association.",
                      data, _layout(title, res["column_a"], "Rows", barmode="group", showlegend=True))
    value, group = res["value_column"], res["group_column"]
    rows = table["rows"][:MAX_BARS]
    data = [{"type": "bar", "x": [str(r[idx["group"]]) for r in rows], "y": [r[idx["mean"]] for r in rows],
             "marker": {"color": BLUE}, "customdata": [[r[idx["n"]], r[idx["median"]]] for r in rows],
             "hovertemplate": "%{x}<br>mean: %{y:,.2f}<br>median: %{customdata[1]:,.2f}<br>n: %{customdata[0]:,}<extra></extra>"}]
    title = f"Mean {value} by {group}"
    return _chart("bar", title,
                  f"Mean of {value} for each {group}. The {res['test']} compares these groups (p = {res['p_value']:.3g}).",
                  data, _layout(title, group, f"Mean {value}"))


def _column_summary(out: dict) -> dict | None:
    table = out.get("table")
    if not table or not table["rows"]:
        return None
    col = out["result"]["column"]
    rows = table["rows"]
    idx = _cols(table)
    data = [{"type": "bar", "x": [str(r[idx["value"]]) for r in rows], "y": [r[idx["count"]] for r in rows],
             "marker": {"color": BLUE}, "hovertemplate": "%{x}: %{y:,} rows<extra></extra>"}]
    title = f"Most common values in {col}"
    return _chart("bar", title, f"The {len(rows)} most frequent values of {col}.", data, _layout(title, col, "Rows"))

def _generated(out: dict) -> dict | None:
    return out["result"].get("chart")

BUILDERS = {
    "group_by_aggregate": _bar_or_donut,
    "analyze_time_series": _time_series,
    "calculate_correlation": _correlation,
    "detect_outliers": _outliers,
    "run_statistical_test": _stat_test,
    "get_column_summary": _column_summary,
    "generate_chart": _generated,
}


def chart_for_output(output: dict) -> dict | None:
    """Return a chart spec for a tool output, or None when no chart suits it (a table is shown instead)."""
    builder = BUILDERS.get(output.get("tool"))
    if builder is None:
        return None
    try:
        return builder(output)
    except (KeyError, IndexError, TypeError, ValueError, StopIteration):
        return None  # a chart problem must never break an analysis response