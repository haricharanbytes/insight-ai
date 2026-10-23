"""Automated EDA report model: a format-independent structure built from the deterministic tools.

Every number in a finding comes from a tool result. No recommendations are produced.
A failing section becomes a note in that section, never a failed report.
"""
import logging
from datetime import datetime, timezone

import pandas as pd

from src.agent.tools import run_tool
from src.analysis.common import AnalysisError
from src.analysis.filtering import apply_filters
from src.data.profiler import profile_dataset
from src.visualization.charts import chart_for_output

logger = logging.getLogger("insightai.eda")

SECTIONS = ("overview", "quality", "numeric", "categories", "trends", "correlations", "groups")
MAX_NUMERIC = 6
MAX_CATEGORIES = 4
PREFERRED = ("sales", "revenue", "amount", "total", "price", "profit", "cost", "income", "value", "quantity")
TITLE_MAX = 120
AUTHOR_MAX = 80

CAVEATS = {
    "numeric": "Outlier counts are statistical indicators, not errors. Nothing was removed or changed.",
    "categories": "Shares are of the rows used; categories with few rows give unstable figures.",
    "trends": "A trend describes the past in this data. It is not a forecast.",
    "correlations": "Correlation measures association, not causation, and can be driven by outliers or shared causes.",
    "groups": "A significant difference does not show why the groups differ. Check the effect size, not only the p-value.",
}


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def _rank(name: str) -> int:
    low = name.lower()
    for i, w in enumerate(PREFERRED):
        if w in low:
            return i
    return len(PREFERRED)


def _section(key: str, title: str) -> dict:
    return {"key": key, "title": title, "ok": True, "findings": [], "tables": [], "charts": [],
            "warnings": [], "caveat": CAVEATS.get(key, ""), "sources": []}


def _fail(sec: dict, message: str) -> dict:
    sec.update(ok=False, findings=[], tables=[], charts=[])
    sec["warnings"].append(message)
    return sec


def _use(sec: dict, out: dict, chart: bool = True) -> None:
    """Attach a tool output's provenance, warnings, and chart to a section."""
    sec["sources"].append({"tool": out["tool"], "method": out["method"], "rows_used": out["rows_used"],
                           "inputs": out["inputs"]})
    sec["warnings"] += [w for w in out.get("warnings", []) if w not in sec["warnings"]]
    if chart:
        c = chart_for_output(out)
        if c:
            sec["charts"].append(c)


def _table(title: str, columns: list, rows: list) -> dict:
    return {"title": title, "columns": [str(c) for c in columns], "rows": rows}


def _freq(span_days) -> str:
    d = span_days or 0
    return "D" if d < 120 else "M" if d <= 730 else "Q" if d <= 2190 else "Y"


# ------------------------------------------------------------------ sections

def _overview(profile: dict, filename: str, n_rows_total: int) -> dict:
    s, sec = profile["summary"], _section("overview", "Overview")
    sec["findings"] = [
        f"The dataset '{filename}' has {s['rows']:,} rows and {s['columns']} columns"
        + (f" (after filters; {n_rows_total:,} before)." if s["rows"] != n_rows_total else "."),
        f"Column types: {s['numeric_columns']} numeric, {s['categorical_columns']} categorical, "
        f"{s['datetime_columns']} date, {s['identifier_columns']} identifier.",
        f"{s['missing_pct']}% of cells are missing ({s['missing_cells']:,} cells) and {s['duplicate_rows']:,} rows "
        "are exact duplicates.",
        f"Overall data-quality status: {s['quality_status']}.",
    ]
    cols = [[c["name"], c["inferred_type"], c["missing_pct"], c["unique_count"]] for c in profile["columns"]]
    sec["tables"].append(_table("Columns", ["Column", "Type", "Missing %", "Unique values"], cols))
    return sec


def _quality(profile: dict) -> dict:
    sec = _section("quality", "Data quality")
    q = profile["quality"]
    counts = q["issue_counts"]
    sec["findings"] = [f"{counts['critical']} critical, {counts['warning']} warning and {counts['info']} informational "
                       "findings were recorded. Nothing in the data was changed."]
    sec["findings"] += [f"{(i['column'] + ': ') if i['column'] else ''}{i['message']}" for i in q["issues"][:25]]
    if q["issues"]:
        sec["tables"].append(_table("Findings", ["Severity", "Column", "Finding"],
                                    [[i["severity"], i["column"] or "", i["message"]] for i in q["issues"][:50]]))
    else:
        sec["findings"].append("No data-quality issues were detected.")
    return sec


def _numeric(df: pd.DataFrame, profile: dict, options: dict) -> dict:
    sec = _section("numeric", "Numeric columns")
    cols = [c for c in profile["columns"] if c["inferred_type"] == "numeric"]
    if not cols:
        return _fail(sec, "The dataset has no numeric columns.")
    rows = []
    for c in cols:
        o = c.get("outliers")
        rows.append([c["name"], c["mean"], c["median"], c["std"], c["min"], c["max"], c["missing_pct"],
                     o["pct"] if o else None])
        line = f"{c['name']}: mean {_fmt(c['mean'])}, median {_fmt(c['median'])}, range {_fmt(c['min'])} to {_fmt(c['max'])}"
        line += f", {o['pct']}% of values lie outside the IQR bounds." if o else "."
        sec["findings"].append(line)
        if c["negative_count"]:
            sec["findings"].append(f"{c['name']} has {c['negative_count']:,} negative values and {c['zero_count']:,} zeros.")
    sec["tables"].append(_table("Summary", ["Column", "Mean", "Median", "Std", "Min", "Max", "Missing %", "Outliers % (IQR)"], rows))
    for c in sorted(cols, key=lambda c: (_rank(c["name"]), cols.index(c)))[:MAX_NUMERIC]:
        try:
            _use(sec, run_tool(df, "generate_chart", {"chart_type": "histogram", "x": c["name"]}))
        except AnalysisError as exc:
            sec["warnings"].append(f"Histogram of {c['name']} skipped: {exc.message}")
    return sec


def _categories(df: pd.DataFrame, profile: dict, main: str | None) -> dict:
    sec = _section("categories", "Categories")
    cats = [c for c in profile["columns"] if c["inferred_type"] in ("categorical", "boolean") and 2 <= c["unique_count"] <= 30]
    cats.sort(key=lambda c: (c["unique_count"], profile["columns"].index(c)))
    if not cats:
        return _fail(sec, "No categorical column with 2 to 30 values was found.")
    for c in cats[:MAX_CATEGORIES]:
        metrics = [{"column": "*", "op": "count"}] + ([{"column": main, "op": "sum"}] if main else [])
        try:
            out = run_tool(df, "group_by_aggregate", {"group_by": [c["name"]], "metrics": metrics, "limit": 15,
                                                      "sort_by": "row_count"})
        except AnalysisError as exc:
            sec["warnings"].append(f"{c['name']} skipped: {exc.message}")
            continue
        t, top = out["table"], out["result"]["top_group"]
        sec["tables"].append(_table(f"Rows by {c['name']}", t["columns"], t["rows"]))
        share = f", {top['share_pct']}% of rows" if "share_pct" in top else ""
        sec["findings"].append(f"{c['name']}: {out['result']['total_groups']} values; the most common is "
                               f"'{top[c['name']]}' with {top['row_count']:,} rows{share}.")
        _use(sec, out)
    if not sec["tables"]:
        return _fail(sec, "No category breakdown could be computed.")
    return sec


def _trends(df: pd.DataFrame, profile: dict, main: str | None) -> dict:
    sec = _section("trends", "Trends over time")
    dates = [c for c in profile["columns"] if c["inferred_type"] == "datetime"]
    if not dates:
        return _fail(sec, "The dataset has no date column.")
    d = dates[0]
    params = {"date_column": d["name"], "freq": _freq(d.get("span_days"))}
    if main:
        params.update(value_column=main, op="sum")
    try:
        out = run_tool(df, "analyze_time_series", params)
    except AnalysisError as exc:
        return _fail(sec, exc.message)
    r = out["result"]
    what = f"total {main}" if main else "row count"
    sec["findings"] = [
        f"{what.capitalize()} per {r['frequency']} from {r['first']['period']} to {r['last']['period']} "
        f"({r['n_periods']} periods).",
        f"The highest value was {_fmt(r['peak']['value'])} in {r['peak']['period']} and the lowest was "
        f"{_fmt(r['lowest']['value'])} in {r['lowest']['period']}.",
    ]
    t = r["trend"]
    if t.get("available"):
        sec["findings"].append(f"A linear trend test gives: {t['direction']} (slope {_fmt(t['slope_per_period'])} per "
                               f"{r['frequency']}, p = {t['p_value']:.3g} if available), indicative only."
                               if t.get("p_value") is not None else f"A linear trend test gives: {t['direction']}.")
    sec["tables"].append(_table("Per period", out["table"]["columns"], out["table"]["rows"]))
    _use(sec, out)
    return sec


def _correlations(df: pd.DataFrame) -> dict:
    sec = _section("correlations", "Correlations")
    try:
        out = run_tool(df, "calculate_correlation", {"limit": 10})
    except AnalysisError as exc:
        return _fail(sec, exc.message)
    for p in out["table"]["rows"][:5]:
        a, b, r_, _, n, strength, direction = p
        sec["findings"].append(f"{a} and {b}: r = {r_:.2f} ({strength}, {direction}), over {n:,} rows.")
    sec["tables"].append(_table("Strongest pairs", out["table"]["columns"], out["table"]["rows"]))
    _use(sec, out)
    return sec


def _groups(df: pd.DataFrame, profile: dict, main: str | None) -> dict:
    sec = _section("groups", "Group differences")
    cats = [c for c in profile["columns"] if c["inferred_type"] in ("categorical", "boolean") and 2 <= c["unique_count"] <= 12]
    cats.sort(key=lambda c: (c["unique_count"], profile["columns"].index(c)))
    if not main or not cats:
        return _fail(sec, "A numeric column and a categorical column with 2 to 12 values are needed.")
    g = cats[0]["name"]
    test = "mannwhitney" if cats[0]["unique_count"] == 2 else "kruskal"
    try:
        out = run_tool(df, "run_statistical_test", {"test": test, "value_column": main, "group_column": g})
    except AnalysisError as exc:
        return _fail(sec, exc.message)
    r = out["result"]
    eff = r["effect_size"]
    verdict = "differ" if r["significant"] else "do not clearly differ"
    sec["findings"] = [
        f"{main} across {g} ({r['n_groups']} groups): the groups {verdict} (p = {r['p_value']:.3g}, {r['test']}).",
        f"Effect size: {eff['name']} = {_fmt(eff['value'])} ({eff['magnitude']}).",
    ]
    sec["tables"].append(_table(f"{main} by {g}", out["table"]["columns"], out["table"]["rows"]))
    _use(sec, out)
    return sec


# ------------------------------------------------------------------ entry point

def build_eda_report(df: pd.DataFrame, filename: str, options: dict | None = None) -> dict:
    """Build the report model. options: title, author, sections (list of keys), filters."""
    options = options or {}
    chosen = options.get("sections") or list(SECTIONS)
    bad = [s for s in chosen if s not in SECTIONS]
    if bad:
        raise AnalysisError(f"Unknown section(s) {bad}. Available: {', '.join(SECTIONS)}.", "invalid_parameter",
                            {"available": list(SECTIONS)})
    title = str(options.get("title") or "Automated EDA report").strip()[:TITLE_MAX]
    author = str(options.get("author") or "").strip()[:AUTHOR_MAX]

    work, applied, fw = apply_filters(df, options.get("filters"))
    if len(work) == 0:
        raise AnalysisError("No rows remain after filtering.", "no_rows", {"filters": applied})
    profile = profile_dataset(work)
    numeric = [c["name"] for c in profile["columns"] if c["inferred_type"] == "numeric"]
    main = min(numeric, key=lambda n: (_rank(n), numeric.index(n))) if numeric else None

    builders = {
        "overview": lambda: _overview(profile, filename, len(df)),
        "quality": lambda: _quality(profile),
        "numeric": lambda: _numeric(work, profile, options),
        "categories": lambda: _categories(work, profile, main),
        "trends": lambda: _trends(work, profile, main),
        "correlations": lambda: _correlations(work),
        "groups": lambda: _groups(work, profile, main),
    }
    sections = []
    for key in SECTIONS:
        if key not in chosen:
            continue
        try:
            sections.append(builders[key]())
        except Exception:
            logger.exception("EDA section %s failed", key)
            sec = _section(key, key.capitalize())
            sections.append(_fail(sec, "This section could not be computed."))

    s = profile["summary"]
    summary = [
        f"The data has {s['rows']:,} rows and {s['columns']} columns; its quality status is {s['quality_status']}.",
        f"{s['missing_pct']}% of cells are missing and {s['duplicate_rows']:,} rows are exact duplicates.",
    ]
    for sec in sections:
        if sec["ok"] and sec["key"] in ("trends", "correlations", "groups") and sec["findings"]:
            summary.append(sec["findings"][0])
    summary.append("These statements describe the data as given; they do not explain causes.")

    return {
        "meta": {"title": title, "author": author, "dataset": filename, "rows": s["rows"], "columns": s["columns"],
                 "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "filters": applied, "main_numeric_column": main},
        "summary": summary,
        "summary_source": "rules",
        "sections": sections,
        "warnings": fw,
        "notes": ["Every number comes from a Python calculation (pandas, SciPy). Nothing in the source data was "
                  "modified or removed.", "This report contains no recommendations."],
    }