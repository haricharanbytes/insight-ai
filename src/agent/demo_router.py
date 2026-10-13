"""Rule-based question router used when no LLM is configured.

It only maps a question to ONE deterministic tool call using column names found in the
question. Answers are templated from tool results, so numbers are always real.
"""
import re
import time

import pandas as pd

from src.analysis.common import AnalysisError

from .tools import run_tool

_FREQ = [("daily", "D"), ("weekly", "W"), ("monthly", "M"), ("month", "M"),
         ("quarter", "Q"), ("yearly", "Y"), ("annual", "Y"), ("year", "Y"), ("week", "W"), ("day", "D")]
_LOWEST = re.compile(r"\b(lowest|least|worst|smallest|bottom|minimum|fewest)\b", re.I)
_AVG = re.compile(r"\b(average|avg|mean)\b", re.I)
_TREND = re.compile(r"\b(trend|over time|monthly|quarterly|yearly|annual|weekly|daily|growth|change[sd]?)\b", re.I)
_CORR = re.compile(r"\b(correlat\w*|relationship|related|associat\w*)\b", re.I)
_OUT = re.compile(r"\b(outliers?|anomal\w*|unusual)\b", re.I)
_QUALITY = re.compile(r"\b(missing|null|duplicates?|data quality|quality)\b", re.I)
_RANK = re.compile(r"\b(top|best|highest|most|largest|biggest|lowest|least|worst|total|sum|average|mean|by|per|each)\b", re.I)

UNSUPPORTED = ("Demo mode (no LLM configured) can answer rankings and totals by a category, "
               "trends over time, correlations, outliers, and data-quality questions, using column "
               "names exactly as they appear in your data. This question needs the full AI analyst: "
               "set LLM_API_KEY and LLM_MODEL in .env.")


def _mentioned(question: str, columns: list[str]) -> list[str]:
    """Columns named in the question, in order of appearance (longest names first to avoid overlaps)."""
    q = question.lower()
    found = []
    for c in sorted(columns, key=len, reverse=True):
        m = re.search(r"(?<![\w])" + re.escape(c.lower()) + r"(?![\w])", q)
        if m and not any(m.start() >= s and m.end() <= e for _, s, e in found):
            found.append((c, m.start(), m.end()))
    return [c for c, _, _ in sorted(found, key=lambda x: x[1])]


def _types(profile: dict) -> dict:
    return {c["name"]: c["inferred_type"] for c in profile["columns"]}


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def _plan(question: str, profile: dict):
    types = _types(profile)
    cols = _mentioned(question, list(types))
    nums = [c for c in cols if types[c] == "numeric"]
    cats = [c for c in cols if types[c] in ("categorical", "boolean")]
    dates = [c for c in cols if types[c] == "datetime"]
    all_dates = [n for n, t in types.items() if t == "datetime"]
    all_nums = [n for n, t in types.items() if t == "numeric"]

    if _QUALITY.search(question):
        return "quality", None, {}
    if _OUT.search(question):
        col = nums[0] if nums else (all_nums[0] if all_nums else None)
        return ("detect_outliers", {"column": col}, {}) if col else None
    if _CORR.search(question):
        params = {"columns": nums} if len(nums) >= 2 else {}
        return "calculate_correlation", params, {}
    if _TREND.search(question):
        date = dates[0] if dates else (all_dates[0] if all_dates else None)
        if date is None:
            return None
        freq = next((f for w, f in _FREQ if w in question.lower()), "M")
        params = {"date_column": date, "freq": freq}
        if nums:
            params["value_column"] = nums[0]
            params["op"] = "mean" if _AVG.search(question) else "sum"
        return "analyze_time_series", params, {}
    if cats and (nums or _RANK.search(question)):
        op = "mean" if _AVG.search(question) else "sum"
        metric = {"column": nums[0], "op": op} if nums else {"column": "*", "op": "count"}
        params = {"group_by": [cats[0]], "metrics": [metric], "ascending": bool(_LOWEST.search(question))}
        return "group_by_aggregate", params, {}
    if nums and not cats and _RANK.search(question):
        op = "mean" if _AVG.search(question) else "sum"
        return "group_by_aggregate", {"metrics": [{"column": nums[0], "op": op}]}, {}
    return None


def _answer(tool: str, out: dict) -> tuple[str, list[str]]:
    res = out["result"]
    follow: list[str] = []
    if tool == "group_by_aggregate":
        if "values" in res:
            text = "Answer: " + ", ".join(f"{k} = {_fmt(v)}" for k, v in res["values"].items()) + "."
        else:
            top, bottom = res["top_group"], res["bottom_group"]
            key = out["inputs"]["group_by"][0]
            metric = next(k for k in top if k not in (key, "share_pct"))
            text = (f"Answer: {top[key]} is first by {metric} ({_fmt(top[metric])}) and "
                    f"{bottom[key]} is last ({_fmt(bottom[metric])}), across {res['total_groups']} groups.")
        follow = ["Show the trend over time.", "Compare with another category column."]
    elif tool == "analyze_time_series":
        p, l, t = res["peak"], res["last"], res["trend"]
        text = (f"Answer: {res['operation']} per {res['frequency']} ran from {res['first']['period']} "
                f"({_fmt(res['first']['value'])}) to {l['period']} ({_fmt(l['value'])}); peak was "
                f"{p['period']} ({_fmt(p['value'])}).")
        if t.get("available"):
            text += f" Trend: {t['direction']} (slope {_fmt(t['slope_per_period'])} per {res['frequency']}, indicative only)."
        follow = ["Break the trend down by a category.", "Look at the lowest period."]
    elif tool == "calculate_correlation":
        s = res["strongest"]
        text = (f"Answer: the strongest relationship is {s['column_a']} vs {s['column_b']} "
                f"(r = {s['r']:.2f}, {s['strength']} {s['direction']}). Correlation is not causation.")
        follow = ["Check outliers in one of these columns."]
    else:  # detect_outliers
        text = (f"Answer: {res['n_outliers']:,} of {res['n_values']:,} values in {res['column']} "
                f"({res['pct_outliers']}%) fall outside {_fmt(res['bounds']['lower'])} to "
                f"{_fmt(res['bounds']['upper'])}. These are indicators, not errors.")
        follow = ["See the correlation between numeric columns."]
    return text, follow


def run_demo(df: pd.DataFrame, profile: dict, question: str) -> dict:
    started = time.monotonic()
    steps, caveats, follow = [], [], []
    plan = _plan(question, profile)

    if plan and plan[0] == "quality":
        s = profile["summary"]
        issues = profile["quality"]["issues"][:5]
        answer = (f"Answer: {s['rows']:,} rows, {s['columns']} columns, {s['missing_pct']}% of cells missing, "
                  f"{s['duplicate_rows']:,} duplicate rows. Quality status: {s['quality_status']}.")
        caveats = [f"{i['column'] + ': ' if i['column'] else ''}{i['message']}" for i in issues]
        follow = ["Ask for outliers in a numeric column."]
    elif plan is None:
        answer, follow = UNSUPPORTED, []
    else:
        tool, params, _ = plan
        step = {"tool": tool, "params": params, "scope_guard_dropped": []}
        try:
            out = run_tool(df, tool, params)
            step.update(ok=True, output=out)
            answer, follow = _answer(tool, out)
            caveats = list(out.get("warnings") or [])
            if out["result"].get("note"):
                caveats.append(out["result"]["note"])
        except AnalysisError as exc:
            step.update(ok=False, error={"code": exc.code, "message": exc.message, "details": exc.details})
            answer = f"Answer: I could not run that analysis: {exc.message}"
        steps.append(step)

    if caveats:
        answer += "\n\nCaveats: " + " ".join(caveats[:4])
    if follow:
        answer += "\n\nNext questions: " + " ".join(follow)
    return {
        "answer": answer, "steps": steps, "grounding": {"unverified_numbers": []},
        "incomplete": plan is None, "mode": "demo", "model": None, "models_used": [],
        "llm_calls": 0, "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "latency_s": round(time.monotonic() - started, 2),
    }