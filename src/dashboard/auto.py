"""Rule-based dashboard builder: a valid spec from the dataset profile (and optional prompt hints)."""
import re

from .spec import Dashboard, parse_dashboard

PREFERRED = ("sales", "revenue", "amount", "total", "price", "profit", "cost", "income", "value", "quantity")
MAX_CATEGORY_UNIQUE = 30


def _mentioned(prompt: str, names: list[str]) -> list[str]:
    """Column names that appear in the prompt, in order of appearance (longest names first to avoid overlaps)."""
    q = (prompt or "").lower()
    spans = []
    for n in sorted(names, key=len, reverse=True):
        m = re.search(r"(?<![\w])" + re.escape(n.lower()) + r"(?![\w])", q)
        if m and not any(m.start() >= s and m.end() <= e for _, s, e in spans):
            spans.append((n, m.start(), m.end()))
    return [n for n, _, _ in sorted(spans, key=lambda x: x[1])]


def _rank(name: str) -> int:
    low = name.lower()
    for i, word in enumerate(PREFERRED):
        if word in low:
            return i
    return len(PREFERRED)


def _freq(date_col: dict) -> str:
    days = date_col.get("span_days") or 0
    if days < 120:
        return "D"
    if days <= 730:
        return "M"
    if days <= 2190:
        return "Q"
    return "Y"


def build_auto_dashboard(profile: dict, prompt: str = "", title: str | None = None) -> Dashboard:
    cols = profile["columns"]
    by_name = {c["name"]: c for c in cols}
    numeric = [c["name"] for c in cols if c["inferred_type"] == "numeric"]
    cats = [c["name"] for c in cols if c["inferred_type"] in ("categorical", "boolean")
            and 2 <= c["unique_count"] <= MAX_CATEGORY_UNIQUE]
    dates = [c for c in cols if c["inferred_type"] == "datetime"]

    named = _mentioned(prompt, [c["name"] for c in cols])
    named_num = [n for n in named if n in numeric]
    named_cat = [n for n in named if n in cats]

    ordered_num = sorted(numeric, key=lambda n: (_rank(n), numeric.index(n)))
    main = named_num[0] if named_num else (ordered_num[0] if ordered_num else None)
    others = [n for n in ordered_num if n != main]
    second = named_num[1] if len(named_num) > 1 else (others[0] if others else None)

    chosen_cats = list(named_cat)
    for n in sorted(cats, key=lambda n: (by_name[n]["unique_count"], cats.index(n))):
        if n not in chosen_cats:
            chosen_cats.append(n)
    chosen_cats = chosen_cats[:2]

    widgets: list[dict] = [{"type": "kpi", "title": "Total rows", "metric": {"column": "*", "op": "count"}, "size": "s"}]
    if main:
        widgets.append({"type": "kpi", "title": f"Total {main}", "metric": {"column": main, "op": "sum"}, "size": "s"})
        widgets.append({"type": "kpi", "title": f"Average {main}", "metric": {"column": main, "op": "mean"}, "size": "s"})
    if second:
        widgets.append({"type": "kpi", "title": f"Total {second}", "metric": {"column": second, "op": "sum"}, "size": "s"})

    metric = {"column": main, "op": "sum"} if main else {"column": "*", "op": "count"}
    what = main or "rows"
    for i, cat in enumerate(chosen_cats):
        if i == 0 and by_name[cat]["unique_count"] <= 6:
            widgets.append({"type": "donut", "title": f"Share of {what} by {cat}", "group_by": cat, "metric": metric})
        widgets.append({"type": "bar", "title": f"{what} by {cat}", "group_by": cat, "metric": metric, "limit": 10})

    if dates:
        d = dates[0]
        line = {"type": "line", "title": f"{what} over time", "date_column": d["name"], "freq": _freq(d), "size": "l"}
        if main:
            line.update(value_column=main, op="sum")
        widgets.append(line)

    if main:
        widgets.append({"type": "histogram", "title": f"Distribution of {main}", "column": main})
    if main and second:
        widgets.append({"type": "scatter", "title": f"{second} vs {main}", "x": main, "y": second})
    if len(numeric) >= 3:
        widgets.append({"type": "heatmap", "title": "Correlations", "size": "l"})
    widgets.append({"type": "table", "title": "Data preview", "columns": [c["name"] for c in cols[:8]], "limit": 8, "size": "l"})

    filters = [{"kind": "category", "column": n, "values": []} for n in chosen_cats]
    if dates:
        filters.append({"kind": "date_range", "column": dates[0]["name"]})

    raw = {
        "title": title or "Auto dashboard",
        "description": "Built automatically from the column types. Edit it or add widgets with a prompt.",
        "widgets": widgets,
        "filters": filters,
    }
    return parse_dashboard(raw, profile)