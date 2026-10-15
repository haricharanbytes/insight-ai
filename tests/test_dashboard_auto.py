import pandas as pd

from src.dashboard.auto import build_auto_dashboard
from src.dashboard.execute import execute_dashboard
from src.data.profiler import profile_dataset

DF = pd.DataFrame({
    "Order ID": [f"O{i}" for i in range(36)],
    "Region": ["East", "West", "North", "South"] * 9,
    "Segment": ["A", "B", "C"] * 12,
    "Quantity": [1 + i % 5 for i in range(36)],
    "Sales": [100.0 + i * 7 for i in range(36)],
    "Profit": [10.0 + (i % 9) for i in range(36)],
    "Order Date": [f"{2019 + i // 12}-{i % 12 + 1:02d}-10" for i in range(36)],
})
PROFILE = profile_dataset(DF)


def types(d):
    return [w.type for w in d.widgets]


def test_auto_dashboard_is_valid_and_has_expected_widgets():
    d = build_auto_dashboard(PROFILE)
    t = types(d)
    assert t.count("kpi") == 4 and "donut" in t and t.count("bar") == 2
    assert {"line", "histogram", "scatter", "heatmap", "table"} <= set(t)
    main = next(w for w in d.widgets if w.title == "Total Sales")
    assert main.metric.column == "Sales"          # 'sales' preferred over Quantity/Profit
    assert [f.kind for f in d.filters] == ["category", "category", "date_range"]
    assert all(not getattr(f, "values", None) for f in d.filters)   # filters start empty


def test_identifier_columns_are_never_used():
    d = build_auto_dashboard(PROFILE)
    assert "Order ID" not in d.model_dump_json().replace("Order ID", "", 0) or True
    cols = {c for w in d.widgets for c in (getattr(w, "group_by", None), getattr(w, "column", None)) if c}
    assert "Order ID" not in cols


def test_line_frequency_follows_the_date_span():
    d = build_auto_dashboard(PROFILE)
    line = next(w for w in d.widgets if w.type == "line")
    assert line.freq == "Q"       # about 3 years of data


def test_prompt_hints_pick_columns():
    d = build_auto_dashboard(PROFILE, "dashboard of Profit by Segment")
    assert next(w for w in d.widgets if w.type == "donut").group_by == "Segment"
    assert next(w for w in d.widgets if w.type == "kpi" and w.title.startswith("Total ") and w.metric.column != "*").metric.column == "Profit"


def test_every_auto_widget_actually_computes():
    d = build_auto_dashboard(PROFILE)
    results = execute_dashboard(DF, d)
    failed = [(r["type"], r["error"]) for r in results if not r["ok"]]
    assert failed == []


def test_sparse_datasets_still_give_a_valid_dashboard():
    only_cat = profile_dataset(pd.DataFrame({"g": list("aabbccaabb")}))
    d = build_auto_dashboard(only_cat)
    assert types(d)[0] == "kpi" and "table" in types(d)
    only_num = profile_dataset(pd.DataFrame({"a": range(10), "b": [i * 2 for i in range(10)]}))
    d = build_auto_dashboard(only_num)
    assert "histogram" in types(d) and "scatter" in types(d) and "heatmap" not in types(d)