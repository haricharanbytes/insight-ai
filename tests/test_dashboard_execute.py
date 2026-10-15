import json

import pandas as pd
import pytest

from src.dashboard.execute import affected_widget_ids, execute_dashboard, execute_widget, widget_call
from src.dashboard.spec import parse_dashboard
from src.data.profiler import profile_dataset

DF = pd.DataFrame({
    "Region": ["East", "West", "North", "South"] * 3,
    "Segment": ["A", "B", "C"] * 4,
    "Sales": [100.0 + i * 10 for i in range(12)],
    "Profit": [10.0 + i for i in range(12)],
    "Order Date": [f"2020-{m:02d}-15" for m in range(1, 13)],
})
PROFILE = profile_dataset(DF)


def dash(*widgets, **extra):
    return parse_dashboard({"title": "T", "widgets": list(widgets), **extra}, PROFILE)


def run_one(widget, **extra):
    d = dash(widget, **extra)
    return execute_widget(DF, d, d.widgets[0])


def test_kpi_value_matches_pandas():
    r = run_one({"type": "kpi", "metric": {"column": "Sales", "op": "sum"}})
    assert r["ok"] and r["value"] == pytest.approx(DF["Sales"].sum()) and r["label"] == "sum_Sales"
    assert r["rows_used"] == 12 and r["title"] == "sum of Sales"


def test_kpi_row_count_and_mean():
    assert run_one({"type": "kpi", "metric": {"column": "*", "op": "count"}})["value"] == 12
    assert run_one({"type": "kpi", "metric": {"column": "Profit", "op": "mean"}})["value"] == pytest.approx(DF["Profit"].mean())


def test_bar_and_donut_charts():
    bar = run_one({"type": "bar", "group_by": "Region", "metric": {"column": "Sales", "op": "mean"}})
    assert bar["ok"] and bar["chart"]["kind"] == "bar"
    donut = run_one({"type": "donut", "group_by": "Segment", "metric": {"column": "Sales", "op": "sum"}})
    assert donut["chart"]["kind"] == "donut"
    pie = donut["chart"]["figure"]["data"][0]
    assert dict(zip(pie["labels"], pie["values"])) == pytest.approx(DF.groupby("Segment")["Sales"].sum().to_dict())


def test_line_histogram_scatter_box_heatmap():
    assert run_one({"type": "line", "date_column": "Order Date", "value_column": "Sales"})["chart"]["kind"] == "line"
    assert run_one({"type": "line", "date_column": "Order Date"})["chart"]["kind"] == "bar"   # row counts
    assert run_one({"type": "histogram", "column": "Profit", "bins": 5})["chart"]["kind"] == "histogram"
    assert run_one({"type": "scatter", "x": "Sales", "y": "Profit"})["chart"]["kind"] == "scatter"
    assert run_one({"type": "box", "column": "Sales", "group_by": "Region"})["chart"]["kind"] == "box"
    assert run_one({"type": "heatmap"})["chart"]["kind"] == "heatmap"


def test_table_with_and_without_filters():
    r = run_one({"type": "table", "columns": ["Region", "Sales"], "limit": 3})
    assert r["ok"] and len(r["table"]["rows"]) == 3 and r["table"]["total_rows"] == 12
    f = run_one({"type": "table", "filters": [{"column": "Region", "op": "eq", "value": "East"}]})
    assert f["table"]["total_rows"] == 3 and f["rows_after"] == 3


def test_global_filters_change_results_and_opt_out_works():
    gf = [{"kind": "category", "column": "Region", "values": ["East"]}]
    kpi = {"type": "kpi", "metric": {"column": "Sales", "op": "sum"}}
    d = dash(kpi, {**kpi, "use_global_filters": False}, filters=gf)
    a, b = execute_dashboard(DF, d)
    assert a["value"] == pytest.approx(DF[DF["Region"] == "East"]["Sales"].sum()) and a["rows_used"] == 3
    assert b["value"] == pytest.approx(DF["Sales"].sum()) and b["rows_used"] == 12


def test_only_ids_recomputes_just_those_widgets():
    kpi = {"type": "kpi", "metric": {"column": "Sales", "op": "sum"}}
    d = dash(kpi, {**kpi, "use_global_filters": False})
    assert affected_widget_ids(d) == {"w1"}
    out = execute_dashboard(DF, d, affected_widget_ids(d))
    assert [r["id"] for r in out] == ["w1"]


def test_one_failing_widget_does_not_break_the_rest():
    d = dash({"type": "kpi", "metric": {"column": "Sales", "op": "sum"}},
             {"type": "kpi", "metric": {"column": "Profit", "op": "sum"},
              "filters": [{"column": "Region", "op": "eq", "value": "Mars"}]})
    ok, bad = execute_dashboard(DF, d)
    assert ok["ok"] is True
    assert bad["ok"] is False and bad["error"]["message"] == "No rows match the current filters."


def test_widget_call_includes_combined_filters():
    d = dash({"type": "bar", "group_by": "Region", "metric": {"column": "Sales", "op": "sum"},
              "filters": [{"column": "Segment", "op": "eq", "value": "A"}]},
             filters=[{"kind": "category", "column": "Region", "values": ["East", "West"]}])
    tool, params = widget_call(d.widgets[0], [{"column": "Region", "op": "in", "value": ["East", "West"]},
                                              {"column": "Segment", "op": "eq", "value": "A"}])
    assert tool == "group_by_aggregate" and len(params["filters"]) == 2


def test_results_are_json_serialisable_and_input_unmodified():
    before = DF.copy()
    d = dash({"type": "kpi", "metric": {"column": "Sales", "op": "sum"}},
             {"type": "bar", "group_by": "Region", "metric": {"column": "Sales", "op": "sum"}},
             {"type": "heatmap"}, {"type": "table"})
    json.dumps(execute_dashboard(DF, d), allow_nan=False)
    pd.testing.assert_frame_equal(DF, before)