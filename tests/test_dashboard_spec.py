import pandas as pd
import pytest

from src.analysis.filtering import apply_filters
from src.dashboard.spec import SpecError, parse_dashboard, widget_filters
from src.data.profiler import profile_dataset

DF = pd.DataFrame({
    "Region": ["East", "West", "North", "South"] * 3,
    "Segment": ["A", "B", "C"] * 4,
    "Sales": [100.0 + i * 10 for i in range(12)],
    "Profit": [10.0 + i for i in range(12)],
    "Order Date": [f"2020-{m:02d}-15" for m in range(1, 13)],
})
PROFILE = profile_dataset(DF)
KPI = {"type": "kpi", "metric": {"column": "Sales", "op": "sum"}}


def spec(*widgets, **extra):
    return {"title": "T", "widgets": list(widgets), **extra}


def problems(raw, profile=PROFILE):
    with pytest.raises(SpecError) as exc:
        parse_dashboard(raw, profile)
    assert exc.value.code == "invalid_dashboard"
    return exc.value.details["problems"]


def test_valid_dashboard_gets_ids_and_defaults():
    d = parse_dashboard(spec(
        {"type": "kpi", "title": "Total sales", "metric": {"column": "Sales", "op": "SUM"}},
        {"type": "bar", "group_by": "Region", "metric": {"column": "Sales", "op": "sum"}},
        {"type": "donut", "group_by": "Segment", "metric": {"column": "*", "op": "count"}},
        {"type": "line", "date_column": "Order Date", "value_column": "Sales"},
        {"type": "histogram", "column": "Profit"},
        {"type": "scatter", "x": "Sales", "y": "Profit"},
        {"type": "box", "column": "Sales", "group_by": "Region"},
        {"type": "heatmap"},
        {"type": "table", "columns": ["Region", "Sales"]},
    ), PROFILE)
    assert [w.id for w in d.widgets] == [f"w{i}" for i in range(1, 10)]
    assert d.widgets[0].metric.op == "sum"
    assert d.widgets[1].limit == 10 and d.widgets[3].freq == "M" and d.widgets[3].op == "sum"


def test_json_roundtrip_is_stable():
    d = parse_dashboard(spec(KPI, {"type": "histogram", "column": "Profit", "bins": 12}), PROFILE)
    dumped = d.model_dump(mode="json")
    assert parse_dashboard(dumped, PROFILE).model_dump(mode="json") == dumped


def test_unknown_column_gets_a_suggestion():
    p = problems(spec({"type": "kpi", "metric": {"column": "Sale", "op": "sum"}}))
    assert p[0]["path"] == "widgets[0].metric.column" and p[0]["code"] == "column_not_found"
    assert "Sales" in p[0]["message"]


def test_type_problems_are_all_reported_together():
    p = problems(spec(
        {"type": "bar", "group_by": "Sales", "metric": {"column": "Profit", "op": "sum"}},
        {"type": "kpi", "metric": {"column": "Region", "op": "sum"}},
        {"type": "donut", "group_by": "Region", "metric": {"column": "Sales", "op": "mean"}},
        {"type": "line", "date_column": "Sales"},
        {"type": "scatter", "x": "Sales", "y": "Sales"},
    ))
    codes = {x["path"]: x["code"] for x in p}
    assert codes["widgets[0].group_by"] == "invalid_column_type"
    assert codes["widgets[1].metric.column"] == "invalid_column_type"
    assert codes["widgets[2].metric.op"] == "invalid_parameter"
    assert codes["widgets[3].date_column"] == "invalid_column_type"
    assert codes["widgets[4].y"] == "invalid_parameter"


def test_shape_errors():
    assert problems(spec({**KPI, "color": "red"}))                       # unknown field
    assert problems(spec({"type": "kpi", "metric": {"column": "Sales", "op": "variance"}}))
    assert problems(spec({"type": "histogram", "column": "Sales", "bins": 2}))
    assert problems(spec(*[{"type": "histogram", "column": "Sales"}] * 21))   # too many widgets
    p = problems(spec({"type": "pie"}))
    assert p[0]["path"].startswith("widgets[0]")
    assert any("duplicate" in x["message"] for x in problems(spec({**KPI, "id": "a"}, {**KPI, "id": "a"})))


def test_count_star_rules():
    assert problems(spec({"type": "kpi", "metric": {"column": "*", "op": "sum"}}))
    parse_dashboard(spec({"type": "kpi", "metric": {"column": "*", "op": "count"}}), PROFILE)


def test_heatmap_needs_two_numeric_columns():
    small = profile_dataset(DF[["Region", "Sales"]])
    assert problems(spec({"type": "heatmap"}), small)


def test_global_filters_become_tool_filters():
    d = parse_dashboard(spec(KPI, filters=[
        {"kind": "category", "column": "Region", "values": ["East", "West"]},
        {"kind": "date_range", "column": "Order Date", "start": "2020-02-01", "end": "2020-06-30"},
        {"kind": "numeric_range", "column": "Sales", "min": 120},
    ]), PROFILE)
    f = [g.to_tool_filters() for g in d.filters]
    assert f[0] == [{"column": "Region", "op": "in", "value": ["East", "West"]}]
    assert f[1] == [{"column": "Order Date", "op": "between", "value": ["2020-02-01", "2020-06-30"]}]
    assert f[2] == [{"column": "Sales", "op": "gte", "value": 120.0}]

    # The generated filters really work with the existing filter engine:
    # East rows are i=0,4,8 and West rows i=1,5,9; dates in Feb-Jun leave i=1,4,5; Sales >= 120 leaves i=4,5.
    kept, _, _ = apply_filters(DF, widget_filters(d, d.widgets[0]))
    assert len(kept) == 2 and sorted(kept["Sales"]) == [140.0, 150.0]


def test_inactive_filters_produce_nothing():
    d = parse_dashboard(spec(KPI, filters=[
        {"kind": "category", "column": "Region", "values": []},
        {"kind": "date_range", "column": "Order Date"},
        {"kind": "numeric_range", "column": "Sales"},
    ]), PROFILE)
    assert widget_filters(d, d.widgets[0]) == []


def test_widget_can_opt_out_of_global_filters_and_keeps_local_ones():
    d = parse_dashboard(spec(
        {**KPI, "use_global_filters": False, "filters": [{"column": "Region", "op": "eq", "value": "East"}]},
        filters=[{"kind": "category", "column": "Segment", "values": ["A"]}],
    ), PROFILE)
    assert widget_filters(d, d.widgets[0]) == [{"column": "Region", "op": "eq", "value": "East"}]


def test_bad_global_and_local_filters():
    p = problems(spec(KPI, filters=[
        {"kind": "category", "column": "Sales", "values": ["x"]},
        {"kind": "date_range", "column": "Region"},
        {"kind": "numeric_range", "column": "Nope"},
    ]))
    assert {x["code"] for x in p} == {"invalid_column_type", "column_not_found"}
    assert problems(spec(KPI, filters=[{"kind": "date_range", "column": "Order Date", "start": "2021-01-01", "end": "2020-01-01"}]))
    assert problems(spec(KPI, filters=[{"kind": "date_range", "column": "Order Date", "start": "not a date"}]))
    assert problems(spec(KPI, filters=[{"kind": "numeric_range", "column": "Sales", "min": 9, "max": 1}]))
    assert problems(spec({**KPI, "filters": [{"column": "Nope", "op": "eq", "value": 1}]}))
    assert problems(spec({**KPI, "filters": [{"column": "Region", "op": "eq"}]}))   # value required