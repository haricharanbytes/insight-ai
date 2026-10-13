import json

import pandas as pd

from src.analysis.aggregation import group_by_aggregate
from src.analysis.correlations import calculate_correlation
from src.analysis.outliers import detect_outliers
from src.analysis.statistics import get_column_summary, run_statistical_test
from src.analysis.timeseries import analyze_time_series
from src.visualization.charts import chart_for_output

DF = pd.DataFrame({
    "Region": ["East", "East", "West", "West", "West", "North", "North", "South", "South", "East"],
    "Sales": [100.0, 200.0, 50.0, 150.0, 80.0, 400.0, 300.0, 120.0, 90.0, 60.0],
    "Profit": [10.0, -20.0, 5.0, 15.0, 7.0, 60.0, 40.0, 12.0, 9.0, 6.0],
    "Order Date": ["2020-01-15", "2020-01-20", "2020-02-10", "2020-03-05", "2020-03-25",
                   "2020-03-30", "2020-05-02", "2020-06-11", "2020-06-20", "2020-07-04"],
})


def chart(out):
    c = chart_for_output(out)
    assert c is not None
    json.dumps(c, allow_nan=False)  # must be serialisable
    assert c["title"] and c["caption"]
    return c


def test_group_chart_is_donut_for_few_nonnegative_sums():
    out = group_by_aggregate(DF, ["Region"], [{"column": "Sales", "op": "sum"}])
    c = chart(out)
    assert c["kind"] == "donut"
    pie = c["figure"]["data"][0]
    assert dict(zip(pie["labels"], pie["values"])) == {"East": 360.0, "West": 280.0, "North": 700.0, "South": 210.0}


def test_group_chart_is_bar_for_means_and_negative_sums():
    assert chart(group_by_aggregate(DF, ["Region"], [{"column": "Sales", "op": "mean"}]))["kind"] == "bar"
    assert chart(group_by_aggregate(DF, ["Region"], [{"column": "Profit", "op": "sum"}]))["kind"] == "bar"


def test_bar_values_come_from_the_table_and_read_top_down():
    out = group_by_aggregate(DF, ["Region"], [{"column": "Sales", "op": "mean"}])
    c = chart(out)
    bar = c["figure"]["data"][0]
    assert bar["y"][-1] == out["table"]["rows"][0][0]  # largest group is drawn on top
    assert sorted(bar["x"]) == sorted(r[1] for r in out["table"]["rows"])


def test_truncated_groups_are_noted_and_never_a_donut():
    out = group_by_aggregate(DF, ["Region"], [{"column": "Sales", "op": "sum"}], limit=2)
    c = chart(out)
    assert c["kind"] == "bar" and "of 4 groups" in c["caption"]


def test_time_series_line_keeps_gaps():
    out = analyze_time_series(DF, "Order Date", "Sales", "mean", "M")
    c = chart(out)
    assert c["kind"] == "line"
    y = c["figure"]["data"][0]["y"]
    assert None in y and c["figure"]["data"][0]["connectgaps"] is False
    assert "not zero" in c["caption"]


def test_time_series_count_is_bar():
    assert chart(analyze_time_series(DF, "Order Date"))["kind"] == "bar"


def test_correlation_heatmap_and_outliers_and_tests():
    assert chart(calculate_correlation(DF, ["Sales", "Profit"]))["kind"] == "heatmap"
    assert chart(detect_outliers(DF, "Sales", threshold=0.5))["kind"] == "box"
    assert chart(run_statistical_test(DF, "kruskal", value_column="Sales", group_column="Region"))["kind"] == "bar"
    assert chart(get_column_summary(DF, "Region"))["kind"] == "bar"
    cat = pd.DataFrame({"a": ["x"] * 10 + ["y"] * 10, "b": ["p"] * 6 + ["q"] * 4 + ["p"] * 3 + ["q"] * 7})
    assert chart(run_statistical_test(cat, "chi_square", column_a="a", column_b="b"))["kind"] == "bar"
                


def test_no_outliers_still_gives_a_chart_with_a_message():
    out = detect_outliers(pd.DataFrame({"a": range(1, 11)}), "a")
    assert "No values lie outside" in chart(out)["caption"]


def test_no_chart_for_tools_without_one_and_bad_output_is_safe():
    assert chart_for_output({"tool": "filter_data"}) is None
    assert chart_for_output({"tool": "group_by_aggregate", "table": None, "inputs": {}, "result": {}}) is None
    assert chart_for_output({"tool": "analyze_time_series"}) is None  # malformed: must not raise