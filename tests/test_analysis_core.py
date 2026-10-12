import pandas as pd
import pytest

from src.analysis.aggregation import group_by_aggregate
from src.analysis.common import AnalysisError
from src.analysis.filtering import apply_filters, filter_data
from src.analysis.timeseries import analyze_time_series


@pytest.fixture
def sales():
    return pd.DataFrame({
        "Region": ["East", "East", "West", "West", "West", "North"],
        "Product": ["A", "B", "A", "B", "B", "A"],
        "Sales": [100.0, 200.0, 50.0, 150.0, None, 400.0],
        "Profit": [10.0, -20.0, 5.0, 15.0, 7.0, 60.0],
        "Order Date": ["2020-01-15", "2020-01-20", "2020-02-10",
                       "2020-03-05", "2020-03-25", "2020-03-30"],
    })


def rows(result):
    t = result["table"]
    return [dict(zip(t["columns"], r)) for r in t["rows"]]


def code_of(fn, *args, **kwargs):
    with pytest.raises(AnalysisError) as exc:
        fn(*args, **kwargs)
    return exc.value.code


# ====================== group_by_aggregate ======================

def test_group_sum_sorted_desc(sales):
    r = group_by_aggregate(sales, ["Region"], [{"column": "Sales", "op": "sum"}])
    assert [x["Region"] for x in rows(r)] == ["North", "East", "West"]
    assert {x["Region"]: x["sum_Sales"] for x in rows(r)} == {"North": 400.0, "East": 300.0, "West": 200.0}
    assert r["result"]["top_group"]["Region"] == "North"
    assert r["rows_used"] == 6


def test_share_pct(sales):
    r = group_by_aggregate(sales, ["Region"], [{"column": "Sales", "op": "sum"}])
    shares = {x["Region"]: x["share_pct"] for x in rows(r)}
    assert shares["North"] == pytest.approx(44.44, abs=0.01)
    assert sum(shares.values()) == pytest.approx(100, abs=0.05)


def test_no_share_when_values_negative(sales):
    r = group_by_aggregate(sales, ["Region"], [{"column": "Profit", "op": "sum"}])
    assert "share_pct" not in r["table"]["columns"]


def test_count_ignores_missing_but_row_count_does_not(sales):
    r = group_by_aggregate(sales, ["Region"], [
        {"column": "Sales", "op": "count"}, {"column": "*", "op": "count"}])
    west = next(x for x in rows(r) if x["Region"] == "West")
    assert west["count_Sales"] == 2
    assert west["row_count"] == 3


def test_mean_median_min_max(sales):
    r = group_by_aggregate(sales, ["Region"], [
        {"column": "Sales", "op": "mean"}, {"column": "Sales", "op": "min"},
        {"column": "Sales", "op": "max"}, {"column": "Sales", "op": "median"}])
    west = next(x for x in rows(r) if x["Region"] == "West")
    assert west["mean_Sales"] == 100.0  # missing value ignored, not treated as 0
    assert west["min_Sales"] == 50.0 and west["max_Sales"] == 150.0
    assert west["median_Sales"] == 100.0


def test_overall_aggregate_without_group(sales):
    r = group_by_aggregate(sales, [], [{"column": "Sales", "op": "sum"}])
    assert r["result"]["values"]["sum_Sales"] == 900.0


def test_all_missing_sum_stays_missing_not_zero():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 2.0, None]})
    r = group_by_aggregate(df, ["g"], [{"column": "v", "op": "sum"}], sort_by="g", ascending=True)
    assert [x["sum_v"] for x in rows(r)] == [3.0, None]
    only_b = group_by_aggregate(df[df["g"] == "b"], [], [{"column": "v", "op": "sum"}])
    assert only_b["result"]["values"]["sum_v"] is None


def test_missing_group_key_is_labelled():
    df = pd.DataFrame({"g": ["a", None, "a"], "v": [1, 2, 3]})
    r = group_by_aggregate(df, ["g"], [{"column": "v", "op": "sum"}])
    assert "(missing)" in [x["g"] for x in rows(r)]
    assert any("missing" in w for w in r["warnings"])


def test_aggregation_with_filters(sales):
    r = group_by_aggregate(sales, ["Region"], [{"column": "Sales", "op": "sum"}],
                           filters=[{"column": "Product", "op": "eq", "value": "A"}])
    assert {x["Region"]: x["sum_Sales"] for x in rows(r)} == {"North": 400.0, "East": 100.0, "West": 50.0}
    assert r["rows_used"] == 3


def test_limit_and_truncated_flag(sales):
    r = group_by_aggregate(sales, ["Region"], [{"column": "Sales", "op": "sum"}], limit=2)
    assert len(r["table"]["rows"]) == 2
    assert r["table"]["truncated"] is True
    assert r["result"]["total_groups"] == 3


def test_invalid_column_has_suggestion(sales):
    with pytest.raises(AnalysisError) as exc:
        group_by_aggregate(sales, ["Region"], [{"column": "Sale", "op": "sum"}])
    assert exc.value.code == "column_not_found"
    assert "Sales" in exc.value.details["suggestions"]["Sale"]


def test_aggregation_errors(sales):
    assert code_of(group_by_aggregate, sales, ["Region"], [{"column": "Product", "op": "sum"}]) == "invalid_column_type"
    assert code_of(group_by_aggregate, sales, ["Region"], [{"column": "Sales", "op": "variance"}]) == "unsupported_operation"
    assert code_of(group_by_aggregate, sales, ["Region"], [{"column": "Region", "op": "count"}]) == "invalid_parameter"
    assert code_of(group_by_aggregate, sales, ["Region"], limit=0) == "invalid_parameter"
    assert code_of(group_by_aggregate, sales, ["Region"], [{"column": "Sales", "op": "sum"}], sort_by="nope") == "column_not_found"
    assert code_of(group_by_aggregate, sales, ["Region"], filters=[{"column": "Region", "op": "eq", "value": "Mars"}]) == "no_rows"


# ====================== filtering ======================

def test_filter_eq(sales):
    r = filter_data(sales, [{"column": "Region", "op": "eq", "value": "East"}])
    assert r["result"]["rows_before"] == 6 and r["result"]["rows_after"] == 2


def test_filter_and_combination(sales):
    r = filter_data(sales, [
        {"column": "Region", "op": "in", "value": ["East", "West"]},
        {"column": "Sales", "op": "gte", "value": 100},
    ])
    assert r["result"]["rows_after"] == 3  # East 100, East 200, West 150


def test_filter_numeric_value_given_as_string(sales):
    r = filter_data(sales, [{"column": "Sales", "op": "gt", "value": "150"}])
    assert r["result"]["rows_after"] == 2  # 200 and 400


def test_filter_text_date_range(sales):
    r = filter_data(sales, [{"column": "Order Date", "op": "between", "value": ["2020-02-01", "2020-03-31"]}])
    assert r["result"]["rows_after"] == 4


def test_ne_excludes_missing(sales):
    r = filter_data(sales, [{"column": "Sales", "op": "ne", "value": 100}])
    assert r["result"]["rows_after"] == 4  # the missing Sales row is not returned


def test_contains_is_case_insensitive(sales):
    r = filter_data(sales, [{"column": "Product", "op": "contains", "value": "a"}])
    assert r["result"]["rows_after"] == 3


def test_null_filters(sales):
    assert filter_data(sales, [{"column": "Sales", "op": "is_null"}])["result"]["rows_after"] == 1
    assert filter_data(sales, [{"column": "Sales", "op": "not_null"}])["result"]["rows_after"] == 5


def test_zero_match_warns_with_examples(sales):
    r = filter_data(sales, [{"column": "Region", "op": "eq", "value": "east"}])
    assert r["result"]["rows_after"] == 0
    assert any("no rows" in w and "example values" in w for w in r["warnings"])


def test_filter_errors(sales):
    assert code_of(filter_data, sales, [{"column": "Sales", "op": "like", "value": 1}]) == "unsupported_operation"
    assert code_of(filter_data, sales, [{"column": "Sales", "op": "gt", "value": "abc"}]) == "invalid_filter_value"
    assert code_of(filter_data, sales, [{"column": "Region", "op": "gt", "value": "A"}]) == "invalid_column_type"
    assert code_of(filter_data, sales, [{"column": "Nope", "op": "eq", "value": 1}]) == "column_not_found"
    assert code_of(filter_data, sales, [{"column": "Sales", "op": "gt"}]) == "invalid_filter_value"
    assert code_of(filter_data, sales, []) == "invalid_parameter"
    assert code_of(filter_data, sales, [{"column": "Sales", "op": "between", "value": [5, 1]}]) == "invalid_filter_value"


def test_inputs_are_not_modified(sales):
    before = sales.copy()
    group_by_aggregate(sales, ["Region"], [{"column": "Sales", "op": "sum"}],
                       filters=[{"column": "Product", "op": "eq", "value": "A"}])
    filter_data(sales, [{"column": "Region", "op": "eq", "value": "East"}])
    analyze_time_series(sales, "Order Date", "Sales")
    apply_filters(sales, [{"column": "Sales", "op": "gt", "value": 1}])
    pd.testing.assert_frame_equal(sales, before)


# ====================== analyze_time_series ======================

def test_monthly_sum_and_changes(sales):
    r = analyze_time_series(sales, "Order Date", "Sales", "sum", "M")
    t = rows(r)
    assert [x["period"] for x in t] == ["2020-01", "2020-02", "2020-03"]
    assert [x["value"] for x in t] == [300.0, 50.0, 550.0]
    assert t[1]["change_pct"] == pytest.approx(-83.333, abs=0.01)
    assert t[2]["change_pct"] == pytest.approx(1000.0)
    assert r["result"]["peak"]["period"] == "2020-03"
    assert r["result"]["lowest"]["period"] == "2020-02"
    assert r["result"]["overall_change"]["absolute"] == 250.0
    assert r["result"]["overall_change"]["pct"] == pytest.approx(83.33, abs=0.01)


def test_row_count_per_period(sales):
    r = analyze_time_series(sales, "Order Date")
    assert [x["value"] for x in rows(r)] == [2.0, 1.0, 3.0]
    assert r["inputs"]["operation"] == "count"


def test_partial_first_period_warning(sales):
    r = analyze_time_series(sales, "Order Date", "Sales", "sum", "M")
    assert any("partial" in w and "first" in w for w in r["warnings"])
    assert not any("last period" in w for w in r["warnings"])


def test_trend_unavailable_with_few_periods(sales):
    r = analyze_time_series(sales, "Order Date", "Sales", "sum", "M")
    assert r["result"]["trend"]["available"] is False


def test_increasing_trend():
    df = pd.DataFrame({
        "when": pd.date_range("2020-01-01", periods=12, freq="MS"),
        "v": [10.0 * i for i in range(1, 13)],
    })
    r = analyze_time_series(df, "when", "v", "sum", "M")
    trend = r["result"]["trend"]
    assert trend["direction"] == "increasing"
    assert trend["slope_per_period"] == pytest.approx(10.0)
    assert trend["r_squared"] == pytest.approx(1.0)


def test_empty_period_sum_is_zero_with_warning():
    df = pd.DataFrame({"d": ["2020-01-05", "2020-03-05"], "v": [10, 20]})
    r = analyze_time_series(df, "d", "v", "sum", "M")
    assert [x["value"] for x in rows(r)] == [10.0, 0.0, 20.0]
    assert any("no records" in w for w in r["warnings"])


def test_empty_period_mean_stays_missing():
    df = pd.DataFrame({"d": ["2020-01-05", "2020-03-05"], "v": [10, 20]})
    r = analyze_time_series(df, "d", "v", "mean", "M")
    assert [x["value"] for x in rows(r)] == [10.0, None, 20.0]


def test_unreadable_dates_excluded_and_reported():
    dates = [f"2020-{m:02d}-15" for m in range(1, 11)] + [None]
    df = pd.DataFrame({"d": dates, "v": range(11)})
    r = analyze_time_series(df, "d", "v", "sum", "M")
    assert r["rows_used"] == 10
    assert any("excluded" in w for w in r["warnings"])


def test_time_series_with_filters(sales):
    r = analyze_time_series(sales, "Order Date", "Sales", "sum", "M",
                            filters=[{"column": "Region", "op": "eq", "value": "West"}])
    assert [x["period"] for x in rows(r)] == ["2020-02", "2020-03"]
    assert [x["value"] for x in rows(r)] == [50.0, 150.0]  # the missing Sales value is ignored
    assert r["rows_used"] == 3


def test_time_series_errors(sales):
    assert code_of(analyze_time_series, sales, "Region") == "invalid_column_type"
    assert code_of(analyze_time_series, sales, "Sales") == "invalid_column_type"
    assert code_of(analyze_time_series, sales, "Order Date", "Product") == "invalid_column_type"
    assert code_of(analyze_time_series, sales, "Order Date", "Sales", "sum", "X") == "invalid_parameter"
    assert code_of(analyze_time_series, sales, "Order Date", "Sales", "variance") == "unsupported_operation"
    assert code_of(analyze_time_series, sales, "Nope") == "column_not_found"


def test_too_many_periods():
    df = pd.DataFrame({"d": ["2000-01-01", "2020-01-01"], "v": [1, 2]})
    assert code_of(analyze_time_series, df, "d", "v", "sum", "D") == "too_many_periods"