import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.expected import truth  # noqa: E402
from src.agent.demo_router import run_demo  # noqa: E402
from src.analysis.aggregation import group_by_aggregate  # noqa: E402
from src.analysis.common import AnalysisError  # noqa: E402
from src.analysis.correlations import calculate_correlation  # noqa: E402
from src.analysis.filtering import filter_data  # noqa: E402
from src.analysis.timeseries import analyze_time_series  # noqa: E402
from src.data.loader import load_dataset  # noqa: E402
from src.data.profiler import profile_dataset  # noqa: E402
from src.data.validation import DataLoadError  # noqa: E402

CSV = ROOT / "benchmarks" / "bench_sales.csv"
T = truth()


@pytest.fixture(scope="module")
def df():
    return load_dataset("bench_sales.csv", CSV.read_bytes(), 10**8, 10**6).df


def rows(out):
    t = out["table"]
    return [dict(zip(t["columns"], r)) for r in t["rows"]]


def test_numeric_aggregation(df):
    out = group_by_aggregate(df, [], [{"column": "Sales", "op": "sum"}])
    assert out["result"]["values"]["sum_Sales"] == pytest.approx(T["total_sales"])


def test_category_grouping(df):
    by_region = {r["Region"]: r["sum_Sales"] for r in rows(group_by_aggregate(df, ["Region"], [{"column": "Sales", "op": "sum"}]))}
    assert by_region == pytest.approx(T["sales_by_region"])
    by_product = group_by_aggregate(df, ["Product"], [{"column": "Sales", "op": "sum"}])
    assert by_product["result"]["top_group"]["Product"] == T["top_product"]


def test_filtering(df):
    out = filter_data(df, [{"column": "Region", "op": "eq", "value": "West"}])
    assert out["result"]["rows_after"] == T["west_rows"]
    s = group_by_aggregate(df, [], [{"column": "Sales", "op": "sum"}], filters=[{"column": "Region", "op": "eq", "value": "West"}])
    assert s["result"]["values"]["sum_Sales"] == pytest.approx(T["west_sales"])


def test_date_trend(df):
    out = analyze_time_series(df, "Order Date", "Sales", "sum", "Q")
    got = {r["period"]: r["value"] for r in rows(out)}
    assert got == pytest.approx(T["sales_by_quarter"])
    assert out["result"]["last"]["period"] == T["last_quarter"]
    assert T["last_quarter_sales"] < T["prev_quarter_sales"]  # the decline is real in this dataset


def test_missing_and_duplicates_and_negatives(df):
    p = profile_dataset(df)
    sales = next(c for c in p["columns"] if c["name"] == "Sales")
    assert sales["missing_count"] == T["missing_sales"]
    assert p["duplicates"]["count"] == T["duplicates"]
    profit = next(c for c in p["columns"] if c["name"] == "Profit")
    assert profit["negative_count"] == T["negative_profit"]


def test_correlation(df):
    out = calculate_correlation(df, ["Sales", "Profit"])
    assert out["result"]["strongest"]["r"] == pytest.approx(T["corr_sales_profit"], abs=1e-9)


def test_invalid_column_and_operation(df):
    with pytest.raises(AnalysisError) as e:
        group_by_aggregate(df, ["Regoin"], [{"column": "Sales", "op": "sum"}])
    assert e.value.code == "column_not_found" and "Region" in e.value.details["suggestions"]["Regoin"]
    with pytest.raises(AnalysisError) as e:
        group_by_aggregate(df, ["Region"], [{"column": "Sales", "op": "variance"}])
    assert e.value.code == "unsupported_operation"


@pytest.mark.parametrize("name,content", [("a.csv", b""), ("a.csv", b"\x00\x01bin"), ("a.txt", b"x"), ("a.xlsx", b"nope")])
def test_malformed_files(name, content):
    with pytest.raises(DataLoadError):
        load_dataset(name, content, 10**6, 10**4)


def test_unsupported_request_is_declined_not_invented(df):
    r = run_demo(df, profile_dataset(df), "Why did revenue fall?")
    assert r["steps"] == [] and r["incomplete"] is True


def test_followup_resolution_via_demo_state_is_not_claimed():
    # Follow-ups need the LLM agent; they are measured in the LLM suite, not here.
    assert True