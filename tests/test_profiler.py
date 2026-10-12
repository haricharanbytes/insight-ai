import json

import pandas as pd
import pytest

from src.data.profiler import profile_dataset


def col(profile, name):
    return next(c for c in profile["columns"] if c["name"] == name)


def codes(profile):
    return {i["code"] for i in profile["quality"]["issues"]}


# ---------- numeric ----------

def test_numeric_summary_and_outlier():
    p = profile_dataset(pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]}))
    a = col(p, "a")
    assert a["inferred_type"] == "numeric"
    assert a["mean"] == pytest.approx(14.5)
    assert a["median"] == pytest.approx(5.5)
    assert a["min"] == 1 and a["max"] == 100
    assert a["outliers"]["count"] == 1
    assert a["outliers"]["high_count"] == 1


def test_zero_negative_missing_are_distinct():
    p = profile_dataset(pd.DataFrame({"v": [0, -5, None, 3, 4, 5, 6, 7, 8, 9]}))
    v = col(p, "v")
    assert v["zero_count"] == 1
    assert v["negative_count"] == 1
    assert v["missing_count"] == 1


# ---------- structure ----------

def test_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9], "b": list("xxyzuvwabc")})
    p = profile_dataset(df)
    assert p["duplicates"]["count"] == 1
    assert "duplicate_rows" in codes(p)


def test_overall_missing_pct():
    df = pd.DataFrame({"a": [1, None, 3, 4], "b": ["x", "y", None, None]})
    p = profile_dataset(df)
    assert p["missing"]["total_cells_missing"] == 3
    assert p["missing"]["pct"] == pytest.approx(37.5)


# ---------- dates ----------

def test_text_dates_detected():
    df = pd.DataFrame({"d": ["11/8/2016", "6/12/2016", "10/11/2015", "6/9/2014"]})
    d = col(profile_dataset(df), "d")
    assert d["inferred_type"] == "datetime"
    assert d["parsed_from_text"] is True
    assert d["min"].startswith("2014-06-09")


def test_native_datetime():
    df = pd.DataFrame({"when": pd.date_range("2021-01-01", periods=5)})
    w = col(profile_dataset(df), "when")
    assert w["inferred_type"] == "datetime"
    assert w["parsed_from_text"] is False
    assert w["min"].startswith("2021-01-01")


def test_numeric_strings_not_dates():
    df = pd.DataFrame({"c": ["20200101", "20200102", "20200103", "20200104"]})
    assert col(profile_dataset(df), "c")["inferred_type"] != "datetime"


def test_unparsed_dates_counted():
    vals = [f"2020-{m:02d}-15" for m in range(1, 10)] + ["not a date"]
    p = profile_dataset(pd.DataFrame({"d": vals}))
    d = col(p, "d")
    assert d["inferred_type"] == "datetime"
    assert d["unparsed_count"] == 1
    assert "unparsed_dates" in codes(p)


# ---------- type inference ----------

def test_identifier_by_name():
    df = pd.DataFrame({
        "Row ID": range(1, 11),
        "Postal Code": [10001, 10002] * 5,
        "Sales": [5.5, 6.1, 7.3, 8.2, 9.9, 3.3, 4.4, 5.0, 6.6, 7.7],
    })
    p = profile_dataset(df)
    assert col(p, "Row ID")["inferred_type"] == "identifier"
    assert col(p, "Postal Code")["inferred_type"] == "identifier"
    assert p["type_groups"]["numeric"] == ["Sales"]


def test_categorical_top_values():
    p = profile_dataset(pd.DataFrame({"cat": ["A"] * 5 + ["B"] * 3 + ["C"] * 2}))
    c = col(p, "cat")
    assert c["inferred_type"] == "categorical"
    assert c["unique_count"] == 3
    assert c["top_values"][0] == {"value": "A", "count": 5, "pct": 50.0}


def test_numeric_text_flagged():
    df = pd.DataFrame({"amount": [f"{1000 + i * 7:,}" for i in range(15)]})
    p = profile_dataset(df)
    assert col(p, "amount")["inferred_type"] == "numeric_text"
    assert "numeric_stored_as_text" in codes(p)


# ---------- quality ----------

def test_empty_column_is_critical():
    p = profile_dataset(pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]}))
    assert col(p, "b")["inferred_type"] == "empty"
    assert "empty_column" in codes(p)
    assert p["quality"]["status"] == "poor"


def test_clean_dataset_is_good():
    p = profile_dataset(pd.DataFrame({"x": list(range(1, 11)), "g": list("aabbaabbaa")}))
    assert p["quality"]["issues"] == []
    assert p["quality"]["status"] == "good"


def test_inconsistent_labels():
    p = profile_dataset(pd.DataFrame({"c": ["North", "north ", "South", "South", "North", "South"]}))
    assert "inconsistent_labels" in codes(p)


def test_blank_and_placeholder_not_counted_as_missing():
    p = profile_dataset(pd.DataFrame({"c": ["a", "b", " ", "-", "a", "b"]}))
    c = col(p, "c")
    assert c["missing_count"] == 0
    assert c["blank_count"] == 1
    assert c["placeholder_count"] == 1
    assert {"blank_strings", "placeholder_values"} <= codes(p)


# ---------- safety ----------

def test_profile_does_not_modify_dataframe():
    df = pd.DataFrame({
        "a": [1, None, 3, 4, 5, 6, 7, 8],
        "d": ["2020-01-01", "2020-02-01", None, "2020-04-01", "2020-05-01",
              "2020-06-01", "2020-07-01", "2020-08-01"],
    })
    before = df.copy()
    profile_dataset(df)
    pd.testing.assert_frame_equal(df, before)


def test_profile_is_json_serializable():
    df = pd.DataFrame({
        "n": [1.5, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "d": pd.date_range("2022-01-01", periods=10),
        "c": list("aabbccddee"),
    })
    json.dumps(profile_dataset(df), allow_nan=False)  # raises on NaN/inf/non-serializable