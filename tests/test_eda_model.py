import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.common import AnalysisError
from src.reports.eda import SECTIONS, build_eda_report

rng = np.random.default_rng(0)
N = 120
DF = pd.DataFrame({
    "Order ID": [f"O{i}" for i in range(N)],
    "Region": ["East", "West", "North", "South"] * 30,
    "Segment": ["A", "B", "C"] * 40,
    "Sales": [100.0 + i * 3 + (i % 7) for i in range(N)],
    "Profit": [10.0 + (i % 11) for i in range(N)],
    "Order Date": [f"{2019 + i // 40}-{(i % 12) + 1:02d}-10" for i in range(N)],
})


def sec(report, key):
    return next(s for s in report["sections"] if s["key"] == key)


def test_full_report_has_every_section_and_is_serialisable():
    r = build_eda_report(DF, "d.csv")
    assert [s["key"] for s in r["sections"]] == list(SECTIONS)
    assert all(s["ok"] for s in r["sections"])
    json.dumps(r, allow_nan=False)
    assert r["summary_source"] == "rules" and r["meta"]["main_numeric_column"] == "Sales"


def test_no_recommendations_anywhere():
    text = json.dumps(build_eda_report(DF, "d.csv")).lower()
    assert "recommend" in text                       # only in the disclaimer note
    assert text.count("recommend") == 1
    assert "should " not in " ".join(build_eda_report(DF, "d.csv")["summary"]).lower()


def test_numbers_in_findings_match_pandas():
    r = build_eda_report(DF, "d.csv")
    line = next(f for f in sec(r, "numeric")["findings"] if f.startswith("Sales:"))
    assert f"mean {DF['Sales'].mean():,.2f}" in line and f"median {DF['Sales'].median():,.2f}" in line
    cat = next(f for f in sec(r, "categories")["findings"] if f.startswith("Segment:"))
    assert "'A' with 40 rows" in cat or "'B' with 40 rows" in cat or "'C' with 40 rows" in cat


def test_sections_carry_provenance_charts_and_caveats():
    r = build_eda_report(DF, "d.csv")
    assert sec(r, "trends")["sources"][0]["tool"] == "analyze_time_series" and sec(r, "trends")["charts"]
    assert sec(r, "correlations")["charts"][0]["kind"] == "heatmap"
    assert "association, not causation" in sec(r, "correlations")["caveat"]
    assert len(sec(r, "numeric")["charts"]) == 2          # Sales and Profit histograms


def test_options_filter_title_author_and_sections():
    r = build_eda_report(DF, "d.csv", {"title": "My <b>report</b>", "author": "Asha", "sections": ["overview", "numeric"],
                                       "filters": [{"column": "Region", "op": "eq", "value": "East"}]})
    assert [s["key"] for s in r["sections"]] == ["overview", "numeric"]
    assert r["meta"]["rows"] == 30 and r["meta"]["author"] == "Asha" and "<b>" in r["meta"]["title"]
    assert "after filters; 120 before" in sec(r, "overview")["findings"][0]


def test_failing_section_becomes_a_note_not_a_failed_report():
    cats_only = pd.DataFrame({"g": list("aabbccaabb") * 5, "h": list("xyxyxyxyxy") * 5})
    r = build_eda_report(cats_only, "c.csv")
    assert sec(r, "numeric")["ok"] is False and sec(r, "trends")["ok"] is False
    assert sec(r, "overview")["ok"] and sec(r, "categories")["ok"]
    assert sec(r, "numeric")["warnings"]


def test_refusals():
    with pytest.raises(AnalysisError) as e:
        build_eda_report(DF, "d.csv", {"sections": ["nope"]})
    assert e.value.code == "invalid_parameter"
    with pytest.raises(AnalysisError) as e:
        build_eda_report(DF, "d.csv", {"filters": [{"column": "Region", "op": "eq", "value": "Mars"}]})
    assert e.value.code == "no_rows"


def test_input_not_modified():
    before = DF.copy()
    build_eda_report(DF, "d.csv")
    pd.testing.assert_frame_equal(DF, before)