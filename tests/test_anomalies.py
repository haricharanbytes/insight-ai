import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.anomalies import detect_anomalies
from src.analysis.common import AnalysisError


def planted(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n), "c": rng.normal(50, 5, n)})
    df.loc[[10, 20, 30], ["a", "b"]] = [[9.0, 9.0], [-9.0, 8.5], [9.5, -9.0]]     # jointly extreme
    df.loc[40, "c"] = 120.0                                                          # extreme in one column
    return df


DF = planted()
PLANTED_ROWS = {11, 21, 31, 41}      # 1-based row numbers


def code_of(df, **kw):
    with pytest.raises(AnalysisError) as exc:
        detect_anomalies(df, **kw)
    return exc.value.code


def test_planted_anomalies_are_found():
    out = detect_anomalies(DF, contamination=0.02, limit=10)
    t = out["table"]
    listed = {r[t["columns"].index("row_number")] for r in t["rows"]}
    assert PLANTED_ROWS <= listed
    assert out["result"]["n_flagged"] == pytest.approx(0.02 * 300, abs=2)
    assert out["rows_used"] == 300


def test_driven_by_names_the_deviating_column():
    out = detect_anomalies(DF, contamination=0.02, limit=10)
    cols = out["table"]["columns"]
    by_row = {r[cols.index("row_number")]: r[cols.index("driven_by")] for r in out["table"]["rows"]}
    assert by_row[41].startswith("c (+")
    assert "a (" in by_row[11] or "b (" in by_row[11]


def test_rows_are_ranked_by_score_descending():
    out = detect_anomalies(DF, contamination=0.05, limit=20)
    i = out["table"]["columns"].index("anomaly_score")
    scores = [r[i] for r in out["table"]["rows"]]
    assert scores == sorted(scores, reverse=True)


def test_row_numbers_use_original_positions_after_filtering_missing():
    d = DF.copy()
    d.loc[0:4, "a"] = np.nan
    out = detect_anomalies(d, contamination=0.02)
    assert out["rows_used"] == 295 and any("5 rows" in w for w in out["warnings"])
    cols = out["table"]["columns"]
    assert {r[cols.index("row_number")] for r in out["table"]["rows"]} >= {21, 31, 41}


def test_caveat_and_reproducible_and_serialisable():
    a, b = detect_anomalies(DF), detect_anomalies(DF)
    assert a["table"]["rows"] == b["table"]["rows"]
    assert "not necessarily errors" in a["result"]["note"] and "setting" in a["result"]["note"]
    json.dumps(a, allow_nan=False)


def test_identifier_columns_ignored_and_limit_capped():
    d = DF.assign(**{"Row ID": range(300)})
    out = detect_anomalies(d, limit=500)
    assert out["result"]["columns"] == ["a", "b", "c"]
    assert any("Row ID" in w for w in out["warnings"]) and any("reduced" in w for w in out["warnings"])


def test_refusals():
    assert code_of(DF.head(30)) == "insufficient_data"
    assert code_of(DF, contamination=0.5) == "invalid_parameter"
    assert code_of(DF, contamination=0) == "invalid_parameter"
    assert code_of(DF, columns=["a"]) == "invalid_parameter"
    assert code_of(DF, columns=["a", "nope"]) == "column_not_found"
    assert code_of(DF.assign(t="x"), columns=["a", "t"]) == "invalid_column_type"
    assert code_of(DF.assign(k=1.0), columns=["a", "k"]) == "no_variation"
    assert code_of(DF, filters=[{"column": "a", "op": "gt", "value": 1e6}]) == "no_rows"
    assert code_of(DF, limit=0) == "invalid_parameter"


def test_input_not_modified():
    before = DF.copy()
    detect_anomalies(DF)
    pd.testing.assert_frame_equal(DF, before)