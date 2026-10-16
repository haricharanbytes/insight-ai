import numpy as np
import pandas as pd
import pytest

from src.analysis.common import AnalysisError
from src.analysis.regression import run_regression

N = 60
x1 = np.arange(N, dtype=float)
x2 = (np.arange(N) % 7).astype(float)
DF = pd.DataFrame({
    "x1": x1, "x2": x2,
    "y": 3.0 + 2.0 * x1 - 1.5 * x2 + np.where(np.arange(N) % 2 == 0, 0.5, -0.5),   # small alternating noise
    "grp": ["a", "b", "c"] * 20,
    "label": ["yes" if (i * 7 + i // 3) % 5 < 2 else "no" for i in range(N)],
})


def rows(out):
    t = out["table"]
    return {r[0]: dict(zip(t["columns"], r)) for r in t["rows"]}


def code_of(**kw):
    with pytest.raises(AnalysisError) as exc:
        run_regression(DF, **kw)
    return exc.value.code


def test_ols_recovers_known_coefficients():
    out = run_regression(DF, "y", ["x1", "x2"])
    t = rows(out)
    assert t["x1"]["coefficient"] == pytest.approx(2.0, abs=0.02)
    assert t["x2"]["coefficient"] == pytest.approx(-1.5, abs=0.1)
    assert t["x1"]["ci_low"] < 2.0 < t["x1"]["ci_high"]
    assert out["result"]["r_squared"] > 0.999 and out["rows_used"] == N
    assert set(out["result"]["significant_terms"]) == {"x1", "x2"}


def test_matches_numpy_least_squares():
    out = run_regression(DF, "y", ["x1", "x2"])
    X = np.column_stack([np.ones(N), x1, x2])
    beta = np.linalg.lstsq(X, DF["y"].to_numpy(), rcond=None)[0]
    t = rows(out)
    assert [t["const"]["coefficient"], t["x1"]["coefficient"], t["x2"]["coefficient"]] == pytest.approx(list(beta))


def test_categorical_predictor_becomes_indicators():
    out = run_regression(DF, "y", ["x1", "grp"])
    assert {"grp=b", "grp=c"} <= set(rows(out))
    assert any("reference category is 'a'" in w for w in out["warnings"])


def test_collinearity_is_warned_and_perfect_collinearity_refused():
    d = DF.assign(x1_copy=DF["x1"] * 2 + 1e-6 * np.arange(N) ** 2)
    out = run_regression(d, "y", ["x1", "x1_copy"])
    assert any("VIF" in w for w in out["warnings"])
    exact = DF.assign(dup=DF["x1"])
    assert code_of.__wrapped__ if False else True
    with pytest.raises(AnalysisError) as exc:
        run_regression(exact, "y", ["x1", "dup"])
    assert exc.value.code in ("test_failed", "no_variation")


def test_logistic_odds_and_caveats():
    out = run_regression(DF, "label", ["x1", "x2"], model="logistic")
    t = rows(out)
    assert t["x1"]["odds_ratio"] == pytest.approx(np.exp(t["x1"]["coefficient"]))
    assert 0 <= out["result"]["accuracy_in_sample"] <= 1
    assert any("overstates" in w for w in out["warnings"])
    assert any("'yes'" in w for w in out["warnings"])      # positive class is named


def test_missing_rows_are_dropped_and_reported():
    d = DF.copy()
    d.loc[0:2, "x2"] = np.nan
    out = run_regression(d, "y", ["x1", "x2"])
    assert out["rows_used"] == N - 3 and any("3 rows" in w for w in out["warnings"])


def test_filters_apply():
    out = run_regression(DF, "y", ["x1"], filters=[{"column": "grp", "op": "eq", "value": "a"}])
    assert out["rows_used"] == 20


def test_refusals():
    assert code_of(target="y", predictors=["x1"], model="ridge") == "unsupported_operation"
    assert code_of(target="grp", predictors=["x1"]) == "invalid_column_type"
    assert code_of(target="grp", predictors=["x1"], model="logistic") == "invalid_column_type"   # 3 classes
    assert code_of(target="y", predictors=["y"]) == "invalid_parameter"
    assert code_of(target="y", predictors=[]) == "invalid_parameter"
    assert code_of(target="y", predictors=["nope"]) == "column_not_found"
    assert code_of(target="y", predictors=["x1", "x2", "grp"],
                   filters=[{"column": "x1", "op": "lt", "value": 10}]) == "insufficient_data"
    flat = DF.assign(c=1.0)
    with pytest.raises(AnalysisError) as exc:
        run_regression(flat, "y", ["x1", "c"])
    assert exc.value.code == "no_variation"


def test_input_not_modified():
    before = DF.copy()
    run_regression(DF, "y", ["x1", "grp"])
    pd.testing.assert_frame_equal(DF, before)