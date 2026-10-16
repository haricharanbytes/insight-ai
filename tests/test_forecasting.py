import numpy as np
import pandas as pd
import pytest

from src.analysis.common import AnalysisError
from src.analysis.forecasting import forecast_series

N = 48
months = pd.period_range("2018-01", periods=N, freq="M")


def frame(values):
    return pd.DataFrame({"d": [f"{p.start_time.date()}" for p in months[: len(values)]], "v": values})


def code_of(df, **kw):
    with pytest.raises(AnalysisError) as exc:
        forecast_series(df, "d", "v", **kw)
    return exc.value.code


LINEAR = frame([100.0 + 5.0 * i for i in range(N)])
SEASONAL = frame([100 + 2.0 * i + 30 * np.sin(2 * np.pi * (i % 12) / 12) for i in range(N)])


def test_linear_trend_is_forecast_well_and_beats_naive():
    out = forecast_series(LINEAR, "d", "v", horizon=6)
    res = out["result"]
    assert res["recommended"] in ("holt_winters", "arima")
    assert res["backtest"][res["recommended"]]["mae"] < res["naive_mae"]
    first = out["table"]["rows"][0]
    cols = out["table"]["columns"]
    row = dict(zip(cols, first))
    assert row["period"] == "2022-01"
    assert row[res["recommended"]] == pytest.approx(100 + 5 * 48, abs=3)      # next point on the line
    assert row["low_95"] <= row[res["recommended"]] <= row["high_95"]


def test_naive_and_seasonal_naive_match_hand_calculation():
    out = forecast_series(SEASONAL, "d", "v", horizon=3, methods=["naive", "seasonal_naive"])
    cols = out["table"]["columns"]
    first = dict(zip(cols, out["table"]["rows"][0]))
    y = SEASONAL["v"].to_numpy()
    assert first["naive"] == pytest.approx(y[-1])
    assert first["seasonal_naive"] == pytest.approx(y[-12])


def test_backtest_scores_match_numpy():
    out = forecast_series(LINEAR, "d", "v", horizon=6, methods=["naive"])
    y = LINEAR["v"].to_numpy()
    h = out["result"]["holdout_periods"]
    expected_mae = np.mean(np.abs(y[-h:] - y[-h - 1]))
    assert out["result"]["backtest"]["naive"]["mae"] == pytest.approx(expected_mae)


def test_random_walk_noise_recommends_nothing():
    rng = np.random.default_rng(0)
    noise = frame(list(100 + rng.normal(0, 10, N)))
    out = forecast_series(noise, "d", "v", horizon=6)
    assert out["result"]["recommended"] is None or \
        out["result"]["backtest"][out["result"]["recommended"]]["mae"] < out["result"]["naive_mae"]
    if out["result"]["recommended"] is None:
        assert "No method beat the naive baseline" in out["result"]["verdict"]


def test_caveats_and_serialisable():
    import json
    out = forecast_series(SEASONAL, "d", "v", horizon=6)
    assert "not a guarantee" in out["result"]["note"]
    assert any("single split" in w for w in out["warnings"])
    json.dumps(out, allow_nan=False)


def test_partial_last_period_is_warned():
    d = frame([100.0 + i for i in range(N)])
    d.loc[N - 1, "d"] = "2021-12-05"
    out = forecast_series(d, "d", "v", horizon=3)
    assert any("only partly covered" in w for w in out["warnings"])


def test_refusals():
    assert code_of(frame([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), horizon=2) == "insufficient_data"
    assert code_of(LINEAR, horizon=20) == "invalid_parameter"
    assert code_of(LINEAR, horizon=0) == "invalid_parameter"
    assert code_of(LINEAR, freq="X") == "invalid_parameter"
    assert code_of(LINEAR, methods=["prophet"]) == "unsupported_operation"
    assert code_of(frame([5.0] * N)) == "no_variation"
    with pytest.raises(AnalysisError):
        forecast_series(LINEAR, "v", "v")          # numeric column as a date
    with pytest.raises(AnalysisError):
        forecast_series(LINEAR, "d", "nope")


def test_input_not_modified():
    before = LINEAR.copy()
    forecast_series(LINEAR, "d", "v", horizon=3)
    pd.testing.assert_frame_equal(LINEAR, before)