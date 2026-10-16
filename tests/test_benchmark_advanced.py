import numpy as np
import pandas as pd
import pytest

from src.analysis.clustering import cluster_data
from src.analysis.forecasting import forecast_series
from src.analysis.posthoc import run_posthoc_tests
from src.analysis.regression import run_regression

rng = np.random.default_rng(42)
N = 200
X1, X2 = rng.normal(0, 1, N), rng.normal(0, 1, N)
Y = 4.0 + 3.0 * X1 - 2.0 * X2 + rng.normal(0, 0.5, N)
REG = pd.DataFrame({"x1": X1, "x2": X2, "y": Y})


def test_regression_matches_normal_equations():
    # Independent route: solve (X'X) b = X'y directly, no statsmodels.
    A = np.column_stack([np.ones(N), X1, X2])
    beta = np.linalg.solve(A.T @ A, A.T @ Y)
    out = run_regression(REG, "y", ["x1", "x2"])
    t = {r[0]: r[1] for r in out["table"]["rows"]}
    assert [t["const"], t["x1"], t["x2"]] == pytest.approx(list(beta), rel=1e-9)
    ss_res = float(((Y - A @ beta) ** 2).sum())
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    assert out["result"]["r_squared"] == pytest.approx(1 - ss_res / ss_tot, rel=1e-9)


def test_forecast_naive_backtest_by_hand():
    vals = [100 + 3 * i + (5 if i % 2 else -5) for i in range(36)]
    d = pd.DataFrame({"d": [f"{p.start_time.date()}" for p in pd.period_range("2019-01", periods=36, freq="M")], "v": vals})
    out = forecast_series(d, "d", "v", horizon=4, methods=["naive"])
    h = out["result"]["holdout_periods"]
    expected = sum(abs(vals[-h + j] - vals[-h - 1]) for j in range(h)) / h
    assert out["result"]["backtest"]["naive"]["mae"] == pytest.approx(expected)


def test_clustering_recovers_known_groups_exactly():
    pts = np.vstack([rng.normal(c, 0.3, (50, 2)) for c in [(0, 0), (8, 8), (16, 0)]])
    truth = np.repeat([0, 1, 2], 50)
    out = cluster_data(pd.DataFrame(pts, columns=["a", "b"]))
    assert out["result"]["k"] == 3 and sorted(out["result"]["sizes"]) == [50, 50, 50]
    # purity: each true group lands in one cluster (checked via cluster means order)
    means = sorted(round(r[out["table"]["columns"].index("mean_a")]) for r in out["table"]["rows"])
    assert means == [0, 8, 16]


def test_posthoc_holm_by_hand():
    base = rng.normal(0, 1, 30)
    d = pd.DataFrame({"g": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
                      "v": np.concatenate([base, base[::-1] + 0.01, base + 3])})
    out = run_posthoc_tests(d, "v", "g")
    rows = [dict(zip(out["table"]["columns"], r)) for r in out["table"]["rows"]]
    raw = sorted(r["p_raw"] for r in rows)
    holm = np.maximum.accumulate([min(1, raw[0] * 3), min(1, raw[1] * 2), min(1, raw[2])])
    assert sorted(r["p_adjusted"] for r in rows) == pytest.approx(list(holm))