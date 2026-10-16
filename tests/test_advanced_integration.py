import json

import numpy as np
import pandas as pd

from src.agent.schemas import TOOL_SCHEMAS, openai_tools
from src.agent.tools import TOOLS, run_tool
from src.visualization.charts import chart_for_output

rng = np.random.default_rng(0)
N = 72
DF = pd.DataFrame({
    "d": [f"{p.start_time.date()}" for p in pd.period_range("2018-01", periods=N, freq="M")],
    "a": rng.normal(0, 1, N), "b": rng.normal(5, 2, N), "c": rng.normal(10, 3, N),
    "g": ["x", "y", "z"] * 24,
    "v": np.concatenate([rng.normal(10, 1, 24), rng.normal(10, 1, 24), rng.normal(13, 1, 24)]),
})
DF["v"] = DF["v"].to_numpy()
DF["y"] = 2 * DF["a"] + DF["b"] + rng.normal(0, 0.1, N)

CALLS = {
    "run_regression": {"target": "y", "predictors": ["a", "b"]},
    "forecast_series": {"date_column": "d", "value_column": "c", "horizon": 6, "methods": ["naive", "holt_winters"]},
    "cluster_data": {"columns": ["a", "b"]},
    "detect_anomalies": {"columns": ["a", "b"]},
    "check_test_assumptions": {"value_column": "v", "group_column": "g"},
    "run_posthoc_tests": {"value_column": "v", "group_column": "g"},
}


def test_all_new_tools_registered_with_schemas():
    assert set(CALLS) <= set(TOOLS) and set(CALLS) <= set(TOOL_SCHEMAS)
    assert len(openai_tools()) == len(TOOLS)


def test_each_new_tool_runs_through_the_registry_and_serialises():
    for name, params in CALLS.items():
        out = run_tool(DF, name, params)
        assert out["tool"] == name and out["warnings"] is not None
        json.dumps(out, allow_nan=False)


def test_charts_exist_for_the_tools_that_have_them():
    for name in ("run_regression", "forecast_series", "cluster_data", "detect_anomalies", "run_posthoc_tests"):
        c = chart_for_output(run_tool(DF, name, CALLS[name]))
        assert c is not None and c["caption"], name
        json.dumps(c, allow_nan=False)
    assert chart_for_output(run_tool(DF, "check_test_assumptions", CALLS["check_test_assumptions"])) is None