"""Tool registry: the deterministic functions the agent (and the API) can call.

Step 6 extends this with explicit JSON input schemas for the LLM.
"""
import inspect

import pandas as pd

from src.analysis.aggregation import group_by_aggregate
from src.analysis.common import AnalysisError
from src.analysis.correlations import calculate_correlation
from src.analysis.filtering import filter_data
from src.analysis.outliers import detect_outliers
from src.analysis.statistics import get_column_summary, run_statistical_test
from src.analysis.timeseries import analyze_time_series
from src.analysis.charting import generate_chart

TOOLS = {
    "get_column_summary": get_column_summary,
    "filter_data": filter_data,
    "group_by_aggregate": group_by_aggregate,
    "analyze_time_series": analyze_time_series,
    "calculate_correlation": calculate_correlation,
    "detect_outliers": detect_outliers,
    "run_statistical_test": run_statistical_test,
    "generate_chart": generate_chart,
}


def list_tools() -> list[str]:
    return sorted(TOOLS)


def run_tool(df: pd.DataFrame, name: str, params: dict) -> dict:
    """Validate the call shape, then run the tool on df. Never modifies df."""
    fn = TOOLS.get(name)
    if fn is None:
        raise AnalysisError(f"Unknown tool '{name}'.", "unknown_tool", {"available": list_tools()})
    if not isinstance(params, dict):
        raise AnalysisError("'params' must be an object.", "invalid_parameter")
    try:
        inspect.signature(fn).bind(df, **params)
    except TypeError as exc:
        raise AnalysisError(f"Invalid parameters for '{name}': {exc}", "invalid_parameter") from None
    return fn(df, **params)