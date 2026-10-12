"""JSON schemas the LLM sees for each tool.

Enums are imported from the analysis modules, so the schemas cannot drift from what the
functions actually accept. tests/test_agent_schemas.py also checks them against the real
function signatures.
"""
from src.analysis.aggregation import AGG_OPS
from src.analysis.correlations import METHODS as CORR_METHODS
from src.analysis.filtering import FILTER_OPS
from src.analysis.outliers import METHODS as OUTLIER_METHODS
from src.analysis.statistics import TESTS
from src.analysis.timeseries import FREQS, TS_OPS

STR = {"type": "string"}


def _strs(description: str) -> dict:
    return {"type": "array", "items": STR, "description": description}


FILTERS = {
    "type": "array",
    "description": "Row filters, combined with AND. Text matching is exact and case-sensitive.",
    "items": {
        "type": "object",
        "properties": {
            "column": {"type": "string", "description": "Exact column name from the dataset."},
            "op": {"type": "string", "enum": list(FILTER_OPS)},
            "value": {"description": "Value to compare. Use a [low, high] list for 'between' and a list for 'in'/'not_in'. Omit for is_null/not_null."},
        },
        "required": ["column", "op"],
    },
}

METRICS = {
    "type": "array",
    "description": "What to calculate per group.",
    "items": {
        "type": "object",
        "properties": {
            "column": {"type": "string", "description": "Exact column name, or '*' with op 'count' to count rows."},
            "op": {"type": "string", "enum": list(AGG_OPS)},
        },
        "required": ["column", "op"],
    },
}

_COL = "Exact column name from the dataset."

TOOL_SCHEMAS: dict[str, dict] = {
    "get_column_summary": {
        "description": "Descriptive statistics for ONE column (counts, missing values, mean/median/min/max for numbers, top values for categories).",
        "parameters": {
            "type": "object",
            "properties": {"column": {"type": "string", "description": _COL}, "filters": FILTERS},
            "required": ["column"],
        },
    },
    "filter_data": {
        "description": "Count and preview the rows that match filters. Use for 'how many orders where...' questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "filters": FILTERS,
                "columns": _strs("Columns to show in the preview (optional)."),
                "limit": {"type": "integer", "description": "Preview rows, max 50."},
            },
            "required": ["filters"],
        },
    },
    "group_by_aggregate": {
        "description": "Totals, averages, counts, etc. per category, sorted. Use for rankings and 'which X has the most/least Y', and for overall totals (leave group_by empty).",
        "parameters": {
            "type": "object",
            "properties": {
                "group_by": _strs("Columns to group by (max 3). Empty for one overall result."),
                "metrics": METRICS,
                "filters": FILTERS,
                "sort_by": {"type": "string", "description": "Result column to sort by, e.g. 'sum_Sales'. Defaults to the first metric."},
                "ascending": {"type": "boolean", "description": "Sort ascending (default false = largest first)."},
                "limit": {"type": "integer", "description": "Max rows returned (default 20)."},
            },
            "required": ["metrics"],
        },
    },
    "analyze_time_series": {
        "description": "How a value changes over time: per-period totals/averages/counts, period-over-period change, peak, lowest and a trend test. Needs a date column.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_column": {"type": "string", "description": "Exact name of a date column."},
                "value_column": {"type": "string", "description": "Numeric column to aggregate. Omit to count rows per period."},
                "op": {"type": "string", "enum": list(TS_OPS)},
                "freq": {"type": "string", "enum": list(FREQS), "description": "D=day, W=week, M=month, Q=quarter, Y=year."},
                "filters": FILTERS,
            },
            "required": ["date_column"],
        },
    },
    "calculate_correlation": {
        "description": "Pairwise correlations between numeric columns. Association only, not causation. Omit 'columns' to use all numeric columns.",
        "parameters": {
            "type": "object",
            "properties": {
                "columns": _strs("2-15 numeric columns (optional)."),
                "method": {"type": "string", "enum": list(CORR_METHODS)},
                "filters": FILTERS,
                "limit": {"type": "integer", "description": "Max pairs returned."},
            },
            "required": [],
        },
    },
    "detect_outliers": {
        "description": "Flag statistically unusual values in a numeric column. Indicators only; nothing is removed.",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": _COL},
                "method": {"type": "string", "enum": list(OUTLIER_METHODS)},
                "threshold": {"type": "number", "description": "Multiplier (IQR default 1.5) or z-score cutoff (default 3)."},
                "filters": FILTERS,
                "limit": {"type": "integer", "description": "Max flagged rows listed, max 50."},
            },
            "required": ["column"],
        },
    },
    "run_statistical_test": {
        "description": (
            "Hypothesis test. ttest/mannwhitney: compare a numeric column between exactly two groups. "
            "anova/kruskal: compare it across two or more groups. chi_square: association between two categorical columns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "test": {"type": "string", "enum": list(TESTS)},
                "value_column": {"type": "string", "description": "Numeric column (not for chi_square)."},
                "group_column": {"type": "string", "description": "Categorical column defining the groups (not for chi_square)."},
                "groups": _strs("Which group values to compare (required for ttest/mannwhitney if the column has more than two)."),
                "column_a": {"type": "string", "description": "First categorical column (chi_square only)."},
                "column_b": {"type": "string", "description": "Second categorical column (chi_square only)."},
                "filters": FILTERS,
                "alpha": {"type": "number", "description": "Significance level, default 0.05."},
            },
            "required": ["test"],
        },
    },
}


def openai_tools() -> list[dict]:
    """Tool definitions in the OpenAI/OpenRouter 'tools' format."""
    return [
        {"type": "function",
         "function": {"name": name, "description": spec["description"], "parameters": spec["parameters"]}}
        for name, spec in TOOL_SCHEMAS.items()
    ]