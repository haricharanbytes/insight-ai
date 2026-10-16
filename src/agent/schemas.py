"""JSON schemas the LLM sees for each tool.

Enums are imported from the analysis modules, so the schemas cannot drift from what the
functions actually accept. tests/test_agent_schemas.py also checks them against the real
function signatures.
"""
from src.analysis.aggregation import AGG_OPS
from src.analysis.charting import CHART_TYPES
from src.analysis.correlations import METHODS as CORR_METHODS
from src.analysis.filtering import FILTER_OPS
from src.analysis.outliers import METHODS as OUTLIER_METHODS
from src.analysis.statistics import TESTS
from src.analysis.timeseries import FREQS, TS_OPS
from src.analysis.posthoc import CORRECTIONS, PAIR_METHODS
from src.analysis.regression import MODELS as REGRESSION_MODELS
from src.analysis.forecasting import METHODS as FORECAST_METHODS

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
    "generate_chart": {
        "description": "Draw a histogram (distribution of one numeric column), scatter plot (two numeric columns) or box plot (one numeric column, optionally per category). For bar and line charts use group_by_aggregate or analyze_time_series instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {"type": "string", "enum": list(CHART_TYPES)},
                "x": {"type": "string", "description": "Numeric column (the values for histogram/box, the x axis for scatter)."},
                "y": {"type": "string", "description": "Numeric column for the y axis (scatter only)."},
                "group_by": {"type": "string", "description": "Categorical column to split a box plot by."},
                "bins": {"type": "integer", "description": "Histogram bins, 5-60 (default 20)."},
                "filters": FILTERS,
            },
            "required": ["chart_type", "x"],
        },
    },
        "run_regression": {
        "description": "Explain a numeric target (linear) or a two-value target (logistic) from several predictors: coefficients, confidence intervals, p-values, fit and diagnostics. Association only, not causation; no predictions for new inputs.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Column to explain."},
                "predictors": _strs("1-10 predictor columns (numeric, boolean, or categorical with at most 8 values)."),
                "model": {"type": "string", "enum": list(REGRESSION_MODELS)},
                "filters": FILTERS,
                "alpha": {"type": "number", "description": "Significance level, default 0.05."},
            },
            "required": ["target", "predictors"],
        },
    },
    "forecast_series": {
        "description": "Forecast a value per period ahead, backtested against a naive baseline. Recommends a method only if it beat naive. Needs a date column and at least 12 periods.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_column": {"type": "string", "description": "Exact name of a date column."},
                "value_column": {"type": "string", "description": "Numeric column to forecast. Omit to forecast row counts."},
                "op": {"type": "string", "enum": list(TS_OPS)},
                "freq": {"type": "string", "enum": list(FREQS), "description": "D, W, M, Q or Y."},
                "horizon": {"type": "integer", "description": "Periods ahead; at most a third of the history."},
                "methods": {"type": "array", "items": {"type": "string", "enum": list(FORECAST_METHODS)}},
                "filters": FILTERS,
            },
            "required": ["date_column"],
        },
    },
    "cluster_data": {
        "description": "Group rows into clusters with k-means on standardized numeric columns; returns cluster sizes and per-cluster means. Clusters are a summary, not natural groups.",
        "parameters": {
            "type": "object",
            "properties": {
                "columns": _strs("2-10 numeric columns (optional; default all numeric non-identifier columns)."),
                "k": {"type": "integer", "description": "Number of clusters 2-8; omit to choose automatically."},
                "filters": FILTERS,
            },
            "required": [],
        },
    },
    "detect_anomalies": {
        "description": "Find rows that are unusual across several numeric columns at once (Isolation Forest). The number flagged is a setting, not a finding.",
        "parameters": {
            "type": "object",
            "properties": {
                "columns": _strs("2-10 numeric columns (optional)."),
                "contamination": {"type": "number", "description": "Share of rows to flag, 0.001-0.2 (default 0.02)."},
                "filters": FILTERS,
                "limit": {"type": "integer", "description": "Rows listed, max 20."},
            },
            "required": [],
        },
    },
    "check_test_assumptions": {
        "description": "Check normality (Shapiro-Wilk) and equal variances (Levene) of a numeric column across groups, and recommend a parametric or rank-based test.",
        "parameters": {
            "type": "object",
            "properties": {
                "value_column": {"type": "string", "description": "Numeric column."},
                "group_column": {"type": "string", "description": "Categorical column defining the groups."},
                "groups": _strs("Which group values to include (optional)."),
                "filters": FILTERS,
            },
            "required": ["value_column", "group_column"],
        },
    },
    "run_posthoc_tests": {
        "description": "Pairwise comparisons between all groups with multiple-comparison correction. Use after a significant anova or kruskal result.",
        "parameters": {
            "type": "object",
            "properties": {
                "value_column": {"type": "string", "description": "Numeric column."},
                "group_column": {"type": "string", "description": "Categorical column defining the groups."},
                "method": {"type": "string", "enum": list(PAIR_METHODS)},
                "correction": {"type": "string", "enum": list(CORRECTIONS)},
                "groups": _strs("Which group values to include (optional, max 12)."),
                "filters": FILTERS,
                "alpha": {"type": "number", "description": "Significance level, default 0.05."},
            },
            "required": ["value_column", "group_column"],
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