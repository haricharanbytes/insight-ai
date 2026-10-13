"""generate_chart tool: histogram / scatter / box built from raw columns with strict limits."""
import numpy as np
import pandas as pd

from src.visualization.charts import BLUE, GREY, NAVY, _chart, _layout

from .common import AnalysisError, make_result, require_columns, require_numeric
from .filtering import apply_filters

CHART_TYPES = ("histogram", "scatter", "box")
MAX_SCATTER_POINTS = 2000
MAX_BOX_GROUPS = 12
MIN_POINTS = 3


def _finite(series: pd.Series) -> pd.Series:
    v = pd.Series(series.to_numpy(dtype=float, na_value=np.nan), index=series.index)
    return v[np.isfinite(v)]


def generate_chart(df: pd.DataFrame, chart_type: str, x: str, y: str | None = None,
                   group_by: str | None = None, bins: int = 20, filters=None) -> dict:
    """Return a tool output whose result['chart'] holds a ready-to-draw chart spec."""
    warnings: list[str] = []
    chart_type = str(chart_type).lower()
    if chart_type not in CHART_TYPES:
        raise AnalysisError(
            f"Unsupported chart_type '{chart_type}'. Supported: {', '.join(CHART_TYPES)}. "
            "For bar and line charts use group_by_aggregate or analyze_time_series.",
            "unsupported_operation", {"supported": list(CHART_TYPES)})

    needed = [x] + ([y] if y else []) + ([group_by] if group_by else [])
    require_columns(df, needed)
    require_numeric(df, x)
    if chart_type == "scatter":
        if not y:
            raise AnalysisError("A scatter chart needs both x and y.", "invalid_parameter")
        if x == y:
            raise AnalysisError("x and y must be different columns.", "invalid_parameter")
        require_numeric(df, y)
    if chart_type != "box" and group_by:
        raise AnalysisError("group_by is only supported for box charts.", "invalid_parameter")
    if chart_type != "scatter" and y:
        raise AnalysisError("y is only used for scatter charts.", "invalid_parameter")
    if isinstance(bins, bool) or not isinstance(bins, int) or not 5 <= bins <= 60:
        raise AnalysisError("'bins' must be an integer between 5 and 60.", "invalid_parameter")

    work, applied, fw = apply_filters(df, filters)
    warnings += fw
    if len(work) == 0:
        raise AnalysisError("No rows remain after filtering.", "no_rows", {"filters": applied})

    inputs = {"chart_type": chart_type, "x": x, "y": y, "group_by": group_by,
              "bins": bins, "filters": applied}

    if chart_type == "histogram":
        vals = _finite(work[x])
        dropped = len(work) - len(vals)
        if dropped:
            warnings.append(f"{dropped:,} rows with a missing or non-finite {x} were excluded.")
        if len(vals) < MIN_POINTS:
            raise AnalysisError(f"Need at least {MIN_POINTS} values in '{x}'.", "insufficient_data")
        counts, edges = np.histogram(vals.to_numpy(), bins=bins)
        title = f"Distribution of {x}"
        spec = _chart("histogram", title,
                      f"How many rows fall in each range of {x} ({bins} equal-width bins, {len(vals):,} values).",
                      [{"type": "bar", "x": [float((a + b) / 2) for a, b in zip(edges[:-1], edges[1:])],
                        "y": [int(c) for c in counts], "width": [float(b - a) for a, b in zip(edges[:-1], edges[1:])],
                        "marker": {"color": BLUE}, "customdata": [[float(a), float(b)] for a, b in zip(edges[:-1], edges[1:])],
                        "hovertemplate": "%{customdata[0]:,.2f} to %{customdata[1]:,.2f}<br>%{y:,} rows<extra></extra>"}],
                      _layout(title, x, "Rows", bargap=0.02))
        result = {"n_values": int(len(vals)), "bins": bins, "min": float(vals.min()), "max": float(vals.max()),
                  "counts": [int(c) for c in counts], "bin_edges": [float(e) for e in edges]}
        used = len(vals)

    elif chart_type == "scatter":
        sub = pd.DataFrame({x: _finite(work[x]), y: _finite(work[y])}).dropna()
        dropped = len(work) - len(sub)
        if dropped:
            warnings.append(f"{dropped:,} rows with a missing value in {x} or {y} were excluded.")
        if len(sub) < MIN_POINTS:
            raise AnalysisError("Need at least 3 rows with both values.", "insufficient_data")
        total = len(sub)
        if total > MAX_SCATTER_POINTS:
            sub = sub.sample(MAX_SCATTER_POINTS, random_state=0)
            warnings.append(f"Showing a random sample of {MAX_SCATTER_POINTS:,} of {total:,} points.")
        title = f"{y} vs {x}"
        spec = _chart("scatter", title,
                      f"Each point is one row. Shows association between {x} and {y}, not causation."
                      + (f" Random sample of {MAX_SCATTER_POINTS:,} of {total:,} rows." if total > MAX_SCATTER_POINTS else ""),
                      [{"type": "scattergl", "mode": "markers", "x": sub[x].tolist(), "y": sub[y].tolist(),
                        "marker": {"color": BLUE, "size": 5, "opacity": 0.55},
                        "hovertemplate": f"{x}: %{{x:,.2f}}<br>{y}: %{{y:,.2f}}<extra></extra>"}],
                      _layout(title, x, y))
        result = {"n_points_total": int(total), "n_points_drawn": int(len(sub)), "sampled": total > MAX_SCATTER_POINTS}
        used = total

    else:  # box
        title = f"Distribution of {x}" + (f" by {group_by}" if group_by else "")
        if group_by:
            sub = pd.DataFrame({"g": work[group_by].astype(str), "v": _finite(work[x])}).dropna()
            groups = sorted(sub["g"].unique())
            if len(groups) > MAX_BOX_GROUPS:
                top = sub["g"].value_counts().head(MAX_BOX_GROUPS).index
                sub = sub[sub["g"].isin(top)]
                groups = sorted(top)
                warnings.append(f"'{group_by}' has many values; showing the {MAX_BOX_GROUPS} with the most rows.")
            data = [{"type": "box", "name": g, "y": sub.loc[sub["g"] == g, "v"].tolist(), "boxpoints": "outliers",
                     "marker": {"color": BLUE}} for g in groups]
            result = {"n_values": int(len(sub)), "groups": groups}
            used = len(sub)
        else:
            vals = _finite(work[x])
            data = [{"type": "box", "name": x, "y": vals.tolist(), "boxpoints": "outliers", "marker": {"color": BLUE}}]
            result = {"n_values": int(len(vals))}
            used = len(vals)
        if used < MIN_POINTS:
            raise AnalysisError("Not enough values to draw a box plot.", "insufficient_data")
        spec = _chart("box", title, "Box = middle 50% of values, line = median, dots = unusual values.",
                      data, _layout(title, group_by or "", x, showlegend=False))

    return make_result(
        "generate_chart",
        method=f"{chart_type} drawn directly from the dataset's '{x}'"
               + (f" and '{y}'" if y else "") + " values; missing values are excluded.",
        inputs=inputs, rows_used=used, result={**result, "chart": spec}, table=None, warnings=warnings)