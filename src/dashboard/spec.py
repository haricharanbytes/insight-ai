"""Dashboard spec: the validated JSON that describes a dashboard.

Shape is checked by pydantic; column names and types are checked against the dataset profile.
All problems are collected and reported together (so an LLM can repair a spec in one round).
Nothing here touches data or the LLM.
"""
import difflib
import math
from typing import Annotated, Any, Literal, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from src.analysis.aggregation import AGG_OPS, NUMERIC_ONLY
from src.analysis.common import AnalysisError
from src.analysis.correlations import METHODS as CORR_METHODS
from src.analysis.filtering import FILTER_OPS
from src.analysis.timeseries import FREQS, TS_OPS

MAX_WIDGETS = 20
MAX_GLOBAL_FILTERS = 6
MAX_LOCAL_FILTERS = 5
MAX_PROBLEMS = 25
CATEGORY_TYPES = {"categorical", "boolean"}
_NO_VALUE_OPS = {"is_null", "not_null"}
_ID = r"^[A-Za-z0-9_-]{1,40}$"


class SpecError(AnalysisError):
    """Raised when a dashboard spec is invalid. details['problems'] lists every problem found."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ building blocks

class Metric(_Base):
    column: str = Field(min_length=1, max_length=200)
    op: str

    @field_validator("op")
    @classmethod
    def _op(cls, v: str) -> str:
        v = v.lower()
        if v not in AGG_OPS:
            raise ValueError(f"unsupported aggregation '{v}'. Supported: {', '.join(AGG_OPS)}")
        return v


class LocalFilter(_Base):
    """A filter attached to one widget (same shape the analysis tools accept)."""
    column: str = Field(min_length=1, max_length=200)
    op: str
    value: Any = None

    @field_validator("op")
    @classmethod
    def _op(cls, v: str) -> str:
        v = v.lower()
        if v not in FILTER_OPS:
            raise ValueError(f"unsupported filter operator '{v}'. Supported: {', '.join(FILTER_OPS)}")
        return v

    @model_validator(mode="after")
    def _needs_value(self):
        if self.op not in _NO_VALUE_OPS and self.value is None:
            raise ValueError(f"filter '{self.op}' on '{self.column}' needs a value")
        return self

    def to_tool_filter(self) -> dict:
        out = {"column": self.column, "op": self.op}
        if self.op not in _NO_VALUE_OPS:
            out["value"] = self.value
        return out


# ------------------------------------------------------------------ widgets

class _WidgetBase(_Base):
    id: str | None = Field(default=None, pattern=_ID)
    title: str = Field(default="", max_length=120)
    size: Literal["s", "m", "l"] = "m"
    use_global_filters: bool = True
    filters: list[LocalFilter] = Field(default_factory=list, max_length=MAX_LOCAL_FILTERS)


class KpiWidget(_WidgetBase):
    type: Literal["kpi"]
    metric: Metric


class _Grouped(_WidgetBase):
    group_by: str = Field(min_length=1, max_length=200)
    metric: Metric
    limit: int = Field(default=10, ge=1, le=25)
    ascending: bool = False


class BarWidget(_Grouped):
    type: Literal["bar"]


class DonutWidget(_Grouped):
    type: Literal["donut"]


class LineWidget(_WidgetBase):
    type: Literal["line"]
    date_column: str = Field(min_length=1, max_length=200)
    value_column: str | None = Field(default=None, max_length=200)  # None = count rows
    op: str = "sum"
    freq: str = "M"

    @field_validator("op")
    @classmethod
    def _op(cls, v: str) -> str:
        v = v.lower()
        if v not in TS_OPS:
            raise ValueError(f"unsupported aggregation '{v}'. Supported: {', '.join(TS_OPS)}")
        return v

    @field_validator("freq")
    @classmethod
    def _freq(cls, v: str) -> str:
        v = v.upper()
        if v not in FREQS:
            raise ValueError(f"unsupported frequency '{v}'. Use one of: {', '.join(FREQS)}")
        return v


class HistogramWidget(_WidgetBase):
    type: Literal["histogram"]
    column: str = Field(min_length=1, max_length=200)
    bins: int = Field(default=20, ge=5, le=60)


class ScatterWidget(_WidgetBase):
    type: Literal["scatter"]
    x: str = Field(min_length=1, max_length=200)
    y: str = Field(min_length=1, max_length=200)


class BoxWidget(_WidgetBase):
    type: Literal["box"]
    column: str = Field(min_length=1, max_length=200)
    group_by: str | None = Field(default=None, max_length=200)


class HeatmapWidget(_WidgetBase):
    type: Literal["heatmap"]
    columns: list[str] | None = Field(default=None, min_length=2, max_length=15)  # None = all numeric
    method: str = "pearson"

    @field_validator("method")
    @classmethod
    def _method(cls, v: str) -> str:
        v = v.lower()
        if v not in CORR_METHODS:
            raise ValueError(f"unsupported method '{v}'. Supported: {', '.join(CORR_METHODS)}")
        return v


class TableWidget(_WidgetBase):
    type: Literal["table"]
    columns: list[str] | None = Field(default=None, min_length=1, max_length=12)  # None = first columns
    limit: int = Field(default=10, ge=1, le=50)


Widget = Annotated[
    Union[KpiWidget, BarWidget, DonutWidget, LineWidget, HistogramWidget,
          ScatterWidget, BoxWidget, HeatmapWidget, TableWidget],
    Field(discriminator="type"),
]


# ------------------------------------------------------------------ global filters

def _check_date(v):
    if v is None:
        return v
    try:
        ts = pd.Timestamp(v)
    except (ValueError, TypeError):
        raise ValueError(f"'{v}' is not a valid date") from None
    if pd.isna(ts):
        raise ValueError(f"'{v}' is not a valid date")
    return v


def _range_filters(column: str, low, high) -> list[dict]:
    if low is not None and high is not None:
        return [{"column": column, "op": "between", "value": [low, high]}]
    if low is not None:
        return [{"column": column, "op": "gte", "value": low}]
    if high is not None:
        return [{"column": column, "op": "lte", "value": high}]
    return []


class CategoryFilter(_Base):
    kind: Literal["category"]
    column: str = Field(min_length=1, max_length=200)
    values: list[str] = Field(default_factory=list, max_length=100)  # empty = no restriction

    def to_tool_filters(self) -> list[dict]:
        return [{"column": self.column, "op": "in", "value": list(self.values)}] if self.values else []


class DateRangeFilter(_Base):
    kind: Literal["date_range"]
    column: str = Field(min_length=1, max_length=200)
    start: str | None = None
    end: str | None = None

    _v = field_validator("start", "end")(lambda cls, v: _check_date(v))

    @model_validator(mode="after")
    def _order(self):
        if self.start and self.end and pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError("date range start is after its end")
        return self

    def to_tool_filters(self) -> list[dict]:
        return _range_filters(self.column, self.start, self.end)


class NumericRangeFilter(_Base):
    kind: Literal["numeric_range"]
    column: str = Field(min_length=1, max_length=200)
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check(self):
        for name, v in (("min", self.min), ("max", self.max)):
            if v is not None and not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min is greater than max")
        return self

    def to_tool_filters(self) -> list[dict]:
        return _range_filters(self.column, self.min, self.max)


GlobalFilter = Annotated[Union[CategoryFilter, DateRangeFilter, NumericRangeFilter], Field(discriminator="kind")]


# ------------------------------------------------------------------ dashboard

class Dashboard(_Base):
    version: Literal[1] = 1
    title: str = Field(default="Dashboard", min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    widgets: list[Widget] = Field(default_factory=list, max_length=MAX_WIDGETS)
    filters: list[GlobalFilter] = Field(default_factory=list, max_length=MAX_GLOBAL_FILTERS)

    @model_validator(mode="after")
    def _ids(self):
        seen: set[str] = set()
        for w in self.widgets:
            if w.id is not None:
                if w.id in seen:
                    raise ValueError(f"duplicate widget id '{w.id}'")
                seen.add(w.id)
        n = 0
        for w in self.widgets:
            if w.id is None:
                while True:
                    n += 1
                    if f"w{n}" not in seen:
                        break
                w.id = f"w{n}"
                seen.add(w.id)
        return self


def widget_filters(dash: Dashboard, widget) -> list[dict]:
    """Tool-ready filters for one widget: active global filters (unless opted out) plus its own."""
    out: list[dict] = []
    if widget.use_global_filters:
        for g in dash.filters:
            out += g.to_tool_filters()
    return out + [f.to_tool_filter() for f in widget.filters]


# ------------------------------------------------------------------ dataset validation

class _Checker:
    def __init__(self, profile: dict):
        self.types = {c["name"]: c["inferred_type"] for c in profile["columns"]}
        self.problems: list[dict] = []

    def add(self, path: str, code: str, message: str) -> None:
        self.problems.append({"path": path, "code": code, "message": message})

    def column(self, path: str, name: str, allowed: set | None = None, label: str = "this field") -> bool:
        if name not in self.types:
            close = difflib.get_close_matches(name, list(self.types), n=3, cutoff=0.6)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            self.add(path, "column_not_found", f"Column '{name}' does not exist.{hint}")
            return False
        if allowed is not None and self.types[name] not in allowed:
            self.add(path, "invalid_column_type",
                     f"'{name}' is {self.types[name]}, but {label} needs {' or '.join(sorted(allowed))}.")
            return False
        return True

    def metric(self, m: Metric, path: str, group_col: str | None = None) -> None:
        if m.column == "*":
            if m.op != "count":
                self.add(f"{path}.column", "invalid_parameter", "'*' (all rows) can only be used with 'count'.")
            return
        allowed = {"numeric"} if m.op in NUMERIC_ONLY else None
        self.column(f"{path}.column", m.column, allowed, f"'{m.op}'")
        if group_col and m.column == group_col:
            self.add(f"{path}.column", "invalid_parameter", "A metric cannot use the grouping column.")

    def widget(self, w, path: str) -> None:
        t = w.type
        if t == "kpi":
            self.metric(w.metric, f"{path}.metric")
        elif t in ("bar", "donut"):
            self.column(f"{path}.group_by", w.group_by, CATEGORY_TYPES, "grouping")
            self.metric(w.metric, f"{path}.metric", w.group_by)
            if t == "donut" and w.metric.op not in ("sum", "count"):
                self.add(f"{path}.metric.op", "invalid_parameter",
                         "A donut shows shares of a total, so it needs 'sum' or 'count'. Use a bar chart for other aggregations.")
        elif t == "line":
            self.column(f"{path}.date_column", w.date_column, {"datetime"}, "a time axis")
            if w.value_column:
                self.column(f"{path}.value_column", w.value_column, {"numeric"}, "a value")
        elif t == "histogram":
            self.column(f"{path}.column", w.column, {"numeric"}, "a histogram")
        elif t == "scatter":
            self.column(f"{path}.x", w.x, {"numeric"}, "a scatter axis")
            self.column(f"{path}.y", w.y, {"numeric"}, "a scatter axis")
            if w.x == w.y:
                self.add(f"{path}.y", "invalid_parameter", "x and y must be different columns.")
        elif t == "box":
            self.column(f"{path}.column", w.column, {"numeric"}, "a box plot")
            if w.group_by:
                self.column(f"{path}.group_by", w.group_by, CATEGORY_TYPES, "grouping")
        elif t == "heatmap":
            if w.columns is None:
                if sum(1 for v in self.types.values() if v == "numeric") < 2:
                    self.add(path, "insufficient_data", "A correlation heatmap needs at least two numeric columns.")
            else:
                if len(set(w.columns)) != len(w.columns):
                    self.add(f"{path}.columns", "invalid_parameter", "Columns must be unique.")
                for i, c in enumerate(w.columns):
                    self.column(f"{path}.columns[{i}]", c, {"numeric"}, "a correlation")
        elif t == "table":
            for i, c in enumerate(w.columns or []):
                self.column(f"{path}.columns[{i}]", c)
        for j, f in enumerate(w.filters):
            self.column(f"{path}.filters[{j}].column", f.column)

    def global_filter(self, g, path: str) -> None:
        needs = {"category": CATEGORY_TYPES, "date_range": {"datetime"}, "numeric_range": {"numeric"}}[g.kind]
        self.column(f"{path}.column", g.column, needs, f"a {g.kind.replace('_', ' ')} filter")


def check_against_dataset(dash: Dashboard, profile: dict) -> list[dict]:
    chk = _Checker(profile)
    for i, w in enumerate(dash.widgets):
        chk.widget(w, f"widgets[{i}]")
    for i, g in enumerate(dash.filters):
        chk.global_filter(g, f"filters[{i}]")
    return chk.problems


# ------------------------------------------------------------------ entry point

def _path(loc) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (("." if out else "") + str(part))
    return out


def _spec_error(problems: list[dict]) -> SpecError:
    problems = problems[:MAX_PROBLEMS]
    summary = "; ".join(f"{p['path'] or 'spec'}: {p['message']}" for p in problems[:3])
    more = f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""
    return SpecError(f"The dashboard spec has {len(problems)} problem(s): {summary}{more}",
                     "invalid_dashboard", {"problems": problems})


def parse_dashboard(raw: Any, profile: dict) -> Dashboard:
    """Validate a raw dict into a Dashboard for this dataset, or raise SpecError listing every problem."""
    try:
        dash = Dashboard.model_validate(raw)
    except ValidationError as exc:
        raise _spec_error([
            {"path": _path(e["loc"]), "code": "invalid_spec", "message": e["msg"].removeprefix("Value error, ")}
            for e in exc.errors()
        ]) from None
    problems = check_against_dataset(dash, profile)
    if problems:
        raise _spec_error(problems)
    return dash