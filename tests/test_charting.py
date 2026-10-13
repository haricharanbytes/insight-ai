import json

import pandas as pd
import pytest

from src.analysis.charting import generate_chart
from src.analysis.common import AnalysisError
from src.visualization.charts import chart_for_output

DF = pd.DataFrame({
    "a": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
    "b": [2.0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
    "g": list("xxxyyyzzzxy"),
})


def code_of(**kw):
    with pytest.raises(AnalysisError) as exc:
        generate_chart(DF, **kw)
    return exc.value.code


def test_histogram_counts_are_real():
    out = generate_chart(DF, "histogram", "a", bins=5)
    res = out["result"]
    assert res["n_values"] == 10 and sum(res["counts"]) == 10 and len(res["counts"]) == 5
    assert any("missing" in w for w in out["warnings"])
    assert chart_for_output(out)["kind"] == "histogram"
    json.dumps(out, allow_nan=False)


def test_scatter_drops_missing_pairs_and_samples_large_data():
    out = generate_chart(DF, "scatter", "a", "b")
    assert out["result"]["n_points_drawn"] == 10 and out["result"]["sampled"] is False
    big = pd.DataFrame({"a": range(5000), "b": range(5000)})
    out = generate_chart(big, "scatter", "a", "b")
    assert out["result"]["n_points_drawn"] == 2000 and out["result"]["n_points_total"] == 5000
    assert any("random sample" in w for w in out["warnings"])
    assert "2,000 of 5,000" in out["result"]["chart"]["caption"]


def test_box_with_and_without_groups():
    assert generate_chart(DF, "box", "b")["result"]["n_values"] == 11
    out = generate_chart(DF, "box", "b", group_by="g")
    assert out["result"]["groups"] == ["x", "y", "z"]
    assert len(out["result"]["chart"]["figure"]["data"]) == 3


def test_errors():
    assert code_of(chart_type="pie", x="a") == "unsupported_operation"
    assert code_of(chart_type="histogram", x="g") == "invalid_column_type"
    assert code_of(chart_type="histogram", x="nope") == "column_not_found"
    assert code_of(chart_type="scatter", x="a") == "invalid_parameter"
    assert code_of(chart_type="scatter", x="a", y="a") == "invalid_parameter"
    assert code_of(chart_type="histogram", x="a", bins=2) == "invalid_parameter"
    assert code_of(chart_type="histogram", x="a", group_by="g") == "invalid_parameter"
    assert code_of(chart_type="histogram", x="a", filters=[{"column": "g", "op": "eq", "value": "q"}]) == "no_rows"


def test_input_not_modified():
    before = DF.copy()
    generate_chart(DF, "scatter", "a", "b")
    pd.testing.assert_frame_equal(DF, before)