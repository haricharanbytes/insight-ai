import json

import pandas as pd
import pytest

from src.analysis.common import AnalysisError
from src.analysis.correlations import calculate_correlation
from src.analysis.outliers import detect_outliers
from src.analysis.statistics import get_column_summary, run_statistical_test


def rows(result):
    t = result["table"]
    return [dict(zip(t["columns"], r)) for r in t["rows"]]


def code_of(fn, *args, **kwargs):
    with pytest.raises(AnalysisError) as exc:
        fn(*args, **kwargs)
    return exc.value.code


# ====================== correlations ======================

@pytest.fixture
def corr_df():
    return pd.DataFrame({
        "x": range(1, 11),
        "y": [2 * i + 1 for i in range(1, 11)],
        "z": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
        "t": list("abcdefghij"),
    })


def test_perfect_correlation_listed_first(corr_df):
    r = calculate_correlation(corr_df, ["x", "y", "z"])
    first = rows(r)[0]
    assert (first["column_a"], first["column_b"]) == ("x", "y")
    assert first["r"] == pytest.approx(1.0)
    assert first["p_value"] < 1e-6
    assert first["strength"] == "very strong" and first["direction"] == "positive"
    assert r["result"]["pairs_computed"] == 3
    assert r["rows_used"] == 10


def test_negative_correlation_direction():
    df = pd.DataFrame({"x": range(1, 11), "w": [-i for i in range(1, 11)]})
    first = rows(calculate_correlation(df, ["x", "w"]))[0]
    assert first["r"] == pytest.approx(-1.0)
    assert first["direction"] == "negative"


def test_matrix_is_symmetric_with_unit_diagonal(corr_df):
    m = calculate_correlation(corr_df, ["x", "y", "z"])["result"]["matrix"]["values"]
    assert m[0][0] == 1.0
    assert m[0][1] == m[1][0]
    assert m[1][2] == m[2][1]


def test_spearman_handles_monotonic_nonlinear():
    df = pd.DataFrame({"x": range(1, 11), "y": [i ** 3 for i in range(1, 11)]})
    pearson = rows(calculate_correlation(df, ["x", "y"], "pearson"))[0]["r"]
    spearman = rows(calculate_correlation(df, ["x", "y"], "spearman"))[0]["r"]
    assert spearman == pytest.approx(1.0)
    assert pearson < 0.99


def test_auto_selection_skips_identifier_columns():
    df = pd.DataFrame({
        "Row ID": range(1, 11),
        "a": range(10),
        "b": [5, 3, 8, 1, 9, 2, 7, 4, 6, 10],
        "t": list("abcdefghij"),
    })
    r = calculate_correlation(df)
    assert r["result"]["columns"] == ["a", "b"]
    assert any("Row ID" in w for w in r["warnings"])


def test_pairwise_complete_rows_only():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, None], "y": [2, 4, 6, 8, 10, 12]})
    first = rows(calculate_correlation(df, ["x", "y"]))[0]
    assert first["n"] == 5
    assert first["r"] == pytest.approx(1.0)


def test_constant_column_skipped():
    df = pd.DataFrame({"x": range(10), "y": [i * i for i in range(10)], "c": [5] * 10})
    r = calculate_correlation(df, ["x", "y", "c"])
    assert r["result"]["pairs_computed"] == 1
    assert r["result"]["pairs_skipped"] == 2
    assert any("skipped" in w for w in r["warnings"])
    assert code_of(calculate_correlation, df, ["x", "c"]) == "insufficient_data"


def test_correlation_errors(corr_df):
    assert code_of(calculate_correlation, corr_df, ["x", "t"]) == "invalid_column_type"
    assert code_of(calculate_correlation, corr_df, ["x"]) == "invalid_parameter"
    assert code_of(calculate_correlation, corr_df, ["x", "x"]) == "invalid_parameter"
    assert code_of(calculate_correlation, corr_df, ["x", "y"], "kendall") == "unsupported_operation"
    assert code_of(calculate_correlation, corr_df, ["x", "nope"]) == "column_not_found"
    assert code_of(calculate_correlation, corr_df, ["x", "y"],
                   filters=[{"column": "t", "op": "eq", "value": "zzz"}]) == "no_rows"


# ====================== outliers ======================

def _outlier_df():
    return pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})


def test_iqr_outlier():
    r = detect_outliers(_outlier_df(), "a")
    res = r["result"]
    assert res["n_outliers"] == 1 and res["high_count"] == 1 and res["low_count"] == 0
    assert res["bounds"]["lower"] == pytest.approx(-3.5)
    assert res["bounds"]["upper"] == pytest.approx(14.5)
    first = rows(r)[0]
    assert first["value"] == 100.0 and first["row_number"] == 10 and first["direction"] == "high"


def test_no_outliers_gives_empty_table():
    r = detect_outliers(pd.DataFrame({"a": range(1, 11)}), "a")
    assert r["result"]["n_outliers"] == 0
    assert r["table"]["rows"] == []


def test_zscore_masking_is_real():
    # The extreme value inflates the std, so |z| is only ~2.84 and a 3-sigma rule misses it.
    assert detect_outliers(_outlier_df(), "a", "zscore")["result"]["n_outliers"] == 0
    assert detect_outliers(_outlier_df(), "a", "zscore", threshold=2.5)["result"]["n_outliers"] == 1


def test_missing_values_ignored_and_reported():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, None]})
    r = detect_outliers(df, "a")
    assert r["rows_used"] == 10
    assert any("missing" in w for w in r["warnings"])


def test_outlier_errors():
    df = _outlier_df().assign(t=list("abcdefghij"))
    assert code_of(detect_outliers, df, "t") == "invalid_column_type"
    assert code_of(detect_outliers, df, "nope") == "column_not_found"
    assert code_of(detect_outliers, df, "a", "magic") == "unsupported_operation"
    assert code_of(detect_outliers, df, "a", threshold=0) == "invalid_parameter"
    assert code_of(detect_outliers, df.head(5), "a") == "insufficient_data"
    constant = pd.DataFrame({"a": [5] * 20})
    assert code_of(detect_outliers, constant, "a") == "method_not_applicable"
    assert code_of(detect_outliers, constant, "a", "zscore") == "method_not_applicable"


# ====================== statistical tests ======================

def two_groups():
    return pd.DataFrame({"g": ["A"] * 5 + ["B"] * 5, "v": [1, 2, 3, 4, 5, 3, 4, 5, 6, 7]})


def three_groups():
    return pd.DataFrame({"g": ["A"] * 3 + ["B"] * 3 + ["C"] * 3, "v": [1, 2, 3, 2, 3, 4, 7, 8, 9]})


def test_welch_ttest():
    r = run_statistical_test(two_groups(), "ttest", value_column="v", group_column="g")
    res = r["result"]
    assert res["statistic"] == pytest.approx(-2.0)
    assert res["degrees_of_freedom"] == pytest.approx(8.0)
    assert res["p_value"] == pytest.approx(0.0805, abs=0.001)
    assert res["mean_difference"] == pytest.approx(-2.0)
    assert res["effect_size"]["value"] == pytest.approx(-1.265, abs=0.001)
    assert res["effect_size"]["magnitude"] == "large"
    assert res["significant"] is False  # p ~ 0.08 > 0.05, despite the large effect (tiny sample)
    assert r["rows_used"] == 10
    assert [x["group"] for x in rows(r)] == ["A", "B"]


def test_ttest_groups_selected_from_three():
    r = run_statistical_test(three_groups(), "ttest", value_column="v", group_column="g", groups=["A", "B"])
    assert r["result"]["mean_difference"] == pytest.approx(-1.0)
    assert r["rows_used"] == 6


def test_mann_whitney_complete_separation():
    df = pd.DataFrame({"g": ["A"] * 5 + ["B"] * 5, "v": list(range(1, 11))})
    res = run_statistical_test(df, "mannwhitney", value_column="v", group_column="g")["result"]
    assert res["statistic"] == pytest.approx(0.0)
    assert res["effect_size"]["value"] == pytest.approx(-1.0)
    assert res["p_value"] == pytest.approx(2 / 252, abs=0.0005)
    assert res["significant"] is True


def test_anova():
    res = run_statistical_test(three_groups(), "anova", value_column="v", group_column="g")["result"]
    assert res["statistic"] == pytest.approx(31.0)
    assert (res["df_between"], res["df_within"]) == (2, 6)
    assert res["effect_size"]["value"] == pytest.approx(62 / 68, abs=0.001)
    assert res["p_value"] < 0.001


def test_kruskal():
    res = run_statistical_test(three_groups(), "kruskal", value_column="v", group_column="g")["result"]
    assert res["statistic"] == pytest.approx(6.0565, abs=0.01)
    assert res["degrees_of_freedom"] == 2


def test_chi_square():
    df = pd.DataFrame({
        "a": ["A"] * 30 + ["B"] * 30,
        "b": ["x"] * 10 + ["y"] * 20 + ["x"] * 20 + ["y"] * 10,
    })
    r = run_statistical_test(df, "chi_square", column_a="a", column_b="b")
    res = r["result"]
    assert res["statistic"] == pytest.approx(6.6667, abs=0.001)
    assert res["dof"] == 1
    assert res["p_value"] == pytest.approx(0.0098, abs=0.001)
    assert res["effect_size"]["value"] == pytest.approx(1 / 3, abs=0.001)
    assert res["significant"] is True
    assert r["rows_used"] == 60


def test_statistical_test_errors():
    g2, g3 = two_groups(), three_groups()
    assert code_of(run_statistical_test, g3, "ttest", value_column="v", group_column="g") == "invalid_parameter"
    assert code_of(run_statistical_test, g3, "ttest", value_column="v", group_column="g", groups=["A", "Z"]) == "group_not_found"
    assert code_of(run_statistical_test, g2, "bayes", value_column="v", group_column="g") == "unsupported_operation"
    assert code_of(run_statistical_test, g2, "ttest", value_column="g", group_column="v") == "invalid_column_type"
    assert code_of(run_statistical_test, g2, "ttest", value_column="v") == "invalid_parameter"
    assert code_of(run_statistical_test, g2, "ttest", value_column="v", group_column="g", alpha=2) == "invalid_parameter"
    assert code_of(run_statistical_test, g2, "ttest", value_column="nope", group_column="g") == "column_not_found"
    assert code_of(run_statistical_test, g2, "chi_square", column_a="g", column_b="g") == "invalid_parameter"
    one_obs = pd.DataFrame({"g": ["A", "B", "B"], "v": [1, 2, 3]})
    assert code_of(run_statistical_test, one_obs, "ttest", value_column="v", group_column="g") == "insufficient_data"
    flat = pd.DataFrame({"g": ["A"] * 3 + ["B"] * 3, "v": [4] * 6})
    assert code_of(run_statistical_test, flat, "ttest", value_column="v", group_column="g") == "no_variation"


# ====================== column summary ======================

def test_column_summary():
    df = pd.DataFrame({
        "Row ID": range(1, 11),
        "cat": list("aabbbcccce"),
        "v": [1.0, 2, 3, 4, 5, None, 7, 8, 9, 10],
    })
    p = get_column_summary(df, "v")["result"]["profile"]
    assert p["missing_count"] == 1
    assert p["mean"] == pytest.approx(49 / 9)

    c = get_column_summary(df, "cat")
    assert c["table"]["rows"][0][0] == "c" and c["table"]["rows"][0][1] == 4

    assert get_column_summary(df, "Row ID")["result"]["profile"]["mean"] == 5.5  # explicit ask gets numeric stats

    f = get_column_summary(df, "v", filters=[{"column": "cat", "op": "eq", "value": "a"}])
    assert f["rows_used"] == 2
    assert f["result"]["profile"]["mean"] == pytest.approx(1.5)

    assert code_of(get_column_summary, df, "nope") == "column_not_found"


# ====================== cross-cutting ======================

def test_all_tool_outputs_are_json_serializable(corr_df):
    outputs = [
        calculate_correlation(corr_df, ["x", "y", "z"]),
        detect_outliers(_outlier_df(), "a"),
        run_statistical_test(two_groups(), "ttest", value_column="v", group_column="g"),
        run_statistical_test(three_groups(), "anova", value_column="v", group_column="g"),
        run_statistical_test(three_groups(), "kruskal", value_column="v", group_column="g"),
        get_column_summary(corr_df, "z"),
    ]
    for out in outputs:
        json.dumps(out, allow_nan=False)


def test_inputs_are_not_modified(corr_df):
    before = corr_df.copy()
    calculate_correlation(corr_df, ["x", "y", "z"])
    detect_outliers(corr_df, "z", threshold=1.0)
    run_statistical_test(corr_df.assign(g=list("AABBAABBAB")), "ttest", value_column="z", group_column="g")
    get_column_summary(corr_df, "t")
    pd.testing.assert_frame_equal(corr_df, before)