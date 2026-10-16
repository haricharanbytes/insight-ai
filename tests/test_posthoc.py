import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.analysis.common import AnalysisError
from src.analysis.posthoc import check_test_assumptions, run_posthoc_tests

rng = np.random.default_rng(0)
_base = rng.normal(10, 1, 40)
DF = pd.DataFrame({
    "g": ["A"] * 40 + ["B"] * 40 + ["C"] * 40,
    "v": np.concatenate([_base, _base[::-1] + 0.01, np.roll(_base, 7) + 4.0]),
})


def rows(out):
    t = out["table"]
    return [dict(zip(t["columns"], r)) for r in t["rows"]]


def code_of(fn, *a, **kw):
    with pytest.raises(AnalysisError) as exc:
        fn(DF, *a, **kw)
    return exc.value.code


def test_posthoc_finds_the_different_group_after_correction():
    out = run_posthoc_tests(DF, "v", "g")
    r = rows(out)
    assert len(r) == 3 and out["result"]["n_pairs"] == 3
    sig = {frozenset((x["group_a"], x["group_b"])) for x in r if x["significant"]}
    assert sig == {frozenset("AC"), frozenset("BC")}
    ab = next(x for x in r if {x["group_a"], x["group_b"]} == {"A", "B"})
    assert ab["significant"] is False


def test_raw_p_matches_scipy_and_holm_matches_hand_calculation():
    out = run_posthoc_tests(DF, "v", "g", correction="holm")
    r = rows(out)
    a, b = DF[DF.g == "A"].v.to_numpy(), DF[DF.g == "B"].v.to_numpy()
    ab = next(x for x in r if {x["group_a"], x["group_b"]} == {"A", "B"})
    assert ab["p_raw"] == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue)
    raws = sorted(x["p_raw"] for x in r)
    holm = [min(1.0, raws[0] * 3), min(1.0, raws[1] * 2), min(1.0, raws[2] * 1)]
    holm = np.maximum.accumulate(holm)
    assert sorted(x["p_adjusted"] for x in r) == pytest.approx(list(holm))


def test_bonferroni_and_fdr_and_mannwhitney():
    bon = rows(run_posthoc_tests(DF, "v", "g", correction="bonferroni"))
    assert all(x["p_adjusted"] == pytest.approx(min(1.0, x["p_raw"] * 3)) for x in bon)
    fdr = run_posthoc_tests(DF, "v", "g", correction="fdr_bh")
    assert all(x["p_adjusted"] >= x["p_raw"] - 1e-12 for x in rows(fdr))
    mw = run_posthoc_tests(DF, "v", "g", method="mannwhitney")
    assert "Mann-Whitney" in mw["method"] and mw["result"]["n_significant_adjusted"] == 2


def test_adjustment_removes_a_false_positive():
    rng2 = np.random.default_rng(3)
    many = pd.DataFrame({"g": np.repeat(list("ABCDEFGH"), 12), "v": rng2.normal(0, 1, 96)})
    out = run_posthoc_tests(many, "v", "g")
    assert out["result"]["n_pairs"] == 28
    assert out["result"]["n_significant_adjusted"] <= out["result"]["n_significant_unadjusted"]


def test_assumption_checks_normal_vs_skewed():
    out = check_test_assumptions(DF, "v", "g")
    assert out["result"]["recommended_test"] == "anova" and out["result"]["equal_variances"] is True
    skew = pd.DataFrame({"g": ["A"] * 15 + ["B"] * 15, "v": np.concatenate([rng.exponential(1, 15) ** 3,
                                                                           rng.exponential(1, 15) ** 3])})
    out = check_test_assumptions(skew, "v", "g")
    assert out["result"]["recommended_test"] == "mannwhitney"
    out = check_test_assumptions(DF.assign(v=np.where(DF.g == "C", DF.v * 5, DF.v)), "v", "g")
    assert out["result"]["equal_variances"] is False


def test_levene_matches_scipy():
    out = check_test_assumptions(DF, "v", "g")
    ref = stats.levene(*[DF[DF.g == k].v for k in "ABC"], center="median")
    assert out["result"]["levene_p"] == pytest.approx(ref.pvalue)


def test_caveats_and_serialisable():
    a, b = check_test_assumptions(DF, "v", "g"), run_posthoc_tests(DF, "v", "g")
    assert "not proof" in a["result"]["note"] and "significant overall test" in b["result"]["note"]
    json.dumps([a, b], allow_nan=False)


def test_refusals():
    assert code_of(run_posthoc_tests, "v", "g", method="bayes") == "unsupported_operation"
    assert code_of(run_posthoc_tests, "v", "g", correction="none") == "unsupported_operation"
    assert code_of(run_posthoc_tests, "v", "g", alpha=2) == "invalid_parameter"
    assert code_of(run_posthoc_tests, "v", "g", groups=["A", "Z"]) == "group_not_found"
    assert code_of(run_posthoc_tests, "v", "g", groups=["A"]) == "insufficient_data"
    assert code_of(run_posthoc_tests, "g", "v") == "invalid_column_type"
    assert code_of(check_test_assumptions, "v", "nope") == "column_not_found"
    tiny = pd.DataFrame({"g": ["A", "A", "B", "B", "B"], "v": [1.0, 2, 3, 4, 5]})
    with pytest.raises(AnalysisError) as exc:
        run_posthoc_tests(tiny, "v", "g")
    assert exc.value.code == "insufficient_data"
    many = pd.DataFrame({"g": np.repeat(list("ABCDEFGHIJKLM"), 4), "v": rng.normal(0, 1, 52)})
    with pytest.raises(AnalysisError) as exc:
        run_posthoc_tests(many, "v", "g")
    assert exc.value.code == "too_many_groups"


def test_input_not_modified():
    before = DF.copy()
    run_posthoc_tests(DF, "v", "g")
    check_test_assumptions(DF, "v", "g")
    pd.testing.assert_frame_equal(DF, before)