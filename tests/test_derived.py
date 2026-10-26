import pytest

from src.agent.derived import verify_derived
from src.agent.evidence import build_index, extract_claims, verify_exact

STEPS = [{"ok": True, "tool": "group_by_aggregate", "output": {
    "tool": "group_by_aggregate", "inputs": {}, "rows_used": 500, "warnings": [],
    "result": {},
    "table": {"columns": ["Category", "sum_Sales", "sum_Profit"],
              "rows": [["Technology", 800.0, 160.0], ["Furniture", 600.0, 30.0], ["Office", 400.0, 80.0]]}}}]
INDEX = build_index(STEPS)


def check(text):
    claims = verify_exact(extract_claims(text), INDEX)
    return verify_derived(claims, INDEX)


def first(text):
    return check(text)[0]


def test_exact_values_are_left_alone():
    c = first("Technology sold 800.00 in total.")
    assert c.status == "exact" and c.formula is None


def test_named_difference_is_derived_with_formula():
    c = first("Technology outsold Furniture by 200.0.")
    assert c.status == "derived" and "800.00 - 600.00" in c.formula and "difference" in c.formula


def test_named_ratio_percent_and_margin():
    c = first("Technology had a margin of 20.0% on sales.")       # 160 / 800
    assert c.status in ("derived", "plausible") and "160.00 / 800.00" in c.formula


def test_percent_change_and_share_and_mean_when_named():
    pc = first("Technology sales are 33.3% above Furniture.")      # (800 - 600) / 600 * 100 = 33.33
    assert pc.status == "derived" and "pct change" in pc.formula
    sh = first("Technology is 57.14% of Technology plus Furniture sales.")     # 800 / 1400 * 100
    assert sh.status == "derived" and "share pct" in sh.formula
    mn = first("The mean of Technology and Furniture sales is 700.0.")
    assert mn.status == "derived" and "mean2" in mn.formula


def test_unnamed_but_long_numbers_are_only_plausible():
    c = first("The gap was 200.00 overall.")
    assert c.status == "plausible"


def test_fabricated_and_short_numbers_stay_unverified():
    assert first("Sales were 1234.56 overall.").status == "unverified"
    assert first("A change of 13.7% happened.").status == "unverified"       # short, unnamed: not searched


def test_quoted_operands_make_it_derived():
    c = check("Sales of 800.00 against 600.00 left a gap of 200.00.")[-1]
    assert c.text == "200.00" and c.status == "derived"


def test_no_claim_matches_across_different_steps():
    two = STEPS + [{"ok": True, "tool": "x", "output": {"tool": "x", "inputs": {}, "rows_used": 9, "warnings": [],
                                                        "result": {"a": 55.0}, "table": None}}]
    idx = build_index(two)
    claims = verify_derived(verify_exact(extract_claims("The sum was 855.00 in total."), idx), idx)
    assert claims[0].status == "unverified"           # 800 + 55 spans two steps: not allowed


def test_pool_is_bounded_and_input_untouched():
    big = [{"ok": True, "tool": "t", "output": {"tool": "t", "inputs": {}, "rows_used": 1, "warnings": [], "result": {},
                                               "table": {"columns": ["k", "v"], "rows": [[f"r{i}", 3.0 + i * 1.37] for i in range(600)]}}}]
    idx = build_index(big)
    n = len(idx)
    verify_derived(verify_exact(extract_claims("Total of 999999.99 here."), idx), idx)
    assert len(idx) == n