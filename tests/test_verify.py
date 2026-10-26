from src.agent.verify import verify_answer

STEP = {"ok": True, "tool": "group_by_aggregate", "output": {
    "tool": "group_by_aggregate", "inputs": {"group_by": ["Region"], "ascending": False}, "rows_used": 500, "warnings": [],
    "result": {"top_group": {"Region": "North", "sum_Sales": 800.0}, "bottom_group": {"Region": "West", "sum_Sales": 400.0}},
    "table": {"columns": ["Region", "sum_Sales"], "rows": [["North", 800.0], ["East", 600.0], ["West", 400.0]]}}}


def test_all_verified_is_high():
    r = verify_answer("North has the highest sales at 800.00.", [STEP])
    assert r["confidence"] == "high" and r["status_label"] == "verified"
    assert r["counts"]["exact"] == 1 and r["qualitative_counts"]["confirmed"] == 1


def test_derived_number_stays_high_and_names_the_formula():
    r = verify_answer("North outsold East by 200.0.", [STEP])
    assert r["confidence"] == "high" and r["claims"][0]["status"] == "derived"
    assert "800.00 - 600.00" in r["claims"][0]["formula"]


def test_unverified_number_is_low():
    r = verify_answer("North sold 1234.56 in total.", [STEP])
    assert r["confidence"] == "low" and r["unverified_numbers"] == ["1234.56"]
    assert any("could not be matched" in x for x in r["reasons"])


def test_contradiction_is_low():
    r = verify_answer("East has the highest sales.", [STEP])
    assert r["confidence"] == "low" and r["qualitative_counts"]["contradicted"] == 1


def test_plausible_and_warnings_give_medium():
    r = verify_answer("The gap was 200.00 overall.", [STEP])
    assert r["confidence"] == "medium" and r["counts"]["plausible"] == 1
    warned = {**STEP, "output": {**STEP["output"], "warnings": ["The first period is partial."]}}
    r = verify_answer("North has 800.00.", [warned])
    assert r["confidence"] == "medium" and any("data-limit" in x for x in r["reasons"])


def test_numbers_without_tools_or_incomplete_are_low():
    assert verify_answer("Sales were 1234.56.", [])["confidence"] == "low"
    assert verify_answer("North has 800.00.", [STEP], incomplete=True)["confidence"] == "low"


def test_text_only_answers():
    assert verify_answer("The Salary column does not exist.", [])["confidence"] == "medium"
    assert verify_answer("Nothing to report.", [STEP])["confidence"] == "high"


def test_failed_steps_do_not_count_as_evidence():
    bad = {"ok": False, "tool": "x", "error": {}}
    assert verify_answer("North has 800.00.", [bad])["confidence"] == "low"