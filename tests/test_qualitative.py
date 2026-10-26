from src.agent.qualitative import check_qualitative

GROUP = {"ok": True, "output": {"tool": "group_by_aggregate",
         "inputs": {"group_by": ["Region"], "ascending": False},
         "result": {"top_group": {"Region": "North", "sum_Sales": 400}, "bottom_group": {"Region": "West", "sum_Sales": 50}},
         "table": {"columns": ["Region", "sum_Sales"], "rows": [["North", 400], ["East", 300], ["West", 50]]}}}
TREND = lambda d, avail=True: {"ok": True, "output": {"tool": "analyze_time_series", "inputs": {}, "table": None,
                                                    "result": {"trend": {"available": avail, "direction": d}}}}
TEST = lambda sig: {"ok": True, "output": {"tool": "run_statistical_test", "inputs": {}, "table": None,
                                          "result": {"significant": sig, "p_value": 0.03 if sig else 0.4}}}


def outcome(text, steps):
    r = check_qualitative(text, steps)
    return r[0].outcome if r else None


def test_top_and_bottom_group():
    assert outcome("North has the highest sales.", [GROUP]) == "confirmed"
    assert outcome("East has the highest sales.", [GROUP]) == "contradicted"
    assert outcome("West has the lowest sales.", [GROUP]) == "confirmed"
    assert outcome("North is not the top region.", [GROUP]) == "contradicted"
    assert outcome("North beat East as the highest.", [GROUP]) == "not_checkable"
    assert outcome("Mars has the highest sales.", [GROUP]) is None        # unknown group: no check


def test_trend_direction():
    assert outcome("Sales are increasing over time.", [TREND("increasing")]) == "confirmed"
    assert outcome("Sales are declining.", [TREND("increasing")]) == "contradicted"
    assert outcome("There is no clear trend.", [TREND("no clear trend")]) == "confirmed"
    assert outcome("Sales are rising.", [TREND("no clear trend")]) == "contradicted"
    assert outcome("Sales are rising.", [TREND("x", avail=False)]) == "not_checkable"
    assert outcome("Sales rose then fell.", [TREND("increasing")]) is None   # mixed directions: skipped


def test_significance():
    assert outcome("The groups differ significantly.", [TEST(True)]) == "confirmed"
    assert outcome("There is no significant difference.", [TEST(True)]) == "contradicted"
    assert outcome("There is no significant difference.", [TEST(False)]) == "confirmed"


def test_no_matching_tool_means_no_check():
    assert check_qualitative("North has the highest sales.", []) == []
    assert check_qualitative("The groups differ significantly.", [GROUP]) == []
    assert check_qualitative("", [GROUP]) == [] and check_qualitative(None, [GROUP]) == []