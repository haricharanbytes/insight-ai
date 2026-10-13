from src.agent.grounding import find_unverified_numbers

OUT = [{"result": {"total": 836154.033, "share": 36.4, "change": -83.333},
        "table": {"rows": [["North", 400.0, 1234567.0]]}, "inputs": {"limit": 20}}]


def test_rounded_and_formatted_numbers_match():
    text = "Technology had 836,154 in sales (36.4%), and one month fell -83.3%."
    assert find_unverified_numbers(text, OUT) == []


def test_suffix_numbers_match():
    assert find_unverified_numbers("About $836K and $1.2M.", OUT) == []


def test_unknown_number_is_flagged():
    assert find_unverified_numbers("Total sales were 12,345.67.", OUT) == ["12,345.67"]


def test_no_tool_results_flags_everything():
    assert find_unverified_numbers("Revenue was 500.5.", []) == ["500.5"]


def test_dates_years_and_small_integers_ignored():
    text = "Between 2014-01-03 and 2017Q4 (in 2016) the top 3 regions, 10 items, on 2017-12-30."
    assert find_unverified_numbers(text, []) == []


def test_wrong_precision_is_flagged():
    assert find_unverified_numbers("The share was 36.9%.", OUT) == ["36.9%"]


def test_duplicates_reported_once():
    assert find_unverified_numbers("It was 999.9 and again 999.9.", []) == ["999.9"]

def test_numbers_in_warnings_and_history_count_as_evidence():
    warns = ["The first period (2014Q1) is partial: the data covers about 94% of it."]
    assert find_unverified_numbers("Coverage was 94%.", [], warns) == []
    prior = ['{"rows": [["2017Q4", 104759.1234]], "inputs": {}}']
    assert find_unverified_numbers("Sales were $104,759.", [], prior) == []
    assert find_unverified_numbers("Sales were $104,760.5.", [], prior) == ["$104,760.5"]