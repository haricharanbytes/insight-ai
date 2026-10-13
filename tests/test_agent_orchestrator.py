import io

import pandas as pd
import pytest

from src.agent.llm_client import LLMError
from src.agent.orchestrator import run_agent
from src.data.profiler import profile_dataset
from tests.fakes import ScriptedClient, text_msg, tool_msg

DF = pd.read_csv(io.StringIO(
    "Region,Sales,Profit\nEast,100,10\nEast,200,-20\nWest,50,5\nWest,150,15\nNorth,400,60\n"))
PROFILE = profile_dataset(DF)
GOOD_ARGS = {"group_by": ["Region"], "metrics": [{"column": "Sales", "op": "sum"}]}


def run(client, question="Which region sells most?", history=None, **kw):
    return run_agent(client, DF, PROFILE, question, history, **kw)


def test_tool_call_then_grounded_answer():
    client = ScriptedClient([tool_msg("group_by_aggregate", GOOD_ARGS),
                             text_msg("Answer: North has the highest total Sales at 400.")])
    r = run(client)
    assert r["answer"].startswith("Answer: North")
    assert [s["ok"] for s in r["steps"]] == [True]
    assert r["steps"][0]["output"]["result"]["top_group"]["Region"] == "North"
    assert r["grounding"]["unverified_numbers"] == []
    assert r["llm_calls"] == 2 and r["usage"]["prompt_tokens"] == 20
    assert r["incomplete"] is False and r["model"] == "fake"
    assert client.calls[0]["tools"]  # tools were offered
    # the tool result was fed back to the model
    assert client.calls[1]["messages"][-1]["role"] == "tool"


def test_invalid_column_is_fed_back_and_model_self_corrects():
    bad = {"group_by": ["Regoin"], "metrics": [{"column": "Sales", "op": "sum"}]}
    client = ScriptedClient([tool_msg("group_by_aggregate", bad, "c1"),
                             tool_msg("group_by_aggregate", GOOD_ARGS, "c2"),
                             text_msg("Answer: North, 400.")])
    r = run(client)
    assert [s["ok"] for s in r["steps"]] == [False, True]
    error_message = client.calls[1]["messages"][-1]["content"]
    assert "column_not_found" in error_message and "Region" in error_message


def test_malformed_arguments_reported_to_model():
    client = ScriptedClient([tool_msg("group_by_aggregate", None, raw_arguments="{bad json"),
                             text_msg("Sorry, I could not run that.")])
    r = run(client)
    assert r["steps"][0]["ok"] is False and r["steps"][0]["error"]["code"] == "invalid_arguments"
    assert "not valid JSON" in client.calls[1]["messages"][-1]["content"]


def test_unknown_tool_does_not_crash():
    client = ScriptedClient([tool_msg("drop_table", {}), text_msg("I cannot do that.")])
    r = run(client)
    assert r["steps"][0]["error"]["code"] == "unknown_tool"
    assert r["answer"] == "I cannot do that."


def test_direct_text_answer_without_tools():
    r = run(ScriptedClient([text_msg("Answer: there is no Salary column. Which column did you mean?")]))
    assert r["steps"] == [] and "no Salary column" in r["answer"]


def test_fabricated_number_is_flagged():
    r = run(ScriptedClient([text_msg("Total sales were 12,345.67.")]))
    assert r["grounding"]["unverified_numbers"] == ["12,345.67"]


def test_round_limit_forces_a_final_answer_without_tools():
    client = ScriptedClient([tool_msg("group_by_aggregate", GOOD_ARGS, f"c{i}") for i in range(3)]
                            + [text_msg("Answer: North, 400.")])
    r = run(client, max_steps=3)
    assert r["answer"] == "Answer: North, 400."
    assert client.calls[-1]["tools"] is None
    assert "final answer now" in client.calls[-1]["messages"][-1]["content"]
    assert len(r["steps"]) == 3


def test_empty_reply_marks_answer_incomplete():
    r = run(ScriptedClient([text_msg(""), text_msg("")]), max_steps=1)
    assert r["incomplete"] is True and "could not produce" in r["answer"]


def test_llm_error_propagates():
    with pytest.raises(LLMError):
        run(ScriptedClient([LLMError("down", "llm_unavailable")]))


def test_history_is_sent_for_followups():
    history = []
    first = ScriptedClient([tool_msg("group_by_aggregate", GOOD_ARGS), text_msg("North, 400.")])
    from src.agent.memory import add_turn
    r1 = run(first)
    add_turn(history, "Which region sells most?", r1["answer"], r1["steps"])

    second = ScriptedClient([text_msg("Answer: North.")])
    run(second, "Show its trend", history)
    msgs = second.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "sum_Sales" in msgs[2]["content"] and msgs[3]["content"] == "Show its trend"


def test_dataframe_is_not_modified():
    before = DF.copy()
    run(ScriptedClient([tool_msg("group_by_aggregate", GOOD_ARGS), text_msg("ok")]))
    pd.testing.assert_frame_equal(DF, before)