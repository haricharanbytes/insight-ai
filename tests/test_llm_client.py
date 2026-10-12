import json

import httpx
import pytest

from src.agent.llm_client import LLMClient, LLMError, parse_tool_calls

SECRET = "sk-secret-123"


def ok_body(content="hi"):
    return {"choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


def make_client(handler, models=("m1", "m2"), retries=1, key=SECRET):
    return LLMClient(key, "https://example.test/api/v1", list(models), max_retries=retries,
                     backoff_seconds=0, transport=httpx.MockTransport(handler))


def model_of(request):
    return json.loads(request.content)["model"]


MSGS = [{"role": "user", "content": "hello"}]


def test_success_returns_message_model_and_usage():
    client = make_client(lambda req: httpx.Response(200, json=ok_body("hello back")))
    r = client.chat(MSGS)
    assert r.message["content"] == "hello back"
    assert r.model == "m1" and r.attempts == 1
    assert r.usage["prompt_tokens"] == 5


def test_request_carries_auth_and_tools():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        seen["url"] = str(req.url)
        return httpx.Response(200, json=ok_body())

    make_client(handler).chat(MSGS, tools=[{"type": "function", "function": {"name": "x"}}])
    assert seen["auth"] == f"Bearer {SECRET}"
    assert seen["body"]["tools"] and seen["body"]["tool_choice"] == "auto"
    assert seen["url"] == "https://example.test/api/v1/chat/completions"


def test_retries_a_429_then_succeeds():
    calls = []

    def handler(req):
        calls.append(model_of(req))
        return httpx.Response(429) if len(calls) == 1 else httpx.Response(200, json=ok_body())

    r = make_client(handler).chat(MSGS)
    assert calls == ["m1", "m1"] and r.model == "m1" and r.attempts == 2


def test_falls_back_to_next_model_after_retries():
    calls = []

    def handler(req):
        calls.append(model_of(req))
        return httpx.Response(500) if model_of(req) == "m1" else httpx.Response(200, json=ok_body())

    r = make_client(handler, retries=1).chat(MSGS)
    assert calls == ["m1", "m1", "m2"]
    assert r.model == "m2"


def test_403_skips_model_without_retrying():
    calls = []

    def handler(req):
        calls.append(model_of(req))
        return httpx.Response(403) if model_of(req) == "m1" else httpx.Response(200, json=ok_body())

    r = make_client(handler, retries=3).chat(MSGS)
    assert calls == ["m1", "m2"] and r.model == "m2"


def test_401_fails_fast_with_auth_code():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(401)

    with pytest.raises(LLMError) as exc:
        make_client(handler).chat(MSGS)
    assert exc.value.code == "llm_auth" and len(calls) == 1


def test_error_body_inside_200_is_treated_as_failure():
    def handler(req):
        if model_of(req) == "m1":
            return httpx.Response(200, json={"error": {"message": "provider down"}})
        return httpx.Response(200, json=ok_body())

    assert make_client(handler, retries=0).chat(MSGS).model == "m2"


def test_all_models_failing_raises_and_never_leaks_the_key():
    with pytest.raises(LLMError) as exc:
        make_client(lambda req: httpx.Response(429)).chat(MSGS)
    assert exc.value.code == "llm_unavailable"
    assert SECRET not in str(exc.value)
    assert "rate limit" in exc.value.message


def test_network_error_is_handled():
    def handler(req):
        raise httpx.ConnectError("boom")

    with pytest.raises(LLMError) as exc:
        make_client(handler, retries=0).chat(MSGS)
    assert exc.value.code == "llm_unavailable"


def test_not_configured():
    with pytest.raises(LLMError) as exc:
        LLMClient("", "https://example.test", ["m"])
    assert exc.value.code == "llm_not_configured"


# ---------------- parse_tool_calls ----------------

def _call(arguments, name="group_by_aggregate", cid="c1"):
    return {"tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}}]}


def test_parse_valid_tool_call():
    out = parse_tool_calls(_call('{"group_by": ["Region"]}'))
    assert out == [{"id": "c1", "name": "group_by_aggregate",
                    "arguments": {"group_by": ["Region"]}, "error": None}]


def test_parse_dict_and_empty_arguments():
    assert parse_tool_calls(_call({"a": 1}))[0]["arguments"] == {"a": 1}
    assert parse_tool_calls(_call(""))[0]["arguments"] == {}


def test_parse_bad_arguments_reports_error():
    assert parse_tool_calls(_call("{not json"))[0]["error"] == "arguments were not valid JSON"
    assert parse_tool_calls(_call("[1, 2]"))[0]["error"] == "arguments must be a JSON object"


def test_parse_no_tool_calls_and_missing_id():
    assert parse_tool_calls({"content": "just text"}) == []
    msg = {"tool_calls": [{"function": {"name": "x", "arguments": "{}"}}]}
    assert parse_tool_calls(msg)[0]["id"] == "call_0"