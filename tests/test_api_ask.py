from fastapi.testclient import TestClient

from main import app
from src.agent.llm_client import LLMError
from tests.fakes import ScriptedClient, text_msg, tool_msg

client = TestClient(app)
CSV = b"Region,Sales,Profit\nEast,100,10\nEast,200,-20\nWest,50,5\nWest,150,15\nNorth,400,60\n"
ARGS = {"group_by": ["Region"], "metrics": [{"column": "Sales", "op": "sum"}]}


def session_with_data():
    sid = client.post("/api/sessions").json()["session_id"]
    h = {"X-Session-ID": sid}
    client.post("/api/upload", files={"file": ("sales.csv", CSV, "text/csv")}, headers=h)
    return h


def use(monkeypatch, fake):
    monkeypatch.setattr("src.api.routes.get_llm_client", lambda: fake)


def test_ask_runs_tools_and_stores_results(monkeypatch):
    h = session_with_data()
    use(monkeypatch, ScriptedClient([tool_msg("group_by_aggregate", ARGS),
                                     text_msg("Answer: North leads with 400.")]))
    r = client.post("/api/ask", json={"question": "Which region sells most?"}, headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Answer: North") and body["mode"] == "llm"
    assert body["grounding"]["unverified_numbers"] == []
    rid = body["steps"][0]["result_id"]
    stored = client.get(f"/api/results/{rid}", headers=h).json()
    assert stored["output"]["result"]["top_group"]["Region"] == "North"


def test_followup_uses_conversation_memory(monkeypatch):
    h = session_with_data()
    use(monkeypatch, ScriptedClient([tool_msg("group_by_aggregate", ARGS), text_msg("North, 400.")]))
    client.post("/api/ask", json={"question": "Which region sells most?"}, headers=h)

    second = ScriptedClient([text_msg("Answer: it is North."), text_msg("Answer: it is North.")])
    use(monkeypatch, second)
    client.post("/api/ask", json={"question": "Show its profit"}, headers=h)
    msgs = second.calls[0]["messages"]
    assert len(msgs) == 4 and "Which region sells most?" in msgs[1]["content"]


def test_ask_needs_dataset_and_llm(monkeypatch):
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post("/api/ask", json={"question": "hi"}, headers={"X-Session-ID": sid})
    assert r.status_code == 409

    h = session_with_data()
    use(monkeypatch, None)
    r = client.post("/api/ask", json={"question": "hi"}, headers=h)
    assert r.status_code == 503 and r.json()["code"] == "llm_not_configured"


def test_ask_validation_and_llm_errors(monkeypatch):
    h = session_with_data()
    use(monkeypatch, ScriptedClient([]))
    assert client.post("/api/ask", json={"question": "   "}, headers=h).json()["code"] == "empty_question"
    assert client.post("/api/ask", json={"question": "x" * 2001}, headers=h).status_code == 422

    use(monkeypatch, ScriptedClient([LLMError("rate limited", "llm_unavailable")]))
    r = client.post("/api/ask", json={"question": "hi"}, headers=h)
    assert r.status_code == 503 and r.json()["code"] == "llm_unavailable"