from fastapi.testclient import TestClient

from main import app
from src.agent.llm_client import LLMError
from src.reports.eda_summary import llm_summary
from tests.fakes import ScriptedClient, text_msg

client = TestClient(app)
CSV = ("Region,Segment,Sales,Profit,Order Date\n" + "\n".join(
    f"{['East','West','North','South'][i % 4]},{['A','B','C'][i % 3]},{100 + i * 7},{10 + i % 9},2020-{i % 12 + 1:02d}-10"
    for i in range(60))).encode()


def session():
    h = {"X-Session-ID": client.post("/api/sessions").json()["session_id"]}
    client.post("/api/upload", files={"file": ("d.csv", CSV, "text/csv")}, headers=h)
    return h


def no_llm(monkeypatch, fake=None):
    monkeypatch.setattr("src.api.eda_routes.get_llm_client", lambda: fake)


def test_three_downloads(monkeypatch):
    no_llm(monkeypatch)
    h = session()
    for ext, magic in (("html", b"<!doctype html>"), ("xlsx", b"PK"), ("pdf", b"%PDF-")):
        r = client.post(f"/api/eda/report.{ext}", json={"title": "T", "author": "A"}, headers=h)
        assert r.status_code == 200 and r.content.startswith(magic)
        assert f"d_eda.{ext}" in r.headers["content-disposition"]


def test_options_and_errors(monkeypatch):
    no_llm(monkeypatch)
    h = session()
    r = client.post("/api/eda/report.html", json={"sections": ["overview"], "filters": [{"column": "Region", "op": "eq", "value": "East"}]}, headers=h)
    assert r.status_code == 200 and "after filters; 60 before" in r.text
    assert client.post("/api/eda/report.html", json={"sections": ["nope"]}, headers=h).status_code == 422
    assert client.post("/api/eda/report.html", json={"filters": [{"column": "Region", "op": "eq", "value": "Mars"}]}, headers=h).json()["code"] == "no_rows"
    sid = client.post("/api/sessions").json()["session_id"]
    assert client.post("/api/eda/report.pdf", json={}, headers={"X-Session-ID": sid}).status_code == 409


def test_ai_summary_without_llm_keeps_rules_and_warns(monkeypatch):
    no_llm(monkeypatch)
    r = client.post("/api/eda/report.html", json={"ai_summary": True}, headers=session())
    assert "needs an LLM" in r.text and "Observed" in r.text


def test_ai_summary_grounding(monkeypatch):
    report = {"summary": ["The data has 60 rows and 5 columns."], "sections": []}
    good, why = llm_summary(ScriptedClient([text_msg("The data has 60 rows and 5 columns.")]), report)
    assert good and why is None
    bad, why = llm_summary(ScriptedClient([text_msg("The data has 61 rows and 5 columns.")]), report)
    assert bad is None and "61" in why
    none, why = llm_summary(ScriptedClient([LLMError("x", "llm_unavailable")]), report)
    assert none is None and "unavailable" in why
    no_llm(monkeypatch, ScriptedClient([text_msg("The data has 60 rows.")]))
    r = client.post("/api/eda/report.html", json={"ai_summary": True, "sections": ["overview"]}, headers=session())
    assert "AI-written" in r.text