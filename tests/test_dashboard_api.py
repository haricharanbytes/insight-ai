import json

from fastapi.testclient import TestClient

from main import app
from tests.fakes import ScriptedClient, json_msg

client = TestClient(app)
CSV = ("Region,Segment,Sales,Profit,Order Date\n" + "\n".join(
    f"{['East','West','North','South'][i % 4]},{['A','B','C'][i % 3]},{100 + i * 7},{10 + i % 9},2020-{i % 12 + 1:02d}-10"
    for i in range(36))).encode()


def session():
    h = {"X-Session-ID": client.post("/api/sessions").json()["session_id"]}
    client.post("/api/upload", files={"file": ("d.csv", CSV, "text/csv")}, headers=h)
    return h


def no_llm(monkeypatch, fake=None):
    monkeypatch.setattr("src.api.dashboard_routes.get_llm_client", lambda: fake)


def test_requires_dataset_and_dashboard(monkeypatch):
    sid = client.post("/api/sessions").json()["session_id"]
    assert client.post("/api/dashboard", json={}, headers={"X-Session-ID": sid}).status_code == 409
    h = session()
    r = client.get("/api/dashboard", headers=h)
    assert r.status_code == 404 and r.json()["code"] == "no_dashboard"


def test_create_auto_without_prompt_or_llm(monkeypatch):
    no_llm(monkeypatch)
    h = session()
    r = client.post("/api/dashboard", json={}, headers=h).json()
    assert r["meta"]["source"] == "auto" and len(r["results"]) == len(r["dashboard"]["widgets"])
    assert all(x["ok"] for x in r["results"])
    r2 = client.post("/api/dashboard", json={"prompt": "profit by region"}, headers=h).json()
    assert r2["meta"]["source"] == "auto" and "No LLM" in r2["meta"]["note"]


def test_create_with_llm_prompt(monkeypatch):
    spec = {"title": "P", "widgets": [{"type": "kpi", "metric": {"column": "Profit", "op": "sum"}}]}
    no_llm(monkeypatch, ScriptedClient([json_msg(spec)]))
    r = client.post("/api/dashboard", json={"prompt": "profit"}, headers=session()).json()
    assert r["meta"]["source"] == "llm" and r["results"][0]["ok"]


def test_filter_flow_recomputes_only_affected_widgets(monkeypatch):
    no_llm(monkeypatch)
    h = session()
    client.post("/api/dashboard", json={}, headers=h)

    # Filter on Region explicitly (the auto dashboard's first category filter is Segment).
    r = client.put("/api/dashboard/filters", headers=h, json={"filters": [
        {"kind": "category", "column": "Region", "values": ["East"]}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recomputed"]

    total = next(x for x in body["results"] if x["title"] == "Total rows")
    assert total["ok"], total                      # shows the real error if it ever fails
    assert total["value"] == 9                     # 36 rows, 1 in 4 is East

    again = client.get("/api/dashboard", headers=h).json()
    assert again["dashboard"]["filters"][0]["column"] == "Region"
    assert again["dashboard"]["filters"][0]["values"] == ["East"]


def test_filter_matching_no_rows_gives_a_clear_widget_error(monkeypatch):
    no_llm(monkeypatch)
    h = session()
    client.post("/api/dashboard", json={}, headers=h)
    r = client.put("/api/dashboard/filters", headers=h, json={"filters": [
        {"kind": "category", "column": "Segment", "values": ["East"]}]})   # no such segment
    assert r.status_code == 200
    total = next(x for x in r.json()["results"] if x["title"] == "Total rows")
    assert total["ok"] is False and total["error"]["message"] == "No rows match the current filters."


def test_invalid_filters_are_rejected_and_not_applied(monkeypatch):
    no_llm(monkeypatch)
    h = session()
    client.post("/api/dashboard", json={}, headers=h)
    bad = client.put("/api/dashboard/filters", headers=h, json={"filters": [
        {"kind": "category", "column": "Sales", "values": ["x"]}]})
    assert bad.status_code == 422 and bad.json()["code"] == "invalid_dashboard"
    assert bad.json()["details"]["problems"][0]["code"] == "invalid_column_type"
    assert client.get("/api/dashboard", headers=h).json()["dashboard"]["filters"][0]["values"] == []


def test_remove_reorder_rename(monkeypatch):
    no_llm(monkeypatch)
    h = session()
    d = client.post("/api/dashboard", json={}, headers=h).json()["dashboard"]
    ids = [w["id"] for w in d["widgets"]]
    r = client.delete(f"/api/dashboard/widgets/{ids[0]}", headers=h).json()
    assert [w["id"] for w in r["dashboard"]["widgets"]] == ids[1:]
    assert client.delete("/api/dashboard/widgets/nope", headers=h).status_code == 404

    rev = list(reversed(ids[1:]))
    r = client.put("/api/dashboard/order", headers=h, json={"order": rev}).json()
    assert [w["id"] for w in r["dashboard"]["widgets"]] == rev
    assert client.put("/api/dashboard/order", headers=h, json={"order": ["w1"]}).status_code == 422

    r = client.put("/api/dashboard/title", headers=h, json={"title": "My board"}).json()
    assert r["dashboard"]["title"] == "My board"


def test_add_widget_by_prompt(monkeypatch):
    no_llm(monkeypatch)
    h = session()
    client.post("/api/dashboard", json={}, headers=h)
    r = client.post("/api/dashboard/widgets", json={"prompt": "add a chart"}, headers=h).json()
    assert r["meta"]["added"] == 0 and "needs the LLM" in r["meta"]["note"]

    new = {"title": "t", "widgets": [{"type": "histogram", "column": "Sales"}]}
    no_llm(monkeypatch, ScriptedClient([json_msg(new)]))
    r = client.post("/api/dashboard/widgets", json={"prompt": "histogram of sales"}, headers=h).json()
    assert r["meta"]["added"] == 1 and r["dashboard"]["widgets"][-1]["type"] == "histogram"


def test_save_and_load_roundtrip_and_bad_files(monkeypatch):
    no_llm(monkeypatch)
    h = session()
    saved = client.post("/api/dashboard", json={}, headers=h).json()["dashboard"]
    f = client.get("/api/dashboard/save", headers=h)
    assert f.status_code == 200 and "attachment" in f.headers["content-disposition"]
    assert json.loads(f.text) == saved

    h2 = session()
    ok = client.post("/api/dashboard/load", headers=h2, files={"file": ("d.json", f.content, "application/json")})
    assert ok.status_code == 200 and ok.json()["dashboard"] == saved

    assert client.post("/api/dashboard/load", headers=h2, files={"file": ("d.json", b"not json")}).json()["code"] == "invalid_json"
    wrong = json.dumps({"title": "t", "widgets": [{"type": "histogram", "column": "Nope"}]}).encode()
    r = client.post("/api/dashboard/load", headers=h2, files={"file": ("d.json", wrong)})
    assert r.status_code == 422 and "does not fit the current dataset" in r.json()["message"]
    big = b"{" + b" " * (200 * 1024 + 10)
    assert client.post("/api/dashboard/load", headers=h2, files={"file": ("d.json", big)}).status_code == 413


def test_new_dataset_clears_dashboard(monkeypatch):
    no_llm(monkeypatch)
    h = session()
    client.post("/api/dashboard", json={}, headers=h)
    client.post("/api/upload", files={"file": ("e.csv", b"a,b\n1,2\n3,4\n", "text/csv")}, headers=h)
    assert client.get("/api/dashboard", headers=h).status_code == 404