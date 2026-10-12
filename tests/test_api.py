import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app
from src.api.sessions import SessionStore

client = TestClient(app)

CSV = b"Region,Sales,Profit\nEast,100,10\nEast,200,-20\nWest,50,5\nWest,150,15\nNorth,400,60\n"


def new_session() -> dict:
    sid = client.post("/api/sessions").json()["session_id"]
    return {"X-Session-ID": sid}


def upload_csv(headers, content=CSV, name="sales.csv"):
    return client.post("/api/upload", files={"file": (name, content, "text/csv")}, headers=headers)


def xlsx_bytes(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    return buf.getvalue()


def test_health_hides_secrets():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert "llm_api_key" not in body and "api_key" not in str(body).lower()


def test_session_required():
    assert client.get("/api/dataset").status_code == 401
    r = client.get("/api/dataset", headers={"X-Session-ID": "does-not-exist"})
    assert r.status_code == 404 and r.json()["code"] == "session_not_found"


def test_dataset_required_before_tools():
    r = client.post("/api/tools/group_by_aggregate", json={"params": {}}, headers=new_session())
    assert r.status_code == 409 and r.json()["code"] == "no_dataset"


def test_upload_csv_returns_profile():
    h = new_session()
    r = upload_csv(h)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "loaded" and body["filename"] == "sales.csv"
    assert body["profile"]["summary"]["rows"] == 5
    assert client.get("/api/dataset", headers=h).json()["profile"]["summary"]["columns"] == 3


def test_preview_limits_rows():
    h = new_session()
    upload_csv(h)
    t = client.get("/api/dataset/preview?rows=2", headers=h).json()
    assert len(t["rows"]) == 2 and t["total_rows"] == 5 and t["truncated"] is True
    assert client.get("/api/dataset/preview?rows=0", headers=h).status_code == 422


def test_bad_uploads_are_rejected_cleanly():
    h = new_session()
    r = upload_csv(h, b"", "empty.csv")
    assert r.status_code == 400 and r.json()["code"] == "empty_file"
    r = upload_csv(h, b"hello", "notes.txt")
    assert r.status_code == 400 and r.json()["code"] == "unsupported_file_type"
    r = upload_csv(h, b"\x00\x01\x02binary", "x.csv")
    assert r.status_code == 400 and r.json()["code"] == "malformed_file"


def test_multi_sheet_excel_flow():
    h = new_session()
    content = xlsx_bytes({"Sales": pd.DataFrame({"x": [1, 2]}), "Costs": pd.DataFrame({"y": [9]})})
    r = client.post("/api/upload", files={"file": ("book.xlsx", content)}, headers=h)
    assert r.json() == {"status": "sheet_selection_required", "filename": "book.xlsx",
                        "sheet_names": ["Sales", "Costs"]}
    assert client.get("/api/dataset", headers=h).status_code == 409  # nothing loaded yet

    r = client.post("/api/upload/sheet", json={"sheet_name": "Costs"}, headers=h)
    assert r.json()["status"] == "loaded" and r.json()["sheet_name"] == "Costs"
    again = client.post("/api/upload/sheet", json={"sheet_name": "Costs"}, headers=h)
    assert again.status_code == 400 and again.json()["code"] == "no_pending_upload"


def test_run_tool_and_fetch_result():
    h = new_session()
    upload_csv(h)
    r = client.post("/api/tools/group_by_aggregate", headers=h, json={"params": {
        "group_by": ["Region"], "metrics": [{"column": "Sales", "op": "sum"}]}})
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["top_group"]["Region"] == "North"
    assert body["result"]["top_group"]["sum_Sales"] == 400.0
    stored = client.get(f"/api/results/{body['result_id']}", headers=h).json()
    assert stored["output"]["tool"] == "group_by_aggregate"


def test_tool_errors():
    h = new_session()
    upload_csv(h)
    r = client.post("/api/tools/group_by_aggregate", headers=h, json={"params": {
        "group_by": ["Region"], "metrics": [{"column": "Sale", "op": "sum"}]}})
    assert r.status_code == 422 and r.json()["code"] == "column_not_found"
    assert "Sales" in r.json()["details"]["suggestions"]["Sale"]

    r = client.post("/api/tools/nope", headers=h, json={"params": {}})
    assert r.status_code == 404 and r.json()["code"] == "unknown_tool"

    r = client.post("/api/tools/group_by_aggregate", headers=h, json={"params": {"bogus": 1}})
    assert r.status_code == 422 and r.json()["code"] == "invalid_parameter"

    r = client.post("/api/tools/group_by_aggregate", headers=h, json={"params": {"df": "x"}})
    assert r.status_code == 422 and r.json()["code"] == "invalid_parameter"


def test_sessions_are_isolated():
    a, b = new_session(), new_session()
    upload_csv(a)
    assert client.get("/api/dataset", headers=a).status_code == 200
    assert client.get("/api/dataset", headers=b).status_code == 409
    rid = client.post("/api/tools/filter_data", headers=a, json={"params": {
        "filters": [{"column": "Region", "op": "eq", "value": "East"}]}}).json()["result_id"]
    assert client.get(f"/api/results/{rid}", headers=b).status_code == 404


def test_new_upload_replaces_dataset_and_clears_results():
    h = new_session()
    upload_csv(h)
    rid = client.post("/api/tools/filter_data", headers=h, json={"params": {
        "filters": [{"column": "Sales", "op": "gt", "value": 0}]}}).json()["result_id"]
    upload_csv(h, b"a,b\n1,2\n3,4\n", "other.csv")
    assert client.get("/api/dataset", headers=h).json()["filename"] == "other.csv"
    assert client.get(f"/api/results/{rid}", headers=h).status_code == 404


def test_clear_dataset_and_delete_session():
    h = new_session()
    upload_csv(h)
    assert client.delete("/api/dataset", headers=h).json() == {"cleared": True}
    assert client.get("/api/dataset", headers=h).status_code == 409
    assert client.delete("/api/sessions", headers=h).json() == {"deleted": True}
    assert client.get("/api/dataset", headers=h).status_code == 404


def test_session_expiry_and_cap():
    store = SessionStore(ttl_seconds=-1)
    s = store.create()
    assert store.get(s.id) is None  # already expired

    small = SessionStore(ttl_seconds=3600, max_sessions=2)
    first, second, third = small.create(), small.create(), small.create()
    assert small.get(first.id) is None  # least recently used was evicted
    assert small.get(second.id) and small.get(third.id)