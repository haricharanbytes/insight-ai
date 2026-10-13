import io

import pandas as pd
from fastapi.testclient import TestClient

from main import app
from src.data.profiler import profile_dataset
from src.reports.exporter import build_report, dataframe_to_csv, safe_cell, table_to_csv
from tests.fakes import ScriptedClient, text_msg, tool_msg

client = TestClient(app)
CSV = b"Region,Sales,Profit\nEast,100,10\nEast,200,-20\nWest,50,5\nWest,150,15\nNorth,400,60\n"


def test_safe_cell_only_touches_dangerous_text():
    assert safe_cell("=SUM(A1)") == "'=SUM(A1)"
    assert safe_cell("+1") == "'+1" and safe_cell("@x") == "'@x" and safe_cell("-cmd") == "'-cmd"
    assert safe_cell("\tTAB") == "'\tTAB"
    assert safe_cell("normal") == "normal" and safe_cell(-5.0) == -5.0 and safe_cell(None) is None


def test_table_csv_escapes_formulas_but_keeps_negative_numbers():
    t = {"columns": ["name", "profit"], "rows": [["=HYPERLINK(\"x\")", -20.0], ["ok", None]]}
    lines = table_to_csv(t).splitlines()
    assert lines[1].startswith("\"'=HYPERLINK") and lines[1].endswith(",-20.0")
    assert lines[2] == "ok,"


def test_dataframe_csv_escapes_text_headers_and_cells():
    df = pd.DataFrame({"=bad": ["@evil", "fine"], "n": [-1, 2]})
    out = dataframe_to_csv(df)
    assert out.splitlines()[0] == "'=bad,n"
    assert "'@evil,-1" in out and "fine,2" in out


def test_report_escapes_html_and_labels_sections():
    profile = profile_dataset(pd.DataFrame({"a": range(10)}))
    out = {"tool": "t", "method": "<script>alert(1)</script>", "inputs": {}, "rows_used": 10,
           "table": None, "warnings": ["<b>warn</b>"], "result": {}}
    items = [{"question": "q <i>", "answer": "line1\nline2 <u>", "unverified": ["9.9"], "tool_steps": [{"output": out, "chart": None}]}]
    doc = build_report("f<1>.csv", profile, [], items)
    assert "<script>alert(1)" not in doc and "&lt;script&gt;" in doc
    assert "&lt;b&gt;warn" in doc and "f&lt;1&gt;.csv" in doc
    assert "Observed" in doc and "Interpretation" in doc and "Generated" in doc and "9.9" in doc


def _session():
    h = {"X-Session-ID": client.post("/api/sessions").json()["session_id"]}
    client.post("/api/upload", files={"file": ("sales.csv", CSV, "text/csv")}, headers=h)
    return h


def test_export_endpoints(monkeypatch):
    h = _session()
    args = {"group_by": ["Region"], "metrics": [{"column": "Sales", "op": "sum"}]}
    monkeypatch.setattr("src.api.routes.get_llm_client",
                        lambda: ScriptedClient([tool_msg("group_by_aggregate", args), text_msg("Answer: North has 400.")]))
    r = client.post("/api/ask", json={"question": "Which region sells most?"}, headers=h).json()
    rid = r["steps"][0]["result_id"]

    csv_resp = client.get(f"/api/export/result/{rid}.csv", headers=h)
    assert csv_resp.status_code == 200 and "attachment" in csv_resp.headers["content-disposition"]
    assert csv_resp.text.splitlines()[0].startswith("Region,sum_Sales") and "North,400,44.44" in csv_resp.text
    
    ds = client.post("/api/export/dataset.csv", json={"filters": [{"column": "Region", "op": "eq", "value": "East"}]}, headers=h)
    assert ds.status_code == 200 and len(ds.text.strip().splitlines()) == 3  # header + 2 East rows
    assert "_filtered" in ds.headers["content-disposition"]
    full = client.post("/api/export/dataset.csv", json={}, headers=h)
    assert len(full.text.strip().splitlines()) == 6

    rep = client.get("/api/export/report.html", headers=h)
    assert rep.status_code == 200 and "Which region sells most?" in rep.text and "Plotly.newPlot" in rep.text


def test_export_errors():
    h = _session()
    assert client.get("/api/export/result/nope.csv", headers=h).status_code == 404
    bad = client.post("/api/export/dataset.csv", json={"filters": [{"column": "Region", "op": "eq", "value": "Mars"}]}, headers=h)
    assert bad.status_code == 422 and bad.json()["code"] == "no_rows"
    assert client.post("/api/export/dataset.csv", json={}, headers={"X-Session-ID": "x"}).status_code == 404