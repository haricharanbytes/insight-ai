from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_security_headers_present():
    r = client.get("/")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


def test_health_does_not_leak_secrets():
    body = str(client.get("/api/health").json()).lower()
    assert "key" not in body.replace("llm_model", "")


def test_internal_errors_do_not_leak_details(monkeypatch):
    sid = client.post("/api/sessions").json()["session_id"]
    h = {"X-Session-ID": sid}
    client.post("/api/upload", files={"file": ("a.csv", b"a,b\n1,2\n3,4\n", "text/csv")}, headers=h)

    def boom(*a, **k):
        raise RuntimeError("SECRET-INTERNAL-DETAIL")

    monkeypatch.setattr("src.api.routes.run_tool", boom)
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/api/tools/filter_data", json={"params": {}}, headers=h)
    assert r.status_code == 500 and "SECRET" not in r.text