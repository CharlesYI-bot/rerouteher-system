"""Smoke test: the app builds and /api/health responds (no DB or models needed for this route)."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_health(monkeypatch):
    monkeypatch.setattr("app.main.Embedder", lambda *args: None)
    monkeypatch.setattr("app.main.EscoTfidfMatcher.load", lambda *args: None)
    # build the app without triggering lifespan (no DB/model needed for /api/health)
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as client:
        # lifespan runs here; if models/DB are absent it logs warnings but still boots
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
