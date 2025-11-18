from fastapi.testclient import TestClient

from app import app


def test_healthcheck():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_chat_endpoint_returns_sources():
    with TestClient(app) as client:
        response = client.post("/chat", json={"text": "Explique o pipeline"})
        assert response.status_code == 200
        payload = response.json()
        assert "reply" in payload
        assert payload["sources"]
