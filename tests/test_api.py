from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_valid_response():
    r = client.post("/predict", json={"url": "http://paypal-login-verify.xyz/account"})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ("phishing", "legitimate")
    assert 0 <= body["probability"] <= 1


def test_predict_rejects_empty_url():
    assert client.post("/predict", json={"url": ""}).status_code == 422