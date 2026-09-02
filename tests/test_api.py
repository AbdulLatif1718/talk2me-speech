from fastapi.testclient import TestClient

from talk2me_speech.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json().get("status", "")


def test_list_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json().get("models", {})
    assert "whisper-small" in models
    assert "xlsr-300m" in models


def test_transcribe_endpoint():
    payload = {
        "audio_path": "data/sample.wav",
        "model": "whisper-small",
        "language": "en",
    }
    response = client.post("/transcribe", json=payload)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "success"
    assert "transcript" in res_json
    assert res_json["model"] == "whisper-small"

