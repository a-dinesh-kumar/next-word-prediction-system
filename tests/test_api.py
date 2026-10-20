
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200

    response_data = response.json()
    assert response_data["status"] == "healthy"
    assert response_data["model_loaded"] is True


def test_homepage_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "NexText" in response.text


def test_prediction_returns_candidates(client):
    response = client.post(
        "/predict",
        json={
            "input_text": "the team won the",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    response_data = response.json()

    assert "predictions" in response_data
    assert len(response_data["predictions"]) == 5

    for prediction in response_data["predictions"]:
        assert "word" in prediction
        assert "probability" in prediction
        assert isinstance(prediction["word"], str)
        assert isinstance(prediction["probability"], (int, float))
        assert 0 <= prediction["probability"] <= 1


def test_prediction_respects_top_k(client):
    response = client.post(
        "/predict",
        json={
            "input_text": "the team won the",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"input_text": "", "top_k": 5},
        {"input_text": "the team won", "top_k": 0},
        {"input_text": "the team won", "top_k": 51},
    ],
)
def test_invalid_prediction_request(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_generation_returns_text(client):
    response = client.post(
        "/generate",
        json={
            "seed_text": "the team won the",
            "num_words": 10,
            "strategy": "greedy",
        },
    )

    assert response.status_code == 200

    response_data = response.json()

    assert isinstance(response_data["generated_text"], str)
    assert response_data["generated_text"].strip()
    assert response_data["strategy"] == "greedy"


def test_sampled_generation(client):
    response = client.post(
        "/generate",
        json={
            "seed_text": "the team won the",
            "num_words": 10,
            "strategy": "sampled",
            "temperature": 0.8,
            "top_k": 10,
            "repetition_penalty": 0.7,
        },
    )

    assert response.status_code == 200
    assert response.json()["strategy"] == "sampled"


@pytest.mark.parametrize(
    "payload",
    [
        {"seed_text": "", "num_words": 10},
        {"seed_text": "the team won", "num_words": 0},
        {"seed_text": "the team won", "num_words": 101},
        {"seed_text": "the team won", "strategy": "unknown"},
    ],
)
def test_invalid_generation_request(client, payload):
    response = client.post("/generate", json=payload)

    assert response.status_code == 422


def test_frontend_assets_are_served(client):
    css_response = client.get("/static/style.css")
    js_response = client.get("/static/app.js")

    assert css_response.status_code == 200
    assert js_response.status_code == 200
