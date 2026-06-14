import pytest
from flask import json

from app.api import app


@pytest.fixture
def client():
    
    #Flask test client fixture.
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_predict_success(client):
    
    #This test returns a valid answer for a valid input.

    payload = {
        "question": "What is the capital of France?",
        "context": "France's capital city is Paris."
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = json.loads(response.data)
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_predict_missing_fields(client):
    
    #This test returns error if question or context is missing.

    payloads = [
        {"question": "Some question"},
        {"context": "Some context"},
        {},
    ]

    for payload in payloads:
        response = client.post("/predict", json=payload)
        assert response.status_code == 400

        data = json.loads(response.data)
        assert "error" in data
