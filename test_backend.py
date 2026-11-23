"""
Unit and Integration Tests for MindScore FastAPI Backend
"""

import pytest
from fastapi.testclient import TestClient
from main import app, DEFAULT_VALUES, DATASET_MEAN

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data


def test_metadata():
    response = client.get("/api/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "ranges" in data
    assert "defaults" in data
    assert data["dataset_mean"] == DATASET_MEAN
    assert "Instagram" in data["categories"]["Most_Used_Platform"]
    assert "India" in data["categories"]["Grouped_country"]


def test_predict_endpoint_valid():
    response = client.post("/api/predict", json=DEFAULT_VALUES)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "band" in data
    assert "diff_from_mean" in data
    assert "what_if" in data
    assert 0.0 <= data["score"] <= 10.0
    assert len(data["what_if"]) > 0
    # Check structure of what-if items
    for item in data["what_if"]:
        assert "label" in item
        assert "delta" in item
        assert "direction" in item
        assert item["direction"] in ["pos", "neg", "zero"]


def test_predict_endpoint_custom_values():
    custom_input = dict(DEFAULT_VALUES)
    custom_input["Sleep_Hours_Per_Night"] = 8.5
    custom_input["Avg_Daily_Usage_Hours"] = 2.0
    custom_input["Physical_Activity_Hours"] = 3.0
    custom_input["Stress_Level"] = "Low"

    response = client.post("/api/predict", json=custom_input)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["score"] <= 10.0


def test_predict_endpoint_invalid_category():
    invalid_input = dict(DEFAULT_VALUES)
    invalid_input["Most_Used_Platform"] = "InvalidPlatformApp"
    response = client.post("/api/predict", json=invalid_input)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation)


def test_what_if_endpoint():
    response = client.post("/api/what-if", json=DEFAULT_VALUES)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_static_endpoints():
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "MindScore" in r_index.text

    r_css = client.get("/style.css")
    assert r_css.status_code == 200

    r_js = client.get("/script.js")
    assert r_js.status_code == 200


if __name__ == "__main__":
    pytest.main(["-v", __file__])
