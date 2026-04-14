"""API tests for /predict."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modelrunner.main import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_get_predict_contract(client: TestClient) -> None:
    r = client.get("/predict")
    assert r.status_code == 200
    data = r.json()
    assert data["openapi"] == "3.1.0"
    assert "/predict" in data["paths"]


def test_post_predict_ok(client: TestClient) -> None:
    r = client.post(
        "/predict",
        json={"feature_a": 1.0, "feature_b": 2.0, "optional_flag": True},
    )
    assert r.status_code == 200
    assert r.json() == {"cluster_id": 0, "distance": 0.0}


def test_post_predict_unknown_key(client: TestClient) -> None:
    r = client.post(
        "/predict",
        json={"feature_a": 1.0, "feature_b": 2.0, "optional_flag": False, "extra": 1},
    )
    assert r.status_code == 422
