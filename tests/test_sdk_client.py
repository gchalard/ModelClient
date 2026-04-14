"""SDK client unit tests (mocked transport; integration is covered in ``test_predict``)."""

from __future__ import annotations

import httpx

from modelrunner.sdk import ModelRunnerClient


def _handler(request: httpx.Request) -> httpx.Response:
    if request.method == "GET" and request.url.path == "/predict":
        return httpx.Response(
            200,
            json={"openapi": "3.1.0", "paths": {"/predict": {}}},
        )
    if request.method == "POST" and request.url.path == "/predict":
        return httpx.Response(200, json={"cluster_id": 0, "distance": 0.0})
    if request.method == "GET" and request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    return httpx.Response(404)


def test_sdk_predict_uses_post_json() -> None:
    transport = httpx.MockTransport(_handler)
    with httpx.Client(transport=transport, base_url="http://example") as hc:
        c = ModelRunnerClient(client=hc)
        out = c.predict(features={"feature_a": 1.0, "feature_b": 2.0})
        assert out == {"cluster_id": 0, "distance": 0.0}


def test_sdk_get_contract() -> None:
    transport = httpx.MockTransport(_handler)
    with httpx.Client(transport=transport, base_url="http://example") as hc:
        c = ModelRunnerClient(client=hc)
        doc = c.get_contract()
        assert doc["openapi"] == "3.1.0"


def test_sdk_explicit_base_url() -> None:
    transport = httpx.MockTransport(_handler)
    with httpx.Client(transport=transport, base_url="http://localhost:9999") as hc:
        c = ModelRunnerClient(client=hc)
        assert c.predict(features={"x": 1.0}) == {"cluster_id": 0, "distance": 0.0}


def test_sdk_constructor_host_port_builds_url() -> None:
    transport = httpx.MockTransport(_handler)
    with httpx.Client(transport=transport, base_url="http://127.0.0.1:8080") as hc:
        c = ModelRunnerClient(client=hc)
        c.predict(features={"a": 1.0})
    # Constructor path (no injected client) — smoke only; requires a reachable server.
    c2 = ModelRunnerClient("127.0.0.1", 8080)
    assert "127.0.0.1:8080" in str(c2._http.base_url)
    c2.close()
