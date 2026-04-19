"""SDK client unit tests (mocked transport; integration is covered in ``test_predict``)."""

from __future__ import annotations

import asyncio

import httpx

from modelrunner.sdk import AsyncModelRunnerClient, ModelRunnerClient

_MOCK_MANIFEST = {
    "schema_version": 1,
    "model": {
        "id": "mock",
        "name": "Mock",
        "version": "1",
        "task": "test",
    },
    "runtime": {
        "adapter": "modelrunner.adapters.dummy:DummyAdapter",
        "artifacts": {"base_dir": "artifacts", "paths": {}},
    },
    "input": {
        "features": [
            {"name": "feature_a", "type": "float", "required": True},
            {"name": "feature_b", "type": "float", "required": True},
        ],
    },
    "output": {
        "fields": [
            {"name": "cluster_id", "type": "int", "required": True},
        ],
    },
    "metadata": {"note": "from-manifest"},
}

_PREDICT_DOC = {
    "openapi": "3.1.0",
    "info": {"title": "Mock", "version": "1"},
    "paths": {
        "/predict": {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "feature_a": {"type": "number"},
                                    "feature_b": {"type": "number"},
                                },
                                "required": ["feature_a", "feature_b"],
                            },
                        },
                    },
                },
                "required": True,
            },
        },
    },
    "metadata": {"note": "top-level"},
    "manifest": _MOCK_MANIFEST,
}


def _handler(request: httpx.Request) -> httpx.Response:
    if request.method == "GET" and request.url.path == "/predict":
        return httpx.Response(
            200,
            json=_PREDICT_DOC,
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


def test_sdk_get_features_metadata_manifest() -> None:
    transport = httpx.MockTransport(_handler)
    with httpx.Client(transport=transport, base_url="http://example") as hc:
        c = ModelRunnerClient(client=hc)
        assert c.get_features() == ["feature_a", "feature_b"]
        assert c.get_metadata() == {"note": "top-level"}
        m = c.get_manifest()
        assert m.model.id == "mock"
        assert m.metadata == {"note": "from-manifest"}


async def _async_sdk_roundtrip() -> None:
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://example") as hc:
        c = AsyncModelRunnerClient(client=hc)
        assert await c.get_features() == ["feature_a", "feature_b"]
        assert await c.get_metadata() == {"note": "top-level"}
        m = await c.get_manifest()
        assert m.model.id == "mock"


def test_async_sdk_get_features_metadata_manifest() -> None:
    asyncio.run(_async_sdk_roundtrip())


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
