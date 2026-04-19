"""HTTP client for the ModelRunner inference API."""

from __future__ import annotations

from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import Any

import httpx

from modelrunner.manifest import Manifest
from modelrunner.sdk.predict_doc import feature_names_from_predict_doc


@dataclass(frozen=True, slots=True)
class ModelRunnerClientConfig:
    """Defaults for :class:`ModelRunnerClient` (scheme, timeouts)."""

    scheme: str = "http"
    timeout: float = 30.0


@dataclass(frozen=True, slots=True)
class AsyncModelRunnerClientConfig:
    """Defaults for :class:`AsyncModelRunnerClient`."""

    scheme: str = "http"
    timeout: float = 30.0


@dataclass(slots=True)
class ModelRunnerClient:
    """Sync client: ``ModelRunnerClient(host, port).predict(features={...})``."""

    host: str | None = None
    port: int | None = None
    base_url: str | None = None
    _: KW_ONLY
    scheme: str | None = None
    timeout: float | None = None
    config: ModelRunnerClientConfig | None = None
    client: InitVar[httpx.Client | None] = None
    _http: httpx.Client = field(init=False, repr=False)
    _owns_client: bool = field(init=False, repr=False)

    def __post_init__(self, client: httpx.Client | None) -> None:
        if client is not None:
            self._http = client
            self._owns_client = False
            return
        if self.config is not None:
            eff_scheme = self.scheme if self.scheme is not None else self.config.scheme
            eff_timeout = self.timeout if self.timeout is not None else self.config.timeout
        else:
            eff_scheme = self.scheme if self.scheme is not None else "http"
            eff_timeout = 30.0 if self.timeout is None else self.timeout
        if self.base_url is not None:
            url = self.base_url.rstrip("/")
        elif self.host is not None and self.port is not None:
            url = f"{eff_scheme}://{self.host}:{self.port}".rstrip("/")
        else:
            msg = "Provide (host, port), or base_url, or client="
            raise ValueError(msg)
        self._http = httpx.Client(base_url=url, timeout=eff_timeout)
        self._owns_client = True

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> ModelRunnerClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get_contract(self) -> dict[str, Any]:
        """GET /predict — OpenAPI contract for this deployment."""
        r = self._http.get("/predict")
        r.raise_for_status()
        return r.json()

    def get_features(self) -> list[str]:
        """Feature names for POST ``/predict``, in manifest order (from the OpenAPI fragment)."""
        return feature_names_from_predict_doc(self.get_contract())

    def get_metadata(self) -> dict[str, Any] | None:
        """Optional deployment ``metadata`` from the manifest (``None`` if unset)."""
        doc = self.get_contract()
        meta = doc.get("metadata")
        return meta if isinstance(meta, dict) else None

    def get_manifest(self) -> Manifest:
        """Full validated :class:`~modelrunner.manifest.Manifest` embedded in GET ``/predict``."""
        doc = self.get_contract()
        raw = doc.get("manifest")
        if not isinstance(raw, dict):
            msg = "GET /predict response has no manifest object; upgrade the ModelRunner server"
            raise ValueError(msg)
        return Manifest.from_dict(raw)

    def predict(self, *, features: dict[str, Any]) -> dict[str, Any]:
        """POST /predict with a ``{feature_name: value}`` body."""
        r = self._http.post("/predict", json=features)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict[str, Any]:
        """GET /health if exposed by the server."""
        r = self._http.get("/health")
        r.raise_for_status()
        return r.json()


@dataclass(slots=True)
class AsyncModelRunnerClient:
    """Async client with the same surface as :class:`ModelRunnerClient`."""

    host: str | None = None
    port: int | None = None
    base_url: str | None = None
    _: KW_ONLY
    scheme: str | None = None
    timeout: float | None = None
    config: AsyncModelRunnerClientConfig | None = None
    client: InitVar[httpx.AsyncClient | None] = None
    _http: httpx.AsyncClient = field(init=False, repr=False)
    _owns_client: bool = field(init=False, repr=False)

    def __post_init__(self, client: httpx.AsyncClient | None) -> None:
        if client is not None:
            self._http = client
            self._owns_client = False
            return
        if self.config is not None:
            eff_scheme = self.scheme if self.scheme is not None else self.config.scheme
            eff_timeout = self.timeout if self.timeout is not None else self.config.timeout
        else:
            eff_scheme = self.scheme if self.scheme is not None else "http"
            eff_timeout = 30.0 if self.timeout is None else self.timeout
        if self.base_url is not None:
            url = self.base_url.rstrip("/")
        elif self.host is not None and self.port is not None:
            url = f"{eff_scheme}://{self.host}:{self.port}".rstrip("/")
        else:
            msg = "Provide (host, port), or base_url, or client="
            raise ValueError(msg)
        self._http = httpx.AsyncClient(base_url=url, timeout=eff_timeout)
        self._owns_client = True

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncModelRunnerClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def get_contract(self) -> dict[str, Any]:
        r = await self._http.get("/predict")
        r.raise_for_status()
        return r.json()

    async def get_features(self) -> list[str]:
        return feature_names_from_predict_doc(await self.get_contract())

    async def get_metadata(self) -> dict[str, Any] | None:
        doc = await self.get_contract()
        meta = doc.get("metadata")
        return meta if isinstance(meta, dict) else None

    async def get_manifest(self) -> Manifest:
        doc = await self.get_contract()
        raw = doc.get("manifest")
        if not isinstance(raw, dict):
            msg = "GET /predict response has no manifest object; upgrade the ModelRunner server"
            raise ValueError(msg)
        return Manifest.from_dict(raw)

    async def predict(self, *, features: dict[str, Any]) -> dict[str, Any]:
        r = await self._http.post("/predict", json=features)
        r.raise_for_status()
        return r.json()

    async def health(self) -> dict[str, Any]:
        r = await self._http.get("/health")
        r.raise_for_status()
        return r.json()
