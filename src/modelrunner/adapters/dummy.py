"""Placeholder adapter for local development and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modelrunner.manifest import Manifest
from modelrunner.ports import ModelPredictor


class DummyAdapter:
    """Returns a fixed payload matching typical clustering output."""

    def load(self, manifest: Manifest, manifest_path: Path) -> None:
        _ = manifest
        _ = manifest_path

    def predict(self, features: list[Any]) -> dict[str, Any]:
        _ = features
        return {"cluster_id": 0, "distance": 0.0}
