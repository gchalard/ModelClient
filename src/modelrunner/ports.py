"""Ports (interfaces) for pluggable model backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from modelrunner.manifest import Manifest


@runtime_checkable
class ModelPredictor(Protocol):
    """Load artifacts once, then predict from an ordered feature vector."""

    def load(self, manifest: Manifest, manifest_path: Path) -> None:
        """Resolve artifact paths and load model state."""

    def predict(self, features: list[Any]) -> dict[str, Any]:
        """`features` matches manifest `input.features` order."""
