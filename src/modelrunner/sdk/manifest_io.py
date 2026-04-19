"""Load, validate, and write manifest YAML so files stay compliant with the server schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modelrunner.manifest import Manifest, manifest_to_yaml


def validate_manifest(data: dict[str, Any]) -> Manifest:
    """Parse a manifest dict (e.g. from YAML/JSON); raises validation errors if invalid."""
    return Manifest.from_dict(data)


def write_manifest(manifest: Manifest, path: str | Path) -> None:
    """Write a validated manifest to ``path`` (UTF-8)."""
    p = Path(path)
    p.write_text(manifest_to_yaml(manifest), encoding="utf-8")
