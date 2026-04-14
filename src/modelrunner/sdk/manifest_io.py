"""Load, validate, and write manifest YAML so files stay compliant with the server schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from modelrunner.manifest import Manifest, manifest_to_plain_dict


def _strip_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


def validate_manifest(data: dict[str, Any]) -> Manifest:
    """Parse a manifest dict (e.g. from YAML/JSON); raises validation errors if invalid."""
    return Manifest.from_dict(data)


def manifest_to_yaml(manifest: Manifest) -> str:
    """Serialize a manifest to YAML text."""
    data = _strip_none(manifest_to_plain_dict(manifest))
    return yaml.safe_dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


def write_manifest(manifest: Manifest, path: str | Path) -> None:
    """Write a validated manifest to ``path`` (UTF-8)."""
    p = Path(path)
    p.write_text(manifest_to_yaml(manifest), encoding="utf-8")
