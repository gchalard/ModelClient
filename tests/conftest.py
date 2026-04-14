"""Pytest setup: manifest path before ``modelrunner.main`` is imported."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MODEL_MANIFEST_PATH",
    str((REPO / "manifests/example_manifest.yaml").resolve()),
)
