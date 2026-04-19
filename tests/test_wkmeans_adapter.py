"""WKMeans adapter using real exports under ``tests/wkmeans/``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from wkmeans import WKMeans

from modelrunner.adapters.wkmeans import WKMeansAdapter
from modelrunner.manifest import load_manifest
from modelrunner.predict.prediction_service import PredictionService

REPO = Path(__file__).resolve().parents[1]
WK_DIR = REPO / "tests" / "wkmeans"
METADATA_FILE = "metadata:2d8a7673f4c29a411a569a9f1a176760413d4d7ab78eb020fe48f22aca794f5e.yaml"


def _feature_names() -> list[str]:
    raw = yaml.safe_load((WK_DIR / "feature_order.yaml").read_text(encoding="utf-8"))
    names = raw["features"]
    assert len(names) == 50
    return names


def _vec_to_payload(vec: np.ndarray) -> dict[str, float]:
    v = np.asarray(vec, dtype=np.float64).ravel()
    names = _feature_names()
    assert v.size == len(names)
    return {names[i]: float(v[i]) for i in range(len(names))}


def test_wkmeans_from_file_direct() -> None:
    """``WKMeans.from_file(metadata_path=…)`` resolves centroids next to the YAML."""
    meta = WK_DIR / METADATA_FILE
    model = WKMeans.from_file(metadata_path=meta)
    assert len(model.centroids) == model.k == 3
    assert len(model.centroids[0]) == 50


def test_wkmeans_predict_each_centroid_maps_to_self() -> None:
    """Same as WKMeans.predict: centroid *i* maps to cluster index *i*."""
    manifest_path = WK_DIR / "manifest.yaml"
    m = load_manifest(manifest_path)
    adapter = WKMeansAdapter()
    adapter.load(m, manifest_path)
    svc = PredictionService(m, adapter)
    ref = WKMeans.from_file(metadata_path=WK_DIR / METADATA_FILE)
    for idx, centroid in enumerate(ref.centroids):
        out = svc.predict(_vec_to_payload(np.asarray(centroid)))
        assert out == {"cluster_id": idx}
