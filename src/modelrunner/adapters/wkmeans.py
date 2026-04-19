"""Adapter for :class:`wkmeans.WKMeans` models exported via ``WKMeans.export`` / ``from_file``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from wkmeans import WKMeans

from modelrunner.manifest import Manifest
from modelrunner.ports import ModelPredictor


class WKMeansAdapter:
    """Loads ``metadata.yaml`` + centroid blob.

    Returns only ``cluster_id``, aligned with ``WKMeans.predict`` (cluster indices only;
    Wasserstein distance is not part of that method’s return value).
    """

    _model: WKMeans

    def load(self, manifest: Manifest, manifest_path: Path) -> None:
        paths = manifest.runtime.artifacts.paths
        meta_rel = paths.extra.get("metadata")
        if not meta_rel:
            msg = "runtime.artifacts.paths.metadata is required for WKMeansAdapter"
            raise ValueError(msg)
        root = (manifest_path.parent / manifest.runtime.artifacts.base_dir).resolve()
        metadata_path = (root / str(meta_rel)).resolve()
        if not metadata_path.is_file():
            msg = f"WKMeans metadata not found: {metadata_path}"
            raise FileNotFoundError(msg)
        self._model = WKMeans.from_file(metadata_path=metadata_path)

    def predict(self, features: list[Any]) -> dict[str, Any]:
        sample = np.asarray(features, dtype=np.float64).ravel()
        if len(self._model.centroids) == 0:
            msg = "WKMeans model has no centroids"
            raise RuntimeError(msg)
        d = int(np.asarray(self._model.centroids[0]).size)
        if sample.size != d:
            msg = f"expected {d} features, got {sample.size}"
            raise ValueError(msg)
        cluster_id = int(self._model.predict([sample])[0])
        return {"cluster_id": cluster_id}
