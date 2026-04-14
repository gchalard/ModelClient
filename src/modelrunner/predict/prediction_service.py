"""Application service: validate payload, order features, call predictor."""

from __future__ import annotations

from typing import Any

from modelrunner.manifest import FeatureSpec, Manifest
from modelrunner.ports import ModelPredictor


class PredictionValidationError(ValueError):
    """Raised when the request body does not match the manifest."""


def _coerce(spec: FeatureSpec, raw: Any) -> Any:
    t = spec.type
    if t == "float":
        return float(raw)
    if t == "int":
        return int(raw)
    if t == "bool":
        if isinstance(raw, bool):
            return raw
        if raw in (0, 1, "0", "1", "true", "false"):
            return raw in (True, 1, "1", "true")
        msg = f"Feature {spec.name!r}: expected a boolean"
        raise PredictionValidationError(msg)
    if t == "string":
        return str(raw)
    msg = f"Unknown type {t!r} for feature {spec.name!r}"
    raise PredictionValidationError(msg)


def _check_constraints(spec: FeatureSpec, value: Any) -> None:
    if spec.constraints is None:
        return
    c = spec.constraints
    if c.enum is not None and str(value) not in set(c.enum):
        msg = f"Feature {spec.name!r}: must be one of {c.enum}"
        raise PredictionValidationError(msg)
    if spec.type in ("float", "int") and isinstance(value, (int, float)):
        if c.min is not None and value < c.min:
            msg = f"Feature {spec.name!r}: must be >= {c.min}"
            raise PredictionValidationError(msg)
        if c.max is not None and value > c.max:
            msg = f"Feature {spec.name!r}: must be <= {c.max}"
            raise PredictionValidationError(msg)


class PredictionService:
    def __init__(self, manifest: Manifest, predictor: ModelPredictor) -> None:
        self._manifest = manifest
        self._predictor = predictor

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def validate_and_order(self, body: dict[str, Any]) -> list[Any]:
        unknown = set(body) - {f.name for f in self._manifest.input.features}
        if unknown:
            msg = f"Unknown feature keys: {sorted(unknown)}"
            raise PredictionValidationError(msg)

        ordered: list[Any] = []
        for spec in self._manifest.input.features:
            if spec.name in body:
                raw = body[spec.name]
            elif spec.default is not None:
                raw = spec.default
            elif spec.required:
                msg = f"Missing required feature {spec.name!r}"
                raise PredictionValidationError(msg)
            else:
                msg = f"Missing optional feature {spec.name!r} (no default in manifest)"
                raise PredictionValidationError(msg)

            value = _coerce(spec, raw)
            _check_constraints(spec, value)
            ordered.append(value)
        return ordered

    def predict(self, body: dict[str, Any]) -> dict[str, Any]:
        ordered = self.validate_and_order(body)
        return self._predictor.predict(ordered)
