"""Manifest schema as dataclasses (single source of truth for /predict)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml

FeatureType = Literal["float", "int", "bool", "string"]


def _expect_mapping(d: Any, what: str) -> dict[str, Any]:
    if not isinstance(d, dict):
        msg = f"{what} must be a mapping, got {type(d).__name__}"
        raise TypeError(msg)
    return d


def _parse_feature_type(raw: Any) -> FeatureType:
    if raw not in ("float", "int", "bool", "string"):
        msg = f"invalid feature type {raw!r}"
        raise ValueError(msg)
    return raw  # type: ignore[return-value]


@dataclass
class ModelInfo:
    id: str
    name: str
    version: str
    task: str
    description: str | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> ModelInfo:
        d = _expect_mapping(raw, "model")
        return cls(
            id=str(d["id"]),
            name=str(d["name"]),
            version=str(d["version"]),
            task=str(d["task"]),
            description=d.get("description"),
        )


@dataclass
class ArtifactPaths:
    """Relative paths under artifacts base_dir; unknown keys are kept in ``extra``."""

    weights: str | None = None
    config: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Any) -> ArtifactPaths:
        if raw is None or raw == {}:
            return cls()
        d = _expect_mapping(raw, "artifacts.paths")
        weights = d.get("weights")
        config = d.get("config")
        extra = {k: v for k, v in d.items() if k not in ("weights", "config")}
        return cls(
            weights=str(weights) if weights is not None else None,
            config=str(config) if config is not None else None,
            extra=extra,
        )


@dataclass
class ArtifactsSpec:
    """Where artifact files live relative to the manifest file."""

    base_dir: str
    paths: ArtifactPaths = field(default_factory=ArtifactPaths)

    @classmethod
    def from_dict(cls, raw: Any) -> ArtifactsSpec:
        d = _expect_mapping(raw, "artifacts")
        if "base_dir" not in d:
            msg = "artifacts.base_dir is required"
            raise ValueError(msg)
        return cls(
            base_dir=str(d["base_dir"]),
            paths=ArtifactPaths.from_mapping(d.get("paths")),
        )


@dataclass
class RuntimeSpec:
    adapter: str
    artifacts: ArtifactsSpec

    @classmethod
    def from_dict(cls, raw: Any) -> RuntimeSpec:
        d = _expect_mapping(raw, "runtime")
        if "adapter" not in d:
            msg = "runtime.adapter is required"
            raise ValueError(msg)
        return cls(
            adapter=str(d["adapter"]),
            artifacts=ArtifactsSpec.from_dict(d.get("artifacts") or {}),
        )


@dataclass
class FeatureConstraints:
    min: float | int | None = None
    max: float | int | None = None
    enum: list[str] | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> FeatureConstraints | None:
        if raw is None:
            return None
        d = _expect_mapping(raw, "constraints")
        enum = list(d["enum"]) if d.get("enum") is not None else None
        inst = cls(
            min=d.get("min"),
            max=d.get("max"),
            enum=enum,
        )
        if inst.min is None and inst.max is None and inst.enum is None:
            return None
        return inst


@dataclass
class FeatureSpec:
    name: str
    type: FeatureType
    required: bool = True
    default: Any | None = None
    constraints: FeatureConstraints | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> FeatureSpec:
        d = _expect_mapping(raw, "feature")
        if "name" not in d or "type" not in d:
            msg = "each feature requires name and type"
            raise ValueError(msg)
        c = d.get("constraints")
        return cls(
            name=str(d["name"]),
            type=_parse_feature_type(d["type"]),
            required=bool(d["required"]) if "required" in d else True,
            default=d.get("default"),
            constraints=FeatureConstraints.from_dict(c),
        )


@dataclass
class InputSpec:
    features: list[FeatureSpec]

    @classmethod
    def from_dict(cls, raw: Any) -> InputSpec:
        d = _expect_mapping(raw, "input")
        feats = d.get("features")
        if not isinstance(feats, list):
            msg = "input.features must be a list"
            raise TypeError(msg)
        return cls(features=[FeatureSpec.from_dict(x) for x in feats])


@dataclass
class OutputFieldSpec:
    name: str
    type: FeatureType
    description: str | None = None
    required: bool = True

    @classmethod
    def from_dict(cls, raw: Any) -> OutputFieldSpec:
        d = _expect_mapping(raw, "output field")
        if "name" not in d or "type" not in d:
            msg = "each output field requires name and type"
            raise ValueError(msg)
        return cls(
            name=str(d["name"]),
            type=_parse_feature_type(d["type"]),
            description=d.get("description"),
            required=bool(d["required"]) if "required" in d else True,
        )


@dataclass
class OutputSpec:
    fields: list[OutputFieldSpec]

    @classmethod
    def from_dict(cls, raw: Any) -> OutputSpec:
        d = _expect_mapping(raw, "output")
        fields = d.get("fields")
        if not isinstance(fields, list):
            msg = "output.fields must be a list"
            raise TypeError(msg)
        return cls(fields=[OutputFieldSpec.from_dict(x) for x in fields])


@dataclass
class Manifest:
    model: ModelInfo
    runtime: RuntimeSpec
    input: InputSpec
    output: OutputSpec
    schema_version: int = 1
    #: Arbitrary JSON-serializable map exposed on ``GET /predict`` (e.g. regime hints).
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> Manifest:
        """Parse and validate a manifest mapping (e.g. from YAML/JSON)."""
        d = _expect_mapping(raw, "manifest root")
        meta = d.get("metadata")
        if meta is not None and not isinstance(meta, dict):
            msg = "metadata must be a mapping when present"
            raise TypeError(msg)
        return cls(
            schema_version=int(d.get("schema_version", 1)),
            model=ModelInfo.from_dict(d.get("model")),
            runtime=RuntimeSpec.from_dict(d.get("runtime")),
            input=InputSpec.from_dict(d.get("input")),
            output=OutputSpec.from_dict(d.get("output")),
            metadata=meta,
        )

    def export(self, path: str | Path | None = None) -> str:
        """Return manifest as YAML; if ``path`` is set, write the same text there (UTF-8)."""
        text = manifest_to_yaml(self)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


def load_manifest(path: str | Path) -> Manifest:
    """Load and validate manifest YAML from disk."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        msg = f"empty manifest: {p}"
        raise ValueError(msg)
    return Manifest.from_dict(raw)


def manifest_to_plain_dict(manifest: Manifest) -> dict[str, Any]:
    """JSON/YAML-serializable dict; merges ``ArtifactPaths.extra`` into ``paths``."""
    data = asdict(manifest)
    paths = data["runtime"]["artifacts"]["paths"]
    extra = paths.pop("extra", {})
    paths.update(extra)
    return data


def _strip_none_manifest(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_none_manifest(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none_manifest(v) for v in obj]
    return obj


def _to_yaml_safe(obj: Any) -> Any:
    """Make nested structures ``yaml.safe_dump``-compatible (e.g. ``StrEnum`` is not a registered representer)."""
    if isinstance(obj, Enum):
        return _to_yaml_safe(obj.value)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_yaml_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_yaml_safe(x) for x in obj)
    return obj


def manifest_to_yaml(manifest: Manifest) -> str:
    """Serialize a manifest to YAML text (no file I/O)."""
    data = _to_yaml_safe(_strip_none_manifest(manifest_to_plain_dict(manifest)))
    return yaml.safe_dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
