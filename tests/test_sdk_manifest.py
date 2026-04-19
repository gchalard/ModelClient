"""SDK manifest helpers."""

from __future__ import annotations

from pathlib import Path

from modelrunner.manifest import (
    ArtifactPaths,
    ArtifactsSpec,
    FeatureSpec,
    InputSpec,
    Manifest,
    ModelInfo,
    OutputFieldSpec,
    OutputSpec,
    RuntimeSpec,
)
from modelrunner.enums import AdapterType
from modelrunner.sdk import manifest_to_yaml, validate_manifest, write_manifest


def test_validate_manifest_roundtrip_dict() -> None:
    raw = {
        "schema_version": 1,
        "model": {
            "id": "x",
            "name": "X",
            "version": "1",
            "task": "test",
        },
        "runtime": {
            "adapter": "modelrunner.adapters.dummy:DummyAdapter",
            "artifacts": {"base_dir": "artifacts", "paths": {}},
        },
        "input": {
            "features": [
                {"name": "a", "type": "float", "required": True},
            ],
        },
        "output": {
            "fields": [
                {"name": "y", "type": "int", "required": True},
            ],
        },
    }
    m = validate_manifest(raw)
    assert m.model.id == "x"


def test_write_and_manifest_to_yaml(tmp_path: Path) -> None:
    m = Manifest(
        model=ModelInfo(id="m", name="M", version="0", task="t"),
        runtime=RuntimeSpec(
            adapter="modelrunner.adapters.dummy:DummyAdapter",
            artifacts=ArtifactsSpec(base_dir="artifacts", paths=ArtifactPaths()),
        ),
        input=InputSpec(
            features=[FeatureSpec(name="f", type="float", required=True)],
        ),
        output=OutputSpec(
            fields=[OutputFieldSpec(name="out", type="float", required=True)],
        ),
    )
    path = tmp_path / "m.yaml"
    write_manifest(m, path)
    text = manifest_to_yaml(m)
    assert "f" in text
    assert path.read_text(encoding="utf-8") == text


def test_manifest_to_yaml_accepts_str_enum_adapter() -> None:
    """``yaml.safe_dump`` does not treat ``StrEnum`` as ``str`` (MRO); normalize before dump."""
    m = Manifest(
        model=ModelInfo(id="m", name="M", version="0", task="t"),
        runtime=RuntimeSpec(
            adapter=AdapterType.WKMEANS,  # type: ignore[arg-type]
            artifacts=ArtifactsSpec(base_dir="artifacts", paths=ArtifactPaths()),
        ),
        input=InputSpec(
            features=[FeatureSpec(name="f", type="float", required=True)],
        ),
        output=OutputSpec(
            fields=[OutputFieldSpec(name="out", type="float", required=True)],
        ),
    )
    text = manifest_to_yaml(m)
    assert str(AdapterType.WKMEANS) in text


def test_manifest_export_roundtrip(tmp_path: Path) -> None:
    m = Manifest(
        model=ModelInfo(id="m", name="M", version="0", task="t"),
        runtime=RuntimeSpec(
            adapter="modelrunner.adapters.dummy:DummyAdapter",
            artifacts=ArtifactsSpec(base_dir="artifacts", paths=ArtifactPaths()),
        ),
        input=InputSpec(
            features=[FeatureSpec(name="f", type="float", required=True)],
        ),
        output=OutputSpec(
            fields=[OutputFieldSpec(name="out", type="float", required=True)],
        ),
        metadata={"k": "v"},
    )
    p = tmp_path / "out.yaml"
    yaml_text = m.export(path=p)
    assert yaml_text == p.read_text(encoding="utf-8")
    assert "k" in yaml_text
    assert m.export() == yaml_text
