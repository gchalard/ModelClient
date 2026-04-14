"""Application settings (env-driven for Docker)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    manifest_path: Path = Field(
        default=Path("manifests/example_manifest.yaml"),
        description="Path to model manifest YAML (relative to cwd or absolute).",
    )


def get_settings() -> Settings:
    return Settings()
