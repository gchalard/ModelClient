"""FastAPI application entry."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from modelrunner.adapters.adapter_loader import load_predictor_class
from modelrunner.manifest import load_manifest
from modelrunner.predict.prediction_service import PredictionService
from modelrunner.predict.predict_routes import router as predict_router
from modelrunner.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    manifest_path = settings.manifest_path
    if not manifest_path.is_absolute():
        manifest_path = (Path.cwd() / manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    predictor_cls = load_predictor_class(manifest.runtime.adapter)
    predictor = predictor_cls()
    predictor.load(manifest, manifest_path)
    app.state.prediction_service = PredictionService(manifest, predictor)
    yield


app = FastAPI(
    title="ModelRunner",
    lifespan=lifespan,
    version="0.1.0",
)
app.include_router(predict_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
