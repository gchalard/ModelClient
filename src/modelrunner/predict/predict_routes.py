"""HTTP routes for /predict (GET contract, POST inference)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Request, status

from modelrunner.contract import build_openapi_contract
from modelrunner.manifest import manifest_to_plain_dict
from modelrunner.predict.prediction_service import PredictionService, PredictionValidationError

router = APIRouter(tags=["predict"])


def _json_safe_metadata(obj: object) -> object:
    """Ensure dict keys are JSON-friendly (YAML may load integer keys)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe_metadata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe_metadata(x) for x in obj]
    return obj


def _service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


@router.get("/predict")
def get_predict_contract(request: Request) -> dict[str, Any]:
    """Return the OpenAPI contract for this deployment plus optional manifest ``metadata``."""
    manifest = _service(request).manifest
    doc = build_openapi_contract(manifest)
    doc["manifest"] = _json_safe_metadata(manifest_to_plain_dict(manifest))
    if manifest.metadata is not None:
        doc["metadata"] = _json_safe_metadata(manifest.metadata)
    return doc


@router.post("/predict")
def post_predict(
    request: Request,
    body: Annotated[dict[str, Any], Body(...)],
) -> dict[str, Any]:
    """Accept ``{feature_name: value}``, order by manifest, run inference."""
    service = _service(request)
    try:
        return service.predict(body)
    except PredictionValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        ) from e
