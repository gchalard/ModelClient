"""HTTP routes for /predict (GET contract, POST inference)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Request, status

from modelrunner.contract import build_openapi_contract
from modelrunner.predict.prediction_service import PredictionService, PredictionValidationError

router = APIRouter(tags=["predict"])


def _service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


@router.get("/predict")
def get_predict_contract(request: Request) -> dict[str, Any]:
    """Return the OpenAPI contract for this deployment (derived from the manifest)."""
    return build_openapi_contract(_service(request).manifest)


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
