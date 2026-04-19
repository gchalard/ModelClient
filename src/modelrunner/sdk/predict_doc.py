"""Parse the JSON returned by GET ``/predict`` (OpenAPI fragment plus extras)."""

from __future__ import annotations

from typing import Any, Mapping


def feature_names_from_predict_doc(doc: Mapping[str, Any]) -> list[str]:
    """Return POST ``/predict`` JSON body property names in contract order.

    Expects the structure produced by :func:`modelrunner.contract.build_openapi_contract`.
    """
    try:
        props = doc["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"][
            "schema"
        ]["properties"]
    except (KeyError, TypeError) as e:
        msg = "GET /predict document is missing the /predict POST requestBody JSON schema"
        raise ValueError(msg) from e
    if not isinstance(props, dict):
        msg = "request schema properties must be a mapping"
        raise TypeError(msg)
    return list(props.keys())
