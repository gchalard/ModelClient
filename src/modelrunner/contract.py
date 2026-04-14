"""Build a minimal OpenAPI 3.1 document from the manifest (GET /predict)."""

from __future__ import annotations

from typing import Any

from modelrunner.manifest import FeatureSpec, FeatureType, Manifest, OutputFieldSpec


def _json_type(ft: FeatureType) -> str:
    return {"float": "number", "int": "integer", "bool": "boolean", "string": "string"}[ft]


def _feature_schema(spec: FeatureSpec) -> dict[str, Any]:
    prop: dict[str, Any] = {"type": _json_type(spec.type)}
    if spec.constraints:
        c = spec.constraints
        if c.min is not None:
            prop["minimum"] = c.min
        if c.max is not None:
            prop["maximum"] = c.max
        if c.enum:
            prop["enum"] = c.enum
    return prop


def _output_field_schema(spec: OutputFieldSpec) -> dict[str, Any]:
    prop: dict[str, Any] = {"type": _json_type(spec.type)}
    if spec.description:
        prop["description"] = spec.description
    return prop


def build_openapi_contract(manifest: Manifest) -> dict[str, Any]:
    """Self-contained OpenAPI document for this model’s `/predict` route."""
    feature_props = {f.name: _feature_schema(f) for f in manifest.input.features}
    required = [f.name for f in manifest.input.features if f.required]
    out_props = {f.name: _output_field_schema(f) for f in manifest.output.fields}
    out_required = [f.name for f in manifest.output.fields if f.required]

    post_request = {
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": feature_props,
                    "required": required,
                },
            },
        },
        "required": True,
    }

    post_response = {
        "description": "Prediction result",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": out_props,
                    "required": out_required,
                },
            },
        },
    }

    return {
        "openapi": "3.1.0",
        "info": {
            "title": manifest.model.name,
            "version": manifest.model.version,
            **({"description": manifest.model.description} if manifest.model.description else {}),
        },
        "paths": {
            "/predict": {
                "get": {
                    "summary": "OpenAPI contract for this model deployment",
                    "operationId": "predict_contract",
                    "responses": {
                        "200": {
                            "description": "OpenAPI document for this service",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "description": "Full OpenAPI 3.1 document",
                                    },
                                },
                            },
                        },
                    },
                },
                "post": {
                    "summary": "Run inference",
                    "operationId": "predict",
                    "requestBody": post_request,
                    "responses": {
                        "200": post_response,
                        "422": {"description": "Validation error"},
                    },
                },
            },
        },
    }
