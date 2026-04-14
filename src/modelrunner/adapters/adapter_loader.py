"""Load adapter classes from manifest import paths."""

from __future__ import annotations

import importlib
from typing import Any, TypeVar

from modelrunner.ports import ModelPredictor

T = TypeVar("T")


def import_object(path: str) -> Any:
    """Import ``module.sub:attr`` style path."""
    if ":" not in path:
        msg = f"Adapter path must be 'module:Class', got: {path!r}"
        raise ValueError(msg)
    module_name, qualname = path.split(":", 1)
    module = importlib.import_module(module_name)
    obj = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def load_predictor_class(path: str) -> type[ModelPredictor]:
    cls = import_object(path)
    if not isinstance(cls, type):
        msg = f"Adapter {path!r} is not a class"
        raise TypeError(msg)
    return cls  # type: ignore[return-value]
