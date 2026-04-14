def main() -> None:
    """CLI entry: ``uv run modelrunner`` or ``python -m modelrunner`` (if configured)."""
    import uvicorn

    uvicorn.run("modelrunner.main:app", host="0.0.0.0", port=8000, reload=False)
