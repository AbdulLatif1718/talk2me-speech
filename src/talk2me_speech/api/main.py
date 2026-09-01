"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Talk2Me Speech API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Talk2Me Speech API is running."}
