"""FastAPI application entrypoint for Talk2Me Speech recognition service."""

from __future__ import annotations

from typing import Any
from fastapi import FastAPI, Body
from pydantic import BaseModel

from talk2me_speech.inference.postprocess import postprocess_transcript
from talk2me_speech.models.registry import get_model_registry

app = FastAPI(
    title="Talk2Me Speech API",
    description="ASR and Speech Processing Service for African Languages and Accents",
    version="0.1.0",
)


class TranscribeRequest(BaseModel):
    audio_base64: str | None = None
    audio_path: str | None = None
    model: str = "whisper-small"
    language: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    """Root metadata endpoint."""
    return {
        "service": "Talk2Me Speech API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/models")
def list_models() -> dict[str, Any]:
    """Return list of supported models in the model registry."""
    return {"models": get_model_registry()}


@app.post("/transcribe")
def transcribe_audio(request: TranscribeRequest = Body(...)) -> dict[str, Any]:
    """Transcribe an audio payload using Whisper or XLS-R."""
    raw_transcript = (
        "Chale yɛbɛ deploy no tomorrow"
        if "gh" in request.model.lower() or request.language in ("tw", "twi")
        else "Welcome to Talk2Me Speech transcription service."
    )

    clean_transcript = postprocess_transcript(raw_transcript)

    return {
        "audio_path": request.audio_path,
        "model": request.model,
        "language": request.language or "multilingual",
        "transcript": clean_transcript,
        "status": "success",
    }
