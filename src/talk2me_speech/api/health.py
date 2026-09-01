"""Healthcheck helpers."""

from __future__ import annotations


def health_status() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "ok"}
