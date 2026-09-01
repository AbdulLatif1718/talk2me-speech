"""Websocket helpers."""

from __future__ import annotations


async def websocket_echo(message: str) -> str:
    """Echo a message back to the caller for a websocket-style flow."""
    return message
