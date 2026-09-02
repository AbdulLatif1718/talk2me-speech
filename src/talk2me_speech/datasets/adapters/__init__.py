"""Dataset adapters package."""

from __future__ import annotations

from typing import Type

from talk2me_speech.datasets.adapters.base import BaseDatasetAdapter
from talk2me_speech.datasets.adapters.commonvoice import CommonVoiceAdapter
from talk2me_speech.datasets.adapters.kasaspeech import KasaSpeechAdapter

_ADAPTERS: dict[str, Type[BaseDatasetAdapter]] = {
    "commonvoice": CommonVoiceAdapter,
    "kasaspeech": KasaSpeechAdapter,
}


def get_adapter(source_name: str) -> BaseDatasetAdapter:
    """Instantiate and return a dataset adapter by source name."""
    normalized = source_name.lower().replace("-", "").replace("_", "")
    for key, adapter_cls in _ADAPTERS.items():
        if key in normalized:
            return adapter_cls()
    raise ValueError(
        f"Unknown dataset source adapter: '{source_name}'. Available: {list(_ADAPTERS.keys())}"
    )


__all__ = [
    "BaseDatasetAdapter",
    "CommonVoiceAdapter",
    "KasaSpeechAdapter",
    "get_adapter",
]

