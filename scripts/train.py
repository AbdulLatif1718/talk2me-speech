#!/usr/bin/env python3
"""Training entrypoint."""

from __future__ import annotations

from talk2me_speech.training.train_whisper import build_train_config


def main() -> None:
    config = build_train_config()
    print("Training config:", config)


if __name__ == "__main__":
    main()
