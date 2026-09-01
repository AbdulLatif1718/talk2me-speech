#!/usr/bin/env python3
"""Evaluation entrypoint."""

from __future__ import annotations

from talk2me_speech.evaluation.wer import compute_wer


def main() -> None:
    score = compute_wer("hello world", "hello there")
    print("WER sample score:", score)


if __name__ == "__main__":
    main()
