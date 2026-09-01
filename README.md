# Talk2Me Speech

Talk2Me Speech is a repository for building and evaluating speech recognition pipelines for Ghanaian and multilingual audio data. It includes dataset preparation, model training, benchmarking, and an inference API for transcription.

## Project structure

- `configs/` contains dataset, model, training, and evaluation YAML configurations.
- `data/` stores raw, interim, processed, manifests, and evaluation artifacts.
- `src/talk2me_speech/` contains the Python package.
- `scripts/` exposes command-line tasks for dataset prep, training, evaluation, and export.
- `tests/` contains smoke tests for core components.
- `models/` stores checkpoints and exported models.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pytest -q
```

## Common commands

```bash
make install
make test
make train
make evaluate
make docker-up
```

## Features

- Audio normalization, resampling, VAD, denoising, and segmentation helpers
- Dataset validation, splitting, augmentation, and manifest generation
- Whisper and XLSR model registry and training entrypoints
- WER/CER benchmarking and code-switch evaluation
- FastAPI-based inference service

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
