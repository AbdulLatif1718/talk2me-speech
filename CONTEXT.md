# Talk2Me Speech - Project Context Guide

## Project Overview

**Talk2Me Speech** is a comprehensive speech recognition pipeline builder focused on Ghanaian and multilingual audio data. It provides end-to-end capabilities for data preparation, model training, evaluation, and inference.

**Purpose**: Build and evaluate state-of-the-art speech recognition systems for low-resource languages and multilingual scenarios, with emphasis on Ghanaian languages and code-switched audio.

**Key Technologies**: PyTorch, Hugging Face Transformers, FastAPI, YAML-based configuration management

---

## Project Structure

```
talk2me-speech/
├── configs/                      # YAML configuration files
│   ├── base.yaml                # Base configuration template
│   ├── datasets/                # Dataset configurations (commonvoice, internal, kasaspeech, v1)
│   ├── evaluation/              # Evaluation configs (code_switch, ghana, noise)
│   ├── models/                  # Model configs (whisper-*, xlsr-*)
│   └── training/                # Training configs (whisper, xlsr)
├── data/                        # Data artifacts
│   ├── raw/                     # Original audio files
│   ├── processed/               # Pre-processed audio
│   ├── interim/                 # Intermediate processing results
│   ├── manifests/               # Dataset manifests (JSON/JSONL file lists)
│   └── evaluation/              # Evaluation results and metrics
├── models/                      # Model artifacts
│   ├── checkpoints/             # Training checkpoints
│   ├── evaluation/              # Evaluation-specific models
│   └── exported/                # Exported/production models
├── notebooks/                   # Jupyter notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_error_analysis.ipynb
│   └── 03_model_comparison.ipynb
├── scripts/                     # CLI scripts
│   ├── prepare_dataset.py       # Data preparation
│   ├── train.py                 # Model training launcher
│   ├── evaluate.py              # Evaluation suite
│   ├── benchmark.py             # Performance benchmarking
│   ├── export_model.py          # Model export for inference
│   └── validate_dataset.py      # Dataset validation
├── src/talk2me_speech/          # Main Python package
│   ├── api/                     # FastAPI inference service
│   ├── audio/                   # Audio processing utilities
│   ├── datasets/                # Dataset loading and management
│   ├── evaluation/              # Evaluation metrics (WER, CER, etc.)
│   ├── feedback/                # User feedback collection
│   ├── inference/               # Inference engines (batch, streaming)
│   ├── models/                  # Model implementations
│   └── training/                # Training logic for different models
├── tests/                       # Unit and integration tests
├── Dockerfile                   # Container configuration
├── docker-compose.yml          # Multi-container setup
├── Makefile                    # Common build/run commands
├── pyproject.toml              # Project metadata and dependencies
└── README.md                   # Quick start guide
```

---

## Core Components

### 1. **Audio Processing** (`src/talk2me_speech/audio/`)
Handles audio preprocessing and augmentation:
- **normalize.py** - Normalize audio levels
- **resample.py** - Change sample rates
- **vad.py** - Voice Activity Detection
- **noise.py** - Noise reduction and denoising
- **segmentation.py** - Split audio into segments

### 2. **Datasets** (`src/talk2me_speech/datasets/`)
Manages dataset loading, validation, and preparation:
- **loader.py** - Load audio datasets
- **manifest.py** - Create and manage manifest files (JSON/JSONL)
- **splitter.py** - Train/val/test splits
- **validator.py** - Validate audio quality and metadata

**Key Datasets**:
- **CommonVoice** - Mozilla's multilingual speech corpus
- **Internal** - Custom Talk2Me dataset
- **KasaSpeech** - Ghanaian language dataset
- **v1** - Versioned dataset release

### 3. **Models** (`src/talk2me_speech/models/`)
Model implementations and registry:
- **whisper.py** - OpenAI Whisper integration
- **xlsr.py** - XLSR (multilingual speech model) integration
- **language_head.py** - Custom language identification head
- **registry.py** - Model factory pattern

**Supported Models**:
- Whisper: small, medium, large
- XLSR: 300m, 1b

### 4. **Training** (`src/talk2me_speech/training/`)
Training pipelines for different models:
- **train_whisper.py** - Whisper fine-tuning
- **train_xlsr.py** - XLSR fine-tuning
- **callbacks.py** - Training callbacks (logging, checkpointing)
- **collator.py** - Data collation for batch training

### 5. **Evaluation** (`src/talk2me_speech/evaluation/`)
Comprehensive evaluation metrics:
- **wer.py** - Word Error Rate
- **cer.py** - Character Error Rate
- **code_switch.py** - Code-switch evaluation
- **named_entities.py** - NER evaluation
- **benchmark.py** - Performance benchmarking

### 6. **Inference** (`src/talk2me_speech/inference/`)
Inference engines for production use:
- **streaming.py** - Real-time streaming transcription
- **batch.py** - Batch transcription processing
- **postprocess.py** - Post-processing transcriptions

### 7. **API** (`src/talk2me_speech/api/`)
FastAPI-based REST/WebSocket service:
- **main.py** - FastAPI application setup
- **health.py** - Health check endpoints
- **websocket.py** - WebSocket for streaming inference

### 8. **Feedback** (`src/talk2me_speech/feedback/`)
User feedback collection and active learning:
- **collector.py** - Collect user corrections
- **corrections.py** - Process correction data
- **dataset_builder.py** - Build datasets from feedback

---

## Configuration System

All configurations are YAML-based for flexibility and reproducibility.

### Configuration Organization
- **base.yaml** - Base defaults (referenced by all configs)
- **datasets/*.yaml** - Dataset-specific settings (paths, sample rates, splits)
- **models/*.yaml** - Model-specific settings (architecture, pretrained weights)
- **training/*.yaml** - Training hyperparameters (learning rate, epochs, batch size)
- **evaluation/*.yaml** - Evaluation settings (metrics, thresholds, datasets)

### Key Configuration Parameters
```yaml
# Typical model config
model:
  name: whisper-large
  pretrained: true
  freeze_encoder: false
  
# Typical training config
training:
  learning_rate: 1e-5
  num_epochs: 10
  batch_size: 32
  warmup_steps: 1000
  save_steps: 500
  
# Typical dataset config
dataset:
  name: commonvoice
  language: en
  split: 80:10:10
  sample_rate: 16000
```

---

## Development Workflow

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .  # Install in editable mode
pytest -q        # Run tests
```

### Common Commands (via Makefile)
```bash
make install      # Install dependencies
make test         # Run test suite
make train        # Run training pipeline
make evaluate     # Run evaluation
make docker-up    # Start API service in Docker
```

### Scripts (CLI Interface)
```bash
# Prepare dataset with validation
python scripts/prepare_dataset.py --config configs/datasets/commonvoice.yaml

# Train a model
python scripts/train.py --config configs/training/whisper.yaml

# Evaluate trained model
python scripts/evaluate.py --model models/checkpoints/whisper-large --config configs/evaluation/code_switch.yaml

# Export model for inference
python scripts/export_model.py --model models/checkpoints/whisper-large --output models/exported/whisper

# Run benchmarks
python scripts/benchmark.py --model models/exported/whisper --dataset data/processed/test
```

---

## Key Architectural Decisions

### 1. **Configuration-Driven Design**
- All behaviors controlled by YAML configs, not hardcoded
- Enables reproducibility and experimentation
- Configs inheritable through base.yaml

### 2. **Model Registry Pattern**
- Models registered in `registry.py`
- Enables swapping models without code changes
- Factory pattern for instantiation

### 3. **Manifest-Based Datasets**
- Datasets represented as JSON/JSONL manifests
- Each entry: `{audio_path, transcript, metadata}`
- Enables lazy loading and efficient processing

### 4. **PyTorch Lightning / HuggingFace Trainer**
- Uses standard trainer classes for consistency
- Automatic checkpoint management
- Built-in distributed training support

### 5. **Streaming Inference**
- Separate streaming inference engine for real-time use
- WebSocket API for low-latency communication
- Compatible with FastAPI

---

## Important Conventions

### Naming Conventions
- **Model checkpoints**: `models/checkpoints/{model_name}-{dataset}-{timestamp}/`
- **Manifests**: `data/manifests/{dataset_name}_{split}.jsonl`
- **Exported models**: `models/exported/{model_name}-{version}/`

### File Paths
- All paths in configs should be relative to project root
- Scripts handle absolute path resolution
- Use `pathlib.Path` for cross-platform compatibility

### Dataset Manifest Format
```json
{
  "audio_path": "data/raw/audio_001.wav",
  "transcript": "hello world",
  "language": "en",
  "duration": 2.5,
  "sample_rate": 16000,
  "metadata": {"source": "commonvoice", "split": "train"}
}
```

### Evaluation Metrics
- **WER** (Word Error Rate) - Primary metric for transcription quality
- **CER** (Character Error Rate) - Secondary metric
- **Code-Switch Evaluation** - Separate track for multilingual audio
- **Noise Robustness** - Evaluation on noisy audio subsets

---

## Current State & Key Decisions

### Trained Models
- Whisper models fine-tuned on multilingual data
- XLSR models for low-resource scenarios
- Checkpoints stored in `models/checkpoints/`

### Evaluation Results
- Code-switch evaluation for Ghanaian multilingual audio
- Ghana-specific test set in `configs/evaluation/ghana.yaml`
- Noise-robust evaluation setup in `configs/evaluation/noise.yaml`

### API Service
- FastAPI-based REST API in `src/talk2me_speech/api/`
- WebSocket support for streaming transcription
- Dockerized with `Dockerfile` and `docker-compose.yml`

### Known Limitations
- Streaming inference optimized for real-time (may trade some accuracy)
- Models trained primarily on English and Ghanaian languages
- Audio preprocessing assumes 16kHz sample rate (configurable)

---

## Dependencies Overview

### Core ML/Audio
- **PyTorch** (2.2+) - Deep learning framework
- **torchaudio** (2.2+) - Audio processing
- **transformers** (4.41+) - Pre-trained models (Whisper, XLSR)
- **librosa** (0.10+) - Audio analysis
- **datasets** (2.18+) - Dataset loading utilities

### Data Processing
- **numpy** (1.24+) - Numerical computing
- **pandas** (2.0+) - Data manipulation
- **soundfile** (0.12+) - Audio I/O

### Infrastructure
- **FastAPI** (0.110+) - Web framework
- **uvicorn** (0.29+) - ASGI server
- **PyYAML** (6.0+) - Config parsing

### Development
- **pytest** (8.0+) - Testing
- **black**, **ruff**, **mypy** - Code quality (dev dependencies)

---

## Testing

### Test Coverage
- **test_audio.py** - Audio processing utilities
- **test_dataset.py** - Dataset loading and validation
- **test_streaming.py** - Streaming inference
- **test_transcription.py** - Transcription pipeline

### Running Tests
```bash
pytest -q              # Quick test
pytest -v              # Verbose
pytest --cov           # With coverage
pytest tests/test_audio.py  # Specific file
```

---

## Troubleshooting & Common Issues

### Audio Processing
- **Sample rate mismatch**: Ensure all audio is resampled to 16kHz via `audio/resample.py`
- **Duration issues**: Check `segmentation.py` for splitting oversized audio
- **Noise problems**: Use `audio/noise.py` for denoising

### Model Training
- **Out of memory**: Reduce `batch_size` in training config
- **Slow convergence**: Adjust `learning_rate` in training config
- **Checkpoint issues**: Clear `models/checkpoints/` if resuming training

### Inference/API
- **Connection timeout**: Check `docker-compose logs` for service issues
- **Slow transcription**: Consider batch processing instead of streaming
- **Memory leaks**: Monitor with `benchmark.py` profiling

---

## For New Contributors

1. **Read this file first** - Understand project structure and goals
2. **Set up environment** - Follow Quick Start in README.md
3. **Run tests** - Verify setup with `make test`
4. **Check Makefile** - Common commands for development
5. **Read component docs** - Each module has clear docstrings
6. **Look at configs** - Configuration examples in `configs/`
7. **Review notebooks** - See usage examples in `notebooks/`

---

## Next Steps & Open Questions

When continuing work on this project:
1. Check `data/evaluation/` for latest benchmark results
2. Review model checkpoints in `models/checkpoints/` for available versions
3. Verify training configs match your dataset in `configs/datasets/`
4. Run `make test` to ensure environment is properly set up
5. Consult `CONTEXT.md` (this file) whenever context is needed

---

**Last Updated**: 2026-09-01
**Project Version**: 0.1.0
**Python**: 3.10+
**Status**: Active Development
