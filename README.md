# Talk2Me Speech

## Overview

**Talk2Me Speech v1** is the research and training infrastructure for developing African-focused speech recognition models for Talk2Me AI.

It is **research infrastructure**, not yet a production ASR service.

### The Core Research Problem

Conventional global ASR systems perform poorly on African speech because they:

1. Train primarily on Western accents and pronunciations
2. Optimize for formal, clear speech
3. Treat code-switching as errors rather than valid linguistic behavior
4. Lack exposure to local vocabulary, expressions, and names
5. Assume single-language input

### The Talk2Me Approach

African speech is legitimate linguistic data. Our core hypothesis:

> Model African accents, code-switching, and multilingual speech as valid patterns—not errors to suppress.

**Example:**

Audio: "Chale yɛbɛ deploy no tomorrow."

✓ Correct transcript: "Chale yɛbɛ deploy no tomorrow."

✗ Wrong: "Charlie we will deploy it tomorrow."

✗ Wrong: "We will deploy tomorrow."

We preserve the natural speech patterns, linguistic characteristics, and code-switching behavior. Translation or standardization defeats the purpose.

---

## Current Scope (v1)

### Implemented ✓

- **Audio preprocessing pipeline**: normalization, resampling to 16kHz, VAD, basic denoising
- **Dataset infrastructure**: manifest creation, loading, basic validation
- **Configuration system**: YAML-based, hierarchical config management
- **Model registry**: Whisper and XLS-R model definitions
- **Basic evaluation metrics**: WER (word error rate), CER (character error rate)
- **Docker deployment**: Containerized API service setup
- **Testing framework**: 8 passing unit tests

### Experimental ⚠️

- **FastAPI inference service**: Basic skeleton with `/health` and `/` endpoints only (no transcription)
- **Streaming inference**: Conceptual structure (not yet integrated)
- **Batch processing**: Data batching utilities (not connected to models)

### Planned 🚀

These are critical components needed before ML experiments can proceed:

- **Real Whisper integration**: Full model loading, preprocessing, fine-tuning loop
- **Real XLS-R integration**: Wav2Vec2/CTC training pipeline
- **Dataset adapters**: CommonVoice, KasaSpeech, internal Talk2Me datasets
- **Speaker-safe splitting**: Prevent speaker leakage between train/test
- **Production inference API**: Streaming `/transcribe` endpoints
- **Experiment tracking**: Metadata, reproducibility, hyperparameter logging
- **Broader African language coverage**: Beyond initial Ghana focus
- **Online feedback learning**: Incorporate user corrections into retraining

---

## Architecture

```
Audio Files
    ↓
Dataset Adapters (Planned)
    ↓
Canonical Talk2Me Schema
    ↓
Validation & Schema Check
    ↓
Audio Preprocessing
    ├─ Normalization
    ├─ Resampling (16kHz)
    ├─ VAD
    └─ Segmentation
    ↓
Speaker-Safe Split (Planned)
    ├─ Train (70%)
    ├─ Validation (15%)
    └─ Test (15%)
    ↓
Model Training (Planned)
    ├─ Whisper Fine-tuning
    └─ XLS-R CTC Training
    ↓
Evaluation
    ├─ WER / CER
    ├─ Language-specific metrics
    ├─ Code-switch evaluation
    └─ Error analysis
    ↓
Model Registry
    ↓
Future: Talk2Me Meeting Integration
```

---

## Dataset Philosophy

### Canonical Schema

All datasets map to this schema:

```
{
  "id": "unique_id",
  "audio_path": "path/to/audio.wav",
  "transcript": "Chale yɛbɛ deploy no tomorrow",
  "duration": 3.2,
  "sample_rate": 16000,
  "source": "commonvoice",
  
  # Optional fields
  "speaker_id": "speaker_001",
  "primary_language": "en",
  "languages": ["en", "tw"],
  "country": "GH",
  "environment": "office",
  "noise_types": ["keyboard", "traffic"],
  "verified": true
}
```

### Data Sources

| Dataset | Focus | Current Status |
| --- | --- | --- |
| **KasaSpeech** | Ghanaian speech, code-switching | Adapter planned |
| **CommonVoice** | Multilingual, African languages | Adapter planned |
| **Talk2Me Internal** | Real meeting recordings | Planned (privacy controls required) |

### Data Privacy & Governance

⚠️ **Important**: Talk2Me will eventually process sensitive organizational meeting recordings.

**v1 constraints:**

- Internal meeting data requires explicit consent
- No automatic use of customer recordings for training
- Dataset governance policies must be in place
- Anonymization requirements TBD

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU training)
- ~10GB disk space (for models and data)

### Setup

```bash
# Clone and navigate
git clone <your-github-repository-url>
cd talk2me-speech

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .

# Verify installation
python -c "import talk2me_speech; print('✓ Installation successful')"

# Run tests
pytest -v
```

---

## Google Colab Setup

### Running on Colab

```python
# Cell 1: Clone and install
!git clone https://github.com/your-org/talk2me-speech.git
%cd talk2me-speech
!pip install -e . -q

# Cell 2: Verify setup
import torch
import talk2me_speech
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Talk2Me Speech v{talk2me_speech.__version__} ready")

# Cell 3: Test pipeline
from talk2me_speech.datasets.loader import load_manifest
from talk2me_speech.audio.normalize import normalize_audio
import numpy as np

# Quick test
audio = np.random.randn(16000).astype(np.float32)
normalized = normalize_audio(audio)
print(f"✓ Audio processing works")
```

### Using Google Drive (Optional)

```python
# Mount Drive for large datasets/checkpoints
from google.colab import drive
drive.mount('/content/drive')

# Your training data can live in:
# /content/drive/MyDrive/talk2me-speech/data/
```

**Note:** Code lives in GitHub; large datasets and checkpoints live in Drive.

---

## Usage

### Prepare Dataset Manifest

The `prepare_dataset.py` script creates a JSONL manifest from audio files. It currently demonstrates the process with a simple example.

**Current status**: Accepts hardcoded paths; CLI argument support is being added.

```bash
python scripts/prepare_dataset.py
```

**Future interface** (being implemented):

```bash
python scripts/prepare_dataset.py \
  --source commonvoice \
  --language en \
  --output data/manifests/commonvoice.jsonl
```

### Validate Dataset

```bash
python scripts/validate_dataset.py
```

**Validation checks**:
- Dataset directory structure
- File existence
- Audio readability (planned)
- Transcript validity (planned)
- Manifest format (planned)

### Baseline Experiment (W0)

The recommended first experiment is a **zero-shot baseline**:

```bash
python scripts/train.py
```

**Current status**: Prints config only (training loop not yet implemented)

**What W0 does**:
1. Load Whisper Small (no fine-tuning)
2. Transcribe test set
3. Record WER/CER
4. Establish baseline for comparison

---

## Experiment Roadmap

Follow this progression to isolate which interventions improve ASR:

### Whisper Experiments

| Experiment | Focus | Expected |
| --- | --- | --- |
| **W0** | Zero-shot baseline | Baseline WER |
| **W1** | + Ghanaian English data | ↓ WER |
| **W2** | + Twi data | ↑ Coverage |
| **W3** | + Ghanaian Pidgin + code-switch | ↓ WER on code-switch |
| **W4** | + code-switch oversampling | ↓ WER on code-switch |
| **W5** | + acoustic augmentation | ↓ WER on noise |

### XLS-R Experiments

| Experiment | Focus | Expected |
| --- | --- | --- |
| **X0** | Zero-shot baseline | Baseline WER |
| **X1** | + Ghanaian English | ↓ WER |
| **X2** | + Twi | ↑ Coverage |
| **X3** | + Joint multilingual/code-switch | ↓ WER |
| **X4** | + acoustic augmentation | ↓ WER on noise |

**Why this progression?**

1. Measure what actually helps
2. Avoid investing time in ineffective interventions
3. Isolate the impact of each change
4. Build scientific understanding, not just higher accuracy

---

## Evaluation

### Implemented Metrics

**Word Error Rate (WER)**
- Basic implementation using Levenshtein distance
- Calculated as: (insertions + deletions + substitutions) / reference_word_count

**Character Error Rate (CER)**
- Same calculation at character level

### Planned Metrics

- **WER by language** (English vs. Twi vs. Pidgin)
- **WER by environment** (office, market, street, etc.)
- **Code-switch subset WER** (samples with mixed languages)
- **Named entity accuracy** (Ghanaian names, places, organizations)
- **Noise robustness** (performance on noisy subsets)

### Evaluation Usage

```bash
python scripts/evaluate.py
```

**Current status**: Runs dummy evaluation on hardcoded data

**Future interface**:

```bash
python scripts/evaluate.py \
  --model models/exported/talk2me-whisper-gh-v0.1 \
  --dataset data/manifests/test.jsonl \
  --config configs/evaluation/ghana.yaml
```

---

## Hardware & Compute

### Recommended Setup

| Experiment | GPU | Batch Size | Training Time |
| --- | --- | --- | --- |
| **W0 Baseline** | None (CPU) | N/A | N/A |
| **W1 Fine-tuning** | NVIDIA T4+ | 8-16 | 2-4 hours |
| **W2-W3** | NVIDIA A100 | 32-64 | 8-16 hours |
| **X0-X1 XLS-R** | NVIDIA A100 | 16-32 | 12-24 hours |

### Google Colab GPU

Colab offers free T4 GPUs (limited hours). These are sufficient for initial experiments (W0, W1) but not for large-scale training.

For Phase 3+ experiments, use:
- Cloud GPU rental (AWS, GCP, Lambda Labs)
- University/institution resources
- On-premises infrastructure

**Colab limitation**: No guarantee of GPU availability or session length. Save checkpoints frequently.

---

## Reproducibility

### Experiment Metadata

Every training run should record:

```json
{
  "experiment_id": "W1_whisper_ghanaian_english_20260902",
  "date": "2026-09-02T14:30:00Z",
  "model": "whisper-small",
  "dataset": "commonvoice_english_ghana_subset",
  "config": "configs/training/whisper.yaml",
  "seed": 42,
  "python_version": "3.10.12",
  "pytorch_version": "2.2.0",
  "transformers_version": "4.41.0",
  "cuda_version": "11.8",
  "gpu": "NVIDIA A100",
  "git_commit": "abc123def456",
  "metrics": {
    "wer": 0.25,
    "cer": 0.10,
    "validation_loss": 0.45
  }
}
```

**Current status**: Metadata recording infrastructure is planned

---

## Model Formats

### Model Naming Convention

```
talk2me-{architecture}-{focus}-v{version}

Examples:
- talk2me-whisper-small-gh-v0.1
- talk2me-whisper-medium-multilingual-v1.0
- talk2me-xlsr-300m-gh-v0.1
```

### Model Stages

| Stage | Status | Example |
| --- | --- | --- |
| Checkpoint | Training in progress | `models/checkpoints/W1_whisper/checkpoint-1000/` |
| Exported | Ready for inference | `models/exported/talk2me-whisper-gh-v0.1/` |
| Production | Validated, deployed | Future |

**Current status**: Model registry exists; no checkpoints or exported models yet

---

## Data Privacy & Governance

### Internal Talk2Me Data

When integrating real Talk2Me meeting recordings:

1. **Consent**: Explicit opt-in from participants
2. **Anonymization**: Remove speaker identities, personal details
3. **Access Control**: Limited to authorized researchers
4. **Retention**: Clear data retention policies
5. **Auditing**: Log all data access
6. **Compliance**: Adhere to privacy regulations (GDPR, local law)

### External Datasets

When using third-party datasets:

1. **License**: Verify commercial-use permissions
2. **Attribution**: Cite sources in publications
3. **Restrictions**: Respect any confidentiality agreements
4. **Redistribution**: Do not redistribute without permission

---

## Troubleshooting

### Installation Issues

**"ModuleNotFoundError: No module named 'talk2me_speech'"**

```bash
# Ensure you're in the repo directory and venv is activated
cd talk2me-speech
source .venv/bin/activate
pip install -e .
```

**"CUDA not found / GPU not available"**

```python
import torch
print(torch.cuda.is_available())  # False on CPU-only systems
```

XLS-R training requires GPU. Whisper can be tested on CPU but will be slow.

### Dataset Issues

**"FileNotFoundError: data/manifests/ not found"**

```bash
mkdir -p data/manifests
python scripts/prepare_dataset.py
```

**Manifest format issues**: Ensure JSONL (one record per line), not JSON array.

### GPU Memory

If CUDA runs out of memory:

1. Reduce `batch_size` in config (8 → 4)
2. Reduce sequence length (truncate long audio)
3. Use `gradient_accumulation_steps` to simulate larger batches
4. Switch to smaller model (Whisper Small vs. Medium)

---

## Repository Structure

```
talk2me-speech/
├── configs/                    # YAML configurations
│   ├── base.yaml
│   ├── models/
│   ├── training/
│   ├── datasets/
│   └── evaluation/
├── data/                       # Data artifacts
│   ├── raw/                    # Original audio
│   ├── processed/              # Preprocessed
│   ├── interim/                # Intermediate
│   └── manifests/              # Dataset JSONL files
├── models/                     # Model artifacts
│   ├── checkpoints/            # Training checkpoints
│   ├── exported/               # Exported models
│   └── evaluation/             # Eval-specific models
├── notebooks/                  # Jupyter notebooks
│   ├── 00_colab_setup.ipynb    # Google Colab environment setup
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_error_analysis.ipynb
│   └── 03_model_comparison.ipynb
├── scripts/                    # CLI entry points
│   ├── prepare_dataset.py
│   ├── validate_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── export_model.py
├── src/talk2me_speech/         # Python package
│   ├── audio/                  # Audio processing
│   ├── datasets/               # Data loading & validation
│   ├── evaluation/             # Metrics
│   ├── models/                 # Model wrappers
│   ├── training/               # Training logic (planned)
│   ├── inference/              # Inference (planned)
│   ├── feedback/               # User feedback (planned)
│   └── api/                    # FastAPI service (stub)
├── tests/                      # Unit tests
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-container setup
├── Makefile                    # Build/run commands
├── pyproject.toml              # Package metadata
├── CONTEXT.md                  # Project context guide
└── README.md                   # This file
```

---

## Roadmap

### Phase 1: Foundation (Current)
✓ Project structure
✓ Configuration system
✓ Basic audio pipeline
⚠️ Placeholder implementations
🚀 Dataset adapters (next)

### Phase 2: Ghana Benchmark
- Complete KasaSpeech adapter
- Ghanaian English dataset preparation
- Speaker-safe splitting verification
- Named entity benchmark

### Phase 3: Baselines
- Whisper Small zero-shot (W0)
- Whisper Small fine-tuning (W1-W2)
- XLS-R baselines (X0-X1)
- Error analysis & metrics

### Phase 4: Ghanaian Optimization
- Ghanaian Pidgin support
- Twi language support
- Code-switching experiments (W3-W4, X3)
- Language-specific evaluation

### Phase 5: Robustness
- Noise augmentation
- Noise robustness evaluation
- Acoustic augmentation strategies
- Environmental variation handling

### Phase 6: Production Inference
- Streaming inference API
- Batch processing API
- Model deployment
- Latency optimization

### Phase 7: Integration
- Talk2Me meeting integration
- Real-time transcription
- Speaker diarization (future)
- Live feedback loop

### Phase 8: Production
- Quality assurance
- Performance monitoring
- Continuous improvement
- Production support

---

## Known Limitations

1. **No real training yet**: Training loops not yet implemented
2. **Limited evaluation**: Basic WER/CER only; missing language-specific metrics
3. **No dataset adapters**: KasaSpeech/CommonVoice require manual setup
4. **Stub API**: Inference endpoints not yet functional
5. **No streaming**: Real-time transcription infrastructure incomplete
6. **No speaker handling**: Speaker-safe splitting not yet implemented

All of these are **planned for upcoming phases**, not missing features.

---

## Quick Start

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pytest -v
```

### Common Commands

```bash
make install      # Install dependencies
make test         # Run tests
make lint         # Lint code
make format       # Format code
make docker-up    # Start API service
make docker-down  # Stop API service
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

**Note on external components:**
- Whisper: OpenAI (MIT)
- XLS-R: Meta (CC-BY-NC-4.0)
- Common Voice: Mozilla (CC0)
- KasaSpeech: Check source for licensing

See individual component licenses before commercial use.
