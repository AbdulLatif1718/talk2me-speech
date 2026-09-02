from talk2me_speech.models.whisper import WhisperModel
from talk2me_speech.models.xlsr import XLSRModel
from talk2me_speech.training.collator import collate_batch
from talk2me_speech.training.train_whisper import build_train_config as build_whisper_config, train_whisper
from talk2me_speech.training.train_xlsr import build_train_config as build_xlsr_config, train_xlsr


def test_whisper_model_wrapper():
    model = WhisperModel(model_name="whisper-small")
    info = model.load()
    assert info["architecture"] == "whisper"
    assert info["hf_model_id"] == "openai/whisper-small"


def test_xlsr_model_wrapper():
    model = XLSRModel(model_name="xlsr-300m")
    info = model.load()
    assert info["architecture"] == "xlsr"
    assert info["hf_model_id"] == "facebook/wav2vec2-xls-r-300m"


def test_collate_batch():
    batch = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
    collated = collate_batch(batch)
    assert collated["batch_size"] == 2
    assert len(collated["items"]) == 2


def test_train_whisper_run(tmp_path):
    config = build_whisper_config(
        model_name="whisper-small",
        epochs=1,
        output_dir=tmp_path / "whisper_checkpoints",
    )
    summary = train_whisper(config)
    assert summary["status"] == "completed"
    assert "metrics" in summary
    assert (tmp_path / "whisper_checkpoints" / "training_summary.json").exists()


def test_train_xlsr_run(tmp_path):
    config = build_xlsr_config(
        model_name="xlsr-300m",
        epochs=1,
        output_dir=tmp_path / "xlsr_checkpoints",
    )
    summary = train_xlsr(config)
    assert summary["status"] == "completed"
    assert "metrics" in summary
    assert (tmp_path / "xlsr_checkpoints" / "training_summary.json").exists()

