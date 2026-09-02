import csv
import json
from talk2me_speech.datasets.adapters import get_adapter, CommonVoiceAdapter, KasaSpeechAdapter
from talk2me_speech.datasets.loader import load_manifest
from talk2me_speech.datasets.manifest import write_manifest
from talk2me_speech.datasets.splitter import split_records, split_speaker_safe, verify_speaker_isolation
from talk2me_speech.datasets.validator import validate_manifest_records, validate_record_schema, validate_dataset_root


def test_load_manifest_returns_empty_for_missing_file(tmp_path):
    manifest = tmp_path / "missing.jsonl"
    assert load_manifest(manifest) == []


def test_split_records_splits_data():
    records = [{"id": i} for i in range(10)]
    train, val = split_records(records, train_ratio=0.8)
    assert len(train) == 8
    assert len(val) == 2


def test_split_speaker_safe_prevents_leakage():
    records = [
        {"id": "1", "speaker_id": "spk_A", "transcript": "a", "audio_path": "a.wav"},
        {"id": "2", "speaker_id": "spk_A", "transcript": "b", "audio_path": "b.wav"},
        {"id": "3", "speaker_id": "spk_B", "transcript": "c", "audio_path": "c.wav"},
        {"id": "4", "speaker_id": "spk_B", "transcript": "d", "audio_path": "d.wav"},
        {"id": "5", "speaker_id": "spk_C", "transcript": "e", "audio_path": "e.wav"},
        {"id": "6", "speaker_id": "spk_D", "transcript": "f", "audio_path": "f.wav"},
    ]
    train, val, test = split_speaker_safe(records, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    assert verify_speaker_isolation(train, val, test) is True


def test_validate_record_schema():
    valid_record = {
        "id": "rec_01",
        "audio_path": "audio/sample.wav",
        "transcript": "Chale let's go",
        "duration": 2.5,
    }
    is_valid, errors = validate_record_schema(valid_record)
    assert is_valid is True
    assert len(errors) == 0

    invalid_record = {
        "id": "rec_02",
        "audio_path": "",
        "transcript": "No audio path provided",
    }
    is_valid, errors = validate_record_schema(invalid_record)
    assert is_valid is False
    assert any("audio_path" in err for err in errors)


def test_validate_manifest_records(tmp_path):
    records = [
        {"id": "1", "audio_path": "a.wav", "transcript": "hello"},
        {"id": "2", "audio_path": "b.wav", "transcript": "world"},
    ]
    valid, errors = validate_manifest_records(records)
    assert valid is True
    assert len(errors) == 0


def test_commonvoice_adapter(tmp_path):
    tsv_file = tmp_path / "validated.tsv"
    with tsv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment"])
        writer.writerow(["client_123", "clip1.mp3", "Welcome to Ghana", "2", "0", "twenties", "male", "Ghanaian English", "en", ""])

    adapter = get_adapter("commonvoice")
    assert isinstance(adapter, CommonVoiceAdapter)

    records = adapter.convert(tmp_path, language="en")
    assert len(records) == 1
    assert records[0]["transcript"] == "Welcome to Ghana"
    assert records[0]["speaker_id"] == "client_123"
    assert records[0]["country"] == "GH"


def test_kasaspeech_adapter(tmp_path):
    csv_file = tmp_path / "metadata.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "audio_file", "transcript", "speaker_id", "languages", "duration"])
        writer.writerow(["kasa_1", "twi_01.wav", "Mepa wo kyɛw", "spk_kasa_1", "twi,en", "3.4"])

    adapter = get_adapter("kasaspeech")
    assert isinstance(adapter, KasaSpeechAdapter)

    records = adapter.convert(tmp_path)
    assert len(records) == 1
    assert records[0]["transcript"] == "Mepa wo kyɛw"
    assert records[0]["speaker_id"] == "spk_kasa_1"
    assert records[0]["country"] == "GH"
    assert "twi" in records[0]["languages"]
