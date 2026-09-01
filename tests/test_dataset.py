from talk2me_speech.datasets.loader import load_manifest
from talk2me_speech.datasets.splitter import split_records


def test_load_manifest_returns_empty_for_missing_file(tmp_path):
    manifest = tmp_path / "missing.jsonl"
    assert load_manifest(manifest) == []


def test_split_records_splits_data():
    records = [{"id": i} for i in range(10)]
    train, val = split_records(records, train_ratio=0.8)
    assert len(train) == 8
    assert len(val) == 2
