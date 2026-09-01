from talk2me_speech.evaluation.cer import compute_cer
from talk2me_speech.evaluation.wer import compute_wer
from talk2me_speech.inference.postprocess import postprocess_transcript


def test_wer_is_zero_on_perfect_match():
    assert compute_wer("hello world", "hello world") == 0.0


def test_cer_is_zero_on_perfect_match():
    assert compute_cer("hello", "hello") == 0.0


def test_postprocess_transcript_removes_extra_spaces():
    assert postprocess_transcript("  hello   world  ") == "hello world"
