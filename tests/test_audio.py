import numpy as np

from talk2me_speech.audio.normalize import normalize_audio
from talk2me_speech.audio.resample import resample_audio


def test_normalize_audio_scales_peak():
    audio = np.array([-1.0, 0.5, 0.25], dtype=np.float32)
    result = normalize_audio(audio, target_dbfs=-6.0)
    assert np.isclose(np.max(np.abs(result)), 0.5011872336272722, rtol=1e-4)


def test_resample_audio_keeps_length_same_rate_when_equal():
    audio = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    result = resample_audio(audio, 16000, 16000)
    assert result.shape == audio.shape
