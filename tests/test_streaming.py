from talk2me_speech.inference.streaming import stream_transcript


def test_stream_transcript_returns_strings():
    chunks = [b"a", b"b", "c"]
    result = stream_transcript(chunks)
    assert result == ["b'a'", "b'b'", "c"]
