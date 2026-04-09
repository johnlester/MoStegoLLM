"""Tests for chunked encoding via the chunk_size parameter."""

from __future__ import annotations

from mostegollm import StegoCodec


class TestChunkedEncoding:
    """Verify encode(chunk_size=...) / decode(list) round-trip behaviour."""

    def test_short_data_single_chunk(self, codec: StegoCodec) -> None:
        """Data smaller than chunk_size produces a single-element list and round-trips."""
        original = b"hello"
        cover_texts = codec.encode(original, chunk_size=2000)
        assert isinstance(cover_texts, list)
        assert len(cover_texts) == 1
        recovered = codec.decode(cover_texts)
        assert recovered == original

    def test_single_chunk_matches_plain_encode(self, codec: StegoCodec) -> None:
        """When data fits in one chunk, result matches the non-chunked encode."""
        original = b"hello"
        cover_single = codec.encode(original)
        cover_long = codec.encode(original, chunk_size=2000)
        assert len(cover_long) == 1
        assert cover_long[0] == cover_single

    def test_multi_chunk_roundtrip(self, codec: StegoCodec) -> None:
        """A payload larger than chunk_size splits into multiple chunks and round-trips."""
        original = b"A" * 50 + b"B" * 50 + b"C" * 50
        cover_texts = codec.encode(original, chunk_size=50)
        assert len(cover_texts) == 3
        recovered = codec.decode(cover_texts)
        assert recovered == original

    def test_multi_chunk_encrypted_roundtrip(self, codec_encrypted: StegoCodec) -> None:
        """Chunked encoding with encryption round-trips correctly."""
        original = b"Secret message that spans multiple chunks!"
        cover_texts = codec_encrypted.encode(original, chunk_size=20)
        assert len(cover_texts) >= 2
        recovered = codec_encrypted.decode(cover_texts)
        assert recovered == original

    def test_prompt_chaining_coherence(self, codec: StegoCodec) -> None:
        """Chunk 1+ uses a chained prompt derived from the previous chunk's cover text."""
        original = b"X" * 100
        cover_texts = codec.encode(original, chunk_size=50)
        assert len(cover_texts) >= 2

        standalone_cover = codec.encode(original[50:100])
        assert cover_texts[1] != standalone_cover

    def test_chunk_size_respected(self, codec: StegoCodec) -> None:
        """Each chunk's plaintext payload does not exceed chunk_size."""
        chunk_size = 30
        original = b"A" * 100
        cover_texts = codec.encode(original, chunk_size=chunk_size)

        for idx, cover_text in enumerate(cover_texts):
            prompt = codec._prompt if idx == 0 else cover_texts[idx - 1][-500:]
            temp_codec = StegoCodec.__new__(StegoCodec)
            temp_codec._model = codec._model
            temp_codec._tokenizer = codec._tokenizer
            temp_codec._device = codec._device
            temp_codec._prompt = prompt
            temp_codec._top_k = codec._top_k
            temp_codec._temperature = codec._temperature
            temp_codec._password = None
            payload = temp_codec.decode(cover_text)
            assert len(payload) <= chunk_size

    def test_str_chunked_roundtrip(self, codec: StegoCodec) -> None:
        """String data can be chunked via encode(text.encode(), chunk_size=...)."""
        original = "Hello, world! " * 10
        cover_texts = codec.encode(original.encode(), chunk_size=50)
        assert len(cover_texts) >= 2
        recovered = codec.decode(cover_texts).decode()
        assert recovered == original

    def test_uneven_last_chunk(self, codec: StegoCodec) -> None:
        """When data doesn't divide evenly, the last chunk is smaller."""
        original = b"A" * 70
        cover_texts = codec.encode(original, chunk_size=30)
        assert len(cover_texts) == 3
        recovered = codec.decode(cover_texts)
        assert recovered == original
