"""API-level tests for StegoCodec."""

from __future__ import annotations

import pytest

from mostegollm import StegoCodec, StegoDecodeError, StegoStats


class TestCodecAPI:
    """Test the public StegoCodec interface."""

    def test_encode_returns_string(self, codec: StegoCodec) -> None:
        """encode() should return a non-empty string."""
        result = codec.encode(b"test")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_decode_returns_bytes(self, codec: StegoCodec) -> None:
        """decode() should return bytes."""
        cover = codec.encode(b"test")
        result = codec.decode(cover)
        assert isinstance(result, bytes)

    def test_encode_with_stats(self, codec: StegoCodec) -> None:
        """encode_with_stats() should return a StegoStats with sane values."""
        data = b"stats test payload"
        stats = codec.encode_with_stats(data)

        assert isinstance(stats, StegoStats)
        assert isinstance(stats.cover_text, str)
        assert len(stats.cover_text) > 0
        assert stats.total_tokens > 0
        assert stats.bits_per_token > 0
        assert stats.payload_size_bytes == len(data)

    def test_stateless_multiple_calls(self, codec: StegoCodec) -> None:
        """Multiple encode/decode calls on the same instance should work."""
        data1 = b"first"
        data2 = b"second"

        cover1 = codec.encode(data1)
        cover2 = codec.encode(data2)

        assert codec.decode(cover1) == data1
        assert codec.decode(cover2) == data2

    def test_wrong_prompt_fails(self, codec: StegoCodec, codec_alt_prompt: StegoCodec) -> None:
        """Decoding with a different prompt should fail or return wrong data."""
        data = b"prompt sensitive"
        cover = codec.encode(data)

        with pytest.raises((StegoDecodeError, AssertionError)):
            codec_alt_prompt.decode(cover)

    def test_chunked_returns_list(self, codec: StegoCodec) -> None:
        """encode(chunk_size=...) returns a list."""
        result = codec.encode(b"test", chunk_size=100)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_decode_accepts_list(self, codec: StegoCodec) -> None:
        """decode() accepts a list of strings."""
        covers = codec.encode(b"test data here", chunk_size=5)
        result = codec.decode(covers)
        assert isinstance(result, bytes)
        assert result == b"test data here"
