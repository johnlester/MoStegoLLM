"""Round-trip encode → decode tests for MoStegoLLM."""

from __future__ import annotations

import pytest

from mostegollm import StegoCodec


class TestRoundTrip:
    """Verify that data survives an encode → decode cycle."""

    def test_short_string(self, codec: StegoCodec) -> None:
        """A short ASCII string should round-trip exactly."""
        original = b"hello"
        cover = codec.encode(original)
        recovered = codec.decode(cover)
        assert recovered == original

    def test_single_byte(self, codec: StegoCodec) -> None:
        """The smallest non-empty payload (1 byte) should round-trip."""
        original = b"\x42"
        cover = codec.encode(original)
        recovered = codec.decode(cover)
        assert recovered == original

    def test_longer_string(self, codec: StegoCodec) -> None:
        """A longer payload should round-trip."""
        original = b"The quick brown fox jumps over the lazy dog."
        cover = codec.encode(original)
        recovered = codec.decode(cover)
        assert recovered == original

    def test_binary_bytes(self, codec: StegoCodec) -> None:
        """Arbitrary binary data (all byte values) should round-trip."""
        original = bytes(range(256))
        cover = codec.encode(original)
        recovered = codec.decode(cover)
        assert recovered == original

    def test_empty_bytes(self, codec: StegoCodec) -> None:
        """An empty payload should round-trip (header still encodes length=0)."""
        original = b""
        cover = codec.encode(original)
        recovered = codec.decode(cover)
        assert recovered == original

    def test_str_convenience(self, codec: StegoCodec) -> None:
        """encode_str / decode_str should round-trip a string."""
        original = "hello world"
        cover = codec.encode_str(original)
        recovered = codec.decode_str(cover)
        assert recovered == original

    def test_determinism(self, codec: StegoCodec) -> None:
        """Encoding the same input twice should produce identical output."""
        data = b"deterministic"
        cover1 = codec.encode(data)
        cover2 = codec.encode(data)
        assert cover1 == cover2
