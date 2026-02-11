"""Edge-case tests for MoStegoLLM."""

from __future__ import annotations

import pytest

from mostegollm import StegoCodec, StegoDecodeError
from mostegollm.utils import (
    HEADER_SIZE,
    StegoEncodeError,
    bits_to_bytes,
    bytes_to_bits,
    pack_header,
    unpack_header,
)


class TestBitConversion:
    """Test bit-level helpers."""

    def test_roundtrip_empty(self) -> None:
        assert bits_to_bytes(bytes_to_bits(b"")) == b""

    def test_roundtrip_single_byte(self) -> None:
        for v in (0x00, 0x01, 0x7F, 0x80, 0xFF):
            assert bits_to_bytes(bytes_to_bits(bytes([v]))) == bytes([v])

    def test_roundtrip_all_bytes(self) -> None:
        data = bytes(range(256))
        assert bits_to_bytes(bytes_to_bits(data)) == data

    def test_bits_to_bytes_padding(self) -> None:
        """Partial last byte should be zero-padded on the right."""
        # 3 bits: 101 → 10100000 = 0xA0
        assert bits_to_bytes([1, 0, 1]) == bytes([0xA0])


class TestHeader:
    """Test header pack/unpack."""

    def test_roundtrip(self) -> None:
        for length in (0, 1, 255, 65535, 1_000_000):
            header = pack_header(length)
            assert len(header) == HEADER_SIZE
            assert unpack_header(header) == length

    def test_bad_magic(self) -> None:
        header = b"\x00\x00\x00\x00\x00\x00"
        with pytest.raises(StegoDecodeError, match="Invalid magic"):
            unpack_header(header)

    def test_negative_length(self) -> None:
        with pytest.raises(StegoEncodeError, match="negative"):
            pack_header(-1)

    def test_too_large(self) -> None:
        with pytest.raises(StegoEncodeError, match="too large"):
            pack_header(0x1_0000_0000)


class TestEdgeCases:
    """Edge cases for the full codec."""

    def test_decode_garbage_text(self, codec: StegoCodec) -> None:
        """Decoding random text should raise StegoDecodeError."""
        with pytest.raises(StegoDecodeError):
            codec.decode("This is just a random sentence with no hidden data.")

    def test_decode_empty_string(self, codec: StegoCodec) -> None:
        """Decoding an empty string should raise StegoDecodeError."""
        with pytest.raises(StegoDecodeError):
            codec.decode("")

    def test_repeated_byte(self, codec: StegoCodec) -> None:
        """Repeated bytes should round-trip."""
        data = b"\xAA" * 20
        cover = codec.encode(data)
        assert codec.decode(cover) == data

    def test_null_bytes(self, codec: StegoCodec) -> None:
        """Null bytes should round-trip."""
        data = b"\x00" * 10
        cover = codec.encode(data)
        assert codec.decode(cover) == data
