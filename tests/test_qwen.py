"""Tests for the Qwen/Qwen2.5-0.5B model with MoStegoLLM."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mostegollm import StegoCryptoError, StegoCodec, StegoDecodeError, StegoStats

QWEN_MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def qwen_codec() -> StegoCodec:
    """Module-scoped codec using Qwen2.5-0.5B (loaded once for all tests)."""
    return StegoCodec(model_name=QWEN_MODEL, device="cpu")


@pytest.fixture(scope="module")
def qwen_codec_encrypted(qwen_codec: StegoCodec) -> StegoCodec:
    """Qwen codec with AES-256-GCM encryption, sharing the loaded model."""
    c = StegoCodec(model_name=QWEN_MODEL, device="cpu", password="qwen-test-pw")
    model, tokenizer, device = qwen_codec._ensure_model()
    c._model = model
    c._tokenizer = tokenizer
    c._device = device
    return c


@pytest.fixture(scope="module")
def qwen_codec_prompted(qwen_codec: StegoCodec) -> StegoCodec:
    """Qwen codec with a custom prompt, sharing the loaded model."""
    c = StegoCodec(model_name=QWEN_MODEL, device="cpu", prompt="In the year 2050,")
    model, tokenizer, device = qwen_codec._ensure_model()
    c._model = model
    c._tokenizer = tokenizer
    c._device = device
    return c


class TestQwenRoundTrip:
    """Verify encode → decode round-trips with Qwen2.5-0.5B."""

    def test_short_string(self, qwen_codec: StegoCodec) -> None:
        original = b"hello"
        cover = qwen_codec.encode(original)
        assert qwen_codec.decode(cover) == original

    def test_single_byte(self, qwen_codec: StegoCodec) -> None:
        original = b"\x42"
        cover = qwen_codec.encode(original)
        assert qwen_codec.decode(cover) == original

    def test_longer_string(self, qwen_codec: StegoCodec) -> None:
        original = b"The quick brown fox jumps over the lazy dog."
        cover = qwen_codec.encode(original)
        assert qwen_codec.decode(cover) == original

    def test_binary_bytes(self, qwen_codec: StegoCodec) -> None:
        original = bytes(range(256))
        cover = qwen_codec.encode(original)
        assert qwen_codec.decode(cover) == original

    def test_empty_bytes(self, qwen_codec: StegoCodec) -> None:
        original = b""
        cover = qwen_codec.encode(original)
        assert qwen_codec.decode(cover) == original

    def test_str_convenience(self, qwen_codec: StegoCodec) -> None:
        original = "hello world"
        cover = qwen_codec.encode_str(original)
        assert qwen_codec.decode_str(cover) == original

    def test_null_bytes(self, qwen_codec: StegoCodec) -> None:
        data = b"\x00" * 10
        cover = qwen_codec.encode(data)
        assert qwen_codec.decode(cover) == data

    def test_repeated_byte(self, qwen_codec: StegoCodec) -> None:
        data = b"\xAA" * 20
        cover = qwen_codec.encode(data)
        assert qwen_codec.decode(cover) == data


class TestQwenDeterminism:
    """Encoding the same input twice should produce identical cover text."""

    def test_deterministic_output(self, qwen_codec: StegoCodec) -> None:
        data = b"deterministic"
        cover1 = qwen_codec.encode(data)
        cover2 = qwen_codec.encode(data)
        assert cover1 == cover2


class TestQwenAPI:
    """Test public API methods with the Qwen model."""

    def test_encode_returns_string(self, qwen_codec: StegoCodec) -> None:
        result = qwen_codec.encode(b"test")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_decode_returns_bytes(self, qwen_codec: StegoCodec) -> None:
        cover = qwen_codec.encode(b"test")
        result = qwen_codec.decode(cover)
        assert isinstance(result, bytes)

    def test_encode_with_stats(self, qwen_codec: StegoCodec) -> None:
        data = b"stats test payload"
        stats = qwen_codec.encode_with_stats(data)
        assert isinstance(stats, StegoStats)
        assert isinstance(stats.cover_text, str)
        assert len(stats.cover_text) > 0
        assert stats.total_tokens > 0
        assert stats.bits_per_token > 0
        assert stats.payload_size_bytes == len(data)

    def test_file_roundtrip(self, qwen_codec: StegoCodec) -> None:
        data = b"file content \x00\xff"
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "input.bin"
            dst = Path(tmpdir) / "output.bin"
            src.write_bytes(data)

            cover = qwen_codec.encode_file(src)
            qwen_codec.decode_file(cover, dst)

            assert dst.read_bytes() == data

    def test_stateless_multiple_calls(self, qwen_codec: StegoCodec) -> None:
        data1 = b"first"
        data2 = b"second"
        cover1 = qwen_codec.encode(data1)
        cover2 = qwen_codec.encode(data2)
        assert qwen_codec.decode(cover1) == data1
        assert qwen_codec.decode(cover2) == data2


class TestQwenEdgeCases:
    """Edge cases with the Qwen model."""

    def test_decode_garbage_text(self, qwen_codec: StegoCodec) -> None:
        with pytest.raises(StegoDecodeError):
            qwen_codec.decode("This is just a random sentence with no hidden data.")

    def test_decode_empty_string(self, qwen_codec: StegoCodec) -> None:
        with pytest.raises(StegoDecodeError):
            qwen_codec.decode("")

    def test_wrong_prompt_fails(
        self, qwen_codec: StegoCodec, qwen_codec_prompted: StegoCodec
    ) -> None:
        data = b"prompt sensitive"
        cover = qwen_codec.encode(data)
        with pytest.raises((StegoDecodeError, AssertionError)):
            qwen_codec_prompted.decode(cover)


class TestQwenEncrypted:
    """Test AES-256-GCM encryption with the Qwen model."""

    def test_encrypted_roundtrip(self, qwen_codec_encrypted: StegoCodec) -> None:
        original = b"encrypted secret"
        cover = qwen_codec_encrypted.encode(original)
        assert qwen_codec_encrypted.decode(cover) == original

    def test_encrypted_str_roundtrip(self, qwen_codec_encrypted: StegoCodec) -> None:
        original = "encrypted string"
        cover = qwen_codec_encrypted.encode_str(original)
        assert qwen_codec_encrypted.decode_str(cover) == original

    def test_wrong_password_fails(self, qwen_codec_encrypted: StegoCodec) -> None:
        cover = qwen_codec_encrypted.encode(b"secret")
        wrong_codec = StegoCodec(model_name=QWEN_MODEL, device="cpu", password="wrong-pw")
        model, tokenizer, device = qwen_codec_encrypted._ensure_model()
        wrong_codec._model = model
        wrong_codec._tokenizer = tokenizer
        wrong_codec._device = device
        with pytest.raises(StegoCryptoError, match="wrong password"):
            wrong_codec.decode(cover)


class TestQwenChunked:
    """Test chunked encoding/decoding with the Qwen model."""

    def test_single_chunk(self, qwen_codec: StegoCodec) -> None:
        original = b"hello"
        cover_texts = qwen_codec.encode_long(original, chunk_size=2000)
        assert len(cover_texts) == 1
        assert qwen_codec.decode_long(cover_texts) == original

    def test_multi_chunk_roundtrip(self, qwen_codec: StegoCodec) -> None:
        original = b"A" * 50 + b"B" * 50 + b"C" * 50
        cover_texts = qwen_codec.encode_long(original, chunk_size=50)
        assert len(cover_texts) == 3
        assert qwen_codec.decode_long(cover_texts) == original

    def test_chunked_str_roundtrip(self, qwen_codec: StegoCodec) -> None:
        original = "Hello, world! " * 10
        cover_texts = qwen_codec.encode_long_str(original, chunk_size=50)
        assert len(cover_texts) >= 2
        assert qwen_codec.decode_long_str(cover_texts) == original

    def test_chunked_encrypted_roundtrip(self, qwen_codec_encrypted: StegoCodec) -> None:
        original = b"Secret chunked message!"
        cover_texts = qwen_codec_encrypted.encode_long(original, chunk_size=20)
        assert len(cover_texts) >= 2
        assert qwen_codec_encrypted.decode_long(cover_texts) == original
