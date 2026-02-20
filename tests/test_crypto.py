"""Tests for the optional encryption layer."""

from __future__ import annotations

import pytest

from mostegollm import StegoCryptoError, StegoCodec
from mostegollm.crypto import decrypt, encrypt


# ---------------------------------------------------------------------------
# Unit tests for encrypt / decrypt
# ---------------------------------------------------------------------------


class TestCryptoUnit:
    """Low-level encrypt/decrypt without the codec."""

    def test_round_trip(self) -> None:
        plaintext = b"hello, world!"
        blob = encrypt(plaintext, "secret")
        assert decrypt(blob, "secret") == plaintext

    def test_empty_payload(self) -> None:
        blob = encrypt(b"", "pw")
        assert decrypt(blob, "pw") == b""

    def test_wrong_password_raises(self) -> None:
        blob = encrypt(b"data", "right")
        with pytest.raises(StegoCryptoError, match="wrong password"):
            decrypt(blob, "wrong")

    def test_tampered_ciphertext_raises(self) -> None:
        blob = bytearray(encrypt(b"data", "pw"))
        # Flip a byte in the ciphertext portion (after salt + nonce)
        blob[30] ^= 0xFF
        with pytest.raises(StegoCryptoError, match="wrong password"):
            decrypt(bytes(blob), "pw")

    def test_truncated_blob_raises(self) -> None:
        with pytest.raises(StegoCryptoError, match="too short"):
            decrypt(b"short", "pw")


# ---------------------------------------------------------------------------
# Encrypted codec round-trip
# ---------------------------------------------------------------------------


class TestEncryptedRoundTrip:
    """Encode → decode with encryption enabled."""

    def test_short_string(self, codec_encrypted: StegoCodec) -> None:
        secret = b"Top Secret!"
        cover = codec_encrypted.encode(secret)
        assert codec_encrypted.decode(cover) == secret

    def test_long_classified_message(self, codec_encrypted: StegoCodec) -> None:
        secret = (
            "EYES ONLY — CIPHER BRANCH SEVEN — PRIORITY ALPHA\n"
            "Transmission Origin: Station Wren / Routing: Lisbon → Vienna"
            " → Dead Drop Edelweiss\n"
            "\n"
            "VERMILION,\n"
            "The nightingale has stopped singing in the garden. You will"
            " know what this means. We have little time and the courier"
            " window is narrowing faster than anticipated — OVERCOAT's"
            " people have been watching the postal routes since Tuesday,"
            " and the man we had at the bridge is no longer answering his"
            " telephone. I do not need to tell you what that suggests.\n"
            "Burn this. Then burn the ash.\n"
            "— CROW\n"
        )
        cover = codec_encrypted.encode_str(secret)
        recovered = codec_encrypted.decode_str(cover)
        assert recovered == secret

    def test_wrong_password_decode_fails(self, codec_encrypted: StegoCodec) -> None:
        cover = codec_encrypted.encode(b"data")
        wrong_pw_codec = StegoCodec(device="cpu", password="wrong-password")
        # Share the already-loaded model to avoid reloading
        wrong_pw_codec._model = codec_encrypted._model
        wrong_pw_codec._tokenizer = codec_encrypted._tokenizer
        wrong_pw_codec._device = codec_encrypted._device
        with pytest.raises(StegoCryptoError):
            wrong_pw_codec.decode(cover)


# ---------------------------------------------------------------------------
# Avalanche effect
# ---------------------------------------------------------------------------


class TestAvalancheEffect:
    """Encryption should cause early, pervasive divergence in cover text."""

    def test_one_char_difference_diverges_early(
        self, codec_encrypted: StegoCodec
    ) -> None:
        msg_a = b"Hello, world!"
        msg_b = b"Hello, World!"  # single-char difference

        cover_a = codec_encrypted.encode(msg_a)
        cover_b = codec_encrypted.encode(msg_b)

        # Tokenize both cover texts so we can compare token-by-token
        tokenizer = codec_encrypted._tokenizer
        assert tokenizer is not None
        ids_a = tokenizer.encode(cover_a)
        ids_b = tokenizer.encode(cover_b)

        # Find first divergence index
        diverge_idx = None
        for i, (ta, tb) in enumerate(zip(ids_a, ids_b)):
            if ta != tb:
                diverge_idx = i
                break

        # With encryption, the ciphertext bytes differ completely, so
        # the cover texts should diverge early.  The first several tokens
        # may correspond to shared prompt context, so we allow a small
        # prefix before requiring divergence.
        assert diverge_idx is not None, "Cover texts are identical — no avalanche"
        assert diverge_idx <= 20, (
            f"Divergence at token {diverge_idx}; expected within first 20 tokens"
        )
