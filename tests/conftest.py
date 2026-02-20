"""Shared pytest fixtures for MoStegoLLM tests."""

from __future__ import annotations

import pytest

from mostegollm import StegoCodec


@pytest.fixture(scope="session")
def codec() -> StegoCodec:
    """Session-scoped codec instance (loads the model once for all tests)."""
    return StegoCodec(device="cpu")


@pytest.fixture(scope="session")
def codec_alt_prompt(codec: StegoCodec) -> StegoCodec:
    """Codec with a different prompt for cross-prompt tests."""
    c = StegoCodec(device="cpu", prompt="Once upon a time in a faraway land,")
    model, tokenizer, device = codec._ensure_model()
    c._model = model
    c._tokenizer = tokenizer
    c._device = device
    return c


@pytest.fixture(scope="session")
def codec_sentence_boundary(codec: StegoCodec) -> StegoCodec:
    """Codec with sentence_boundary=True for sentence-ending tests."""
    c = StegoCodec(device="cpu", sentence_boundary=True)
    model, tokenizer, device = codec._ensure_model()
    c._model = model
    c._tokenizer = tokenizer
    c._device = device
    return c


@pytest.fixture(scope="session")
def codec_encrypted(codec: StegoCodec) -> StegoCodec:
    """Codec with AES-256-GCM encryption enabled, sharing the model from *codec*."""
    c = StegoCodec(device="cpu", password="test-password")
    # Force model load on the base codec and share the instance to avoid loading
    # a second copy (which introduces non-determinism on some platforms).
    model, tokenizer, device = codec._ensure_model()
    c._model = model
    c._tokenizer = tokenizer
    c._device = device
    return c
