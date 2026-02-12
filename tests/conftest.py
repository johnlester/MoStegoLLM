"""Shared pytest fixtures for MoStegoLLM tests."""

from __future__ import annotations

import pytest

from mostegollm import StegoCodec


@pytest.fixture(scope="session")
def codec() -> StegoCodec:
    """Session-scoped codec instance (loads the model once for all tests)."""
    return StegoCodec(device="cpu")


@pytest.fixture(scope="session")
def codec_alt_prompt() -> StegoCodec:
    """Codec with a different prompt for cross-prompt tests."""
    return StegoCodec(device="cpu", prompt="Once upon a time in a faraway land,")
