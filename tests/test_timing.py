"""Timing and sentence-boundary tests for MoStegoLLM encoding."""

from __future__ import annotations

import time

from mostegollm import StegoCodec


def test_encode_short_timing(codec: StegoCodec) -> None:
    """Time encoding a trivially short payload."""
    data = b"hi"

    start = time.perf_counter()
    codec.encode(data)
    elapsed = time.perf_counter() - start

    print(f"\nEncoding {len(data)} bytes took {elapsed:.3f}s")


def test_sentence_boundary_ending(codec_sentence_boundary: StegoCodec) -> None:
    """Cover text should end at a sentence boundary when sentence_boundary=True."""
    data = b"hello"
    cover = codec_sentence_boundary.encode(data)
    stripped = cover.rstrip()
    assert len(stripped) > 0, "Cover text should not be empty"
    assert stripped[-1] in ".!?", (
        f"Expected cover text to end with '.', '!' or '?' but got: ...{stripped[-20:]!r}"
    )


def test_sentence_boundary_roundtrip(
    codec_sentence_boundary: StegoCodec, codec: StegoCodec
) -> None:
    """Round-trip should succeed with sentence_boundary=True."""
    data = b"secret message"
    cover = codec_sentence_boundary.encode(data)
    recovered = codec.decode(cover)
    assert recovered == data
