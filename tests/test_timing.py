"""Timing test for MoStegoLLM encoding."""

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
