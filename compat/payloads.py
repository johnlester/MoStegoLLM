"""Deterministic canonical payloads and prompts for cross-compatibility vectors.

NOTHING here may use os.urandom or unseeded randomness: a reproducibility corpus
must itself be byte-for-byte reproducible. (The intermittent Qwen decode bug
traced to a random-payload test — see the spec.)"""

from __future__ import annotations

import random


def _seeded(n: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(n))


CANONICAL_PROMPTS: list[str] = [
    "According to experts,",
    "In a quiet village near",
    "The data shows that",
]

CANONICAL_PAYLOADS: dict[str, bytes] = {
    "ascii-short": b"hello",
    "single-byte": b"\x42",
    "all-zero": b"\x00" * 8,
    "all-ff": b"\xff" * 8,
    "binary-edge": b"\x00\x01\x02\x03\xfe\xff",
    "text": b"The quick brown fox jumps over the lazy dog.",
    "seeded-32": _seeded(32, 0),
    "seeded-64": _seeded(64, 1),
}
