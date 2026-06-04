"""Offline cross-compatibility guard: decode the committed golden corpus on
whatever this machine is. Runs on every `pytest` invocation (no cloud)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mostegollm.compat import read_vectors, verify_vector

GOLDEN = Path(__file__).parent.parent / "compat" / "golden_vectors.jsonl"


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden corpus not generated yet")
def test_golden_vectors_all_decode(codec):
    model, tok, dev = codec._ensure_model()
    vectors = read_vectors(GOLDEN)
    assert vectors, "golden corpus is empty"

    failures = []
    for v in vectors:
        result = verify_vector(v, model=model, tokenizer=tok, device=dev)
        if not result.ok:
            failures.append(
                f"{v.prompt!r} / {v.payload_hex[:12]}: {result.failure_class} — {result.detail}"
            )
    assert not failures, f"{len(failures)}/{len(vectors)} golden vectors failed:\n" + "\n".join(
        failures[:10]
    )
