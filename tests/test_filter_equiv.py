"""The optimized (batched) BPE filter must be byte-identical to the naive one.

This is the correctness gate for the encode/decode hot-path optimization: the
keep-set the filter produces is what makes cover text re-tokenize back to the
generated ids, so any divergence from the original per-token logic would break
round-trip / cross-platform decoding.
"""

from __future__ import annotations

from mostegollm.encoder import (
    _filter_tokens,
    _get_merge_cache,
    _scan_non_roundtrip,
    get_non_roundtrip_tokens,
)


def _naive_filter(tokenizer, prev, ids, logits, non_rt):
    """The original per-token reference implementation."""
    keep_ids, keep_logits = [], []
    for tid, lg in zip(ids, logits):
        if tid in non_rt:
            continue
        if prev is not None:
            text = tokenizer.decode([prev, tid])
            if tokenizer.encode(text, add_special_tokens=False) != [prev, tid]:
                continue
        keep_ids.append(tid)
        keep_logits.append(lg)
    if not keep_ids:
        return ids, logits
    return keep_ids, keep_logits


def test_batched_filter_matches_naive(codec):
    _, tok, _ = codec._ensure_model()
    non_rt = get_non_roundtrip_tokens(tok)
    ids = list(range(100, 460))  # 360 real candidate token ids
    logits = [float(-i) for i in range(len(ids))]  # strictly descending, like top-k

    for prev in (None, 13, 262, 1000, 5000):
        cache: dict = {}
        opt = _filter_tokens(tok, prev, ids, logits, non_rt, cache)
        naive = _naive_filter(tok, prev, ids, logits, non_rt)
        assert opt == naive, f"mismatch for prev={prev}"
        # Warm-cache path returns the identical result.
        assert _filter_tokens(tok, prev, ids, logits, non_rt, cache) == opt


def test_merge_cache_persists_per_tokenizer(codec):
    _, tok, _ = codec._ensure_model()
    c1 = _get_merge_cache(tok)
    c2 = _get_merge_cache(tok)
    assert c1 is c2  # same persistent dict for the same tokenizer


def test_batched_non_roundtrip_scan_matches_naive(codec):
    _, tok, _ = codec._ensure_model()
    ids = list(range(0, 9000))  # spans multiple scan chunks (chunk=4096)
    batched = _scan_non_roundtrip(tok, ids)
    naive = {tid for tid in ids if tok.encode(tok.decode([tid]), add_special_tokens=False) != [tid]}
    assert batched == naive
