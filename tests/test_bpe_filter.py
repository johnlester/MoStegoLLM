"""Tests for the BPE round-trip token filter (new (token_ids, logits) signature)."""

from __future__ import annotations

from mostegollm import StegoCodec
from mostegollm.encoder import _filter_tokens, get_non_roundtrip_tokens


def test_filter_drops_non_roundtrip_tokens(codec: StegoCodec):
    _model, tokenizer, _device = codec._ensure_model()
    non_rt = get_non_roundtrip_tokens(tokenizer)
    bad = next(iter(non_rt)) if non_rt else None
    if bad is None:
        return  # tokenizer has no non-round-trip tokens; nothing to assert
    ids = [bad, 100, 101]
    logits = [9.0, 8.0, 7.0]
    kept_ids, kept_logits = _filter_tokens(tokenizer, None, ids, logits, non_rt, {})
    assert bad not in kept_ids
    assert len(kept_ids) == len(kept_logits)


def test_filter_preserves_correspondence(codec: StegoCodec):
    _model, tokenizer, _device = codec._ensure_model()
    non_rt = get_non_roundtrip_tokens(tokenizer)
    # Real text tokens (deduped) so the normal keep-path runs, not the fallback.
    ids = list(
        dict.fromkeys(tokenizer.encode("hello world example text", add_special_tokens=False))
    )[:4]
    logits = [4.0, 3.0, 2.0, 1.0][: len(ids)]
    kept_ids, kept_logits = _filter_tokens(tokenizer, None, ids, logits, non_rt, {})
    assert kept_ids  # some normal tokens survive (not the empty-fallback path)
    original = dict(zip(ids, logits))
    for tid, lg in zip(kept_ids, kept_logits):
        assert original[tid] == lg


def test_filter_falls_back_when_all_removed(codec: StegoCodec):
    _model, tokenizer, _device = codec._ensure_model()
    non_rt = get_non_roundtrip_tokens(tokenizer)
    if not non_rt:
        return
    bad = list(non_rt)[:3]
    ids = bad
    logits = [3.0, 2.0, 1.0][: len(bad)]
    kept_ids, kept_logits = _filter_tokens(tokenizer, None, ids, logits, non_rt, {})
    assert kept_ids == ids  # fallback to unfiltered rather than empty
    assert kept_logits == logits  # fallback returns the original logits unchanged
