"""Special tokens must never be selectable cover-text tokens.

Regression test for the round-trip bug where the encoder picked a special token
(e.g. BOS/EOS) as content. Special tokens pass the *local* single-/pairwise
round-trip checks, but when the cover text is decoded to a string and re-encoded
by the decoder, the tokenizer's word-boundary normalization around the special
token shifts the whole sequence — eventually selecting a token outside the
reproducible top-k and breaking decode. Excluding special tokens from the
candidate distribution on both sides is the fix; both encode and decode source
their exclusion set from ``get_non_roundtrip_tokens``.
"""

from __future__ import annotations

from mostegollm.encoder import _filter_tokens, get_non_roundtrip_tokens


def test_all_special_ids_are_excluded(codec):
    """Every tokenizer special id is in the non-round-trip (excluded) set."""
    _, tok, _ = codec._ensure_model()
    non_rt = get_non_roundtrip_tokens(tok)
    special_ids = [sid for sid in tok.all_special_ids if sid is not None]
    assert special_ids, "tokenizer reports no special ids — test is vacuous"
    missing = [sid for sid in special_ids if sid not in non_rt]
    assert not missing, f"special ids not excluded from candidates: {missing}"


def test_filter_drops_special_token_candidate(codec):
    """A special token appearing in the top-k is filtered out of the candidates."""
    _, tok, _ = codec._ensure_model()
    non_rt = get_non_roundtrip_tokens(tok)
    eos = tok.eos_token_id
    assert eos is not None
    ids = [eos, 100, 101, 102]
    logits = [0.0, -1.0, -2.0, -3.0]
    keep_ids, keep_logits = _filter_tokens(tok, None, ids, logits, non_rt, {})
    assert eos not in keep_ids
    assert len(keep_ids) == len(keep_logits)
