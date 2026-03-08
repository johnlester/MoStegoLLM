"""Tests for the BPE merge filter that prevents tokenizer round-trip mismatches."""

from __future__ import annotations

import pytest

from mostegollm import StegoCodec
from mostegollm.encoder import (
    WHOLE,
    _filter_distribution,
    get_non_roundtrip_tokens,
)


class TestNonRoundtripTokens:
    """Tests for get_non_roundtrip_tokens."""

    def test_returns_frozenset(self, codec: StegoCodec) -> None:
        _, tokenizer, _ = codec._ensure_model()
        result = get_non_roundtrip_tokens(tokenizer)
        assert isinstance(result, frozenset)

    def test_nonempty(self, codec: StegoCodec) -> None:
        """SmolLM should have some non-round-tripping tokens (byte fallbacks)."""
        _, tokenizer, _ = codec._ensure_model()
        result = get_non_roundtrip_tokens(tokenizer)
        assert len(result) > 0

    def test_cached(self, codec: StegoCodec) -> None:
        """Repeated calls return the same object (cached)."""
        _, tokenizer, _ = codec._ensure_model()
        a = get_non_roundtrip_tokens(tokenizer)
        b = get_non_roundtrip_tokens(tokenizer)
        assert a is b

    def test_tokens_actually_fail_roundtrip(self, codec: StegoCodec) -> None:
        """Every token in the set should indeed fail to round-trip."""
        _, tokenizer, _ = codec._ensure_model()
        bad_tokens = get_non_roundtrip_tokens(tokenizer)
        # Spot-check a sample
        for tid in list(bad_tokens)[:20]:
            text = tokenizer.decode([tid])
            re_encoded = tokenizer.encode(text, add_special_tokens=False)
            assert re_encoded != [tid], f"Token {tid} round-trips but is marked bad"


class TestFilterDistribution:
    """Tests for _filter_distribution."""

    def test_removes_non_rt_tokens(self) -> None:
        """Non-round-tripping tokens should be removed."""
        token_ids = [10, 20, 30]
        cum_probs = [0, WHOLE // 3, 2 * WHOLE // 3, WHOLE]
        non_rt = frozenset({20})
        merge_cache: dict[tuple[int, int], bool] = {}

        # Dummy tokenizer not needed since prev_token_id is None and
        # non_rt check happens before merge check.
        class FakeTokenizer:
            pass

        filtered_ids, filtered_cum = _filter_distribution(
            FakeTokenizer(),  # type: ignore[arg-type]
            None, token_ids, cum_probs, non_rt, merge_cache,
        )
        assert 20 not in filtered_ids
        assert 10 in filtered_ids
        assert 30 in filtered_ids
        assert filtered_cum[0] == 0
        assert filtered_cum[-1] == WHOLE

    def test_removes_mergeable_tokens(self, codec: StegoCodec) -> None:
        """Tokens that merge with the previous token should be removed."""
        _, tokenizer, _ = codec._ensure_model()
        non_rt = get_non_roundtrip_tokens(tokenizer)

        # newline token (198 in SmolLM) merges with itself
        newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
        # Build a distribution containing the newline token
        token_ids = [newline_id, 100, 200]
        cum_probs = [0, WHOLE // 3, 2 * WHOLE // 3, WHOLE]
        merge_cache: dict[tuple[int, int], bool] = {}

        filtered_ids, filtered_cum = _filter_distribution(
            tokenizer, newline_id, token_ids, cum_probs, non_rt, merge_cache,
        )
        # The newline-after-newline should be filtered out (it merges)
        assert newline_id not in filtered_ids
        assert filtered_cum[0] == 0
        assert filtered_cum[-1] == WHOLE

    def test_preserves_valid_tokens(self, codec: StegoCodec) -> None:
        """Tokens that don't merge and do round-trip should be preserved."""
        _, tokenizer, _ = codec._ensure_model()
        non_rt = get_non_roundtrip_tokens(tokenizer)

        # Use common tokens like "the" and "a" which shouldn't merge
        the_id = tokenizer.encode(" the", add_special_tokens=False)[-1]
        a_id = tokenizer.encode(" a", add_special_tokens=False)[-1]
        token_ids = [the_id, a_id]
        cum_probs = [0, WHOLE // 2, WHOLE]
        merge_cache: dict[tuple[int, int], bool] = {}

        filtered_ids, filtered_cum = _filter_distribution(
            tokenizer, the_id, token_ids, cum_probs, non_rt, merge_cache,
        )
        # At least one token should survive
        assert len(filtered_ids) > 0
        assert filtered_cum[-1] == WHOLE

    def test_empty_filter_falls_back(self) -> None:
        """If all tokens would be filtered, return original distribution."""
        token_ids = [10, 20]
        cum_probs = [0, WHOLE // 2, WHOLE]
        non_rt = frozenset({10, 20})  # all tokens are bad
        merge_cache: dict[tuple[int, int], bool] = {}

        class FakeTokenizer:
            pass

        filtered_ids, filtered_cum = _filter_distribution(
            FakeTokenizer(),  # type: ignore[arg-type]
            None, token_ids, cum_probs, non_rt, merge_cache,
        )
        # Fallback: return original
        assert filtered_ids == token_ids
        assert filtered_cum == cum_probs


class TestBPEFilterRoundTrip:
    """End-to-end tests verifying the BPE filter prevents decode failures."""

    @pytest.mark.parametrize("payload", [
        b"\x00" * 10,
        b"\xff" * 10,
        b"newlines\n\n\ntest",
        b"dots...and,,commas",
        bytes(range(256)),
        b"A" * 50,
    ])
    def test_various_payloads_roundtrip(
        self, codec: StegoCodec, payload: bytes
    ) -> None:
        """Various payloads that previously triggered BPE issues should round-trip."""
        cover = codec.encode(payload)
        recovered = codec.decode(cover)
        assert recovered == payload

    def test_random_bytes_roundtrip(self, codec: StegoCodec) -> None:
        """Random binary data should round-trip reliably."""
        import hashlib
        for i in range(3):
            payload = hashlib.sha256(f"test-{i}".encode()).digest()[:16]
            cover = codec.encode(payload)
            recovered = codec.decode(cover)
            assert recovered == payload, f"Failed on random payload {i}"

    def test_seed_phrase_varies_with_input(self, codec: StegoCodec) -> None:
        """Different inputs should produce different seed phrases (usually)."""
        covers = set()
        for i in range(10):
            cover = codec.encode(f"message-{i}".encode())
            # Extract first 20 chars as a proxy for the seed phrase
            covers.add(cover[:20])
        # With 256 seeds and 10 inputs, we should get at least 2 different openings
        assert len(covers) >= 2
