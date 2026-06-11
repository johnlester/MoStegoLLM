"""top_k must be in the supported [2, K] range.

Below 2, a step has one candidate that owns all of [0, WHOLE) — zero entropy, no
bits encoded (the old behaviour ground to MAX_TOKENS then raised). Above K, the
candidate count exceeds the fixed CUM rank-interval schedule and step_coding
indexes past its end (IndexError). Both must fail fast with a clear Stego error.
"""

from __future__ import annotations

import pytest

from mostegollm import StegoCodec
from mostegollm.coding import K
from mostegollm.utils import StegoDecodeError, StegoEncodeError


def _codec_with_top_k(codec: StegoCodec, k: int) -> StegoCodec:
    """A codec with custom top_k, sharing the session model (no reload)."""
    c = StegoCodec(device="cpu", top_k=k)
    model, tok, dev = codec._ensure_model()
    c._model, c._tokenizer, c._device = model, tok, dev
    return c


@pytest.mark.parametrize("bad", [1, 0, -5])
def test_encode_rejects_top_k_below_2(codec, bad):
    c = _codec_with_top_k(codec, bad)
    with pytest.raises(StegoEncodeError, match="top_k"):
        c.encode(b"hi")


@pytest.mark.parametrize("bad", [K + 1, 512])
def test_encode_rejects_top_k_above_K(codec, bad):
    c = _codec_with_top_k(codec, bad)
    with pytest.raises(StegoEncodeError, match="top_k"):
        c.encode(b"hi")


def test_decode_rejects_bad_top_k(codec):
    # Produce a valid cover text at the default top_k, then try to decode it with
    # an out-of-range top_k: should raise a clear StegoDecodeError, not IndexError.
    cover = codec.encode(b"hi")
    c = _codec_with_top_k(codec, K + 1)
    with pytest.raises(StegoDecodeError, match="top_k"):
        c.decode(cover)


def test_valid_top_k_still_roundtrips(codec):
    c = _codec_with_top_k(codec, 64)
    assert c.decode(c.encode(b"hi")) == b"hi"
