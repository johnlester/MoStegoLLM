"""Property-based fuzz tests for the arithmetic coder round-trip."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from mostegollm import StegoCodec

# Module-level codec shared across all hypothesis examples.
# Cannot use session-scoped fixture with @given, so we lazy-init here.
_codec: StegoCodec | None = None


def _get_codec() -> StegoCodec:
    global _codec
    if _codec is None:
        _codec = StegoCodec(device="cpu")
        _codec._ensure_model()
    return _codec


@settings(max_examples=20, deadline=None)
@given(data=st.binary(min_size=0, max_size=100))
def test_roundtrip_arbitrary_bytes(data: bytes) -> None:
    """Any byte sequence should survive an encode -> decode round-trip."""
    codec = _get_codec()
    cover = codec.encode(data)
    assert isinstance(cover, str)
    recovered = codec.decode(cover)
    assert recovered == data
