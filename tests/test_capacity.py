"""Test encoding capacity limits."""

from __future__ import annotations

from mostegollm import StegoCodec


class TestCapacityLimits:
    """Verify behaviour near the MAX_TOKENS limit."""

    def test_medium_payload_succeeds(self, codec: StegoCodec) -> None:
        """A 500-byte payload should encode successfully."""
        data = bytes(range(256)) + bytes(range(244))  # 500 bytes
        cover = codec.encode(data)
        assert isinstance(cover, str)
        recovered = codec.decode(cover)
        assert recovered == data

    def test_max_tokens_error_message(self) -> None:
        """Exceeding MAX_TOKENS should raise StegoEncodeError with clear message."""
        from mostegollm.encoder import MAX_TOKENS

        assert MAX_TOKENS == 8192
