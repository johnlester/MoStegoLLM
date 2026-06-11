"""Tests for cover-story prompt modes (auto-topic and custom prompt)."""

from __future__ import annotations

import pytest

from mostegollm import StegoCodec, StegoDecodeError
from mostegollm.seeds import TOPICS


def _share_model(codec: StegoCodec, **kwargs) -> StegoCodec:
    """Build a StegoCodec reusing an already-loaded model (avoids reloading)."""
    c = StegoCodec(device="cpu", **kwargs)
    model, tokenizer, device = codec._ensure_model()
    c._model, c._tokenizer, c._device = model, tokenizer, device
    return c


class TestModeA:
    """Auto-topic mode: opener from the codebook, recovered on decode."""

    def test_topic_roundtrip(self, codec_topic: StegoCodec) -> None:
        data = b"meet me at noon"
        cover = codec_topic.encode(data)
        assert isinstance(cover, str)
        assert codec_topic.decode(cover) == data

    @pytest.mark.parametrize("topic", list(TOPICS))
    def test_every_topic_roundtrips(self, codec: StegoCodec, topic: str) -> None:
        c = _share_model(codec, topic=topic)
        data = b"payload"
        cover = c.encode(data)
        assert isinstance(cover, str)
        assert cover.startswith(tuple(TOPICS[topic]))
        assert c.decode(cover) == data

    def test_default_mode_still_roundtrips(self, codec: StegoCodec) -> None:
        data = b"no topic given"
        cover = codec.encode(data)
        assert codec.decode(cover) == data


class TestModeC:
    """Custom-prompt mode: prompt is prepended and stripped by known length."""

    def test_custom_prompt_roundtrip(self, codec_alt_prompt: StegoCodec) -> None:
        data = b"hidden payload"
        cover = codec_alt_prompt.encode(data)
        assert isinstance(cover, str)
        assert cover.startswith("Once upon a time in a faraway land,")
        assert codec_alt_prompt.decode(cover) == data

    def test_decode_rejects_mismatched_prefix(self, codec_alt_prompt: StegoCodec) -> None:
        with pytest.raises(StegoDecodeError):
            codec_alt_prompt.decode("Some text that does not start with the prompt.")


class TestModeConflict:
    """topic and prompt are mutually exclusive."""

    def test_topic_plus_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="topic"):
            StegoCodec(device="cpu", topic="cooking", prompt="A custom prompt,")

    def test_unknown_topic_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown topic"):
            StegoCodec(device="cpu", topic="not-a-topic")
