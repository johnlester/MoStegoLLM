"""MoStegoLLM — Steganographic text encoding/decoding using LLM token distributions.

Encode arbitrary binary data into natural-looking English sentences and decode
those sentences back to the original data, using a HuggingFace language model as
the shared "key" between encoder and decoder.

Example::

    from mostegollm import StegoCodec

    codec = StegoCodec()
    cover = codec.encode(b"secret message")
    assert codec.decode(cover) == b"secret message"
"""

from .codec import StegoCodec
from .model import ModelInfo, get_model_info, list_models
from .utils import (
    StegoDecodeError,
    StegoEncodeError,
    StegoError,
    StegoModelError,
    StegoStats,
)

__all__ = [
    "ModelInfo",
    "StegoCodec",
    "StegoDecodeError",
    "StegoEncodeError",
    "StegoError",
    "StegoModelError",
    "StegoStats",
    "get_model_info",
    "list_models",
]
