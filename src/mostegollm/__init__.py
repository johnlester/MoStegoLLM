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
from .utils import (
    StegoCryptoError,
    StegoDecodeError,
    StegoEncodeError,
    StegoError,
    StegoModelError,
    StegoStats,
)

__all__ = [
    "StegoCodec",
    "StegoCryptoError",
    "StegoDecodeError",
    "StegoEncodeError",
    "StegoError",
    "StegoModelError",
    "StegoStats",
]
