"""Utility functions: exceptions, header packing, bit manipulation helpers."""

from __future__ import annotations

import struct
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Library-specific exceptions
# ---------------------------------------------------------------------------


class StegoError(Exception):
    """Base exception for MoStegoLLM."""


class StegoEncodeError(StegoError):
    """Raised when encoding fails."""


class StegoDecodeError(StegoError):
    """Raised when decoding fails."""


class StegoModelError(StegoError):
    """Raised when model loading or inference fails."""


# ---------------------------------------------------------------------------
# Header format
# ---------------------------------------------------------------------------
# 2-byte magic  |  4-byte payload length (big-endian uint32)
# 0x53 0x54     |  <length>
# Total: 6 bytes = 48 bits
MAGIC = b"\x53\x54"
HEADER_FORMAT = ">2sI"  # 2-byte magic + 4-byte unsigned int
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 6 bytes
HEADER_BITS = HEADER_SIZE * 8  # 48 bits


def pack_header(payload_length: int) -> bytes:
    """Pack a header containing magic bytes and the payload length.

    Args:
        payload_length: Length of the original payload in bytes.

    Returns:
        Packed header bytes.

    Raises:
        StegoEncodeError: If payload_length is negative or too large.
    """
    if payload_length < 0:
        raise StegoEncodeError("Payload length cannot be negative")
    if payload_length > 0xFFFFFFFF:
        raise StegoEncodeError("Payload too large (max 4 GiB)")
    return struct.pack(HEADER_FORMAT, MAGIC, payload_length)


def unpack_header(header_bytes: bytes) -> int:
    """Unpack a header and return the payload length.

    Args:
        header_bytes: Raw header bytes (must be exactly HEADER_SIZE bytes).

    Returns:
        The original payload length in bytes.

    Raises:
        StegoDecodeError: If the header is malformed or the magic bytes don't match.
    """
    if len(header_bytes) != HEADER_SIZE:
        raise StegoDecodeError(
            f"Header must be exactly {HEADER_SIZE} bytes, got {len(header_bytes)}"
        )
    magic, length = struct.unpack(HEADER_FORMAT, header_bytes)
    if magic != MAGIC:
        raise StegoDecodeError(
            f"Invalid magic bytes: expected {MAGIC!r}, got {magic!r}. "
            "The text may not have been encoded with MoStegoLLM, "
            "or a different prompt was used."
        )
    return length


# ---------------------------------------------------------------------------
# Bit-stream helpers
# ---------------------------------------------------------------------------


def bytes_to_bits(data: bytes) -> list[int]:
    """Convert bytes to a list of bits (MSB first per byte).

    Args:
        data: Input bytes.

    Returns:
        List of 0/1 integers.
    """
    bits: list[int] = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    """Convert a list of bits back to bytes (MSB first per byte).

    Pads with zeros on the right if len(bits) is not a multiple of 8.

    Args:
        bits: List of 0/1 integers.

    Returns:
        Reconstructed bytes.
    """
    # Pad to multiple of 8
    padded = bits + [0] * ((8 - len(bits) % 8) % 8)
    result = bytearray()
    for i in range(0, len(padded), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | padded[i + j]
        result.append(byte)
    return bytes(result)


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StegoStats:
    """Diagnostics returned by ``StegoCodec.encode_with_stats``.

    Attributes:
        cover_text: The generated steganographic cover text.
        bits_per_token: Average number of payload bits encoded per token.
        total_tokens: Total number of tokens generated (excluding the prompt).
        payload_size_bytes: Size of the original payload in bytes.
    """

    cover_text: str
    bits_per_token: float
    total_tokens: int
    payload_size_bytes: int
