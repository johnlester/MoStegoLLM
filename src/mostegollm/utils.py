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


class StegoCryptoError(StegoError):
    """Raised when encryption or decryption fails (wrong password, tampered data)."""


# ---------------------------------------------------------------------------
# Header format
# ---------------------------------------------------------------------------
# 2-byte magic  |  4-byte payload length (big-endian uint32)  |  4-byte CRC-32
# 0x53 0x54     |  <length>                                   |  <crc32>
# Total: 10 bytes = 80 bits
MAGIC = b"\x53\x54"
HEADER_FORMAT = ">2sII"  # 2-byte magic + 4-byte uint32 length + 4-byte uint32 CRC
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 10 bytes
HEADER_BITS = HEADER_SIZE * 8  # 80 bits


def pack_header(payload_length: int, crc32: int = 0) -> bytes:
    """Pack a header containing magic bytes, payload length, and CRC-32.

    Args:
        payload_length: Length of the original payload in bytes.
        crc32: CRC-32 checksum of the payload.

    Returns:
        Packed header bytes.

    Raises:
        StegoEncodeError: If payload_length is negative or too large.
    """
    if payload_length < 0:
        raise StegoEncodeError("Payload length cannot be negative")
    if payload_length > 0xFFFFFFFF:
        raise StegoEncodeError("Payload too large (max 4 GiB)")
    return struct.pack(HEADER_FORMAT, MAGIC, payload_length, crc32 & 0xFFFFFFFF)


def unpack_header(header_bytes: bytes) -> tuple[int, int]:
    """Unpack a header and return the payload length and CRC-32.

    Args:
        header_bytes: Raw header bytes (must be exactly HEADER_SIZE bytes).

    Returns:
        A tuple of (payload_length, crc32).

    Raises:
        StegoDecodeError: If the header is malformed or the magic bytes don't match.
    """
    if len(header_bytes) != HEADER_SIZE:
        raise StegoDecodeError(
            f"Header must be exactly {HEADER_SIZE} bytes, got {len(header_bytes)}"
        )
    magic, length, crc = struct.unpack(HEADER_FORMAT, header_bytes)
    if magic != MAGIC:
        raise StegoDecodeError(
            f"Invalid magic bytes: expected {MAGIC!r}, got {magic!r}. "
            "The text may not have been encoded with MoStegoLLM, "
            "or a different prompt was used."
        )
    return length, crc


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
