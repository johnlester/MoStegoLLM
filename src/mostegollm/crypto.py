"""Encryption layer for steganographic payloads (AES-256-GCM + PBKDF2)."""

from __future__ import annotations

import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from .utils import StegoCryptoError

_SALT_LEN = 16
_NONCE_LEN = 12
_KEY_LEN = 32  # AES-256
_KDF_ITERATIONS = 600_000


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit key from *password* and *salt* using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_KEY_LEN,
        salt=salt,
        iterations=_KDF_ITERATIONS,
    )
    return kdf.derive(password.encode("utf-8"))


def encrypt(data: bytes, password: str) -> bytes:
    """Encrypt *data* with AES-256-GCM using a key derived from *password*.

    Returns:
        ``salt (16) || nonce (12) || ciphertext || tag (16)``
    """
    salt = os.urandom(_SALT_LEN)
    nonce = os.urandom(_NONCE_LEN)
    key = _derive_key(password, salt)
    ciphertext = AESGCM(key).encrypt(nonce, data, None)
    return salt + nonce + ciphertext


def decrypt(blob: bytes, password: str) -> bytes:
    """Decrypt a blob produced by :func:`encrypt`.

    Raises:
        StegoCryptoError: On wrong password, tampered data, or truncated blob.
    """
    min_len = _SALT_LEN + _NONCE_LEN + 16  # at least tag
    if len(blob) < min_len:
        raise StegoCryptoError(
            f"Encrypted blob too short ({len(blob)} bytes, minimum {min_len})"
        )
    salt = blob[:_SALT_LEN]
    nonce = blob[_SALT_LEN : _SALT_LEN + _NONCE_LEN]
    ciphertext = blob[_SALT_LEN + _NONCE_LEN :]
    key = _derive_key(password, salt)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, None)
    except Exception as exc:
        raise StegoCryptoError("Decryption failed (wrong password or tampered data)") from exc
