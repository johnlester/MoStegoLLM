"""StegoCodec — the single public entry point for steganographic encoding/decoding."""

from __future__ import annotations

import pathlib
from typing import Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .crypto import decrypt as _decrypt, encrypt as _encrypt
from .decoder import decode as _decode
from .encoder import TOP_K, encode as _encode
from .model import DEFAULT_PROMPT, PRIMARY_MODEL, ModelInfo, get_model_info, list_models, load_model
from .utils import (
    HEADER_BITS,
    StegoDecodeError,
    StegoEncodeError,
    StegoModelError,
    StegoStats,
)


DEFAULT_CHUNK_SIZE = 2000  # bytes of plaintext per chunk
DEFAULT_CONTEXT_SIZE = 500  # characters of prior cover text used as prompt for next chunk


class StegoCodec:
    """Steganographic codec that hides binary data in LLM-generated text.

    Uses arithmetic coding over a language model's token probability
    distributions to encode arbitrary bytes into natural-looking English
    prose and decode them back.

    Example::

        codec = StegoCodec()
        cover = codec.encode(b"secret")
        assert codec.decode(cover) == b"secret"

    Args:
        model_name: HuggingFace model identifier. Defaults to
            ``HuggingFaceTB/SmolLM-135M``; falls back to TinyLlama if the
            primary model is inaccessible.
        device: ``'auto'`` (default), ``'cpu'``, ``'cuda'``, etc.
        prompt: Seed text that prefixes every generation.  Encoder and
            decoder **must** use the same prompt.
        top_k: Number of most-probable tokens considered at each step.
        temperature: Softmax temperature (``1.0`` = unmodified).
        sentence_boundary: If ``True``, continue generating tokens past the
            data-recoverable point until the cover text ends at a sentence
            boundary (``.``, ``!``, or ``?``).
        token: HuggingFace API token for gated models.  When ``None``,
            falls back to the ``HF_TOKEN`` environment variable / ``.env``.
        password: Optional password for AES-256-GCM encryption.  When set,
            data is encrypted before encoding and decrypted after decoding,
            adding a confidentiality layer on top of steganographic hiding.
    """

    def __init__(
        self,
        model_name: str = PRIMARY_MODEL,
        device: str = "auto",
        prompt: str = DEFAULT_PROMPT,
        top_k: int = TOP_K,
        temperature: float = 1.0,
        sentence_boundary: bool = False,
        token: str | None = None,
        password: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device_str = device
        self._prompt = prompt
        self._top_k = top_k
        self._temperature = temperature
        self._sentence_boundary = sentence_boundary
        self._token = token
        self._password = password

        # Lazy-loaded on first use
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._device: torch.device | None = None

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    @classmethod
    def list_models(cls) -> tuple[ModelInfo, ...]:
        """Return metadata for all recommended models."""
        return list_models()

    @classmethod
    def get_model_info(cls, model_name: str) -> ModelInfo | None:
        """Look up a model by name. Returns ``None`` if not in the registry."""
        return get_model_info(model_name)

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device]:
        """Load the model/tokenizer on first use and return them."""
        if self._model is None:
            self._model, self._tokenizer, self._device = load_model(
                self._model_name, self._device_str, token=self._token
            )
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._device is not None
        return self._model, self._tokenizer, self._device

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(self, data: bytes) -> str:
        """Encode binary data into steganographic cover text.

        Args:
            data: Arbitrary bytes to hide.

        Returns:
            A string of natural-looking English text that encodes *data*.

        Raises:
            StegoEncodeError: If encoding fails.
            StegoModelError: If the model cannot be loaded.
        """
        if self._password is not None:
            data = _encrypt(data, self._password)
        model, tokenizer, device = self._ensure_model()
        cover_text, _token_ids, _total_bits = _encode(
            data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=self._prompt,
            top_k=self._top_k,
            temperature=self._temperature,
            sentence_boundary=self._sentence_boundary,
        )
        return cover_text

    def decode(self, cover_text: str) -> bytes:
        """Decode steganographic cover text back to the original bytes.

        Args:
            cover_text: Text previously produced by :meth:`encode`.

        Returns:
            The original binary payload.

        Raises:
            StegoDecodeError: If decoding fails (wrong prompt, corrupted text, …).
            StegoCryptoError: If decryption fails (wrong password, tampered data).
            StegoModelError: If the model cannot be loaded.
        """
        model, tokenizer, device = self._ensure_model()
        payload = _decode(
            cover_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=self._prompt,
            top_k=self._top_k,
            temperature=self._temperature,
        )
        if self._password is not None:
            payload = _decrypt(payload, self._password)
        return payload

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def encode_str(self, text: str, encoding: str = "utf-8") -> str:
        """Encode a string into steganographic cover text.

        Args:
            text: The string to encode.
            encoding: Character encoding (default ``utf-8``).

        Returns:
            Steganographic cover text.
        """
        return self.encode(text.encode(encoding))

    def decode_str(self, cover_text: str, encoding: str = "utf-8") -> str:
        """Decode steganographic cover text back to a string.

        Args:
            cover_text: Text previously produced by :meth:`encode_str`.
            encoding: Character encoding (default ``utf-8``).

        Returns:
            The original string.
        """
        return self.decode(cover_text).decode(encoding)

    def encode_file(self, path: Union[str, pathlib.Path]) -> str:
        """Read a file and encode its contents.

        Args:
            path: Path to the input file.

        Returns:
            Steganographic cover text encoding the file's bytes.
        """
        data = pathlib.Path(path).read_bytes()
        return self.encode(data)

    def decode_file(
        self, cover_text: str, output_path: Union[str, pathlib.Path]
    ) -> None:
        """Decode cover text and write the result to a file.

        Args:
            cover_text: Text previously produced by :meth:`encode` or
                :meth:`encode_file`.
            output_path: Where to write the recovered bytes.
        """
        data = self.decode(cover_text)
        pathlib.Path(output_path).write_bytes(data)

    # ------------------------------------------------------------------
    # Chunked (long data) API
    # ------------------------------------------------------------------

    def encode_long(
        self,
        data: bytes,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        context_size: int = DEFAULT_CONTEXT_SIZE,
    ) -> list[str]:
        """Encode arbitrarily large data by splitting it into independently encoded chunks.

        Each chunk is encoded as a standalone steganographic message.  When
        ``password`` is set, each chunk is encrypted independently (separate
        salt/nonce).  Consecutive chunks use the tail of the previous chunk's
        cover text as the prompt so that the generated prose reads coherently.

        Args:
            data: Arbitrary bytes to hide.
            chunk_size: Maximum plaintext bytes per chunk (before encryption).
            context_size: Number of trailing characters from the previous
                chunk's cover text to use as the prompt for the next chunk.

        Returns:
            A list of cover-text strings, one per chunk.
        """
        model, tokenizer, device = self._ensure_model()

        # Split plaintext into chunks, then optionally encrypt each one.
        chunks: list[bytes] = [
            data[i : i + chunk_size] for i in range(0, len(data), chunk_size)
        ]
        if self._password is not None:
            chunks = [_encrypt(chunk, self._password) for chunk in chunks]

        cover_texts: list[str] = []
        for idx, chunk in enumerate(chunks):
            prompt = (
                self._prompt
                if idx == 0
                else cover_texts[idx - 1][-context_size:]
            )
            cover_text, _token_ids, _total_bits = _encode(
                chunk,
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                top_k=self._top_k,
                temperature=self._temperature,
                sentence_boundary=self._sentence_boundary,
            )
            cover_texts.append(cover_text)

        return cover_texts

    def decode_long(
        self,
        cover_texts: list[str],
        context_size: int = DEFAULT_CONTEXT_SIZE,
    ) -> bytes:
        """Decode a list of cover texts produced by :meth:`encode_long`.

        Args:
            cover_texts: Cover-text strings in the same order produced by
                :meth:`encode_long`.
            context_size: Must match the value used during encoding.

        Returns:
            The original binary payload.
        """
        model, tokenizer, device = self._ensure_model()

        payloads: list[bytes] = []
        for idx, cover_text in enumerate(cover_texts):
            prompt = (
                self._prompt
                if idx == 0
                else cover_texts[idx - 1][-context_size:]
            )
            payload = _decode(
                cover_text,
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                top_k=self._top_k,
                temperature=self._temperature,
            )
            if self._password is not None:
                payload = _decrypt(payload, self._password)
            payloads.append(payload)

        return b"".join(payloads)

    def encode_long_str(
        self,
        text: str,
        encoding: str = "utf-8",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        context_size: int = DEFAULT_CONTEXT_SIZE,
    ) -> list[str]:
        """Encode a string using chunked encoding.

        Args:
            text: The string to encode.
            encoding: Character encoding (default ``utf-8``).
            chunk_size: Maximum plaintext bytes per chunk.
            context_size: Trailing characters used as prompt for next chunk.

        Returns:
            A list of cover-text strings, one per chunk.
        """
        return self.encode_long(
            text.encode(encoding), chunk_size=chunk_size, context_size=context_size
        )

    def decode_long_str(
        self,
        cover_texts: list[str],
        encoding: str = "utf-8",
        context_size: int = DEFAULT_CONTEXT_SIZE,
    ) -> str:
        """Decode chunked cover texts back to a string.

        Args:
            cover_texts: Cover-text strings produced by :meth:`encode_long_str`.
            encoding: Character encoding (default ``utf-8``).
            context_size: Must match the value used during encoding.

        Returns:
            The original string.
        """
        return self.decode_long(cover_texts, context_size=context_size).decode(encoding)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def encode_with_stats(self, data: bytes) -> StegoStats:
        """Encode data and return diagnostics alongside the cover text.

        Args:
            data: Arbitrary bytes to hide.

        Returns:
            A :class:`~mostegollm.utils.StegoStats` with the cover text and
            encoding statistics.
        """
        original_size = len(data)
        if self._password is not None:
            data = _encrypt(data, self._password)
        model, tokenizer, device = self._ensure_model()
        cover_text, token_ids, total_bits = _encode(
            data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=self._prompt,
            top_k=self._top_k,
            temperature=self._temperature,
            sentence_boundary=self._sentence_boundary,
        )
        total_tokens = len(token_ids)
        payload_bits = total_bits - HEADER_BITS
        bits_per_token = total_bits / total_tokens if total_tokens > 0 else 0.0

        return StegoStats(
            cover_text=cover_text,
            bits_per_token=bits_per_token,
            total_tokens=total_tokens,
            payload_size_bytes=original_size,
        )
