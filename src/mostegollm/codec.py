"""StegoCodec — the single public entry point for steganographic encoding/decoding."""

from __future__ import annotations

import pathlib
from typing import Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .decoder import decode as _decode
from .encoder import TOP_K, encode as _encode
from .model import DEFAULT_PROMPT, PRIMARY_MODEL, load_model
from .utils import (
    HEADER_BITS,
    StegoDecodeError,
    StegoEncodeError,
    StegoModelError,
    StegoStats,
)


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
            ``meta-llama/Llama-3.2-1B``; falls back to TinyLlama if the
            gated model is inaccessible.
        device: ``'auto'`` (default), ``'cpu'``, ``'cuda'``, etc.
        prompt: Seed text that prefixes every generation.  Encoder and
            decoder **must** use the same prompt.
        top_k: Number of most-probable tokens considered at each step.
        temperature: Softmax temperature (``1.0`` = unmodified).
    """

    def __init__(
        self,
        model_name: str = PRIMARY_MODEL,
        device: str = "auto",
        prompt: str = DEFAULT_PROMPT,
        top_k: int = TOP_K,
        temperature: float = 1.0,
    ) -> None:
        self._model_name = model_name
        self._device_str = device
        self._prompt = prompt
        self._top_k = top_k
        self._temperature = temperature

        # Lazy-loaded on first use
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._device: torch.device | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device]:
        """Load the model/tokenizer on first use and return them."""
        if self._model is None:
            self._model, self._tokenizer, self._device = load_model(
                self._model_name, self._device_str
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
        model, tokenizer, device = self._ensure_model()
        cover_text, _token_ids, _total_bits = _encode(
            data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=self._prompt,
            top_k=self._top_k,
            temperature=self._temperature,
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
            StegoModelError: If the model cannot be loaded.
        """
        model, tokenizer, device = self._ensure_model()
        return _decode(
            cover_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=self._prompt,
            top_k=self._top_k,
            temperature=self._temperature,
        )

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
        model, tokenizer, device = self._ensure_model()
        cover_text, token_ids, total_bits = _encode(
            data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=self._prompt,
            top_k=self._top_k,
            temperature=self._temperature,
        )
        total_tokens = len(token_ids)
        payload_bits = total_bits - HEADER_BITS
        bits_per_token = total_bits / total_tokens if total_tokens > 0 else 0.0

        return StegoStats(
            cover_text=cover_text,
            bits_per_token=bits_per_token,
            total_tokens=total_tokens,
            payload_size_bytes=len(data),
        )
