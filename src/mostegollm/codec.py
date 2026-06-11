"""StegoCodec — the single public entry point for steganographic encoding/decoding."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .crypto import decrypt as _decrypt, encrypt as _encrypt
from .decoder import decode as _decode
from .encoder import TOP_K, encode as _encode
from .model import DEFAULT_PROMPT, PRIMARY_MODEL, ModelInfo, get_model_info, list_models, load_model
from .seeds import list_topics, match_seed, select_seed
from .utils import (
    StegoDecodeError,
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
        topic: Cover-story topic for the auto opener (Mode A).  When set, the
            prepended opener is chosen from that topic's phrases so the cover
            text reads as if it's about that subject.  Mutually exclusive with
            ``prompt``.  Use :meth:`list_topics` for valid names.
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
        topic: str | None = None,
        top_k: int = TOP_K,
        temperature: float = 1.0,
        sentence_boundary: bool = False,
        token: str | None = None,
        password: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device_str = device
        self._prompt = prompt
        if topic is not None:
            if prompt:
                raise ValueError(
                    "topic and prompt are mutually exclusive: a custom prompt is "
                    "its own opener, so a topic would be ignored."
                )
            valid = list_topics()
            if topic not in valid:
                raise ValueError(f"Unknown topic {topic!r}. Valid topics: {', '.join(valid)}")
        self._topic = topic
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

    def encode(
        self,
        data: bytes,
        *,
        chunk_size: int | None = None,
        context_size: int = DEFAULT_CONTEXT_SIZE,
    ) -> str | list[str]:
        """Encode binary data into steganographic cover text.

        Args:
            data: Arbitrary bytes to hide.
            chunk_size: If set, split *data* into chunks of this many bytes
                and return a list of cover texts (one per chunk).  When
                ``None`` (default), encode as a single piece and return a
                plain string.
            context_size: When chunking, number of trailing characters from
                the previous chunk's cover text to use as the prompt for the
                next chunk.

        Returns:
            A string of natural-looking English text that encodes *data*
            when *chunk_size* is ``None``, or a list of such strings when
            chunking is enabled.

        Raises:
            StegoEncodeError: If encoding fails.
            StegoModelError: If the model cannot be loaded.
        """
        if chunk_size is not None:
            return self._encode_chunked(data, chunk_size=chunk_size, context_size=context_size)

        opener = self._prompt if self._prompt else select_seed(data, self._topic)

        if self._password is not None:
            data = _encrypt(data, self._password)
        model, tokenizer, device = self._ensure_model()
        cover_text, _token_ids, _total_bits = _encode(
            data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=opener,
            top_k=self._top_k,
            temperature=self._temperature,
            sentence_boundary=self._sentence_boundary,
        )
        return opener + cover_text

    def decode(
        self,
        cover_text: str | list[str],
        *,
        context_size: int = DEFAULT_CONTEXT_SIZE,
    ) -> bytes:
        """Decode steganographic cover text back to the original bytes.

        Args:
            cover_text: Text previously produced by :meth:`encode`.  May be
                a single string or a list of strings (from chunked encoding).
            context_size: When decoding a list, must match the value used
                during encoding.

        Returns:
            The original binary payload.

        Raises:
            StegoDecodeError: If decoding fails (wrong prompt, corrupted text, ...).
            StegoCryptoError: If decryption fails (wrong password, tampered data).
            StegoModelError: If the model cannot be loaded.
        """
        if isinstance(cover_text, list):
            return self._decode_chunked(cover_text, context_size=context_size)

        # Recover the opener and strip it: by codebook match (Mode A) or by the
        # known prompt prefix (Mode C). Both are byte-exact string splits.
        if not self._prompt:
            try:
                opener, cover_text = match_seed(cover_text)
            except ValueError as exc:
                raise StegoDecodeError(str(exc)) from exc
        else:
            opener = self._prompt
            if not cover_text.startswith(opener):
                raise StegoDecodeError(
                    "Cover text does not start with the configured prompt. "
                    "The wrong prompt was supplied, or the text was not produced "
                    "by this prompt."
                )
            cover_text = cover_text[len(opener) :]
        prompt = opener

        model, tokenizer, device = self._ensure_model()
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
        result = self.encode(text.encode(encoding))
        assert isinstance(result, str)
        return result

    def decode_str(self, cover_text: str, encoding: str = "utf-8") -> str:
        """Decode steganographic cover text back to a string.

        Args:
            cover_text: Text previously produced by :meth:`encode_str`.
            encoding: Character encoding (default ``utf-8``).

        Returns:
            The original string.
        """
        return self.decode(cover_text).decode(encoding)

    # ------------------------------------------------------------------
    # Private chunked helpers
    # ------------------------------------------------------------------

    def _encode_chunked(
        self,
        data: bytes,
        *,
        chunk_size: int,
        context_size: int,
    ) -> list[str]:
        """Encode data by splitting into independently encoded chunks.

        Each chunk is encoded as a standalone steganographic message.  When
        ``password`` is set, each chunk is encrypted independently (separate
        salt/nonce).  Consecutive chunks use the tail of the previous chunk's
        cover text as the prompt so that the generated prose reads coherently.
        """
        model, tokenizer, device = self._ensure_model()

        # Split plaintext into chunks, then optionally encrypt each one.
        chunks: list[bytes] = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
        if self._password is not None:
            chunks = [_encrypt(chunk, self._password) for chunk in chunks]

        cover_texts: list[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                opener = self._prompt if self._prompt else select_seed(data, self._topic)
                prompt = opener
                prefix = opener
            else:
                prompt = cover_texts[idx - 1][-context_size:]
                prefix = ""

            cover_text, _ids, _bits = _encode(
                chunk,
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                top_k=self._top_k,
                temperature=self._temperature,
                sentence_boundary=self._sentence_boundary,
            )
            cover_texts.append(prefix + cover_text)

        return cover_texts

    def _decode_chunked(
        self,
        cover_texts: list[str],
        *,
        context_size: int,
    ) -> bytes:
        """Decode a list of cover texts produced by chunked encoding."""
        model, tokenizer, device = self._ensure_model()

        payloads: list[bytes] = []
        for idx, cover_text in enumerate(cover_texts):
            if idx == 0:
                if not self._prompt:
                    try:
                        opener, cover_text = match_seed(cover_text)
                    except ValueError as exc:
                        raise StegoDecodeError(str(exc)) from exc
                else:
                    opener = self._prompt
                    if not cover_text.startswith(opener):
                        raise StegoDecodeError(
                            "Cover text does not start with the configured prompt."
                        )
                    cover_text = cover_text[len(opener) :]
                prompt = opener
            else:
                prompt = cover_texts[idx - 1][-context_size:]

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
        opener = self._prompt if self._prompt else select_seed(data, self._topic)
        if self._password is not None:
            data = _encrypt(data, self._password)
        model, tokenizer, device = self._ensure_model()

        cover_text, token_ids, total_bits = _encode(
            data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=opener,
            top_k=self._top_k,
            temperature=self._temperature,
            sentence_boundary=self._sentence_boundary,
        )
        cover_text = opener + cover_text
        total_tokens = len(token_ids)
        bits_per_token = total_bits / total_tokens if total_tokens > 0 else 0.0

        return StegoStats(
            cover_text=cover_text,
            bits_per_token=bits_per_token,
            total_tokens=total_tokens,
            payload_size_bytes=original_size,
        )
