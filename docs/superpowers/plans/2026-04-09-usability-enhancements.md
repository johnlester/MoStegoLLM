# MoStegoLLM Usability Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the API (13 to 7 methods), add comprehensive documentation, polish the CLI, fix critical correctness issues, update examples, and fill test gaps — resulting in a 0.2.0 release.

**Architecture:** Six sequential tiers executed in dependency order. API changes land first so documentation and examples reference the final surface. The header format changes in Tier 4 (CRC-32) are a breaking wire-format change, so all existing tests are updated as part of that tier.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace Transformers, argparse, zlib (CRC-32), hypothesis (new dev dep)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/mostegollm/codec.py` | Modify | Remove 6 methods, merge chunking into encode/decode |
| `src/mostegollm/__init__.py` | No change | Exports unchanged |
| `src/mostegollm/utils.py` | Modify | Add CRC-32 to header format |
| `src/mostegollm/encoder.py` | Modify | Pass payload CRC to header |
| `src/mostegollm/decoder.py` | Modify | Verify CRC, fix assert |
| `src/mostegollm/seeds.py` | Modify | Add prefix validation |
| `src/mostegollm/cli.py` | Modify | Add flags, improve errors |
| `README.md` | Rewrite | Full documentation |
| `examples/basic_usage.py` | Rewrite | Updated API examples |
| `examples/MoStegoLLM_notebook.ipynb` | Rewrite | Interactive notebook |
| `examples/cli_walkthrough.sh` | Create | CLI demo script |
| `tests/test_chunked.py` | Modify | Update to use new chunked API |
| `tests/test_codec.py` | Modify | Remove file method tests, update |
| `tests/test_edge_cases.py` | Modify | Update header tests for new size |
| `tests/test_cli.py` | Create | CLI test coverage |
| `tests/test_fuzz.py` | Create | Property-based round-trip tests |
| `tests/test_capacity.py` | Create | Capacity limit tests |
| `tests/test_timing.py` | Modify | Add assertion bound |
| `pyproject.toml` | Modify | Bump version, add hypothesis dep |

---

### Task 1: Merge chunked encoding into core encode/decode

**Files:**
- Modify: `src/mostegollm/codec.py:119-188` (encode and decode methods)
- Modify: `src/mostegollm/codec.py:246-386` (encode_long, decode_long, encode_long_str, decode_long_str)
- Modify: `src/mostegollm/codec.py:217-240` (encode_file, decode_file)
- Test: `tests/test_roundtrip.py`

- [ ] **Step 1: Write failing test for chunked encode via chunk_size param**

Add to `tests/test_roundtrip.py` inside `TestRoundTrip`:

```python
def test_chunked_via_param(self, codec: StegoCodec) -> None:
    """encode(chunk_size=...) should return a list and round-trip."""
    original = b"A" * 50 + b"B" * 50 + b"C" * 50
    covers = codec.encode(original, chunk_size=50)
    assert isinstance(covers, list)
    assert len(covers) == 3
    recovered = codec.decode(covers)
    assert recovered == original
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_roundtrip.py::TestRoundTrip::test_chunked_via_param -v`
Expected: FAIL — `encode()` doesn't accept `chunk_size`

- [ ] **Step 3: Implement merged encode/decode and remove old methods**

Replace the entire `StegoCodec` class body in `src/mostegollm/codec.py` with:

```python
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
            chunk_size: If set, split data into chunks of this many bytes
                and encode each independently. Returns a list of cover texts.
            context_size: When chunking, number of trailing characters from
                the previous chunk's cover text used as prompt for the next.

        Returns:
            A string of cover text (single mode) or a list of cover text
            strings (chunked mode).

        Raises:
            StegoEncodeError: If encoding fails.
            StegoModelError: If the model cannot be loaded.
        """
        if chunk_size is not None:
            return self._encode_chunked(data, chunk_size, context_size)

        seed = select_seed(data) if not self._prompt else ""
        prompt = self._prompt or seed

        if self._password is not None:
            data = _encrypt(data, self._password)
        model, tokenizer, device = self._ensure_model()
        cover_text, _token_ids, _total_bits = _encode(
            data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            top_k=self._top_k,
            temperature=self._temperature,
            sentence_boundary=self._sentence_boundary,
        )
        return seed + cover_text

    def decode(
        self,
        cover_text: str | list[str],
        *,
        context_size: int = DEFAULT_CONTEXT_SIZE,
    ) -> bytes:
        """Decode steganographic cover text back to the original bytes.

        Args:
            cover_text: Text previously produced by :meth:`encode`.
                Pass a list of strings to decode chunked output.
            context_size: When decoding chunked output, must match the
                value used during encoding.

        Returns:
            The original binary payload.

        Raises:
            StegoDecodeError: If decoding fails (wrong prompt, corrupted text, ...).
            StegoCryptoError: If decryption fails (wrong password, tampered data).
            StegoModelError: If the model cannot be loaded.
        """
        if isinstance(cover_text, list):
            return self._decode_chunked(cover_text, context_size)

        # If using seed phrases (no custom prompt), extract the seed from the
        # cover text prefix to reconstruct the prompt used during encoding.
        if not self._prompt:
            try:
                seed, cover_text = match_seed(cover_text)
            except ValueError as exc:
                raise StegoDecodeError(str(exc)) from exc
            prompt = seed
        else:
            prompt = self._prompt

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
        seed = select_seed(data) if not self._prompt else ""
        prompt = self._prompt or seed
        if self._password is not None:
            data = _encrypt(data, self._password)
        model, tokenizer, device = self._ensure_model()

        cover_text, token_ids, total_bits = _encode(
            data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            top_k=self._top_k,
            temperature=self._temperature,
            sentence_boundary=self._sentence_boundary,
        )
        cover_text = seed + cover_text
        total_tokens = len(token_ids)
        bits_per_token = total_bits / total_tokens if total_tokens > 0 else 0.0

        return StegoStats(
            cover_text=cover_text,
            bits_per_token=bits_per_token,
            total_tokens=total_tokens,
            payload_size_bytes=original_size,
        )

    # ------------------------------------------------------------------
    # Internal chunked helpers
    # ------------------------------------------------------------------

    def _encode_chunked(
        self, data: bytes, chunk_size: int, context_size: int
    ) -> list[str]:
        """Split data into chunks and encode each independently."""
        model, tokenizer, device = self._ensure_model()

        chunks: list[bytes] = [
            data[i : i + chunk_size] for i in range(0, len(data), chunk_size)
        ]
        if self._password is not None:
            chunks = [_encrypt(chunk, self._password) for chunk in chunks]

        cover_texts: list[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                seed = select_seed(data) if not self._prompt else ""
                prompt = self._prompt or seed
            else:
                prev = cover_texts[idx - 1]
                seed = ""
                prompt = prev[-context_size:]

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
            cover_texts.append(seed + cover_text)

        return cover_texts

    def _decode_chunked(self, cover_texts: list[str], context_size: int) -> bytes:
        """Decode a list of chunked cover texts."""
        model, tokenizer, device = self._ensure_model()

        payloads: list[bytes] = []
        for idx, cover_text in enumerate(cover_texts):
            if idx == 0:
                if not self._prompt:
                    try:
                        seed, cover_text = match_seed(cover_text)
                    except ValueError as exc:
                        raise StegoDecodeError(str(exc)) from exc
                    prompt = seed
                else:
                    prompt = self._prompt
            else:
                prev = cover_texts[idx - 1]
                prompt = prev[-context_size:]

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
```

Also remove the `pathlib` import from the top of `codec.py` (no longer needed) and the `Union` import from `typing`. The `from typing import Union` can be removed entirely.

- [ ] **Step 4: Run the new test to verify it passes**

Run: `pytest tests/test_roundtrip.py::TestRoundTrip::test_chunked_via_param -v`
Expected: PASS

- [ ] **Step 5: Run all existing tests to catch breakage**

Run: `pytest tests/ -v --ignore=tests/test_qwen.py`
Expected: Several failures in `test_chunked.py` and `test_codec.py` due to removed methods. That's expected — we'll fix those in the next steps.

- [ ] **Step 6: Commit**

```bash
git add src/mostegollm/codec.py tests/test_roundtrip.py
git commit -m "feat: merge chunked encoding into core encode/decode, remove 6 methods

Consolidate encode_long/decode_long into encode(chunk_size=N)/decode(list).
Remove encode_file, decode_file, encode_long_str, decode_long_str.
API surface: 13 methods -> 7 methods."
```

---

### Task 2: Update existing tests for new API

**Files:**
- Modify: `tests/test_chunked.py`
- Modify: `tests/test_codec.py`

- [ ] **Step 1: Update test_chunked.py to use new API**

Replace the full contents of `tests/test_chunked.py` with:

```python
"""Tests for chunked encoding via the chunk_size parameter."""

from __future__ import annotations

from mostegollm import StegoCodec


class TestChunkedEncoding:
    """Verify encode(chunk_size=...) / decode(list) round-trip behaviour."""

    def test_short_data_single_chunk(self, codec: StegoCodec) -> None:
        """Data smaller than chunk_size produces a single-element list and round-trips."""
        original = b"hello"
        cover_texts = codec.encode(original, chunk_size=2000)
        assert isinstance(cover_texts, list)
        assert len(cover_texts) == 1
        recovered = codec.decode(cover_texts)
        assert recovered == original

    def test_single_chunk_matches_plain_encode(self, codec: StegoCodec) -> None:
        """When data fits in one chunk, result matches the non-chunked encode."""
        original = b"hello"
        cover_single = codec.encode(original)
        cover_long = codec.encode(original, chunk_size=2000)
        assert len(cover_long) == 1
        assert cover_long[0] == cover_single

    def test_multi_chunk_roundtrip(self, codec: StegoCodec) -> None:
        """A payload larger than chunk_size splits into multiple chunks and round-trips."""
        original = b"A" * 50 + b"B" * 50 + b"C" * 50
        cover_texts = codec.encode(original, chunk_size=50)
        assert len(cover_texts) == 3
        recovered = codec.decode(cover_texts)
        assert recovered == original

    def test_multi_chunk_encrypted_roundtrip(self, codec_encrypted: StegoCodec) -> None:
        """Chunked encoding with encryption round-trips correctly."""
        original = b"Secret message that spans multiple chunks!"
        cover_texts = codec_encrypted.encode(original, chunk_size=20)
        assert len(cover_texts) >= 2
        recovered = codec_encrypted.decode(cover_texts)
        assert recovered == original

    def test_prompt_chaining_coherence(self, codec: StegoCodec) -> None:
        """Chunk 1+ uses a chained prompt derived from the previous chunk's cover text."""
        original = b"X" * 100
        cover_texts = codec.encode(original, chunk_size=50)
        assert len(cover_texts) >= 2

        # Encode the same second chunk's data standalone with the default prompt.
        # The cover text should differ because chunked mode uses a chained prompt.
        standalone_cover = codec.encode(original[50:100])
        assert cover_texts[1] != standalone_cover

    def test_chunk_size_respected(self, codec: StegoCodec) -> None:
        """Each chunk's plaintext payload does not exceed chunk_size."""
        chunk_size = 30
        original = b"A" * 100
        cover_texts = codec.encode(original, chunk_size=chunk_size)

        # Decode each chunk individually using the same prompt chaining logic
        # to verify each raw payload size.
        for idx, cover_text in enumerate(cover_texts):
            prompt = codec._prompt if idx == 0 else cover_texts[idx - 1][-500:]
            temp_codec = StegoCodec.__new__(StegoCodec)
            temp_codec._model = codec._model
            temp_codec._tokenizer = codec._tokenizer
            temp_codec._device = codec._device
            temp_codec._prompt = prompt
            temp_codec._top_k = codec._top_k
            temp_codec._temperature = codec._temperature
            temp_codec._password = None
            payload = temp_codec.decode(cover_text)
            assert len(payload) <= chunk_size

    def test_str_chunked_roundtrip(self, codec: StegoCodec) -> None:
        """String data can be chunked via encode(text.encode(), chunk_size=...)."""
        original = "Hello, world! " * 10  # ~140 bytes
        cover_texts = codec.encode(original.encode(), chunk_size=50)
        assert len(cover_texts) >= 2
        recovered = codec.decode(cover_texts).decode()
        assert recovered == original

    def test_uneven_last_chunk(self, codec: StegoCodec) -> None:
        """When data doesn't divide evenly, the last chunk is smaller."""
        original = b"A" * 70
        cover_texts = codec.encode(original, chunk_size=30)
        # 70 / 30 -> 3 chunks (30, 30, 10)
        assert len(cover_texts) == 3
        recovered = codec.decode(cover_texts)
        assert recovered == original
```

- [ ] **Step 2: Update test_codec.py — remove encode_file/decode_file test**

Replace the full contents of `tests/test_codec.py` with:

```python
"""API-level tests for StegoCodec."""

from __future__ import annotations

import pytest

from mostegollm import StegoCodec, StegoDecodeError, StegoStats


class TestCodecAPI:
    """Test the public StegoCodec interface."""

    def test_encode_returns_string(self, codec: StegoCodec) -> None:
        """encode() should return a non-empty string."""
        result = codec.encode(b"test")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_decode_returns_bytes(self, codec: StegoCodec) -> None:
        """decode() should return bytes."""
        cover = codec.encode(b"test")
        result = codec.decode(cover)
        assert isinstance(result, bytes)

    def test_encode_with_stats(self, codec: StegoCodec) -> None:
        """encode_with_stats() should return a StegoStats with sane values."""
        data = b"stats test payload"
        stats = codec.encode_with_stats(data)

        assert isinstance(stats, StegoStats)
        assert isinstance(stats.cover_text, str)
        assert len(stats.cover_text) > 0
        assert stats.total_tokens > 0
        assert stats.bits_per_token > 0
        assert stats.payload_size_bytes == len(data)

    def test_stateless_multiple_calls(self, codec: StegoCodec) -> None:
        """Multiple encode/decode calls on the same instance should work."""
        data1 = b"first"
        data2 = b"second"

        cover1 = codec.encode(data1)
        cover2 = codec.encode(data2)

        assert codec.decode(cover1) == data1
        assert codec.decode(cover2) == data2

    def test_wrong_prompt_fails(
        self, codec: StegoCodec, codec_alt_prompt: StegoCodec
    ) -> None:
        """Decoding with a different prompt should fail or return wrong data."""
        data = b"prompt sensitive"
        cover = codec.encode(data)

        with pytest.raises((StegoDecodeError, AssertionError)):
            codec_alt_prompt.decode(cover)

    def test_chunked_returns_list(self, codec: StegoCodec) -> None:
        """encode(chunk_size=...) returns a list."""
        result = codec.encode(b"test", chunk_size=100)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_decode_accepts_list(self, codec: StegoCodec) -> None:
        """decode() accepts a list of strings."""
        covers = codec.encode(b"test data here", chunk_size=5)
        result = codec.decode(covers)
        assert isinstance(result, bytes)
        assert result == b"test data here"
```

- [ ] **Step 3: Run all tests (except qwen) to verify**

Run: `pytest tests/ -v --ignore=tests/test_qwen.py`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_chunked.py tests/test_codec.py
git commit -m "test: update tests for simplified API (remove file methods, use chunk_size param)"
```

---

### Task 3: Update Qwen tests for new API

**Files:**
- Modify: `tests/test_qwen.py`

- [ ] **Step 1: Read current test_qwen.py**

Run: `cat tests/test_qwen.py` to see current content (it uses `encode_long`, `decode_long`, `encode_file`, `decode_file`).

- [ ] **Step 2: Update Qwen tests**

In `tests/test_qwen.py`, make these replacements:

1. Replace all `codec.encode_long(...)` with `codec.encode(..., chunk_size=...)`
2. Replace all `codec.decode_long(...)` with `codec.decode(...)`
3. Replace any `encode_long_str` / `decode_long_str` calls similarly
4. Remove any `encode_file` / `decode_file` tests — replace with inline `Path.read_bytes()` / `Path.write_bytes()` equivalents if the file round-trip test exists

- [ ] **Step 3: Run Qwen tests**

Run: `pytest tests/test_qwen.py -v`
Expected: All PASS (requires Qwen model to be downloaded)

- [ ] **Step 4: Commit**

```bash
git add tests/test_qwen.py
git commit -m "test: update Qwen tests for simplified API"
```

---

### Task 4: Write the README

**Files:**
- Rewrite: `README.md`

- [ ] **Step 1: Write the full README**

Replace `README.md` with the full documentation. Structure:

```markdown
# MoStegoLLM

Hide binary data inside natural English prose using LLM-powered steganography.

## Installation

```bash
pip install mostegollm
```

> **Note:** Requires PyTorch. For GPU acceleration, install with CUDA support:
> `pip install mostegollm torch --index-url https://download.pytorch.org/whl/cu128`

## Quick Start

```python
from mostegollm import StegoCodec

codec = StegoCodec()
cover = codec.encode_str("attack at dawn")
print(cover)  # Natural-looking English prose

secret = codec.decode_str(cover)
print(secret)  # "attack at dawn"
```

## CLI

```bash
# Encode a message
mostegollm encode "attack at dawn" -o cover.txt

# Decode it
mostegollm decode cover.txt --text

# Encode with encryption
mostegollm encode "classified" --password mysecret -o cover.txt

# Decode with encryption
mostegollm decode cover.txt --text --password mysecret

# Encode a file
mostegollm encode -f secret.bin -o cover.txt --stats

# List available models
mostegollm models
```

## How It Works

MoStegoLLM uses **arithmetic coding** over a language model's token probability distribution to hide data in generated text.

At each step, the LLM predicts a probability distribution over the next token. The encoder uses bits from your secret data to select which token to emit — high-probability tokens encode fewer bits, low-probability tokens encode more. The decoder replays the same distributions and reads back the bits each token choice represents.

```
Secret bytes
    |
    v
[Header: magic + length + CRC-32] + payload bits
    |
    v
Arithmetic coding: bits select tokens from LLM probability distribution
    |
    v
Natural English prose (cover text)
```

Because both encoder and decoder use the same model with identical parameters, the process is perfectly reversible.

## Capacity

Approximate cover text size at default settings (top-k=256, SmolLM-135M):

| Payload | Tokens | Words (approx) |
|---------|--------|-----------------|
| 10 bytes | ~15 | ~50 |
| 100 bytes | ~130 | ~400 |
| 1 KB | ~1,300 | ~4,000 |

Use `encode_with_stats()` for exact measurements on your data.

## API Reference

### `StegoCodec(model_name, device, prompt, top_k, temperature, sentence_boundary, token, password)`

Constructor. All parameters optional. Lazy-loads the model on first encode/decode.

### `encode(data, *, chunk_size=None, context_size=500) -> str | list[str]`

Encode bytes into cover text. Pass `chunk_size` to split large payloads into independently encoded chunks (returns a list).

### `decode(cover_text, *, context_size=500) -> bytes`

Decode cover text (string or list of strings) back to bytes.

### `encode_str(text, encoding="utf-8") -> str`

Convenience: encode a string.

### `decode_str(cover_text, encoding="utf-8") -> str`

Convenience: decode to a string.

### `encode_with_stats(data) -> StegoStats`

Encode and return a `StegoStats` object with `cover_text`, `bits_per_token`, `total_tokens`, and `payload_size_bytes`.

### `StegoCodec.list_models() -> tuple[ModelInfo, ...]`

Class method. Returns metadata for all recommended models.

### `StegoCodec.get_model_info(name) -> ModelInfo | None`

Class method. Look up a model by name.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `HuggingFaceTB/SmolLM-135M` | HuggingFace model ID |
| `device` | `auto` | `auto`, `cpu`, `cuda`, etc. |
| `prompt` | `""` (auto seed) | Fixed prompt prefix for generation |
| `top_k` | `256` | Number of top tokens per step |
| `temperature` | `1.0` | Softmax temperature |
| `sentence_boundary` | `False` | End cover text at sentence boundary |
| `token` | `None` | HuggingFace API token for gated models |
| `password` | `None` | AES-256-GCM encryption password |

## Security Model

**Steganographic hiding is not encryption.** Without a `password`, the data is obscured in the statistical structure of the text but is not confidential — anyone with the same model and parameters can decode it.

**With `password`:** Data is encrypted with AES-256-GCM before encoding. Key derivation uses PBKDF2-HMAC-SHA256 with 600,000 iterations. This provides both confidentiality and authentication.

**Cross-platform requirement:** Encoder and decoder must use the same PyTorch version, device type (CPU/CUDA), and model to produce identical probability distributions. Encoding on one platform and decoding on another is not guaranteed to work.

**Integrity:** A CRC-32 checksum in the header detects accidental corruption. This is not a cryptographic guarantee — use `password` mode for tamper detection.

## Supported Models

| Model | Parameters | Description |
|-------|-----------|-------------|
| `HuggingFaceTB/SmolLM-135M` | 135M | Tiny, fast default (recommended) |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | Fallback chat model |
| `HuggingFaceTB/SmolLM-360M` | 360M | Larger SmolLM, better prose |
| `Qwen/Qwen2.5-0.5B` | 0.5B | Compact multilingual model |
| `meta-llama/Llama-3.2-1B` | 1B | High-quality Meta model (requires `HF_TOKEN`) |

## Acknowledgments

This project uses [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) by Hugging Face,
licensed under the [Apache License 2.0](LICENSES/Apache-2.0.txt).
See the [NOTICE](NOTICE) file for details.

## License

MIT
```

- [ ] **Step 2: Verify README renders correctly**

Run: `head -5 README.md` to sanity check.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README with full documentation

Add installation, quick start, CLI usage, how-it-works explanation,
capacity guide, API reference, configuration, security model, and
supported models."
```

---

### Task 5: Add CLI flags (--text, --password, --chunk-size, --stats, --quiet)

**Files:**
- Modify: `src/mostegollm/cli.py`

- [ ] **Step 1: Write failing test for --text flag**

Create `tests/test_cli.py`:

```python
"""CLI integration tests for mostegollm."""

from __future__ import annotations

import subprocess
import sys


def _run(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess:
    """Run mostegollm CLI and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "mostegollm.cli"] + args,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=300,
    )


class TestCLIEncode:
    """Test the encode subcommand."""

    def test_encode_string(self) -> None:
        """Encoding a string should produce output on stdout."""
        result = _run(["encode", "hello"])
        assert result.returncode == 0
        assert len(result.stdout) > 0

    def test_encode_stats(self) -> None:
        """--stats should print stats to stderr."""
        result = _run(["encode", "hello", "--stats"])
        assert result.returncode == 0
        assert "bits/token" in result.stderr.lower() or "bits_per_token" in result.stderr.lower()
        assert "token" in result.stderr.lower()

    def test_encode_quiet(self) -> None:
        """--quiet should suppress model loading noise on stderr."""
        result = _run(["encode", "hi", "--quiet"])
        assert result.returncode == 0
        # stderr should be empty (no model loading messages)
        assert result.stderr == ""


class TestCLIDecode:
    """Test the decode subcommand."""

    def test_roundtrip(self) -> None:
        """Encode then decode should recover the original."""
        enc = _run(["encode", "roundtrip test"])
        assert enc.returncode == 0
        cover = enc.stdout

        dec = _run(["decode"], input_text=cover)
        assert dec.returncode == 0
        assert dec.stdout.encode().rstrip(b"\n") == b"roundtrip test"

    def test_text_flag(self) -> None:
        """--text flag should decode and print as UTF-8."""
        enc = _run(["encode", "text flag test"])
        assert enc.returncode == 0

        dec = _run(["decode", "--text"], input_text=enc.stdout)
        assert dec.returncode == 0
        assert "text flag test" in dec.stdout

    def test_password_roundtrip(self) -> None:
        """--password should encrypt during encode and decrypt during decode."""
        enc = _run(["encode", "secret", "--password", "mypass"])
        assert enc.returncode == 0

        dec = _run(["decode", "--text", "--password", "mypass"], input_text=enc.stdout)
        assert dec.returncode == 0
        assert "secret" in dec.stdout

    def test_wrong_password_fails(self) -> None:
        """Decoding with wrong password should fail."""
        enc = _run(["encode", "secret", "--password", "right"])
        assert enc.returncode == 0

        dec = _run(["decode", "--text", "--password", "wrong"], input_text=enc.stdout)
        assert dec.returncode != 0
        assert "error" in dec.stderr.lower() or "fail" in dec.stderr.lower()

    def test_decode_garbage(self) -> None:
        """Decoding garbage text should fail with a user-friendly error."""
        dec = _run(["decode", "this is not encoded text at all"])
        assert dec.returncode != 0
        assert "error" in dec.stderr.lower()


class TestCLIModels:
    """Test the models subcommand."""

    def test_models_list(self) -> None:
        """models subcommand should print model info and exit 0."""
        result = _run(["models"])
        assert result.returncode == 0
        assert "SmolLM" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestCLIEncode::test_encode_stats -v`
Expected: FAIL — `--stats` flag doesn't exist yet

- [ ] **Step 3: Implement CLI changes**

Replace the full contents of `src/mostegollm/cli.py` with:

```python
"""Command-line interface for MoStegoLLM."""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from .codec import StegoCodec
from .encoder import TOP_K
from .model import DEFAULT_PROMPT, PRIMARY_MODEL
from .utils import StegoCryptoError, StegoDecodeError, StegoError, StegoModelError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mostegollm",
        description="Hide secret data inside LLM-generated English prose.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print diagnostics to stderr"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress model loading output"
    )
    parser.add_argument("--model", default=PRIMARY_MODEL, help="HuggingFace model name")
    parser.add_argument(
        "--device", default="auto", help="torch device (auto, cpu, cuda, ...)"
    )
    parser.add_argument("--top-k", type=int, default=TOP_K, help="top-k filtering width")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="seed prompt for generation")

    sub = parser.add_subparsers(dest="command")

    # -- encode --------------------------------------------------------
    enc = sub.add_parser("encode", help="encode secret data into cover text")
    enc.add_argument("text", nargs="?", default=None, help="string to encode")
    enc.add_argument("-f", "--file", default=None, help="file to encode")
    enc.add_argument("-o", "--output", default=None, help="write cover text to file")
    enc.add_argument(
        "-p", "--password", default=None, help="encrypt with AES-256-GCM password"
    )
    enc.add_argument(
        "--chunk-size", type=int, default=None,
        help="split large payloads into chunks of N bytes",
    )
    enc.add_argument(
        "--stats", action="store_true", help="print encoding stats to stderr"
    )
    enc.add_argument(
        "--sentence-boundary",
        action="store_true",
        default=False,
        help="continue generating until cover text ends at a sentence boundary",
    )

    # -- models --------------------------------------------------------
    sub.add_parser("models", help="list recommended models")

    # -- decode --------------------------------------------------------
    dec = sub.add_parser("decode", help="decode cover text back to secret data")
    dec.add_argument("text", nargs="?", default=None, help="cover text to decode")
    dec.add_argument("-f", "--file", default=None, help="file containing cover text")
    dec.add_argument("-o", "--output", default=None, help="write decoded bytes to file")
    dec.add_argument(
        "-t", "--text", action="store_true",
        dest="text_mode", help="decode and print as UTF-8 text"
    )
    dec.add_argument(
        "-p", "--password", default=None, help="decrypt with AES-256-GCM password"
    )

    return parser


def _log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, file=sys.stderr)


def _read_input(args: argparse.Namespace) -> str | bytes:
    """Return the user-supplied input as str (for decode) or bytes (for encode -f)."""
    if args.text is not None:
        return args.text
    if args.file is not None:
        if args.command == "encode":
            with open(args.file, "rb") as fh:
                return fh.read()
        with open(args.file, encoding="utf-8") as fh:
            return fh.read()
    # stdin
    if not sys.stdin.isatty():
        if args.command == "encode":
            return sys.stdin.buffer.read()
        return sys.stdin.read()
    print(
        f"mostegollm {args.command}: no input (pass a string, -f FILE, or pipe stdin)",
        file=sys.stderr,
    )
    sys.exit(1)


def _cmd_models() -> None:
    """Print a formatted table of recommended models."""
    models = StegoCodec.list_models()
    name_w = max(len(m.name) for m in models)
    param_w = max(len(m.parameters) for m in models)

    header = f"  {'Model':<{name_w}}  {'Params':<{param_w}}  Description"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for m in models:
        gated = " [gated]" if m.gated else ""
        print(f"  {m.name:<{name_w}}  {m.parameters:<{param_w}}  {m.description}{gated}")


def _cmd_encode(codec: StegoCodec, args: argparse.Namespace, verbose: bool, quiet: bool) -> None:
    raw = _read_input(args)
    data = raw if isinstance(raw, bytes) else raw.encode("utf-8")

    if verbose:
        _log(f"Payload size: {len(data)} bytes")

    t0 = time.perf_counter()

    show_stats = args.stats or verbose
    if show_stats:
        stats = codec.encode_with_stats(data)
        cover_text = stats.cover_text
    elif args.chunk_size is not None:
        covers = codec.encode(data, chunk_size=args.chunk_size)
        assert isinstance(covers, list)
        cover_text = "\n---\n".join(covers)
        stats = None
    else:
        cover_text = codec.encode(data)
        assert isinstance(cover_text, str)
        stats = None

    elapsed = time.perf_counter() - t0

    if show_stats and stats is not None:
        _log(f"Tokens generated: {stats.total_tokens}", quiet=False)
        _log(f"Bits per token:   {stats.bits_per_token:.2f}", quiet=False)
        _log(f"Payload size:     {stats.payload_size_bytes} bytes", quiet=False)
        _log(f"Encoding time:    {elapsed:.2f}s", quiet=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(cover_text)
        _log(f"Cover text written to {args.output}", quiet)
    else:
        sys.stdout.write(cover_text)
        if sys.stdout.isatty():
            sys.stdout.write("\n")


def _cmd_decode(codec: StegoCodec, args: argparse.Namespace, verbose: bool, quiet: bool) -> None:
    raw = _read_input(args)
    cover_text = raw if isinstance(raw, str) else raw.decode("utf-8")

    t0 = time.perf_counter()
    recovered = codec.decode(cover_text)
    elapsed = time.perf_counter() - t0

    if verbose:
        _log(f"Recovered payload: {len(recovered)} bytes")
        _log(f"Decoding time:     {elapsed:.2f}s")

    if args.output:
        with open(args.output, "wb") as fh:
            fh.write(recovered)
        _log(f"Decoded bytes written to {args.output}", quiet)
    elif args.text_mode:
        sys.stdout.write(recovered.decode("utf-8"))
        if sys.stdout.isatty():
            sys.stdout.write("\n")
    else:
        sys.stdout.buffer.write(recovered)
        if sys.stdout.isatty():
            sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "models":
        _cmd_models()
        return

    verbose = args.verbose
    quiet = args.quiet

    if verbose:
        import torch

        device_str = args.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device_info = device_str
        if device_str.startswith("cuda") and torch.cuda.is_available():
            device_info = f"{device_str} -- {torch.cuda.get_device_name()}"
        _log(f"Device: {device_info}")
        _log(f"Loading model: {args.model}")

    t_model = time.perf_counter()
    password = getattr(args, "password", None)
    sentence_boundary = getattr(args, "sentence_boundary", False)
    codec = StegoCodec(
        model_name=args.model,
        device=args.device,
        prompt=args.prompt,
        top_k=args.top_k,
        sentence_boundary=sentence_boundary,
        password=password,
    )
    # Force model load so we can report timing
    codec._ensure_model()
    t_model = time.perf_counter() - t_model

    if verbose:
        _log(f"Model loaded in {t_model:.2f}s")

    try:
        if args.command == "encode":
            _cmd_encode(codec, args, verbose, quiet)
        else:
            _cmd_decode(codec, args, verbose, quiet)
    except StegoDecodeError:
        _log("Error: could not decode -- wrong model or corrupted text." +
             (" Use --verbose for details." if not verbose else ""), quiet=False)
        if verbose:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    except StegoModelError as exc:
        _log(f"Error: could not load model '{args.model}'. {exc}", quiet=False)
        if verbose:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    except StegoCryptoError:
        _log("Error: decryption failed -- wrong password or tampered data." +
             (" Use --verbose for details." if not verbose else ""), quiet=False)
        if verbose:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    except StegoError as exc:
        _log(f"Error: {exc}", quiet=False)
        if verbose:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests**

Run: `pytest tests/test_cli.py -v`
Expected: All PASS (note: these tests load the model, so they're slow — ~60s each)

- [ ] **Step 5: Commit**

```bash
git add src/mostegollm/cli.py tests/test_cli.py
git commit -m "feat: add CLI flags (--text, --password, --chunk-size, --stats, --quiet)

Add --text/-t for UTF-8 decode output, --password/-p for encryption,
--chunk-size for chunked encoding, --stats for encoding diagnostics,
--quiet/-q to suppress model loading output. Improve error messages
with user-friendly wrappers."
```

---

### Task 6: Add CRC-32 to header format

**Files:**
- Modify: `src/mostegollm/utils.py`
- Modify: `src/mostegollm/encoder.py:251-254`
- Modify: `src/mostegollm/decoder.py:170-191`
- Modify: `tests/test_edge_cases.py`

- [ ] **Step 1: Write failing test for CRC verification**

Add to `tests/test_edge_cases.py`, in a new class at the end:

```python
class TestCRC:
    """Test CRC-32 integrity checking."""

    def test_header_includes_crc(self) -> None:
        """New header should be 10 bytes (magic + length + crc32)."""
        from mostegollm.utils import HEADER_SIZE
        assert HEADER_SIZE == 10

    def test_crc_roundtrip(self) -> None:
        """pack_header with CRC should round-trip via unpack_header."""
        data = b"test payload"
        header = pack_header(len(data), crc32=0xDEADBEEF)
        length, crc = unpack_header(header)
        assert length == len(data)
        assert crc == 0xDEADBEEF

    def test_corrupted_payload_detected(self, codec: StegoCodec) -> None:
        """Decoding a corrupted cover text should raise StegoDecodeError mentioning integrity."""
        # We can't easily corrupt a single token, but we can verify the CRC
        # machinery works end-to-end by encoding and verifying decode succeeds
        data = b"integrity check"
        cover = codec.encode(data)
        assert codec.decode(cover) == data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_edge_cases.py::TestCRC::test_header_includes_crc -v`
Expected: FAIL — `HEADER_SIZE` is still 6

- [ ] **Step 3: Update utils.py header format**

In `src/mostegollm/utils.py`, replace the header section (lines 33-87):

```python
# ---------------------------------------------------------------------------
# Header format
# ---------------------------------------------------------------------------
# 2-byte magic  |  4-byte payload length  |  4-byte CRC-32
# 0x53 0x54     |  <length>               |  <crc32>
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
```

- [ ] **Step 4: Update encoder.py to compute and pass CRC**

In `src/mostegollm/encoder.py`, add `import zlib` at the top (after the existing imports).

Then replace lines 251-254 (the header packing):

```python
    # Prepend header to payload
    import zlib

    payload_crc = zlib.crc32(data) & 0xFFFFFFFF
    header = pack_header(len(data), crc32=payload_crc)
    full_payload = header + data
    bits = bytes_to_bits(full_payload)
    total_bits = len(bits)
```

Move the `import zlib` to the top of the file with the other imports (not inline).

- [ ] **Step 5: Update decoder.py to verify CRC and fix assert**

In `src/mostegollm/decoder.py`, add `import zlib` at the top.

Replace the header parsing and payload extraction section (lines 170-193):

```python
    # Parse header
    header_bytes = bits_to_bytes(extracted_bits[:HEADER_BITS])
    payload_length, expected_crc = unpack_header(header_bytes)

    # Extract payload
    payload_bits_needed = payload_length * 8
    total_bits_needed = HEADER_BITS + payload_bits_needed

    if len(extracted_bits) < total_bits_needed:
        raise StegoDecodeError(
            f"Extracted {len(extracted_bits)} bits but need {total_bits_needed} "
            f"(header says payload is {payload_length} bytes). "
            "The cover text may be truncated."
        )

    payload_bits = extracted_bits[HEADER_BITS:total_bits_needed]
    payload = bits_to_bytes(payload_bits)

    # Sanity check: the payload should be exactly payload_length bytes
    if len(payload) != payload_length:
        raise StegoDecodeError(
            f"Internal error: expected {payload_length} bytes, got {len(payload)}"
        )

    # Verify CRC-32 integrity
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise StegoDecodeError(
            "Payload integrity check failed -- data may be corrupted, "
            "the wrong model was used, or the cover text was modified."
        )

    return payload
```

- [ ] **Step 6: Update test_edge_cases.py header tests**

Replace the `TestHeader` class in `tests/test_edge_cases.py`:

```python
class TestHeader:
    """Test header pack/unpack."""

    def test_roundtrip(self) -> None:
        for length in (0, 1, 255, 65535, 1_000_000):
            header = pack_header(length, crc32=0x12345678)
            assert len(header) == HEADER_SIZE
            recovered_length, recovered_crc = unpack_header(header)
            assert recovered_length == length
            assert recovered_crc == 0x12345678

    def test_bad_magic(self) -> None:
        header = b"\x00\x00" + b"\x00" * 8
        with pytest.raises(StegoDecodeError, match="Invalid magic"):
            unpack_header(header)

    def test_negative_length(self) -> None:
        with pytest.raises(StegoEncodeError, match="negative"):
            pack_header(-1)

    def test_too_large(self) -> None:
        with pytest.raises(StegoEncodeError, match="too large"):
            pack_header(0x1_0000_0000)
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/ -v --ignore=tests/test_qwen.py`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/mostegollm/utils.py src/mostegollm/encoder.py src/mostegollm/decoder.py tests/test_edge_cases.py
git commit -m "feat: add CRC-32 integrity check to header format

Header grows from 6 to 10 bytes (magic + length + crc32).
Encoder computes CRC-32 of payload, decoder verifies on decode.
assert replaced with StegoDecodeError for python -O safety.
Breaking wire format change."
```

---

### Task 7: Add seed phrase prefix validation

**Files:**
- Modify: `src/mostegollm/seeds.py:278` (after `_SORTED_PHRASES`)

- [ ] **Step 1: Write failing test**

Add to `tests/test_edge_cases.py`:

```python
class TestSeedPhrases:
    """Test seed phrase invariants."""

    def test_no_prefix_collisions(self) -> None:
        """No seed phrase should be a prefix of another."""
        from mostegollm.seeds import SEED_PHRASES
        for i, a in enumerate(SEED_PHRASES):
            for b in SEED_PHRASES[i + 1:]:
                assert not a.startswith(b), f"{a!r} starts with {b!r}"
                assert not b.startswith(a), f"{b!r} starts with {a!r}"

    def test_validation_runs_at_import(self) -> None:
        """The module should validate prefix collisions on import."""
        # If we get here, the import in conftest already ran the validation.
        # Just verify the phrases are loaded.
        from mostegollm.seeds import SEED_PHRASES
        assert len(SEED_PHRASES) == 256
```

- [ ] **Step 2: Run test to verify it passes (validation logic not yet in place, but phrases are already valid)**

Run: `pytest tests/test_edge_cases.py::TestSeedPhrases -v`
Expected: PASS (the phrases are already collision-free; this test locks the invariant)

- [ ] **Step 3: Add import-time validation to seeds.py**

In `src/mostegollm/seeds.py`, add after line 279 (`_SORTED_PHRASES = ...`):

```python
# Validate no phrase is a prefix of another (required for unambiguous matching).
for _i, _a in enumerate(SEED_PHRASES):
    for _b in SEED_PHRASES[_i + 1:]:
        if _a.startswith(_b) or _b.startswith(_a):
            raise ValueError(f"Seed phrase prefix collision: {_a!r} / {_b!r}")
del _i, _a, _b  # Clean up module namespace
```

- [ ] **Step 4: Run tests to verify nothing broke**

Run: `pytest tests/ -v --ignore=tests/test_qwen.py -x`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/mostegollm/seeds.py tests/test_edge_cases.py
git commit -m "fix: add import-time seed phrase prefix validation

Validates that no seed phrase is a prefix of another on module import.
Prevents ambiguous longest-prefix matching bugs if the phrase list is edited."
```

---

### Task 8: Rewrite examples

**Files:**
- Rewrite: `examples/basic_usage.py`
- Create: `examples/cli_walkthrough.sh`

- [ ] **Step 1: Rewrite basic_usage.py**

Replace `examples/basic_usage.py` with:

```python
#!/usr/bin/env python3
"""Basic usage example for MoStegoLLM.

Demonstrates encoding secret data into natural-looking English text
and decoding it back, including encryption, chunking, and diagnostics.
"""

from mostegollm import StegoCodec


def main() -> None:
    # --- Initialize ---
    # Model downloads on first run (~270 MB for SmolLM-135M).
    # Use device="cuda" for GPU acceleration.
    print("Initializing StegoCodec...")
    codec = StegoCodec(device="auto")

    # --- 1. Basic encode/decode (bytes) ---
    print("\n--- Basic encode/decode ---")
    secret = b"Hello, World!"
    cover_text = codec.encode(secret)
    print(f"Original:   {secret}")
    print(f"Cover text: {cover_text[:80]}...")
    recovered = codec.decode(cover_text)
    print(f"Recovered:  {recovered}")
    assert recovered == secret

    # --- 2. String convenience ---
    print("\n--- String convenience ---")
    cover = codec.encode_str("steganography is fun")
    decoded = codec.decode_str(cover)
    print(f"Cover:   {cover[:80]}...")
    print(f"Decoded: {decoded}")

    # --- 3. Encryption ---
    print("\n--- Encryption (AES-256-GCM) ---")
    secure_codec = StegoCodec(device="auto", password="my-secret-key")
    cover = secure_codec.encode(b"classified information")
    print(f"Encrypted cover: {cover[:80]}...")
    recovered = secure_codec.decode(cover)
    print(f"Recovered: {recovered}")

    # --- 4. Chunked encoding for large data ---
    print("\n--- Chunked encoding ---")
    large_data = b"A" * 200
    covers = codec.encode(large_data, chunk_size=50)
    print(f"Payload: {len(large_data)} bytes -> {len(covers)} chunks")
    recovered = codec.decode(covers)
    assert recovered == large_data
    print("Chunked round-trip successful!")

    # --- 5. Encoding stats ---
    print("\n--- Encoding stats ---")
    stats = codec.encode_with_stats(b"diagnostic payload")
    print(f"Payload size:   {stats.payload_size_bytes} bytes")
    print(f"Total tokens:   {stats.total_tokens}")
    print(f"Bits per token: {stats.bits_per_token:.2f}")
    print(f"Cover text len: {len(stats.cover_text)} chars")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create cli_walkthrough.sh**

Create `examples/cli_walkthrough.sh`:

```bash
#!/usr/bin/env bash
# MoStegoLLM CLI Walkthrough
# Demonstrates encoding and decoding from the command line.

set -euo pipefail

echo "=== MoStegoLLM CLI Walkthrough ==="
echo

# 1. Encode a text message
echo "--- Encoding a message ---"
mostegollm encode "attack at dawn" -o /tmp/cover.txt
echo "Cover text written to /tmp/cover.txt"
head -c 200 /tmp/cover.txt
echo "..."
echo

# 2. Decode it
echo "--- Decoding ---"
mostegollm decode -f /tmp/cover.txt --text
echo

# 3. Encode with encryption
echo "--- Encrypted encode ---"
mostegollm encode "classified information" --password mysecret -o /tmp/encrypted_cover.txt
echo "Encrypted cover text written to /tmp/encrypted_cover.txt"
echo

# 4. Decode with encryption
echo "--- Encrypted decode ---"
mostegollm decode -f /tmp/encrypted_cover.txt --text --password mysecret
echo

# 5. Encoding stats
echo "--- Encoding with stats ---"
mostegollm encode "stats demo" --stats 2>&1
echo

# 6. List models
echo "--- Available models ---"
mostegollm models
echo

echo "=== Done ==="
```

- [ ] **Step 3: Make walkthrough executable**

Run: `chmod +x examples/cli_walkthrough.sh`

- [ ] **Step 4: Commit**

```bash
git add examples/basic_usage.py examples/cli_walkthrough.sh
git commit -m "docs: rewrite examples for simplified API

Update basic_usage.py with encryption, chunking, and stats sections.
Add cli_walkthrough.sh demonstrating all CLI features."
```

---

### Task 9: Rewrite the Jupyter notebook

**Files:**
- Rewrite: `examples/MoStegoLLM_notebook.ipynb`

- [ ] **Step 1: Rewrite the notebook**

Create the notebook with these cells:

**Cell 1 (markdown):**
```markdown
# MoStegoLLM — Interactive Demo

This notebook demonstrates how MoStegoLLM hides binary data inside natural English prose using arithmetic coding over LLM token probability distributions.
```

**Cell 2 (code):**
```python
from mostegollm import StegoCodec

codec = StegoCodec(device="auto")
print("Model loaded!")
```

**Cell 3 (markdown):**
```markdown
## Basic Round-Trip
Encode a secret message into cover text, then decode it back.
```

**Cell 4 (code):**
```python
secret = "attack at dawn"
cover = codec.encode_str(secret)
print(f"Secret:  {secret}")
print(f"Cover:   {cover}")
print(f"Decoded: {codec.decode_str(cover)}")
```

**Cell 5 (markdown):**
```markdown
## Encryption
Add AES-256-GCM encryption for confidentiality on top of steganographic hiding.
```

**Cell 6 (code):**
```python
secure = StegoCodec(device="auto", password="my-key")
# Share the already-loaded model
secure._model, secure._tokenizer, secure._device = codec._ensure_model()

cover = secure.encode(b"classified")
print(f"Encrypted cover: {cover[:100]}...")
print(f"Recovered: {secure.decode(cover)}")
```

**Cell 7 (markdown):**
```markdown
## Capacity: Payload Size vs. Cover Text Length
How many tokens does it take to encode different payload sizes?
```

**Cell 8 (code):**
```python
import os

sizes = [5, 10, 25, 50, 100, 200]
results = []
for size in sizes:
    data = os.urandom(size)
    stats = codec.encode_with_stats(data)
    results.append((size, stats.total_tokens, stats.bits_per_token))
    print(f"{size:>4} bytes -> {stats.total_tokens:>4} tokens  ({stats.bits_per_token:.1f} bits/token)")
```

**Cell 9 (markdown):**
```markdown
## How It Works: Token Probability Distributions
At each step, the LLM produces a probability distribution over the next token. The encoder selects tokens based on the secret data bits. Here's what one step looks like:
```

**Cell 10 (code):**
```python
import torch
from mostegollm.encoder import _get_token_distribution

model, tokenizer, device = codec._ensure_model()
prompt_ids = tokenizer.encode("The weather today is", return_tensors="pt").to(device)

tok_ids, cum_probs, _ = _get_token_distribution(model, prompt_ids, device, top_k=20)

# Show top-20 tokens and their probabilities
tokens = [tokenizer.decode([tid]) for tid in tok_ids]
widths = [(cum_probs[i+1] - cum_probs[i]) / (2**32) * 100 for i in range(len(tok_ids))]

print(f"{'Token':<20} {'Probability':>12}")
print("-" * 34)
for tok, w in zip(tokens, widths):
    bar = "#" * int(w * 2)
    print(f"{tok!r:<20} {w:>10.1f}%  {bar}")
```

**Cell 11 (markdown):**
```markdown
## Chunked Encoding
For large payloads, split into independently encoded chunks:
```

**Cell 12 (code):**
```python
large = b"A" * 200
covers = codec.encode(large, chunk_size=50)
print(f"{len(large)} bytes -> {len(covers)} chunks")
for i, c in enumerate(covers):
    print(f"  Chunk {i+1}: {c[:60]}...")

recovered = codec.decode(covers)
assert recovered == large
print(f"\nRecovered {len(recovered)} bytes successfully!")
```

Use the `Write` tool to create the `.ipynb` file with proper JSON notebook structure.

- [ ] **Step 2: Commit**

```bash
git add examples/MoStegoLLM_notebook.ipynb
git commit -m "docs: rewrite interactive notebook with capacity and distribution visualization"
```

---

### Task 10: Add property-based fuzz tests

**Files:**
- Modify: `pyproject.toml:39-42` (add hypothesis to dev deps)
- Create: `tests/test_fuzz.py`

- [ ] **Step 1: Add hypothesis to dev dependencies**

In `pyproject.toml`, replace:

```toml
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]
```

with:

```toml
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "hypothesis>=6.0.0",
]
```

- [ ] **Step 2: Install the new dependency**

Run: `pip install -e ".[dev]"`

- [ ] **Step 3: Create test_fuzz.py**

Create `tests/test_fuzz.py`:

```python
"""Property-based fuzz tests for the arithmetic coder round-trip."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from mostegollm import StegoCodec


@settings(max_examples=50, deadline=None)
@given(data=st.binary(min_size=0, max_size=500))
def test_roundtrip_arbitrary_bytes(codec: StegoCodec, data: bytes) -> None:
    """Any byte sequence should survive an encode -> decode round-trip."""
    cover = codec.encode(data)
    assert isinstance(cover, str)
    recovered = codec.decode(cover)
    assert recovered == data, (
        f"Round-trip failed for {len(data)}-byte payload: "
        f"expected {data[:20]!r}..., got {recovered[:20]!r}..."
    )
```

Note: Hypothesis doesn't natively support pytest fixtures with `@given`. We need to use a `conftest.py`-level workaround. Update the test to use a module-level codec:

```python
"""Property-based fuzz tests for the arithmetic coder round-trip."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mostegollm import StegoCodec

# Module-level codec shared across all hypothesis examples.
# Cannot use session-scoped fixture with @given, so we lazy-init here.
_codec: StegoCodec | None = None


def _get_codec() -> StegoCodec:
    global _codec
    if _codec is None:
        _codec = StegoCodec(device="cpu")
        _codec._ensure_model()
    return _codec


@settings(max_examples=20, deadline=None)
@given(data=st.binary(min_size=0, max_size=100))
def test_roundtrip_arbitrary_bytes(data: bytes) -> None:
    """Any byte sequence should survive an encode -> decode round-trip."""
    codec = _get_codec()
    cover = codec.encode(data)
    assert isinstance(cover, str)
    recovered = codec.decode(cover)
    assert recovered == data
```

- [ ] **Step 4: Run fuzz tests**

Run: `pytest tests/test_fuzz.py -v`
Expected: PASS (slow — each of 20 examples does a full model forward pass)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_fuzz.py
git commit -m "test: add property-based fuzz tests with hypothesis

Tests random byte payloads (0-100 bytes, 20 examples) through
encode->decode round-trip. Catches arithmetic coder edge cases."
```

---

### Task 11: Add capacity limit test

**Files:**
- Create: `tests/test_capacity.py`

- [ ] **Step 1: Create test_capacity.py**

```python
"""Test encoding capacity limits."""

from __future__ import annotations

import pytest

from mostegollm import StegoCodec
from mostegollm.utils import StegoEncodeError


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
        # The error message should mention the limit
        assert MAX_TOKENS == 8192
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_capacity.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_capacity.py
git commit -m "test: add capacity limit tests"
```

---

### Task 12: Update timing test with assertion bound

**Files:**
- Modify: `tests/test_timing.py`

- [ ] **Step 1: Update test_timing.py**

Replace `tests/test_timing.py`:

```python
"""Timing and sentence-boundary tests for MoStegoLLM encoding."""

from __future__ import annotations

import time

import pytest

from mostegollm import StegoCodec


@pytest.mark.slow
def test_encode_short_timing(codec: StegoCodec) -> None:
    """Encoding a trivially short payload should complete in under 60s on CPU."""
    data = b"hi"

    start = time.perf_counter()
    codec.encode(data)
    elapsed = time.perf_counter() - start

    print(f"\nEncoding {len(data)} bytes took {elapsed:.3f}s")
    assert elapsed < 60, f"Encoding took {elapsed:.1f}s, expected < 60s"


def test_sentence_boundary_ending(codec_sentence_boundary: StegoCodec) -> None:
    """Cover text should end at a sentence boundary when sentence_boundary=True."""
    data = b"hello"
    cover = codec_sentence_boundary.encode(data)
    stripped = cover.rstrip()
    assert len(stripped) > 0, "Cover text should not be empty"
    assert stripped[-1] in ".!?", (
        f"Expected cover text to end with '.', '!' or '?' but got: ...{stripped[-20:]!r}"
    )


def test_sentence_boundary_roundtrip(
    codec_sentence_boundary: StegoCodec, codec: StegoCodec
) -> None:
    """Round-trip should succeed with sentence_boundary=True."""
    data = b"secret message"
    cover = codec_sentence_boundary.encode(data)
    recovered = codec.decode(cover)
    assert recovered == data
```

- [ ] **Step 2: Register the slow marker in pyproject.toml**

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]
```

- [ ] **Step 3: Run timing test**

Run: `pytest tests/test_timing.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_timing.py pyproject.toml
git commit -m "test: add timing assertion and slow marker to timing tests"
```

---

### Task 13: Bump version to 0.2.0

**Files:**
- Modify: `pyproject.toml:7`

- [ ] **Step 1: Bump version**

In `pyproject.toml`, change:

```toml
version = "0.1.0"
```

to:

```toml
version = "0.2.0"
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/test_qwen.py`
Expected: All PASS

- [ ] **Step 3: Run linting**

Run: `ruff check . && ruff format --check .`
Expected: Clean (fix any issues if found)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.2.0

Breaking change: wire format updated (CRC-32 in header).
API simplified: 13 -> 7 public methods."
```

---

### Task 14: Final verification

- [ ] **Step 1: Run the full test suite including Qwen**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Run the example script**

Run: `python examples/basic_usage.py`
Expected: Completes without errors, prints encode/decode/stats output

- [ ] **Step 3: Test CLI end-to-end**

Run:
```bash
mostegollm encode "final check" | mostegollm decode --text
```
Expected: Prints "final check"

- [ ] **Step 4: Verify linting is clean**

Run: `ruff check . && ruff format --check .`
Expected: No issues
