# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MoStegoLLM is a steganographic text encoding/decoding library that hides arbitrary binary data inside coherent English prose using arithmetic coding over LLM token probability distributions.

## Commands

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_roundtrip.py

# Run a single test by name
pytest tests/ -k "test_short_string_roundtrip"

# Tests with coverage
pytest tests/ --cov=src

# Lint and format (ruff, configured in pyproject.toml)
ruff check .
ruff format .
```

## Architecture

The library implements arithmetic coding-based steganography over LLM token distributions. Data flows as:

**Encoding:** secret bytes → 80-bit / 10-byte header (magic `0x5354` + 4-byte length + 4-byte CRC-32) + payload bits → arithmetic coding selects tokens from LLM probability distribution → cover text

**Decoding:** cover text → tokenize → reconstruct same probability distributions → extract bits from token choices → validate header → verify CRC-32 → recover original bytes

### Key modules (`src/mostegollm/`)

- **`codec.py`** — `StegoCodec`: public API, lazy-loads model on first use. Wraps encoder/decoder with convenience methods (`encode`, `decode`, `encode_str`, `decode_str`, `encode_with_stats`). Optional AES-256-GCM encryption (`password=`), seed-phrase mode (default empty prompt), and chunked encoding (`chunk_size=`, returns/accepts `list[str]`).
- **`encoder.py`** — 32-bit precision arithmetic coding that maps secret bits to token selections from top-k filtered distributions. Handles interval narrowing and MSB renormalization. BPE round-trip filtering (`get_non_roundtrip_tokens`, `_filter_distribution`) so re-tokenization is stable. Beyond the payload, `next_bit` feeds a deterministic xorshift32 stream (not zeros) to keep generation natural — critical for `sentence_boundary`.
- **`decoder.py`** — Reverse of encoder: reconstructs probability intervals, determines which bits correspond to each token choice, validates header, verifies CRC-32.
- **`model.py`** — Model loading with fallback (primary: `HuggingFaceTB/SmolLM-135M`, fallback: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`). Sets determinism via `torch.manual_seed(0)`. Exposes a model registry (`list_models`, `get_model_info`).
- **`crypto.py`** — AES-256-GCM payload encryption with PBKDF2-HMAC-SHA256 key derivation (600k iterations); blob layout `salt(16) || nonce(12) || ciphertext || tag(16)`.
- **`seeds.py`** — 256 prefix-free seed phrases. In default (empty-prompt) mode a phrase is chosen deterministically from `sha256(data)` and prepended so the decoder can recover the prompt (`select_seed`, `match_seed`).
- **`cli.py`** — `mostegollm` console entry point (`encode`/`decode`/`models`). Global flags use `argparse.SUPPRESS` defaults so they work before *or* after the subcommand; chunked output is joined/split on `CHUNK_SEPARATOR`.
- **`utils.py`** — Exception hierarchy (`StegoError` → `StegoEncodeError`/`StegoDecodeError`/`StegoModelError`/`StegoCryptoError`), header pack/unpack, bit conversion helpers, `StegoStats` dataclass.

### Critical implementation details

- Encoder and decoder must produce **identical** probability distributions for round-trip correctness. Any change to top-k filtering, temperature, or renormalization must be mirrored in both.
- Uses **float64** for probability calculations before converting to integer intervals.
- Top-k default is 256; each token gets at least 1-unit width to prevent zero-width intervals.
- Tests use a **session-scoped** fixture (`conftest.py`) to load the model once across all tests.

## Code style

- Python >=3.10, `from __future__ import annotations` throughout
- Line length: 100 (ruff)
- Google-style docstrings
- Full type annotations (PEP 561 `py.typed` marker present)
