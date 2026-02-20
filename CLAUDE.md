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

**Encoding:** secret bytes → 48-bit header (magic `0x5354` + 4-byte length) + payload bits → arithmetic coding selects tokens from LLM probability distribution → cover text

**Decoding:** cover text → tokenize → reconstruct same probability distributions → extract bits from token choices → validate header → recover original bytes

### Key modules (`src/mostegollm/`)

- **`codec.py`** — `StegoCodec`: public API, lazy-loads model on first use. Wraps encoder/decoder with convenience methods (`encode`, `decode`, `encode_str`, `decode_str`, `encode_file`, `decode_file`).
- **`encoder.py`** — 32-bit precision arithmetic coding that maps secret bits to token selections from top-k filtered distributions. Handles interval narrowing and MSB renormalization.
- **`decoder.py`** — Reverse of encoder: reconstructs probability intervals, determines which bits correspond to each token choice, validates header.
- **`model.py`** — Model loading with fallback (primary: `HuggingFaceTB/SmolLM-135M`, fallback: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`). Sets determinism via `torch.manual_seed(0)`.
- **`utils.py`** — Exception hierarchy (`StegoError` → `StegoEncodeError`/`StegoDecodeError`/`StegoModelError`), header pack/unpack, bit conversion helpers, `StegoStats` dataclass.

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
