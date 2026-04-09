# MoStegoLLM Usability Enhancements — Design Spec

**Date:** 2026-04-09
**Goal:** Improve usability and adoption for developers, CLI users, and researchers, with critical correctness fixes woven in.
**Approach:** Outside-in — simplify the API, then document, polish CLI, fix correctness, update examples, fill test gaps. Each tier builds on the previous.

---

## Tier 1: API Simplification

**Problem:** The current `StegoCodec` exposes 13 public methods. Several are trivial wrappers that add surface area without value.

**Changes:**

### Remove (4 methods)

| Method | Reason |
|--------|--------|
| `encode_long_str` | Trivial: `encode_long(text.encode())` |
| `decode_long_str` | Trivial: `decode_long(covers).decode()` |
| `encode_file` | Trivial: `encode(Path(p).read_bytes())`. Belongs in CLI, not library. |
| `decode_file` | Trivial: `Path(p).write_bytes(decode(cover))`. Belongs in CLI, not library. |

### Merge chunking into core methods (absorbs 2 methods)

`encode` gains an optional `chunk_size` parameter:

```python
def encode(
    self,
    data: bytes,
    *,
    chunk_size: int | None = None,
    context_size: int = 500,
) -> str | list[str]:
```

- `chunk_size=None` (default): returns `str` — current behavior.
- `chunk_size=N`: splits data into N-byte chunks, returns `list[str]` — replaces `encode_long`.

`decode` accepts both forms:

```python
def decode(
    self,
    cover_text: str | list[str],
    *,
    context_size: int = 500,
) -> bytes:
```

- `str` input: current behavior.
- `list[str]` input: decodes each chunk and concatenates — replaces `decode_long`.

This eliminates `encode_long` and `decode_long` as separate methods.

### Resulting API (7 methods)

| Method | Purpose |
|--------|---------|
| `encode(data, *, chunk_size, context_size)` | bytes → cover text (single or chunked) |
| `decode(cover_text, *, context_size)` | cover text → bytes (single or chunked) |
| `encode_str(text, encoding)` | string convenience for encode |
| `decode_str(cover_text, encoding)` | string convenience for decode |
| `encode_with_stats(data)` | encode + return `StegoStats` |
| `list_models()` | classmethod: model registry listing |
| `get_model_info(name)` | classmethod: model lookup |

### Migration

The removed methods were added in 0.1.0 and have no external dependents. No deprecation period needed — remove in the same release that adds the consolidated API.

---

## Tier 2: README & Documentation

**Problem:** The README is 9 lines. No installation instructions, no usage examples, no security model, no capacity guidance.

**README.md structure (single source of truth, no separate docs site):**

1. **One-line description** — "Hide binary data inside natural English prose using LLM-powered steganography"
2. **Quick install** — `pip install mostegollm`, note torch/CUDA optional
3. **30-second example** — encode a string, decode it, 4 lines of Python
4. **CLI quick start** — `mostegollm encode "secret"` / `mostegollm decode cover.txt`
5. **How it works** — 1-2 paragraph plain-English explanation of arithmetic coding over token distributions. Text-based diagram showing: `secret bytes → header + payload bits → arithmetic coding selects tokens → cover text`. No image dependencies.
6. **Capacity guide** — table mapping payload sizes to approximate cover text length at default settings (top_k=256, ~6-8 bits/token). Example rows:
   - 10 bytes → ~15 tokens → ~50 words
   - 100 bytes → ~130 tokens → ~400 words
   - 1 KB → ~1300 tokens → ~4000 words
   Values derived empirically from `encode_with_stats` on representative payloads.
7. **API reference** — the 7 methods from Tier 1, each with signature and one-line description
8. **Configuration** — model selection (`model_name`), encryption (`password`), custom prompts, top-k, temperature, sentence boundary
9. **Security model** — explicit about:
   - Steganographic hiding ≠ encryption (without `password`, data is obscured but not confidential)
   - `password` mode: AES-256-GCM, PBKDF2 600K iterations
   - Cross-platform caveat: encoder and decoder must use the same PyTorch version, device type, and model
   - No payload integrity check in plain mode (CRC-32 added in Tier 4, but not a cryptographic guarantee)
10. **Supported models** — table from `MODEL_REGISTRY` (name, parameters, notes)
11. **License** — MIT, with Apache-2.0 attribution for SmolLM

---

## Tier 3: CLI Polish

**Problem:** CLI lacks encryption support, text decode mode, stats output, and has bare exception messages.

### New flags

| Flag | Command | Behavior |
|------|---------|----------|
| `--text` / `-t` | `decode` | Decode and print as UTF-8 instead of raw bytes |
| `--password` / `-p` | `encode`, `decode` | Enable AES-256-GCM encryption/decryption |
| `--chunk-size` | `encode` | Split large payloads into chunks (pairs with Tier 1 API) |
| `--stats` | `encode` | Print bits/token, total tokens, payload size to stderr |
| `--quiet` / `-q` | global | Suppress model loading progress for scripting/piping |

### Error message improvements

Wrap exceptions in user-friendly messages:

- `StegoDecodeError` → "Error: could not decode — wrong model or corrupted text. Use --verbose for details."
- `StegoModelError` → "Error: could not load model '<name>'. Check your internet connection or try --model cpu-friendly-alternative."
- `StegoCryptoError` → "Error: decryption failed — wrong password or tampered data."

Verbose mode (`-v`) prints the full traceback.

### Not adding

- Interactive mode, TUI, or password prompts (keep pipe-friendly)
- Shell completion (low ROI at this stage)

---

## Tier 4: Critical Correctness Fixes

**Problem:** Three small but high-risk issues identified in the architecture review.

### 4a. Payload CRC-32 in plain mode

**Current header:** `magic (2 bytes) + length (4 bytes)` = 6 bytes = 48 bits

**New header:** `magic (2) + length (4) + crc32 (4)` = 10 bytes = 80 bits

The CRC-32 is computed over the payload bytes as they are arithmetic-coded (i.e., post-encryption if `password` is set). This means the decoder can verify integrity without needing the password — it checks the CRC immediately after extracting bits, before attempting decryption. On decode:
- Extract payload from arithmetic-coded bits
- Compute CRC-32 of extracted payload
- Compare against header value
- Raise `StegoDecodeError("Payload integrity check failed — data may be corrupted")` on mismatch

When `password` is set, the GCM authentication tag provides additional cryptographic integrity during the subsequent decryption step.

**Impact on existing encoded text:** This is a breaking change to the wire format. Existing cover text encoded with the old 48-bit header will not decode with the new codec. This is acceptable at version 0.1.0 — no stability guarantees yet. Bump to 0.2.0 on release.

**Files changed:** `utils.py` (header pack/unpack, `HEADER_SIZE`/`HEADER_BITS` constants), `encoder.py` (pass CRC to header), `decoder.py` (verify CRC after decode).

### 4b. assert → StegoDecodeError in decoder.py

**Current (`decoder.py:189`):**
```python
assert len(payload) == payload_length, (
    f"Internal error: expected {payload_length} bytes, got {len(payload)}"
)
```

**New:**
```python
if len(payload) != payload_length:
    raise StegoDecodeError(
        f"Internal error: expected {payload_length} bytes, got {len(payload)}"
    )
```

One-line change. Ensures the check survives `python -O`.

### 4c. Seed phrase prefix validation at import time

Add to `seeds.py`, at module level after the `SEED_PHRASES` list:

```python
# Validate no phrase is a prefix of another (required for unambiguous matching).
for i, a in enumerate(SEED_PHRASES):
    for b in SEED_PHRASES[i + 1:]:
        if a.startswith(b) or b.startswith(a):
            raise AssertionError(f"Seed phrase prefix collision: {a!r} / {b!r}")
```

Runs once at import. O(n^2) over 256 phrases = 32K comparisons — negligible.

---

## Tier 5: Examples & Guides

**Problem:** Examples need updating after API changes. Researcher audience needs a deeper "how it works" demonstration.

### Rewrite `examples/basic_usage.py`

Sections:
1. Basic encode/decode (bytes)
2. String convenience (`encode_str` / `decode_str`)
3. Encryption with password
4. Chunked encoding via `chunk_size` parameter
5. Stats inspection via `encode_with_stats`

Inline comments explain each step. ~60-80 lines.

### Rewrite `examples/MoStegoLLM_notebook.ipynb`

Same structure as `basic_usage.py` but with:
- Markdown cells explaining the "why" between code cells
- A cell showing capacity: encode payloads of varying sizes, plot payload bytes vs. cover text tokens
- A cell visualizing token probability distributions for one encoding step — bar chart of top-20 token probabilities with the chosen token highlighted. This is the "aha moment" for understanding arithmetic coding steganography.

### Add `examples/cli_walkthrough.sh`

Shell script demonstrating:
```bash
# Encode a text message
mostegollm encode "attack at dawn" -o cover.txt

# Decode it
mostegollm decode cover.txt --text

# Encode with encryption
mostegollm encode "classified" --password mysecret -o encrypted_cover.txt

# Decode with encryption
mostegollm decode encrypted_cover.txt --text --password mysecret

# Encode a file
mostegollm encode -f secret.bin -o cover.txt --stats

# List available models
mostegollm models
```

---

## Tier 6: Test Gaps

**Problem:** CLI has zero test coverage. No property-based testing. Capacity limit untested.

### 6a. CLI tests (`tests/test_cli.py`)

Test via `subprocess.run` against the `mostegollm` entry point:
- Basic round-trip: `encode` piped to `decode`
- `--text` flag: decode outputs UTF-8 string
- `--password` flag: encrypted round-trip
- `--stats` flag: stderr contains "bits/token", "tokens", "payload"
- `--quiet` flag: stderr is empty (no model loading noise)
- Error cases: decode garbage text, missing file, wrong password
- `models` subcommand: outputs table, exit code 0

### 6b. Property-based fuzzing (`tests/test_fuzz.py`)

Add `hypothesis` to `[dev]` dependencies.

```python
@given(st.binary(min_size=0, max_size=500))
def test_roundtrip_arbitrary_bytes(codec, data):
    cover = codec.encode(data)
    assert codec.decode(cover) == data
```

Use `@settings(max_examples=50)` to keep CI time reasonable (each example requires a full model forward pass). Targets the arithmetic coder's interval boundary conditions.

### 6c. Capacity limit test (`tests/test_capacity.py`)

Encode a payload large enough to approach `MAX_TOKENS` (8192). With ~8 bits/token, that's ~8 KB. Test that:
- A payload near the limit succeeds
- A payload over the limit raises `StegoEncodeError` with a message containing "maximum token limit"

### 6d. Timing assertion (optional)

Convert `test_timing.py` to assert `elapsed < 60` seconds for the 2-byte payload on CPU. Use `pytest.mark.slow` so it can be skipped in fast CI runs. Skip this if CI environment has unpredictable performance.

---

## Execution Order

Tiers are sequential — each builds on the previous:

```
Tier 1 (API) → Tier 2 (README) → Tier 3 (CLI) → Tier 4 (Correctness) → Tier 5 (Examples) → Tier 6 (Tests)
```

**Rationale:**
- Tier 1 before Tier 2: don't document an API you're about to change
- Tier 2 before Tier 3: README establishes the vocabulary and examples the CLI section references
- Tier 3 before Tier 4: CLI error messages reference the correctness guarantees (e.g., "payload integrity check failed")
- Tier 4 before Tier 5: examples should demonstrate the current behavior (including CRC)
- Tier 6 last: tests validate everything that came before

**Version bump:** Release as 0.2.0 after all tiers complete. The header format change (Tier 4a) is a breaking change to the wire format.

---

## Out of Scope

These are real issues identified in the review but are too large or speculative for this phase:

- **Cross-platform determinism** — needs its own design (environment manifest, token ID serialization, version pinning)
- **Streaming/async API** — capability expansion, not usability
- **Steganalysis resistance** — research-grade feature, different audience
- **Thread safety** — no demand signal yet
- **Docs site** — premature at 0.1.0 scale
