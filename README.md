# MoStegoLLM

Hide binary data inside natural English prose using LLM-powered steganography.

## Installation

```bash
pip install mostegollm
```

Requires PyTorch. For GPU support:

```bash
pip install mostegollm torch --index-url https://download.pytorch.org/whl/cu128
```

## Quick Start

```python
from mostegollm import StegoCodec

codec = StegoCodec()
cover = codec.encode_str("my secret message")
print(cover)  # natural English prose
print(codec.decode_str(cover))  # "my secret message"
```

## CLI Quick Start

Encode a string:

```bash
mostegollm encode "my secret message"
```

Decode cover text:

```bash
mostegollm decode --text "The research findings suggest..."
```

Encode with encryption:

```bash
mostegollm encode --password "hunter2" "my secret message"
```

Encode a binary file:

```bash
mostegollm encode -f secret.bin -o cover.txt
```

List available models:

```bash
mostegollm models
```

## How It Works

MoStegoLLM hides arbitrary binary data inside natural-looking English text using
arithmetic coding over a language model's token probability distributions. At each
step in the generation, the model produces a ranked probability distribution over its
vocabulary. The encoder partitions the [0, 1) interval among the top-k tokens in
proportion to their probabilities, then uses the secret bits to select which token to
emit — exactly as in range coding. The decoder, given the same prompt and model
parameters, reconstructs the same distributions and reads back the bit decisions from
each token choice.

The payload is prefixed with a compact header containing a magic marker and a 4-byte
length field. This header lets the decoder know exactly how many bytes to extract and
validates that the text was produced by MoStegoLLM. Payload bits are combined with the
header and fed through the arithmetic coder to produce a stream of tokens that, when
detokenized, reads as ordinary English prose.

```
Secret bytes
  --> [Header: magic + length] + payload bits
  --> Arithmetic coding over LLM token distributions
  --> Natural English prose
```

## Capacity Guide

Encoding capacity depends on the model's entropy per token. The figures below are
approximate for the default model (SmolLM-135M) at default settings.

| Payload size | Tokens generated | Approximate word count |
|-------------|-----------------|----------------------|
| 10 bytes    | ~15 tokens      | ~50 words            |
| 100 bytes   | ~130 tokens     | ~400 words           |
| 1 KB        | ~1300 tokens    | ~4000 words          |

Use `encode_with_stats()` for exact measurements.

## API Reference

### `StegoCodec`

```python
StegoCodec(
    model_name=...,
    device="auto",
    prompt="",
    top_k=256,
    temperature=1.0,
    sentence_boundary=False,
    token=None,
    password=None,
)
```

Constructor — creates a codec instance. The model is loaded lazily on first use.

---

#### `encode(data, *, chunk_size=None, context_size=500) -> str | list[str]`

Encode binary data into steganographic cover text. When `chunk_size` is set, splits
`data` into chunks and returns a `list[str]`; otherwise returns a single `str`.

---

#### `decode(cover_text, *, context_size=500) -> bytes`

Decode cover text back to the original bytes. Accepts a single string or a list of
strings produced by chunked encoding.

---

#### `encode_str(text, encoding="utf-8") -> str`

Convenience wrapper — encodes a string by converting it to bytes first.

---

#### `decode_str(cover_text, encoding="utf-8") -> str`

Convenience wrapper — decodes cover text and returns the result as a string.

---

#### `encode_with_stats(data) -> StegoStats`

Encode data and return a `StegoStats` object containing the cover text plus encoding
diagnostics: `cover_text`, `bits_per_token`, `total_tokens`, `payload_size_bytes`.

---

#### `StegoCodec.list_models()` (classmethod)

Return a tuple of `ModelInfo` objects describing all models in the registry.

---

#### `StegoCodec.get_model_info(name)` (classmethod)

Look up a model by its HuggingFace identifier. Returns `None` if not in the registry.

---

## Configuration

| Parameter          | Default                        | Description                                                                      |
|--------------------|-------------------------------|----------------------------------------------------------------------------------|
| `model_name`       | `HuggingFaceTB/SmolLM-135M`   | HuggingFace model identifier                                                     |
| `device`           | `"auto"`                      | PyTorch device (`"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"`, …)                    |
| `prompt`           | `""`                          | Seed text prepended to every generation; encoder and decoder must match          |
| `top_k`            | `256`                         | Number of most-probable tokens considered at each step                           |
| `temperature`      | `1.0`                         | Softmax temperature (`1.0` = unmodified); encoder and decoder must match         |
| `sentence_boundary`| `False`                       | Continue generating past data-end until cover text ends at `.`, `!`, or `?`     |
| `token`            | `None`                        | HuggingFace API token for gated models; falls back to `HF_TOKEN` env var         |
| `password`         | `None`                        | Password for AES-256-GCM encryption applied before encoding / after decoding     |

## Security Model

**Steganographic hiding is not the same as encryption.** Without a password, the
existence of hidden data can be detected by anyone who runs the same model with the
same prompt and compares token probabilities to expected distributions. The cover text
looks like natural prose, but a motivated adversary with access to the model can
recover the payload.

**Password mode** adds a confidentiality layer: when `password` is supplied, the
payload is encrypted with AES-256-GCM before encoding and decrypted after decoding.
Key derivation uses PBKDF2-HMAC-SHA256 with 600,000 iterations and a random salt. An
attacker who obtains the cover text cannot recover the payload without the password.

**Cross-platform compatibility:** As of 0.3.0, decoding no longer requires the
same PyTorch version or device type as encoding — the coder assigns
arithmetic-coding intervals by token *rank* (a quantity stable across
floating-point regimes) rather than by probability magnitude. You still need
the **same model weights**. Any rare residual divergence is caught by the
CRC-32 integrity check, so a mismatch fails loudly rather than returning corrupt
data.

### Verified portability

Integer rank-interval coding makes cover text portable **across devices and
PyTorch versions at a fixed model dtype** — but **not across dtypes**. A live
Modal matrix (`compat/`, see `compat/results/smoke-matrix.md`) measured this:

| | cpu-fp32 | t4-fp32 | t4-fp16 |
|---|---|---|---|
| **cpu-fp32** | ✓ | ✓ | ✗ |
| **t4-fp32** | ✓ | ✓ | ✗ |
| **t4-fp16** | ✗ | ✗ | ✓ |

CPU↔GPU at fp32 round-trips perfectly (100% top-k ordering agreement); fp16 ↔
fp32 fails (0% agreement) because fp16's ~1e-3 quantization error is the same
order as the `GUARD=1e-3` merge threshold, flipping token ordering. **Encode and
decode must use the same model dtype.**

A committed corpus of reference test vectors (`compat/golden_vectors.jsonl`) is
decoded on every test run as a regression guard (`tests/test_golden_vectors.py`);
regenerate with `python -m compat.generate_golden`. The cloud matrix runs via
`modal run -m compat.modal_app`. **Apple Silicon / MPS and AMD ROCm are not yet
in the automated matrix.**

**Integrity:** Each encoded message includes a CRC-32 checksum in the header for basic
integrity validation. This is not a cryptographic MAC and does not protect against
deliberate tampering.

## Supported Models

| Model                                  | Parameters | Notes                              |
|----------------------------------------|------------|------------------------------------|
| `HuggingFaceTB/SmolLM-135M`           | 135M       | Default; fast and lightweight      |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B       | Fallback if default is unavailable |
| `HuggingFaceTB/SmolLM-360M`           | 360M       | Better prose quality               |
| `Qwen/Qwen2.5-0.5B`                   | 0.5B       | Compact multilingual model         |
| `meta-llama/Llama-3.2-1B`            | 1B         | High quality; requires `HF_TOKEN`  |

## Acknowledgments

This project uses [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) by
Hugging Face, licensed under the
[Apache License 2.0](LICENSES/Apache-2.0.txt). See the [NOTICE](NOTICE) file for
details.

## License

MIT
