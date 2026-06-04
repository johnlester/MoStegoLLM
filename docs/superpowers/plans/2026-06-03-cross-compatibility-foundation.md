# Cross-Compatibility Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the locally-testable foundation for cross-compatibility testing — a portable "test vector" primitive, a classifying verifier, a deterministic golden corpus, and an offline regression guard that runs in every `pytest` invocation.

**Architecture:** A new shipped module `src/mostegollm/compat.py` defines the `TestVector` record (encode→cover_text + the metadata a different environment needs to decode it), JSONL IO, `make_vector` (encode + capture env), and `verify_vector` (decode + classify any failure as re-tokenization drift, logit divergence, or load error). A top-level `compat/` package holds the deterministic canonical payloads and a script that generates the committed `compat/golden_vectors.jsonl`. `tests/test_golden_vectors.py` decodes that corpus on every test run.

**Tech Stack:** Python ≥3.10, PyTorch, HuggingFace `transformers`/`tokenizers`, `pytest`. Reuses `mostegollm.encoder.encode`, `mostegollm.decoder.decode`, `mostegollm.crypto`.

**Scope note:** This is **Plan 1 (foundation)**. The cloud-orchestration layer from the spec — Modal/RunPod cells, the N×N prove run, `dump_step_logits` + the GUARD-margin report, `floor_check.py` — is **Plan 2** and builds on the primitives here. Everything in this plan is verifiable on a single machine with no cloud.

**Spec:** `docs/superpowers/specs/2026-06-03-cloud-cross-compatibility-testing-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/mostegollm/compat.py` (create) | `TestVector`, `VerifyResult`, `FailureClass`; JSONL IO; env capture; `make_vector`; `verify_vector`. Shipped + importable in any environment. |
| `tests/test_compat.py` (create) | Unit tests: serialization round-trip, env capture, make→verify happy path, failure classification, encrypted path. Uses the session `codec` fixture. |
| `compat/__init__.py` (create) | Marks `compat/` a package (for `python -m compat.generate_golden`). |
| `compat/payloads.py` (create) | Deterministic `CANONICAL_PAYLOADS` + `CANONICAL_PROMPTS`. No `os.urandom`. |
| `compat/generate_golden.py` (create) | Reference-encodes the canonical set → writes `compat/golden_vectors.jsonl`. |
| `compat/golden_vectors.jsonl` (generated, committed) | The reference encoder's output; the offline guard's input. |
| `tests/test_golden_vectors.py` (create) | Offline regression guard: decode every golden vector, assert recovery. |
| `README.md` (modify) | Document the portability envelope + Apple-MPS gap (brief). |

---

## Task 1: `TestVector` record + JSONL IO

**Files:**
- Create: `src/mostegollm/compat.py`
- Test: `tests/test_compat.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compat.py
"""Tests for the cross-compatibility test-vector layer."""

from __future__ import annotations

from pathlib import Path

from mostegollm.compat import TestVector, read_vectors, write_vectors


def _sample_vector() -> TestVector:
    return TestVector(
        schema="mostegollm-testvector/1",
        library_version="0.3.0",
        model="HuggingFaceTB/SmolLM-135M",
        model_revision="abc123",
        prompt="According to experts,",
        settings={"top_k": 256, "temperature": 1.0, "sentence_boundary": False, "password": None},
        payload_sha256="00" * 32,
        payload_hex="68656c6c6f",
        cover_text="Some cover text with unicode café 🙂.",
        generated_token_ids=[1, 2, 3, 4],
        source_env={"torch": "2.5.1", "device": "cpu", "os": "Linux", "arch": "x86_64"},
    )


def test_testvector_json_round_trip():
    v = _sample_vector()
    restored = TestVector.from_json(v.to_json())
    assert restored == v


def test_write_read_vectors_round_trip(tmp_path: Path):
    vectors = [_sample_vector(), _sample_vector()]
    path = tmp_path / "vectors.jsonl"
    write_vectors(vectors, path)
    loaded = read_vectors(path)
    assert loaded == vectors
    # JSONL: one object per line, blank lines tolerated on read.
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compat.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mostegollm.compat'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mostegollm/compat.py
"""Cross-compatibility test vectors: encode here, decode (and verify) elsewhere.

A TestVector is the only artifact that crosses an environment boundary. It carries
the cover text plus everything a *different* PyTorch/device/dtype needs to decode
it and check the result. See
docs/superpowers/specs/2026-06-03-cloud-cross-compatibility-testing-design.md
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from os import PathLike

SCHEMA = "mostegollm-testvector/1"


@dataclass(frozen=True)
class TestVector:
    """A self-describing encode result, portable across environments.

    Attributes:
        payload_sha256 / payload_hex: of the *plaintext* payload to recover
            (for encrypted vectors the cover encodes the ciphertext, but the
            recovered/expected value is the plaintext).
        generated_token_ids: the encoder's token IDs; enables the
            re-tokenization-drift check on decode.
        model_revision: HF commit sha pinning vocab/merges (may be None).
    """

    schema: str
    library_version: str
    model: str
    model_revision: str | None
    prompt: str
    settings: dict
    payload_sha256: str
    payload_hex: str
    cover_text: str
    generated_token_ids: list[int]
    source_env: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "TestVector":
        return cls(**json.loads(line))


def write_vectors(vectors: list[TestVector], path: str | PathLike) -> None:
    """Write vectors as JSONL (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for v in vectors:
            f.write(v.to_json() + "\n")


def read_vectors(path: str | PathLike) -> list[TestVector]:
    """Read a JSONL vector file; blank lines are skipped."""
    out: list[TestVector] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(TestVector.from_json(line))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compat.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mostegollm/compat.py tests/test_compat.py
git commit -m "feat(compat): TestVector record + JSONL IO"
```

---

## Task 2: Environment-capture helpers

**Files:**
- Modify: `src/mostegollm/compat.py`
- Test: `tests/test_compat.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_compat.py
from mostegollm.compat import current_env, lib_version, model_commit_hash


def test_current_env_has_required_keys():
    env = current_env("cpu")
    for key in ("torch", "transformers", "tokenizers", "device", "os", "arch"):
        assert key in env and isinstance(env[key], str)
    assert env["device"] == "cpu"


def test_lib_version_is_string():
    v = lib_version()
    assert isinstance(v, str) and v  # non-empty


def test_model_commit_hash_handles_missing(codec):
    # A real model may or may not expose a commit hash; the helper must never raise.
    model, _tok, _dev = codec._ensure_model()
    result = model_commit_hash(model)
    assert result is None or isinstance(result, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compat.py -k "env or lib_version or commit_hash" -v`
Expected: FAIL — `ImportError: cannot import name 'current_env'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/mostegollm/compat.py (after the imports, before TestVector)
import platform

import torch


def current_env(device: object) -> dict[str, str]:
    """Capture the identifying facts of the current runtime environment."""
    import tokenizers
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "device": str(device),
        "os": platform.system(),
        "arch": platform.machine(),
    }


def lib_version() -> str:
    """Installed mostegollm version, or 'unknown' if not resolvable."""
    try:
        from importlib.metadata import version

        return version("mostegollm")
    except Exception:
        return "unknown"


def model_commit_hash(model: object) -> str | None:
    """Best-effort HF commit sha for the loaded model (None if unavailable)."""
    return getattr(getattr(model, "config", None), "_commit_hash", None)
```

Note: `import torch` / `import platform` go at the top of the file with the other
imports; shown here inline for clarity.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compat.py -k "env or lib_version or commit_hash" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mostegollm/compat.py tests/test_compat.py
git commit -m "feat(compat): environment-capture helpers"
```

---

## Task 3: `make_vector` + `verify_vector` happy path

**Files:**
- Modify: `src/mostegollm/compat.py`
- Test: `tests/test_compat.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_compat.py
import pytest

from mostegollm.compat import make_vector, verify_vector


@pytest.mark.parametrize(
    "payload",
    [b"hello", b"\x42", b"\x00\x01\x02\x03\xfe\xff", b"The quick brown fox."],
)
def test_make_then_verify_round_trips(codec, payload):
    model, tok, dev = codec._ensure_model()
    vector = make_vector(
        payload,
        model=model,
        tokenizer=tok,
        device=dev,
        prompt="According to experts,",
        model_name=codec._model_name,
    )
    # The vector is self-consistent: re-tokenizing the cover reproduces the IDs.
    assert tok.encode(vector.cover_text, add_special_tokens=False) == vector.generated_token_ids
    result = verify_vector(vector, model=model, tokenizer=tok, device=dev)
    assert result.ok, result.detail
    assert result.failure_class is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compat.py -k make_then_verify -v`
Expected: FAIL — `ImportError: cannot import name 'make_vector'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/mostegollm/compat.py
import hashlib
from dataclasses import dataclass  # already imported; ensure present
from enum import Enum

from .crypto import decrypt as _decrypt
from .crypto import encrypt as _encrypt
from .decoder import decode as _decode
from .encoder import encode as _encode
from .utils import StegoCryptoError, StegoDecodeError

_DEFAULT_SETTINGS = {"top_k": 256, "temperature": 1.0, "sentence_boundary": False}


class FailureClass(str, Enum):
    RETOK_DRIFT = "retokenization_drift"
    LOGIT_DIVERGENCE = "logit_divergence"
    LOAD_ERROR = "load_error"


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    failure_class: FailureClass | None
    detail: str
    decoder_env: dict


def make_vector(
    payload: bytes,
    *,
    model: object,
    tokenizer: object,
    device: object,
    prompt: str,
    model_name: str,
    model_revision: str | None = None,
    settings: dict | None = None,
    password: str | None = None,
) -> TestVector:
    """Encode *payload* and capture everything needed to decode it elsewhere."""
    s = {**_DEFAULT_SETTINGS, **(settings or {})}
    to_encode = _encrypt(payload, password) if password else payload
    cover, ids, _bits = _encode(
        to_encode,
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        top_k=s["top_k"],
        temperature=s["temperature"],
        sentence_boundary=s["sentence_boundary"],
    )
    return TestVector(
        schema=SCHEMA,
        library_version=lib_version(),
        model=model_name,
        model_revision=model_revision if model_revision is not None else model_commit_hash(model),
        prompt=prompt,
        settings={**s, "password": password},
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_hex=payload.hex(),
        cover_text=cover,
        generated_token_ids=list(ids),
        source_env=current_env(device),
    )


def verify_vector(
    vector: TestVector,
    *,
    model: object,
    tokenizer: object,
    device: object,
) -> VerifyResult:
    """Decode *vector* in the current environment and classify any failure."""
    env = current_env(device)

    # (1) Single-version invariant: the cover must re-tokenize to the IDs the
    # encoder generated. A mismatch is re-tokenization drift, not a coder bug.
    retok = tokenizer.encode(vector.cover_text, add_special_tokens=False)
    if retok != vector.generated_token_ids:
        return VerifyResult(
            False,
            FailureClass.RETOK_DRIFT,
            f"re-tokenization mismatch: {len(retok)} vs {len(vector.generated_token_ids)} tokens",
            env,
        )

    # (2) Decode + integrity. Re-tokenization matched, so any desync here is the
    # coder diverging (logit ordering past the GUARD margin).
    password = vector.settings.get("password")
    try:
        decoded = _decode(
            vector.cover_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=vector.prompt,
            top_k=vector.settings.get("top_k", 256),
            temperature=vector.settings.get("temperature", 1.0),
        )
        if password:
            decoded = _decrypt(decoded, password)
    except (StegoDecodeError, StegoCryptoError) as exc:
        return VerifyResult(
            False,
            FailureClass.LOGIT_DIVERGENCE,
            f"decode failed though re-tokenization matched: {exc}",
            env,
        )

    if hashlib.sha256(decoded).hexdigest() != vector.payload_sha256:
        return VerifyResult(
            False,
            FailureClass.LOGIT_DIVERGENCE,
            "decoded payload hash mismatch (silent desync)",
            env,
        )
    return VerifyResult(True, None, "ok", env)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compat.py -k make_then_verify -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mostegollm/compat.py tests/test_compat.py
git commit -m "feat(compat): make_vector + classifying verify_vector"
```

---

## Task 4: Failure classification

**Files:**
- Test: `tests/test_compat.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_compat.py
import dataclasses

from mostegollm.compat import FailureClass


def test_verify_detects_retokenization_drift(codec):
    model, tok, dev = codec._ensure_model()
    good = make_vector(b"hello", model=model, tokenizer=tok, device=dev,
                       prompt="According to experts,", model_name=codec._model_name)
    # Corrupt the recorded IDs so re-tokenizing the cover no longer matches.
    bad = dataclasses.replace(good, generated_token_ids=good.generated_token_ids + [999999])
    result = verify_vector(bad, model=model, tokenizer=tok, device=dev)
    assert not result.ok
    assert result.failure_class == FailureClass.RETOK_DRIFT


def test_verify_detects_payload_mismatch(codec):
    model, tok, dev = codec._ensure_model()
    good = make_vector(b"hello", model=model, tokenizer=tok, device=dev,
                       prompt="According to experts,", model_name=codec._model_name)
    # Cover + IDs are valid (re-tokenization passes) but the expected hash is wrong:
    # this is exactly how a silent coder desync would present.
    bad = dataclasses.replace(good, payload_sha256="de" * 32)
    result = verify_vector(bad, model=model, tokenizer=tok, device=dev)
    assert not result.ok
    assert result.failure_class == FailureClass.LOGIT_DIVERGENCE
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `pytest tests/test_compat.py -k "drift or payload_mismatch" -v`
Expected: PASS — the classification logic from Task 3 already handles both branches. (If either fails, fix `verify_vector`, not the test.)

- [ ] **Step 3: No new implementation needed**

These tests pin the behavior added in Task 3. If they pass, proceed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_compat.py
git commit -m "test(compat): pin failure classification branches"
```

---

## Task 5: Encrypted-vector round trip

**Files:**
- Test: `tests/test_compat.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_compat.py
def test_encrypted_vector_round_trips(codec):
    model, tok, dev = codec._ensure_model()
    vector = make_vector(b"secret payload", model=model, tokenizer=tok, device=dev,
                         prompt="According to experts,", model_name=codec._model_name,
                         password="pw-123")
    assert vector.settings["password"] == "pw-123"
    # payload_sha256 is of the PLAINTEXT, not the ciphertext that the cover encodes.
    import hashlib as _h
    assert vector.payload_sha256 == _h.sha256(b"secret payload").hexdigest()
    result = verify_vector(vector, model=model, tokenizer=tok, device=dev)
    assert result.ok, result.detail
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_compat.py -k encrypted_vector -v`
Expected: PASS — the password path in Task 3 handles encrypt-on-make / decrypt-on-verify.

- [ ] **Step 3: No new implementation needed**

If it fails, the bug is in the `password` handling of `make_vector`/`verify_vector`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_compat.py
git commit -m "test(compat): encrypted vector round-trip"
```

---

## Task 6: Canonical deterministic payloads

**Files:**
- Create: `compat/__init__.py`
- Create: `compat/payloads.py`
- Test: `tests/test_compat.py` (import-and-shape check; no `compat` on pythonpath needed — see note)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_compat.py
def test_canonical_payloads_are_deterministic():
    # Imported lazily and by file location so the test does not depend on `compat`
    # being on sys.path (pytest only adds src/).
    import importlib.util
    from pathlib import Path

    spec_path = Path(__file__).parent.parent / "compat" / "payloads.py"
    spec = importlib.util.spec_from_file_location("compat_payloads", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Re-loading yields identical bytes (no os.urandom anywhere).
    mod2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod2)
    assert mod.CANONICAL_PAYLOADS == mod2.CANONICAL_PAYLOADS
    assert all(isinstance(v, bytes) for v in mod.CANONICAL_PAYLOADS.values())
    assert len(mod.CANONICAL_PROMPTS) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compat.py -k canonical_payloads -v`
Expected: FAIL — `FileNotFoundError` / spec is None (no `compat/payloads.py`)

- [ ] **Step 3: Write minimal implementation**

```python
# compat/__init__.py
"""Cross-compatibility orchestration package (deterministic payloads, golden
vector generation, and — in Plan 2 — cloud matrix runners)."""
```

```python
# compat/payloads.py
"""Deterministic canonical payloads and prompts for cross-compatibility vectors.

NOTHING here may use os.urandom or unseeded randomness: a reproducibility corpus
must itself be byte-for-byte reproducible. (The intermittent Qwen decode bug
traced to a random-payload test — see the spec.)"""

from __future__ import annotations

import random


def _seeded(n: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(n))


CANONICAL_PROMPTS: list[str] = [
    "According to experts,",
    "In a quiet village near",
    "The data shows that",
]

CANONICAL_PAYLOADS: dict[str, bytes] = {
    "ascii-short": b"hello",
    "single-byte": b"\x42",
    "all-zero": b"\x00" * 8,
    "all-ff": b"\xff" * 8,
    "binary-edge": b"\x00\x01\x02\x03\xfe\xff",
    "text": b"The quick brown fox jumps over the lazy dog.",
    "seeded-32": _seeded(32, 0),
    "seeded-64": _seeded(64, 1),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compat.py -k canonical_payloads -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add compat/__init__.py compat/payloads.py tests/test_compat.py
git commit -m "feat(compat): deterministic canonical payloads + prompts"
```

---

## Task 7: Golden-vector generation script + committed corpus

**Files:**
- Create: `compat/generate_golden.py`
- Create (generated, committed): `compat/golden_vectors.jsonl`

- [ ] **Step 1: Write the generation script**

```python
# compat/generate_golden.py
"""Generate the committed reference golden corpus.

Run from the repo root:  python -m compat.generate_golden
The reference environment is CPU + the default model (SmolLM-135M), matching the
session test fixture so tests/test_golden_vectors.py decodes it deterministically.
"""

from __future__ import annotations

from pathlib import Path

from mostegollm import StegoCodec
from mostegollm.compat import make_vector, model_commit_hash, write_vectors

from compat.payloads import CANONICAL_PAYLOADS, CANONICAL_PROMPTS

OUT = Path(__file__).parent / "golden_vectors.jsonl"


def main() -> None:
    codec = StegoCodec(device="cpu")
    model, tok, dev = codec._ensure_model()
    revision = model_commit_hash(model)
    vectors = [
        make_vector(
            payload,
            model=model,
            tokenizer=tok,
            device=dev,
            prompt=prompt,
            model_name=codec._model_name,
            model_revision=revision,
        )
        for prompt in CANONICAL_PROMPTS
        for payload in CANONICAL_PAYLOADS.values()
    ]
    write_vectors(vectors, OUT)
    print(f"wrote {len(vectors)} vectors to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the corpus**

Run: `python -m compat.generate_golden`
Expected: `wrote 24 vectors to .../compat/golden_vectors.jsonl` (3 prompts × 8 payloads)

- [ ] **Step 3: Sanity-check the output**

Run: `wc -l compat/golden_vectors.jsonl && head -c 200 compat/golden_vectors.jsonl`
Expected: `24` lines; first line is a JSON object starting with `{"schema": "mostegollm-testvector/1"`

- [ ] **Step 4: Commit**

```bash
git add compat/generate_golden.py compat/golden_vectors.jsonl
git commit -m "feat(compat): golden-vector generator + committed reference corpus"
```

---

## Task 8: Offline regression guard

**Files:**
- Create: `tests/test_golden_vectors.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_golden_vectors.py
"""Offline cross-compatibility guard: decode the committed golden corpus on
whatever this machine is. Runs on every `pytest` invocation (no cloud)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mostegollm.compat import read_vectors, verify_vector

GOLDEN = Path(__file__).parent.parent / "compat" / "golden_vectors.jsonl"


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden corpus not generated yet")
def test_golden_vectors_all_decode(codec):
    model, tok, dev = codec._ensure_model()
    vectors = read_vectors(GOLDEN)
    assert vectors, "golden corpus is empty"

    failures = []
    for v in vectors:
        result = verify_vector(v, model=model, tokenizer=tok, device=dev)
        if not result.ok:
            failures.append(
                f"{v.prompt!r} / {v.payload_hex[:12]}: {result.failure_class} — {result.detail}"
            )
    assert not failures, f"{len(failures)}/{len(vectors)} golden vectors failed:\n" + "\n".join(
        failures[:10]
    )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_golden_vectors.py -v`
Expected: PASS (1 passed) — the reference corpus decodes on the same CPU+model env that generated it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_golden_vectors.py
git commit -m "test(compat): offline golden-corpus regression guard"
```

---

## Task 9: Lint, full-suite verification, and docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/test_compat.py tests/test_golden_vectors.py -v`
Expected: all PASS. (Optionally `pytest tests/ -q` for the whole suite; note `test_qwen.py` has a pre-existing intermittent failure tracked separately.)

- [ ] **Step 2: Lint and format**

Run: `ruff check . && ruff format --check .`
Expected: no errors. Fix any reported issues, then re-run.

- [ ] **Step 3: Add a README note**

Add a short subsection under the existing cross-platform/portability section:

```markdown
### Verified portability (foundation)

Cover text is portable across PyTorch versions, devices, and dtypes by design
(integer rank-interval coding). A committed corpus of reference test vectors
(`compat/golden_vectors.jsonl`) is decoded on every test run as a regression
guard (`tests/test_golden_vectors.py`); regenerate it with
`python -m compat.generate_golden`. Broad multi-environment proof (NVIDIA archs,
AMD ROCm, dtype matrix) runs in the cloud layer (Plan 2). **Apple Silicon / MPS
is not covered by the automated matrix** — verify manually on a Mac if needed.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(compat): document portability guard + Apple-MPS gap"
```

---

## Self-Review

**Spec coverage:**
- Test-vector primitive → Task 1 (record/IO), Task 2 (env), Task 3 (make). ✓
- `verify_vector` + failure classification → Task 3 + Task 4. ✓ (Re-tokenization-drift check embedded in every verify, per the spec correction.)
- Deterministic payloads, no `os.urandom` → Task 6. ✓
- Golden corpus + offline guard → Task 7, Task 8. ✓
- README portability + Apple-MPS gap → Task 9. ✓
- **Deferred to Plan 2 (documented):** Modal/RunPod cells, N×N prove run, `dump_step_logits` + GUARD-margin report, `floor_check.py`, the generated matrix report. These are cloud-only and not unit-testable here.

**Placeholder scan:** No TBD/TODO; every code step is complete. ✓

**Type consistency:** `TestVector` fields used identically across Tasks 1/3/4/5/7/8. `make_vector`/`verify_vector` signatures match every call site (`model_name=codec._model_name`, keyword-only model/tokenizer/device). `FailureClass.RETOK_DRIFT` / `LOGIT_DIVERGENCE` used consistently. `VerifyResult.ok/failure_class/detail/decoder_env` consistent. ✓
