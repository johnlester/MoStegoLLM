# Handoff — reproducible-coding branch

**Date:** 2026-06-02
**Branch:** `reproducible-coding` (off `main`)
**Status:** Feature implemented and cross-platform-proven. **One** intermittent
test failure outstanding; needs fix + a green full-suite run on a machine where
the model runs reliably (this dev environment was killing model processes —
exit 144 — making fix-and-verify unreliable, so I stopped rather than ship
unverified).

## What's done and proven

Spec: `docs/superpowers/specs/2026-06-02-cross-platform-reproducible-coding-design.md`
Plan: `docs/superpowers/plans/2026-06-02-cross-platform-reproducible-coding.md`

Tasks 1–7 are committed and reviewed:
- `coding.py` — fixed integer `CUM`, `GUARD`, `step_coding` (rank intervals,
  run-merging). Unit-tested (`tests/test_coding.py`), spec- and quality-reviewed.
- `encoder.py` / `decoder.py` — rewritten to rank-interval coding. Spec review
  confirmed the arithmetic state machine (renorm, flush, sentence_boundary,
  xorshift padding, CRC) is preserved byte-for-byte. `tests/test_bpe_filter.py`
  updated to the new `_filter_tokens` signature.
- `model.py` — TF32 disabled / determinism hardened.
- **`tests/test_cross_platform.py` — the headline proof, all 10 cases pass:**
  CPU-encode→GPU-decode and float32→float64 round-trip across 5 payloads
  (incl. binary edge cases). The exact scenario the old caveat warned about
  (CPU↔GPU) now works.

Full suite result on this branch: **112 passed, 1 failed** (see below).

## The outstanding failure

`tests/test_qwen.py::TestQwenChunked::test_chunked_encrypted_roundtrip`

- Decode raises `StegoDecodeError: Invalid magic bytes` on one chunk.
- **Intermittent / data-dependent.** The test encrypts with `os.urandom`
  (`crypto.encrypt`), so the payload differs every run; the failing payload is
  not reproducible from the test as written.
- **Qwen-only.** SmolLM (the default model) was robust: 0 failures across
  hundreds of fuzz trials. Qwen: 0 failures in 80 fixed-prompt trials — the rate
  under simple conditions is very low; the one observed failure was under the
  combined path (seed-phrase mode + chained-prompt chunking + random payload).
- **CRC-caught.** It fails loudly (`StegoDecodeError`) — it never returns
  corrupted bytes. This is a fail-closed bug, not silent corruption.
- **Almost certainly pre-existing.** The old (0.2.0) coder used the same
  adjacent-pair-only BPE filter and the same random-payload test, so it was
  likely already flaky at some rate; the rank-interval change may have shifted
  the rate but did not introduce the failure class. (Not confirmed — would need
  fuzzing `main`'s coder on the same scenario.)

### Root-cause analysis (strongly reasoned, NOT empirically classified)

On a single machine, encode and decode run identical model forward passes and
identical integer interval math, so the **only** path by which decode can
diverge from encode is **re-tokenization drift**: chunked decode re-tokenizes
the cover *string* (`tokenizer.encode(cover_text)`), and if that yields
different token IDs than the encoder generated, decode desyncs. The BPE filter
(`encoder._filter_tokens` + `get_non_roundtrip_tokens`) only guarantees (a) each
token round-trips individually and (b) no merge with the *immediately previous*
token — it does **not** catch 3+-token BPE re-tokenization interactions.

I verified re-tokenization was identical on several *non-failing* Qwen samples,
but could not capture a *failing* sample in this environment to confirm drift on
it (model processes kept getting killed). So treat re-tokenization drift as the
leading hypothesis, not a proven fact. (A pure same-machine coder asymmetry is
nearly ruled out by the determinism argument above, but verify when you
reproduce.)

### How to reproduce deterministically

Make the payload deterministic so a failing case is reproducible, then bisect:

```python
import random
from mostegollm.model import load_model
from mostegollm.encoder import encode
from mostegollm.decoder import decode
model, tok, dev = load_model("Qwen/Qwen2.5-0.5B", "cpu")
rng = random.Random(0)
for trial in range(2000):
    n = rng.randint(20, 80)
    payload = bytes(rng.randrange(256) for _ in range(n))
    cover, ids, _ = encode(payload, model=model, tokenizer=tok, device=dev,
                           prompt="In a quiet village near")
    retok = tok.encode(cover, add_special_tokens=False)
    if retok != ids:
        print("RE-TOKENIZATION DRIFT", trial, n, payload.hex()); break
    if decode(cover, model=model, tokenizer=tok, device=dev,
              prompt="In a quiet village near") != payload:
        print("CODER ASYMMETRY (retok matched but decode wrong)", trial); break
```
If the first branch fires → re-tokenization drift (expected). If the second
fires with `retok == ids` → a genuine coder bug (investigate `step_coding`).
The chained-prompt chunked path (`codec._encode_chunked`) may have a higher rate
than the fixed-prompt case above; reproduce there too if needed.

## Recommended fixes (in order)

1. **Make the test deterministic regardless of the root-cause fix.** A
   round-trip test should not depend on `os.urandom`. Seed or fix the payload
   so the suite is reproducible. (Do this even after fixing the coder.)

2. **Cheap improvement — emit the run's top-ranked member, not `min(token_id)`.**
   In `coding.step_coding`, the run representative is currently `min(members)`.
   Emitting the *highest-logit* member instead (the first in sorted order)
   yields more natural cover text and likely fewer re-tokenization surprises
   (the most-probable token is the "expected" continuation). Decode is
   unaffected — it looks up the observed token in `token_to_run`, which maps
   *all* members to the run, so any member works as the emitted token. Verify it
   reduces (it won't necessarily eliminate) the drift rate.

3. **Robust fix — encode-time re-tokenization guarantee.** After generating,
   have the encoder verify `tokenizer.encode(cover) == generated_ids`; on
   mismatch, fail at encode time (or fall back) rather than emitting undecodable
   cover text. This restores a guarantee the removed seed-retry mechanic
   (commit `6f0c659`, removed in `66e28ea`) used to provide as a side effect.
   Cost: one extra `tokenizer.encode` per encode (cheap) and a fallback strategy
   for the mismatch case (raise `StegoEncodeError`, or — for chunked/seed mode —
   retry the chunk with a different seed/prompt). This is the only option that
   *guarantees* round-trip; it's real work and needs reliable model verification.

## Verification before merge

On a machine where the model runs reliably:
```bash
.venv/bin/python -m pytest tests/ -q        # must be fully green
.venv/bin/python -m pytest tests/test_cross_platform.py -v   # 10 passed, none skipped
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

## Finalization state

The 0.3.0 finalization (version bump in `pyproject.toml`, `CHANGELOG.md`, README
cross-platform caveat rewrite) is committed on this branch, but the branch is
**NOT ready to tag/merge** until the Qwen test is resolved and the full suite is
verified green. Adjust the version if you prefer (e.g. keep 0.2.x until fixed).
