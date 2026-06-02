# Cross-Platform Reproducible Coding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make steganographic decoding reproducible across PyTorch versions, devices (CPU/GPU), and BLAS libraries by replacing float-magnitude arithmetic-coding intervals with fixed integer intervals assigned by token rank.

**Architecture:** A new `coding.py` holds integer constants (`WHOLE`, width schedule `W[]`, `GUARD`) and the pure, model-free interval logic. Per step, encoder and decoder take the model's top-K tokens, sort by `(logit DESC, token_id ASC)`, merge near-tie runs within `GUARD`, and assign each run a fixed integer interval from the schedule. Because interval boundaries are integer constants and a token's rank is invariant under sub-ε float noise, both sides build identical intervals. The CRC-32 already in the header detects any rare residual.

**Tech Stack:** Python 3.10+, PyTorch, transformers, pytest (+ hypothesis). Reference spec: `docs/superpowers/specs/2026-06-02-cross-platform-reproducible-coding-design.md`.

---

## File Structure

- **Create** `src/mostegollm/coding.py` — `PRECISION`, `WHOLE`, `HALF`, `QUARTER`, `K`, `GUARD`, the integer width schedule `_WIDTHS`, the fixed cumulative array `CUM`, `StepCoding`, and `step_coding(token_ids, logits)`. Pure, no torch, no model.
- **Modify** `src/mostegollm/encoder.py` — replace `_get_token_distribution` with `_get_topk_logits` (returns logits, no softmax) and `_filter_distribution` with `_filter_tokens` (filters `(token_ids, logits)`); rewrite the token-selection portion of `encode()` to navigate `step_coding` intervals. Import arithmetic constants from `coding.py`.
- **Modify** `src/mostegollm/decoder.py` — use the shared helpers + `step_coding`; look up the observed token's run interval.
- **Modify** `src/mostegollm/model.py` — determinism hardening (deterministic algorithms, disable TF32).
- **Modify** `src/mostegollm/__init__.py` & `pyproject.toml` — version 0.3.0.
- **Create** `tests/test_coding.py` — pure unit tests for `coding.py`.
- **Create** `tests/test_cross_platform.py` — CPU↔GPU and float32↔float64 round-trip (the empirical proof).
- **Modify** `tests/test_bpe_filter.py` — update to the new `_filter_tokens` signature.
- **Create** `CHANGELOG.md`; **modify** `README.md` — document the breaking change + new guarantee.

---

## Task 1: Coding constants and fixed `CUM`

**Files:**
- Create: `src/mostegollm/coding.py`
- Test: `tests/test_coding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coding.py
"""Unit tests for the reproducible rank-interval coding (no model needed)."""

from __future__ import annotations

from mostegollm import coding


def test_constants_are_integers():
    assert coding.WHOLE == 1 << 32
    assert coding.HALF == coding.WHOLE >> 1
    assert coding.QUARTER == coding.WHOLE >> 2
    assert all(isinstance(w, int) for w in coding._WIDTHS)
    assert len(coding._WIDTHS) == coding.K


def test_widths_are_non_increasing_and_positive():
    assert all(w >= 1 for w in coding._WIDTHS)
    assert all(a >= b for a, b in zip(coding._WIDTHS, coding._WIDTHS[1:]))
    assert coding._WIDTHS[0] > coding._WIDTHS[-1]


def test_cum_is_well_formed():
    cum = coding.CUM
    assert len(cum) == coding.K + 1
    assert cum[0] == 0
    assert cum[-1] == coding.WHOLE
    assert all(cum[i] < cum[i + 1] for i in range(coding.K))  # strictly increasing


def test_cum_is_a_fixed_constant():
    # Same object/values every access; never recomputed from floats.
    assert coding.CUM is coding.CUM
    assert coding.CUM == coding._build_cum(coding._WIDTHS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_coding.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mostegollm.coding'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mostegollm/coding.py
"""Reproducible rank-interval coding.

Interval boundaries are fixed integer constants assigned to tokens by rank, so
encode and decode build bit-identical intervals across PyTorch versions,
devices, and BLAS libraries. All constants are generated with integer-only
arithmetic (Python int is arbitrary-precision and platform-independent), so the
schedule itself cannot diverge across platforms.

See docs/superpowers/specs/2026-06-02-cross-platform-reproducible-coding-design.md
"""

from __future__ import annotations

from dataclasses import dataclass

# Arithmetic-coding precision (shared by encoder and decoder).
PRECISION = 32
WHOLE = 1 << PRECISION  # 2^32
HALF = WHOLE >> 1
QUARTER = WHOLE >> 2

# Top-k width considered at each step.
K = 256

# Gap (in logit units) below which adjacent sorted tokens merge into one run.
# Must be >> the cross-platform logit divergence (~3e-5 measured CPU<->GPU);
# 1e-3 is ~35x that margin. Tunable; see the design spec.
GUARD = 1e-3

# Width schedule: geometric decay via an integer-only recurrence. Model-agnostic.
_DECAY_NUM = 7
_DECAY_DEN = 8
_SEED = 1 << 40


def _build_widths(k: int) -> list[int]:
    """Integer geometric-decay widths: w[i] = max(1, w[i-1] * NUM // DEN)."""
    widths: list[int] = []
    w = _SEED
    for _ in range(k):
        widths.append(w)
        w = max(1, (w * _DECAY_NUM) // _DECAY_DEN)
    return widths


def _build_cum(widths: list[int]) -> list[int]:
    """Fixed cumulative boundaries scaled to WHOLE, each interval width >= 1.

    Reserves one unit per position (which guarantees a strictly increasing
    result) and distributes the remaining ``WHOLE - len(widths)`` units by
    weight. Integer-only (multiply + floor-divide), so the array is identical on
    every platform. Constructed so it can never overflow WHOLE — unlike a naive
    scale-then-bump approach, where the many width-1 tail entries would cascade
    the +1 fixups past WHOLE.
    """
    n = len(widths)
    total = sum(widths)
    free = WHOLE - n  # > 0 since n (= K = 256) << WHOLE
    cum = [0]
    acc = 0
    for i, w in enumerate(widths):
        acc += w
        cum.append((i + 1) + (acc * free) // total)
    cum[-1] = WHOLE  # already exact (total divides total*free); set for clarity
    return cum


_WIDTHS = _build_widths(K)
# CUM is a FIXED constant array of length K+1 (not rescaled per step). A token at
# sorted position i always owns [CUM[i], CUM[i+1]); the final occupied run is
# extended to WHOLE (see step_coding). This makes a low-rank emitted token's
# interval independent of how many tokens survive filtering (m), so top-K cutoff
# instability cannot desync the emitted token.
CUM = _build_cum(_WIDTHS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_coding.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mostegollm/coding.py tests/test_coding.py
git commit -m "feat(coding): integer rank-interval constants and fixed CUM"
```

---

## Task 2: `step_coding` — sort, run-merge, assign intervals

**Files:**
- Modify: `src/mostegollm/coding.py`
- Test: `tests/test_coding.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_coding.py

def test_step_coding_partitions_whole():
    # well-separated logits (gaps >> GUARD): each token its own run
    ids = [10, 20, 30, 40]
    logits = [5.0, 4.0, 3.0, 2.0]
    sc = coding.step_coding(ids, logits)
    assert len(sc.intervals) == 4
    assert sc.intervals[0][0] == 0
    assert sc.intervals[-1][1] == coding.WHOLE
    # contiguous, non-overlapping
    for (lo, hi, _), (lo2, _, _) in zip(sc.intervals, sc.intervals[1:]):
        assert hi == lo2
        assert lo < hi
    # every token maps to a run interval
    assert set(sc.token_to_run) == {10, 20, 30, 40}


def test_step_coding_is_sort_order_independent():
    a = coding.step_coding([30, 10, 20], [3.0, 5.0, 4.0])
    b = coding.step_coding([10, 20, 30], [5.0, 4.0, 3.0])
    assert a.intervals == b.intervals
    assert a.token_to_run == b.token_to_run


def test_run_merging_groups_near_ties():
    # tokens 1 and 2 are a near-tie (gap < GUARD) -> one run; token 3 separate
    ids = [1, 2, 3]
    logits = [5.0, 5.0 - coding.GUARD / 2, 1.0]
    sc = coding.step_coding(ids, logits)
    assert len(sc.intervals) == 2  # merged pair + singleton
    # the merged run is represented by the lowest token_id (1)
    assert sc.intervals[0][2] == 1
    # both members of the merged run share the same interval
    assert sc.token_to_run[1] == sc.token_to_run[2]
    assert sc.token_to_run[3] != sc.token_to_run[1]


def test_emitted_run_is_stable_under_sub_guard_noise():
    # Perturbing logits by < GUARD must not change any run interval.
    ids = [7, 3, 9, 1]
    base = [5.0, 3.0, 1.0, 0.5]
    noisy = [v + (0.4 * coding.GUARD) * ((-1) ** i) for i, v in enumerate(base)]
    assert coding.step_coding(ids, base).intervals == coding.step_coding(ids, noisy).intervals
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_coding.py -q`
Expected: FAIL with `AttributeError: module 'mostegollm.coding' has no attribute 'step_coding'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mostegollm/coding.py

@dataclass(frozen=True)
class StepCoding:
    """Per-step coding: a partition of [0, WHOLE) and a token->interval lookup.

    Attributes:
        intervals: list of (lo, hi, representative_token_id), sorted by lo,
            tiling [0, WHOLE). The encoder navigates `value` into one of these
            and emits its representative.
        token_to_run: maps every surviving token_id to its run's (lo, hi). The
            decoder looks up the observed token here.
    """

    intervals: list[tuple[int, int, int]]
    token_to_run: dict[int, tuple[int, int]]


def step_coding(token_ids: list[int], logits: list[float]) -> StepCoding:
    """Build the rank-interval coding for one step.

    Tokens are sorted by (logit DESC, token_id ASC); adjacent tokens whose logit
    gap is < GUARD are merged into a run represented by its lowest token_id.
    Each run is assigned a fixed integer interval from the constant `CUM` array
    (the final occupied run extends to WHOLE).
    """
    order = sorted(range(len(token_ids)), key=lambda i: (-logits[i], token_ids[i]))
    s_ids = [token_ids[i] for i in order]
    s_log = [logits[i] for i in order]
    m = len(s_ids)

    intervals: list[tuple[int, int, int]] = []
    token_to_run: dict[int, tuple[int, int]] = {}
    i = 0
    while i < m:
        j = i
        while j + 1 < m and (s_log[j] - s_log[j + 1]) < GUARD:
            j += 1
        lo = CUM[i]
        # The final occupied run absorbs up to WHOLE so the partition always
        # covers [0, WHOLE) regardless of how many tokens survived (m).
        hi = WHOLE if j == m - 1 else CUM[j + 1]
        members = s_ids[i : j + 1]
        rep = min(members)
        intervals.append((lo, hi, rep))
        for t in members:
            token_to_run[t] = (lo, hi)
        i = j + 1
    return StepCoding(intervals, token_to_run)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_coding.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mostegollm/coding.py tests/test_coding.py
git commit -m "feat(coding): step_coding with run-merging and stable rank intervals"
```

---

## Task 3: Encoder helpers — `_get_topk_logits` and `_filter_tokens`

**Files:**
- Modify: `src/mostegollm/encoder.py`
- Test: `tests/test_bpe_filter.py`

- [ ] **Step 1: Write the failing test** (replace the body of `tests/test_bpe_filter.py` with the new-signature version)

```python
# tests/test_bpe_filter.py
"""Tests for the BPE round-trip token filter (new (token_ids, logits) signature)."""

from __future__ import annotations

from mostegollm import StegoCodec
from mostegollm.encoder import _filter_tokens, get_non_roundtrip_tokens


def test_filter_drops_non_roundtrip_tokens(codec: StegoCodec):
    _model, tokenizer, _device = codec._ensure_model()
    non_rt = get_non_roundtrip_tokens(tokenizer)
    bad = next(iter(non_rt)) if non_rt else None
    if bad is None:
        return  # tokenizer has no non-round-trip tokens; nothing to assert
    ids = [bad, 100, 101]
    logits = [9.0, 8.0, 7.0]
    kept_ids, kept_logits = _filter_tokens(tokenizer, None, ids, logits, non_rt, {})
    assert bad not in kept_ids
    assert len(kept_ids) == len(kept_logits)


def test_filter_preserves_correspondence(codec: StegoCodec):
    _model, tokenizer, _device = codec._ensure_model()
    non_rt = get_non_roundtrip_tokens(tokenizer)
    ids = [100, 101, 102, 103]
    logits = [4.0, 3.0, 2.0, 1.0]
    kept_ids, kept_logits = _filter_tokens(tokenizer, None, ids, logits, non_rt, {})
    # surviving logits stay paired with their token_ids in order
    original = dict(zip(ids, logits))
    for tid, lg in zip(kept_ids, kept_logits):
        assert original[tid] == lg


def test_filter_falls_back_when_all_removed(codec: StegoCodec):
    _model, tokenizer, _device = codec._ensure_model()
    non_rt = get_non_roundtrip_tokens(tokenizer)
    if not non_rt:
        return
    bad = list(non_rt)[:3]
    ids = bad
    logits = [3.0, 2.0, 1.0][: len(bad)]
    kept_ids, kept_logits = _filter_tokens(tokenizer, None, ids, logits, non_rt, {})
    assert kept_ids == ids  # fallback to unfiltered rather than empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bpe_filter.py -q`
Expected: FAIL with `ImportError: cannot import name '_filter_tokens'`

- [ ] **Step 3: Write minimal implementation** — in `src/mostegollm/encoder.py`, update imports and replace `_get_token_distribution` and `_filter_distribution`.

Change the imports block near the top to import arithmetic constants from `coding`:

```python
from .coding import HALF, K, PRECISION, QUARTER, WHOLE, step_coding
from .utils import (
    StegoEncodeError,
    bytes_to_bits,
    pack_header,
)
```

Delete the local `PRECISION`/`WHOLE`/`HALF`/`QUARTER` definitions (now imported). Keep `MAX_TOKENS`, `MAX_EXTRA_TOKENS`. Replace `TOP_K = 256` with `TOP_K = K` (kept for backward-compatible imports).

Replace `_get_token_distribution(...)` with:

```python
def _get_topk_logits(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    device: torch.device,
    top_k: int = K,
    temperature: float = 1.0,
    past_key_values: DynamicCache | None = None,
) -> tuple[list[int], list[float], DynamicCache]:
    """Return the top-k token ids and their (temperature-scaled) float64 logits.

    No softmax: the reproducible coder uses logit *order* and *gaps*, not
    probability magnitudes.
    """
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[0, -1, :]

    if temperature != 1.0:
        logits = logits / temperature

    logits_f64 = logits.to(dtype=torch.float64)
    actual_k = min(top_k, logits_f64.shape[0])
    top = torch.topk(logits_f64, actual_k)
    token_ids = top.indices.cpu().tolist()
    logit_vals = top.values.cpu().tolist()
    return token_ids, logit_vals, outputs.past_key_values
```

Replace `_filter_distribution(...)` with:

```python
def _filter_tokens(
    tokenizer: PreTrainedTokenizerBase,
    prev_token_id: int | None,
    token_ids: list[int],
    logits: list[float],
    non_rt_tokens: frozenset[int],
    merge_cache: dict[tuple[int, int], bool],
) -> tuple[list[int], list[float]]:
    """Drop tokens that break a decode->encode round-trip, keeping logits paired.

    Filters (1) tokens that don't round-trip individually and (2) tokens that
    would BPE-merge with *prev_token_id*. These checks are pure string ops, so
    they are identical across platforms. Falls back to the unfiltered lists if
    everything would be removed.
    """
    keep_ids: list[int] = []
    keep_logits: list[float] = []
    for tid, lg in zip(token_ids, logits):
        if tid in non_rt_tokens:
            continue
        if prev_token_id is not None:
            pair = (prev_token_id, tid)
            if pair not in merge_cache:
                text = tokenizer.decode([prev_token_id, tid])
                re_enc = tokenizer.encode(text, add_special_tokens=False)
                merge_cache[pair] = re_enc != [prev_token_id, tid]
            if merge_cache[pair]:
                continue
        keep_ids.append(tid)
        keep_logits.append(lg)
    if not keep_ids:
        return token_ids, logits
    return keep_ids, keep_logits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bpe_filter.py -q`
Expected: PASS (3 passed). (`encode`/`decode` are still broken at this point — they reference the removed functions; that is fixed in Tasks 4–5.)

- [ ] **Step 5: Commit**

```bash
git add src/mostegollm/encoder.py tests/test_bpe_filter.py
git commit -m "refactor(encoder): logit-based top-k + token filter for rank coding"
```

---

## Task 4: Rewrite `encode()` token selection to use `step_coding`

**Files:**
- Modify: `src/mostegollm/encoder.py` (the `encode()` function body, distribution + selection portion only)
- Test: `tests/test_roundtrip.py` (existing; used as integration check after Task 5)

- [ ] **Step 1: Replace the per-step distribution + token-selection block inside `encode()`.**

Find the block that currently calls `_get_token_distribution` + `_filter_distribution`, computes `range_size`, finds `chosen_idx`, and narrows the interval (the section from `token_ids, cum_probs, past_kv = _get_token_distribution(` through the `low = sym_low` / `high = sym_high` lines). Replace it with:

```python
        # Reproducible rank-interval coding: top-k logits -> BPE filter ->
        # sorted/run-merged fixed integer intervals.
        token_ids, logits, past_kv = _get_topk_logits(
            model,
            next_input,
            device,
            top_k=top_k,
            temperature=temperature,
            past_key_values=past_kv,
        )
        token_ids, logits = _filter_tokens(
            tokenizer, prev_token_id, token_ids, logits, non_rt_tokens, merge_cache
        )
        step = step_coding(token_ids, logits)

        range_size = high - low
        if range_size <= 0:
            raise StegoEncodeError(
                "Arithmetic coding interval collapsed (range_size <= 0). "
                "This indicates a numerical precision issue."
            )

        # Navigate `value` into one interval; default to the last.
        ilo, ihi, chosen_token_id = step.intervals[-1]
        sym_low = low + (range_size * ilo) // WHOLE
        sym_high = low + (range_size * ihi) // WHOLE
        for cand_lo, cand_hi, rep in step.intervals:
            cand_high = low + (range_size * cand_hi) // WHOLE
            if value < cand_high:
                chosen_token_id = rep
                sym_low = low + (range_size * cand_lo) // WHOLE
                sym_high = cand_high
                break
        low = sym_low
        high = sym_high
```

Then ensure the lines that follow (renormalization `while True:` loop, `generated_token_ids.append(chosen_token_id)`, `next_input = torch.tensor([[chosen_token_id]], ...)`, `tokens_generated += 1`, `prev_token_id = chosen_token_id`, and all `data_recoverable`/`sentence_boundary` logic) are **unchanged**. The variable `chosen_token_id` is now set by the navigation above, so delete the old `chosen_token_id = token_ids[chosen_idx]` line.

- [ ] **Step 2: Verify the module imports and encode path are consistent**

Run: `python -c "import mostegollm.encoder"`
Expected: no error (clean import; `cum_probs`/`_get_token_distribution` no longer referenced).

- [ ] **Step 3: Commit (encode half; round-trip verified after decode in Task 5)**

```bash
git add src/mostegollm/encoder.py
git commit -m "feat(encoder): emit tokens via fixed rank intervals"
```

---

## Task 5: Rewrite `decode()` to use `step_coding`

**Files:**
- Modify: `src/mostegollm/decoder.py`
- Test: `tests/test_roundtrip.py`

- [ ] **Step 1: Update `decoder.py` imports**

Replace the imports from `.encoder` and `.utils` / add `.coding`:

```python
from .coding import HALF, QUARTER, WHOLE, step_coding
from .encoder import (
    TOP_K,
    _filter_tokens,
    _get_topk_logits,
    get_non_roundtrip_tokens,
)
from .utils import (
    HEADER_BITS,
    StegoDecodeError,
    bits_to_bytes,
    unpack_header,
)
```

- [ ] **Step 2: Replace the per-token distribution + interval-narrowing block inside `decode()`.**

Find the block from `tok_ids, cum_probs, past_kv = _get_token_distribution(` through the `low = sym_low` / `high = sym_high` lines (the part that finds index `j` and narrows). Replace with:

```python
        # Same reproducible coding the encoder used.
        tok_ids, logits, past_kv = _get_topk_logits(
            model,
            next_input,
            device,
            top_k=top_k,
            temperature=temperature,
            past_key_values=past_kv,
        )
        tok_ids, logits = _filter_tokens(
            tokenizer, prev_token_id, tok_ids, logits, non_rt_tokens, merge_cache
        )
        step = step_coding(tok_ids, logits)

        try:
            ilo, ihi = step.token_to_run[token_id]
        except KeyError:
            raise StegoDecodeError(
                f"Token ID {token_id} ('{tokenizer.decode([token_id])}') at step {step_idx} "
                f"was not found in the reproducible top-{top_k} distribution. "
                "This usually means the cover text was corrupted, the wrong model "
                "was used, or the prompt does not match."
            )

        range_size = high - low
        sym_low = low + (range_size * ilo) // WHOLE
        sym_high = low + (range_size * ihi) // WHOLE
        low = sym_low
        high = sym_high
```

Note: rename the loop variable `step` (used by the old code as the enumerate index) to `step_idx` to avoid colliding with the `step` StepCoding object: change `for step, token_id in enumerate(cover_token_ids):` to `for step_idx, token_id in enumerate(cover_token_ids):`. Keep the renormalization/bit-emission `while True:` loop and the flush/header/CRC logic **unchanged**.

- [ ] **Step 3: Run the round-trip suite (integration check for Tasks 4+5)**

Run: `pytest tests/test_roundtrip.py tests/test_codec.py -q`
Expected: PASS — same-machine round-trips still work with the new coder.

- [ ] **Step 4: Commit**

```bash
git add src/mostegollm/decoder.py
git commit -m "feat(decoder): decode via shared rank intervals"
```

---

## Task 6: Determinism hardening in `model.py`

**Files:**
- Modify: `src/mostegollm/model.py` (the `_setup_determinism` function)

- [ ] **Step 1: Replace `_setup_determinism` body**

```python
def _setup_determinism() -> None:
    """Configure PyTorch for maximum determinism.

    Reduces cross-platform logit divergence (belt-and-suspenders; the rank-
    interval coder does not depend on these for correctness).
    """
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    # Disable TF32 so matmul uses full float32 precision on GPU.
    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
    torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
```

- [ ] **Step 2: Verify import**

Run: `python -c "import mostegollm.model"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add src/mostegollm/model.py
git commit -m "chore(model): disable TF32 and harden determinism"
```

---

## Task 7: Cross-platform round-trip tests (the empirical proof)

**Files:**
- Create: `tests/test_cross_platform.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_cross_platform.py
"""Cross-platform reproducibility: encode here, decode under a different
floating-point regime. These are the empirical proof of the rank-interval
design (they fail on the old magnitude-based coder)."""

from __future__ import annotations

import copy

import pytest
import torch

from mostegollm.decoder import decode
from mostegollm.encoder import encode

PROMPT = "According to experts,"
PAYLOADS = [
    b"cross-platform compatibility test",
    b"hello",
    b"\x00\x01\x02\x03\xfe\xff",
    b"The quick brown fox jumps over the lazy dog.",
    b"a",
]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_float32_to_float64_roundtrip(codec, payload):
    """Encoding with float32 matmul must decode with a float64 model copy."""
    model, tok, dev = codec._ensure_model()
    cover, ids, _ = encode(payload, model=model, tokenizer=tok, device=dev, prompt=PROMPT)
    model64 = copy.deepcopy(model).double()
    recovered = decode(cover, model=model64, tokenizer=tok, device=dev, prompt=PROMPT,
                       token_ids=ids)
    assert recovered == payload


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("payload", PAYLOADS)
def test_cpu_encode_gpu_decode_roundtrip(codec, payload):
    """Encoding on CPU must decode on GPU (the caveat scenario)."""
    model, tok, _ = codec._ensure_model()  # codec fixture is CPU
    cover, ids, _ = encode(payload, model=model, tokenizer=tok,
                           device=torch.device("cpu"), prompt=PROMPT)
    gpu = torch.device("cuda")
    model_gpu = copy.deepcopy(model).to(gpu)
    recovered = decode(cover, model=model_gpu, tokenizer=tok, device=gpu, prompt=PROMPT,
                       token_ids=ids)
    assert recovered == payload
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_cross_platform.py -q`
Expected: PASS — both cross-regime round-trips succeed (the f64 test runs everywhere; the GPU test runs where CUDA is present).

- [ ] **Step 3: Commit**

```bash
git add tests/test_cross_platform.py
git commit -m "test: cross-platform (f32<->f64, CPU<->GPU) round-trip proof"
```

---

## Task 8: Capacity check, full suite, version bump, docs

**Files:**
- Modify: `src/mostegollm/__init__.py`, `pyproject.toml`, `README.md`
- Create: `CHANGELOG.md`

- [ ] **Step 1: Measure capacity (new vs expectation) with a throwaway script**

Run:
```bash
PYTHONPATH=src python - <<'PY'
from mostegollm import StegoCodec
c = StegoCodec(device="cpu")
for payload in (b"hello", b"the quick brown fox jumps over the lazy dog"):
    s = c.encode_with_stats(payload)
    print(f"{len(payload):>3}B -> {s.total_tokens:>4} tokens, {s.bits_per_token:.2f} bits/token")
PY
```
Expected: prints non-zero `bits/token` (record the value). If it is dramatically lower than ~2 bits/token, note it for a follow-up `GUARD`/`_DECAY_*` tune; do not block this task.

- [ ] **Step 2: Run the full suite**

Run: `pytest tests/ -q`
Expected: PASS (all existing round-trip, chunked, encrypted, edge-case, fuzz, timing, CLI, coding, and cross-platform tests). Investigate and fix any failure before proceeding.

- [ ] **Step 3: Bump version to 0.3.0**

In `pyproject.toml` change `version = "0.2.0"` to `version = "0.3.0"`.

If `src/mostegollm/__init__.py` defines `__version__`, update it to `"0.3.0"`; otherwise leave it.

- [ ] **Step 4: Add CHANGELOG and update README caveat**

Create `CHANGELOG.md`:

```markdown
# Changelog

## 0.3.0

### Changed (breaking)
- Encoding now uses reproducible **rank-interval coding** instead of
  probability-magnitude arithmetic coding. Cover text produced by 0.2.0 and
  earlier **cannot be decoded** by 0.3.0, and vice versa.

### Added
- Cross-platform reproducibility: cover text encoded on one PyTorch
  version / device (CPU or GPU) now decodes on another. Decoding no longer
  requires an identical floating-point environment. Any rare residual
  divergence is detected by the existing CRC-32 (decode raises; never returns
  wrong bytes).
- TF32 disabled and deterministic algorithms enabled to further reduce
  cross-platform logit divergence.
```

In `README.md`, replace the "Cross-platform compatibility caveat" paragraph with:

```markdown
**Cross-platform compatibility:** As of 0.3.0, decoding no longer requires the
same PyTorch version or device type as encoding — the coder assigns
arithmetic-coding intervals by token *rank* (a quantity stable across
floating-point regimes) rather than by probability magnitude. You still need
the **same model weights**. Any rare residual divergence is caught by the
CRC-32 integrity check, so a mismatch fails loudly rather than returning corrupt
data.
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/mostegollm/__init__.py README.md CHANGELOG.md
git commit -m "chore: bump to 0.3.0; document reproducible coding"
```

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** rank-interval core (Tasks 1–2, 4–5), integer-only schedule (Task 1), `GUARD` run-merging (Task 2), determinism hardening (Task 6), CPU↔GPU + f32↔f64 proof (Task 7), capacity measurement + 0.3.0 + docs (Task 8). The CRC residual safety net is unchanged in `decoder.py` and exercised by existing tests.
- **Shared code:** `_get_topk_logits`, `_filter_tokens`, `get_non_roundtrip_tokens` live in `encoder.py` and are imported by `decoder.py` (matching the existing pattern), so the two sides cannot drift.
- **Names used consistently:** `_get_topk_logits`, `_filter_tokens`, `step_coding`, `StepCoding.intervals`, `StepCoding.token_to_run`, `CUM` (fixed array), `WHOLE/HALF/QUARTER` (imported from `coding.py`), `TOP_K = K`.
- **Out of scope (do not touch):** `crypto.py`, `seeds.py`, `cli.py`, `utils.py` header format. Sentence-boundary, chunked, and encrypted paths are unchanged.
- **Tuning deferred:** `_DECAY_NUM/_DECAY_DEN` (capacity vs naturalness) and `GUARD` (residual) start at the values in Task 1; tune only if Task 8's capacity check or a residual measurement warrants it.
