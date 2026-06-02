# Cross-Platform Reproducible Coding — Design Spec

**Date:** 2026-06-02
**Status:** Approved (pending spec review)
**Target version:** 0.3.0 (breaks compatibility with 0.2.0 cover texts)

## Problem

Decoding currently requires the *exact same* PyTorch version, device type, and
model weights as encoding. This is empirically confirmed, and stronger than the
docs imply: any floating-point divergence in the logits breaks decode at the
first byte.

### Why it breaks (root cause)

Decode succeeds only if encoder and decoder build **bit-identical** integer
intervals at *every* token step. The current pipeline is:

```
logits (float32 matmul) → float64 softmax → top-k renormalize → int(p · 2³²) → cumulative
```

Two structural problems:

1. **Amplification.** `int(p · 2³²)` turns float32's ~10⁻⁷ relative error into
   ~10⁴ integer units of `cum_probs` drift.
2. **Cascade.** Arithmetic coding is stateful: one differing `cum_probs` entry
   at step *k* desyncs the `low/high` registers, corrupting every step after *k*.

### Measured evidence (SmolLM-135M, top-256)

| Test | Result |
|------|--------|
| Same precision (f32→f32) | decodes |
| Cross-precision (f32→f64), same machine | **fails at byte 0** |
| CPU-encode → GPU-decode | **fails at byte 0** |
| Steps with differing `cum_probs` (f32 vs f64) | **89/89** |
| Steps with top-k **reorder** | **2/89** |
| max `cum_probs` integer drift | ~13,000 / 2³² |
| max \|Δlogit\| CPU vs GPU | **2.88×10⁻⁵** (mean 5.4×10⁻⁶) |
| median adjacent top-k logit gap | **0.0085** (~300× the noise) |

**Key asymmetry:** probability *magnitudes* are 0% stable, but *rank order* is
~98% stable, and the noise (~3×10⁻⁵) is ~300× smaller than where tokens
actually sit. This is what makes a rank-based scheme cheap and effective here.

## Goal

Make reproducible decoding the **default** (single code path; no opt-in mode).
Encoding on one machine/PyTorch/device must decode on another.

### Robustness target (honest framing)

Literal provability is **not attainable** for any deterministic function of
floating-point logits: every scheme eventually compares a float to a threshold,
and that comparison has an irreducible *vulnerable band* the width of the noise.
We therefore target:

> **Empirically zero** cross-platform failures with a wide safety margin
> (validated directly on CPU↔GPU), and any rare residual is **CRC-detected,
> never silent.**

## Design: reproducible rank-interval coding

Replace probability-magnitude intervals with **fixed, constant integer
intervals assigned by rank**.

### Constants (a new module, e.g. `coding.py`)

- `K` — top-k width (e.g. 256).
- `W[0..K-1]` — integer width schedule, derived empirically from the model's
  *average* top-K softmax shape so bits/token stays close to the current scheme.
- `CUM[0..K]` — prefix sums of `W`, with `CUM[0] = 0` and `CUM[K] = 2³² = WHOLE`.
- `GUARD` — gap threshold for run-merging; a constant ≫ measured ε.
  Initial value `1e-3` (~35× the observed max CPU↔GPU \|Δlogit\|).

`W`, `CUM`, `GUARD` are module constants — identical on every machine, **never
computed from floats at runtime.**

### Per-step procedure (identical in encoder and decoder)

1. Run the model; get logits for the current position.
2. Take the top-`K` tokens by logit.
3. Sort by `(logit DESC, token_id ASC)`. The `token_id` tie-break is
   deterministic and platform-independent.
4. **Run-merging:** scan adjacent sorted tokens; whenever the gap to the next
   token is `< GUARD`, they join the same *run*. Each run is represented by its
   lowest `token_id`.
5. Interval assignment (fixed `CUM`, no float widths):
   - A standalone token at sorted position `i` owns `[CUM[i], CUM[i+1])`.
   - A run spanning positions `a..b` owns `[CUM[a], CUM[b+1])`, owned by its
     representative token.

### Encode

Arithmetic-navigate the `value` register (filled from the secret bit stream,
plus the deterministic xorshift32 padding past the payload — unchanged from the
sentence-boundary fix) to the interval/run that contains it, and emit that token
(or the run's representative). Interval narrowing and MSB renormalization are
unchanged.

### Decode

Observe the emitted token, find its sorted position, determine its run, and emit
bits for the run's `[CUM[a], CUM[b+1])`. The decoder makes **no probability-
magnitude comparison** — only the sort and the local `GUARD` gap test.

## Why this is reproducible

- `CUM` is a fixed constant array, so interval *boundaries* never depend on
  float-derived widths or on which tokens are present (no "occupancy"
  instability).
- A token's sorted position = the count of tokens above it. This is invariant
  unless another token *crosses* it, which requires a gap `< ε`.
- Run-merging absorbs exactly those near-ties: tokens within `GUARD` of each
  other share one interval, so a sub-`GUARD` reordering inside a run does not
  change the run's `[CUM[a], CUM[b+1])`.
- Far-away reorderings (above or below the emitted token's run) don't change the
  count of tokens above it, so they don't move its interval.

### Residual (the one remaining vulnerability)

The only float threshold left is the per-gap `gap < GUARD` test. A failure
requires a **run-boundary gap** of the emitted token's run to land within ~ε of
`GUARD` on one platform but not the other. This is:

- **Local** — only gaps at the emitted token's own run boundaries matter.
- **Rare** — the vulnerable band is ~2ε wide; `GUARD` is chosen where the gap
  density is low and ≫ ε.
- **Detected** — the CRC-32 in the header fails closed; decode raises
  `StegoDecodeError`, never returns wrong bytes.

We will **measure** the empirical residual on CPU↔GPU rather than claim a proof.

## Determinism hardening (belt-and-suspenders)

In model setup, additionally: `torch.use_deterministic_algorithms(True)` where
feasible, disable TF32 (`torch.backends.cuda.matmul.allow_tf32 = False`,
`torch.backends.cudnn.allow_tf32 = False`). These shrink ε; the scheme does not
depend on them for correctness.

## Capacity & text quality

Widths come from rank, not the true distribution, so:

- **Capacity:** bits/token ≈ the entropy of the `W` schedule. By fitting `W` to
  the model's average top-K shape, this should stay close to current. To be
  **measured** (new vs old bits/token) during implementation and reported.
- **Quality:** tokens are sampled per the fixed schedule rather than the exact
  model distribution; near-ties resolve to the lowest `token_id`. Because the
  schedule tracks the average shape and near-ties are cosmetically irrelevant,
  generated prose should remain natural. Spot-checked during implementation.

## Compatibility & versioning

- **Breaks** decoding of any cover text produced by ≤ 0.2.0. Accepted.
- Bump to **0.3.0**. Header format (magic + 4-byte length + 4-byte CRC-32) is
  unchanged.
- Document the break prominently in README and a CHANGELOG entry; update the
  cross-platform caveat to describe the new guarantee and its residual.

## Testing

1. **CPU↔GPU regression test** (new, marked `slow`/gpu-gated): encode N random
   payloads on CPU, decode on GPU, assert 0 failures. This is the empirical
   proof — it fails on the current code and must pass on the new code.
2. **float32↔float64 test:** encode with the f32 model, decode with an f64 copy;
   assert success.
3. **Residual probe:** over many steps/payloads, count run-boundary gaps within
   ε of `GUARD`; report the measured residual rate.
4. **Capacity report:** bits/token, new vs old, on a fixed payload set.
5. All existing round-trip, property/fuzz, chunked, encrypted, and
   sentence-boundary tests must still pass (with regenerated expectations where
   they assert exact behavior).

## Module impact

- **New** `coding.py` — `K`, `W[]`, `CUM[]`, `GUARD`, and the
  sort/run-merge/interval helpers shared by encoder and decoder.
- **`encoder.py` / `decoder.py`** — replace `_get_token_distribution`'s
  magnitude-based `cum_probs` with the rank-interval construction from
  `coding.py`; keep the BPE round-trip filter, interval narrowing, and
  renormalization. The two sides continue to share the construction so they
  cannot drift.
- **`model.py`** — determinism hardening.
- **`codec.py`, `crypto.py`, `seeds.py`, `cli.py`** — unaffected.

## Open implementation risks (validate empirically)

- Exact `W[]` schedule and the resulting bits/token.
- Final `GUARD` value vs. measured residual (tune for ~0 on CPU↔GPU with margin).
- top-K cutoff instability (the `K`-th token can flip membership): confirm the
  fixed-`CUM` design makes this harmless for emitted tokens, or guard the cutoff.
- Interaction of run-merging with the BPE round-trip filter ordering.
