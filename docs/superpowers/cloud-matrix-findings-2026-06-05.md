# Cloud cross-compatibility matrix — first results (2026-06-05)

First live run of the Modal smoke matrix (`modal run -m compat.modal_app`).
Raw report: `compat/results/smoke-matrix.md`; raw data: `compat/results/smoke-raw.json`.
Model: SmolLM-135M. 3 cells, full 3×3 encode×decode over 24 (prompt × payload) cases each.

## Result

| enc \ dec | cpu-fp32 | t4-fp32 | t4-fp16 |
|---|---|---|---|
| **cpu-fp32** | ✓ 24/24 | ✓ 24/24 | ✗ 4/24 |
| **t4-fp32** | ✓ 24/24 | ✓ 24/24 | ✗ 4/24 |
| **t4-fp16** | ✗ 4/24 | ✗ 4/24 | ✓ 24/24 |

Top-k ordering agreement vs `cpu-fp32` (reference sequence, 28 steps):
cpu-fp32 100% · t4-fp32 100% · t4-fp16 0%.

## Interpretation

**Device axis is solved.** `cpu-fp32 ↔ t4-fp32` round-trips perfectly in both
directions with 100% per-step top-k ordering agreement. This is the empirical
proof the 0.3.0 design targeted: encode on CPU, decode on a different GPU vendor
arch (and vice versa) at the same dtype works. `GUARD = 1e-3` comfortably absorbs
the ~3e-5 CPU↔GPU logit divergence measured earlier.

**Dtype axis is NOT compatible.** fp16 cross-decodes with fp32 fail (4/24), and
the white-box check shows *why*: 0% top-k ordering agreement — fp16's ~1e-3
relative quantization error is the *same order of magnitude* as `GUARD`, so the
sorted token order (and therefore the rank intervals) flips at essentially every
step. fp16 only decodes its own fp16-encoded vectors (24/24 self).

The surviving 4/24 fp16-cross cases are the shortest payloads (e.g. single byte,
all-zero), where the encoder emits too few tokens for divergence to accumulate
past the header/CRC before the payload is already recoverable.

## Consequence for the design

The portability envelope is **"same model dtype; any device / PyTorch version"** —
not "any dtype." README updated to state this (the prior "portable across …
dtypes by design" claim was an overclaim this run falsified).

Options if cross-dtype interop is ever wanted (not currently a goal):
- Pin a canonical dtype for encode/decode (simplest; what we effectively require now).
- Raise `GUARD` above fp16's error band (~1e-2) — costs capacity (more tokens merge
  into runs, fewer bits per token) and still wouldn't help bf16's larger error.
- Have `verify_vector` warn/skip when decoder dtype ≠ the vector's recorded dtype
  (the vector already records `source_env.dtype`); turns silent cross-dtype
  failures into an explicit, classified outcome.

## Matrix mechanics validated

The end-to-end pipeline works: per-cell `generate` → pooled corpus → per-cell
`verify` (N×N) → per-cell `dump` → Markdown report. HF-cache Volume meant only the
first cell downloaded the model. Ready to extend (more NVIDIA archs, torch-version
cells, and the deferred RunPod AMD/ROCm cell) by adding entries to `compat/cells.py:CELLS`.
