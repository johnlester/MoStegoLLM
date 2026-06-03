# Cloud Cross-Compatibility Testing — Design Spec

**Date:** 2026-06-03
**Status:** Draft (pending spec review)
**Depends on:** `2026-06-02-cross-platform-reproducible-coding-design.md` (the
rank-interval coder whose portability this spec sets out to *prove* and *guard*).

## Problem

The `reproducible-coding` branch claims cover text encoded on one
PyTorch/device/dtype decodes correctly on another. Today that claim is only
backed by `tests/test_cross_platform.py`, which **simulates** divergence in a
single process: it copies the loaded model into float64 and onto the GPU and
round-trips between those copies. That is a good smoke test, but it does not
exercise *genuinely different* environments — different PyTorch versions, GPU
vendors (NVIDIA arch variants, AMD ROCm), CPU ISAs, or dtype regimes on real
hardware. We have no empirical proof of the portability envelope, and no
regression guard against future changes (a model swap, a `GUARD` tweak, a coder
refactor) silently narrowing it.

We want two things:

1. **Prove** the envelope once, empirically, across a real hardware/version/dtype
   matrix.
2. **Bake in** a cheap, recurring guard so regressions surface automatically.

## Non-goals

- **Apple MPS.** Neither Modal nor RunPod offers Apple Silicon. MPS is documented
  as a manual-on-a-Mac gap, not silently claimed as covered.
- **Tokenizer-version drift as an encode-divergence axis.** See "What we
  *measured*" below — for the byte-level-BPE tokenizers this project uses, encode
  output does not change across transformers/`tokenizers` versions. We do **not**
  build a tokenizer-version cell to test encode drift. (We keep a cheap
  *floor-version load* check instead; see Components.)
- A general-purpose distributed test framework. This is scoped to the codec's
  reproducibility claim.

## What we *measured* (grounding two design decisions)

Two assumptions were checked empirically before writing this spec, rather than
assumed:

**1. Logit divergence is real but tiny (from the prior spec).** CPU vs GPU
`|Δlogit|` ≤ 2.88×10⁻⁵ on SmolLM-135M; `GUARD = 1e-3` is ~35× that margin. The
matrix must *confirm* this holds on hardware/dtype combos we have not measured
(H100 bf16, AMD ROCm fp16, etc.) — that is the core scientific question.

**2. Tokenizer encode output does NOT drift across library versions for these
models.** Diffed token IDs over a 15-string adversarial corpus:

| Model | Tokenizer | Versions | `tokenizers` backend | Differing strings |
|-------|-----------|----------|----------------------|-------------------|
| SmolLM-135M | `GPT2TokenizerFast` | tf 4.35.2 → 4.45.2 | 0.15.2 → 0.20.3 | **0** |
| Qwen2.5-0.5B | `Qwen2TokenizerFast` | tf 4.37.2 → 4.45.2 | 0.15.2 → 0.20.3 | **0** |

Both are **byte-level BPE** (GPT-2 lineage): no normalization step, fixed GPT-2
pre-tokenization regex, vocab+merges pinned by `model_revision`. The era's known
tokenizer-encode changes (Llama/T5 `legacy=`, SentencePiece `add_prefix_space`)
structurally cannot affect them. **Consequence:** the re-tokenization drift that
actually breaks this codec is *intrinsic and single-version* — a generated token
re-tokenizing differently in 3+-token context on one library version — not a
cross-version phenomenon. The guard for it belongs in *every* cell as a white-box
check, not in a dedicated tokenizer-version cell.

One real version finding of a *different kind*: **transformers 4.35 cannot load
Qwen2.5 at all** (`Qwen2Tokenizer` didn't exist until ~4.37). That is model-class
*availability*, motivating a floor-version load check rather than an
encode-drift cell.

## Core abstraction: the test vector

The only artifact that crosses an environment boundary is a **test vector** — a
self-describing JSON record (stored one-per-line as JSONL):

```json
{ "schema": "mostegollm-testvector/1",
  "library_version": "0.3.0",
  "model": "HuggingFaceTB/SmolLM-135M", "model_revision": "<HF commit sha>",
  "prompt": "According to experts,",
  "settings": {"top_k": 256, "temperature": 1.0, "sentence_boundary": false,
               "password": null},
  "payload_sha256": "<hex>", "payload_hex": "<hex>",
  "cover_text": "...",
  "generated_token_ids": [ ... ],
  "source_env": {"torch": "2.5.1", "transformers": "4.45.2", "tokenizers": "0.20.3",
                 "device": "H100", "dtype": "float32", "os": "linux", "arch": "x86_64"} }
```

- `payload_sha256` lets a decoder verify recovery without trusting `payload_hex`.
- `generated_token_ids` enables the **re-tokenization-drift check**
  (`tokenizer.encode(cover_text, add_special_tokens=False) == generated_token_ids`),
  the single-version invariant identified above.
- `model_revision` pins the exact vocab/merges so a tokenizer change cannot be
  confused for a coder change.
- `source_env` makes every result traceable to the exact environment that
  produced it.

Everything else is "produce vectors" or "consume (decode/verify) vectors."

## Architecture: two phases

```
                    PROVE PHASE (one-time, N×N)
  each cell:  generate → vectors/<cell>.jsonl  (+ white-box logit dump)
                                │  barrier: merge all into one pooled corpus
                                ▼
  each cell:  decode the ENTIRE pooled corpus → (encoder_cell × decoder_cell × payload) result
                                ▼
            N×N matrix report   +   GUARD-margin table

                    BAKE-IN PHASE (recurring, star)
  reference cell encodes → compat/golden_vectors.jsonl  (committed to repo)
        ├─► tests/test_golden_vectors.py   (decoded on every local pytest run)
        └─► scheduled Modal job: decode golden corpus on {1 GPU, 1 CPU,
            1 bumped-deps image} → regression guard
```

The prove phase is fan-out → barrier → fan-out → synthesize. The bake-in phase
collapses it to one trusted encoder + many decoders and commits the encoder's
output so the guarantee is checkable even offline.

## Components & locations

| Component | Location | Purpose |
|-----------|----------|---------|
| `TestVector` dataclass, `make_vector`, `verify_vector`, `dump_step_logits`, JSONL IO | **`src/mostegollm/compat.py`** (shipped, importable in any cloud image) | Pure, locally-testable. No cloud dependency. |
| Canonical payloads, prompts, cell definitions | `compat/payloads.py` | Fixed and **deterministic** (no `os.urandom` — a direct lesson from the local intermittent bug). |
| Modal orchestration | `compat/modal_app.py` | Per-cell `modal.Image`s; deploy `modal deploy -m compat.modal_app`. |
| RunPod/Lambda job (AMD ROCm) | `compat/runpod_job.py` | The exotic hardware the others can't reach. |
| Matrix aggregation + report | `compat/report.py` | Renders the N×N grid + GUARD-margin table. |
| Floor-version load check | `compat/floor_check.py` (or a Modal fn) | Asserts the model *instantiates* on min-supported transformers (catches the "Qwen needs ≥4.37" class). Cheap, CPU. |
| Committed golden corpus | `compat/golden_vectors.jsonl` | Reference encoder output for the bake-in guard. |
| Offline guard | `tests/test_golden_vectors.py` | Decodes golden corpus on every `pytest` run. |

`make_vector` / `verify_vector` / `dump_step_logits` live **inside the package**
because the cloud verifier must `import mostegollm` anyway; making them
first-class functions means a cell just `pip install mostegollm` + import,
and they get local unit tests for free (`tests/test_compat.py`).

## The matrix (initial; tunable)

| Cell | Platform | Hardware | torch | transformers | dtype | Tests primarily |
|------|----------|----------|-------|--------------|-------|-----------------|
| `ref` | Modal | CPU x86 | pinned 2.5 | pinned 4.45 | fp32 | reference encoder |
| `gpu-ampere` | Modal | A10G | 2.5 | 4.45 | fp16 | GUARD margin |
| `gpu-hopper` | Modal | H100 | 2.5 | 4.45 | bf16 | GUARD margin (lowest-precision dtype) |
| `gpu-old-torch` | Modal | A10G | **2.1** | 4.45 | fp32 | kernel/BLAS logit diffs |
| `cpu-fp64` | Modal | CPU x86 | 2.5 | 4.45 | **fp64** | dtype extreme |
| `amd-rocm` | RunPod | MI-series | rocm build | 4.45 | fp16 | cross-vendor GPU |
| `floor-load` | Modal | CPU x86 | 2.5 | **min-supported** | — | model *loads* on floor version |

Notes:
- `floor-load` replaces the originally-proposed tokenizer-version cell (which we
  measured to be a no-op for these models).
- Each Modal cell is a distinct `modal.Image` (different pinned `torch`/
  `transformers`); every image calls `.add_local_python_source("mostegollm")`;
  model weights live in a `modal.Volume` (not baked into the image, no
  `@modal.build`); deploy is `modal deploy -m compat.modal_app`.
- Crypto (`password=`) is an *optional* extra dimension, not core: AES-GCM is
  platform-independent, and the cover text already bakes in the ciphertext, so
  decode of a stored vector round-trips deterministically regardless of platform.

## Failure classification & reporting

When a decode fails, `verify_vector` classifies it (reusing the diagnostic logic
written while chasing the local intermittent bug):

1. **Re-tokenization drift** — `tokenizer.encode(cover_text) != generated_token_ids`
   → BPE round-trip instability (intrinsic, single-version).
2. **Logit-ordering divergence** — re-tokenization matched but decode desyncs →
   `GUARD` margin exceeded by hardware/dtype. Quantified by the white-box dump.
3. **Load/version error** — model or dependency failure (not a coding bug);
   `floor-load` surfaces these explicitly.

Outputs:
- **N×N grid** (encoder cell × decoder cell), each entry ✓ or a class tag.
- **GUARD-margin table** — per cell-pair, the smallest *decisive* logit gap
  observed (the tightest gap between adjacent ranks that the coder actually
  relied on) vs `GUARD = 1e-3`. This tells us whether `1e-3` is comfortable or
  one model-swap from breaking — the real scientific payoff.

## Determinism requirements

- Canonical payloads are fixed bytes (seeded if pseudo-random), **never**
  `os.urandom` — the intermittent Qwen failure traces to a random-payload test,
  and a reproducibility corpus must itself be reproducible.
- `torch.manual_seed(0)` and TF32-disabled determinism (already in `model.py`)
  apply in every cell.
- `model_revision` is pinned in every vector.

## Testing strategy

- `tests/test_compat.py` — local unit tests: `make_vector` → `verify_vector`
  round-trips on the default model; `dump_step_logits` shape/values; JSONL IO.
  No cloud needed.
- `tests/test_golden_vectors.py` — decodes the committed `golden_vectors.jsonl`
  on whatever the dev machine is, every `pytest` run (offline guard).
- Cloud cells run `verify_vector` over the pooled corpus; the run *fails* if any
  vector mis-decodes, and emits the classified report regardless.

## Deliverables

1. `src/mostegollm/compat.py` + `tests/test_compat.py`.
2. `compat/` orchestration: `payloads.py`, `modal_app.py`, `runpod_job.py`,
   `report.py`, `floor_check.py`.
3. `compat/golden_vectors.jsonl` + `tests/test_golden_vectors.py`.
4. A generated results report (N×N matrix + GUARD-margin table) committed under
   `docs/` from the prove run.
5. README update: the verified portability envelope and the Apple-MPS gap.

## Open questions

- Exact min-supported transformers floor to assert in `floor-load` (≥4.37 for
  Qwen2.5; SmolLM works on 4.35 — pick the higher to support both, or per-model).
- Whether the prove-phase matrix runs as a single Modal app orchestrating RunPod,
  or as two independently-launched runs whose vectors are merged offline.
- Report format: Markdown (diff-friendly in git) vs HTML (richer). Default
  Markdown for the committed artifact.
