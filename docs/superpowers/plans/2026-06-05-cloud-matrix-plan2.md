# Cloud Matrix (Plan 2) — Modal, minimal smoke

**Date:** 2026-06-05
**Spec:** `docs/superpowers/specs/2026-06-03-cloud-cross-compatibility-testing-design.md`
**Builds on:** Plan 1 foundation (`src/mostegollm/compat.py`, `compat/payloads.py`) on main.

## Scope (this iteration)
Modal-only. RunPod/AMD deferred behind a clean seam. First real prove-run is a
**minimal smoke** of 3 cells, run live (GPU cost accepted):

| cell id | hardware | device | dtype | isolates |
|---------|----------|--------|-------|----------|
| `cpu-fp32` | Modal CPU | cpu | float32 | reference |
| `t4-fp32` | Modal T4 | cuda | float32 | device (vs cpu-fp32) |
| `t4-fp16` | Modal T4 | cuda | float16 | dtype (vs t4-fp32) |

Default model SmolLM-135M. `t4-fp32` sitting between the other two splits a
device-driven divergence from a dtype-driven one.

## Architecture
Fan-out → barrier → fan-out → synthesize, driven from a `@app.local_entrypoint`:
1. **generate** on each cell → tagged `TestVector` dicts (cell id + dtype in `source_env`).
2. barrier: pool all vectors locally.
3. **verify** the *entire pool* on each cell → per (encoder_cell × decoder_cell × payload) result.
4. **dump** per-step top-k ordering for one reference (prompt, token_ids) on each cell.
5. synthesize → Markdown report: N×N decode matrix + cross-cell distribution-agreement.

## Files
- `src/mostegollm/compat.py` (modify) — add `dump_step_logits(prompt, token_ids, *, model, tokenizer, device, top_k, temperature)` returning per-step filtered top-k `(ids, logits)`. TDD.
- `compat/cells.py` (create) — `CELLS` list, `load_model_for_cell(cell)` (uses `mostegollm.model.load_model` for determinism, then casts dtype), and `execute(task, cell, payload)` doing generate/verify/dump. The importable, testable core.
- `compat/report.py` (create) — pure aggregation: results → Markdown matrix + agreement table. TDD with synthetic data.
- `compat/modal_app.py` (create) — image (`pip_install` deps + `add_local_python_source("mostegollm","compat")`), thin `run_cpu`/`run_gpu(gpu="T4")` wrappers around `cells.execute`, and the `local_entrypoint` orchestration. Run: `modal run -m compat.modal_app`.
- `compat/results/` + `docs/` — committed smoke report.

## Determinism in cells
`load_model` (sets `manual_seed(0)`, TF32 off) → cast to cell dtype (`.half()` / `.to(bfloat16)` / leave fp32). `make_vector` then records the vector; cell tags `source_env` with `dtype` + `cell_id` via `dataclasses.replace`.

## Verification
- TDD: `dump_step_logits` (replay a known seq, assert shape/first-token), `report.py` (synthetic results → expected Markdown).
- Live: `modal run -m compat.modal_app` produces the 3×3 matrix; success = every (enc,dec) pair recovers all payloads, OR a classified failure surfaced in the report. Negligible-cost CPU import first, then the T4 cells.

## Deferred (next iterations)
RunPod AMD/ROCm cell; torch-version & transformers-floor cells (need per-cell images); richer GUARD-margin (run-structure hashing); HF-cache `modal.Volume` (smoke re-downloads the 135M model per cell — cheap).
