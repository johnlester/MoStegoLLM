"""Benchmark the non-roundtrip vocab scan: naive per-token vs batched, on the
default model and a larger-vocab model. Tokenizer-only — no model, no GPU.

Run:  .venv/bin/python scratch_scan_bench.py
"""

from __future__ import annotations

import time

from transformers import AutoTokenizer

from mostegollm.encoder import _scan_non_roundtrip


def _naive(tok, ids):
    bad = set()
    for tid in ids:
        if tok.encode(tok.decode([tid]), add_special_tokens=False) != [tid]:
            bad.add(tid)
    return bad


for name in ("HuggingFaceTB/SmolLM-135M", "Qwen/Qwen2.5-0.5B"):
    tok = AutoTokenizer.from_pretrained(name)
    ids = list(range(tok.vocab_size))

    t = time.perf_counter()
    naive = _naive(tok, ids)
    t_naive = time.perf_counter() - t

    t = time.perf_counter()
    batched = set(_scan_non_roundtrip(tok, ids))
    t_batched = time.perf_counter() - t

    ok = "OK" if naive == batched else "MISMATCH!"
    speedup = t_naive / t_batched if t_batched else float("inf")
    print(
        f"{name:32s} vocab={tok.vocab_size:>7d}  "
        f"naive={t_naive:6.2f}s  batched={t_batched:6.2f}s  "
        f"{speedup:5.1f}x  set-equal={ok}"
    )
