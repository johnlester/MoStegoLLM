"""Cell definitions and the per-cell execution core for the cloud matrix.

A *cell* is one (hardware, device, dtype) environment. :func:`execute` runs one
of three tasks on the current machine using that cell's model dtype — generate
test vectors, verify a pooled corpus, or dump per-step top-k orderings. It is
pure Python (no Modal), so it is unit-testable and reusable by any runner.

Vectors cross the Modal boundary as JSON strings (robust, no pickle/version
coupling); verify/dump results are plain dicts/lists.
"""

from __future__ import annotations

import dataclasses

import torch

from compat.payloads import CANONICAL_PAYLOADS, CANONICAL_PROMPTS
from mostegollm.compat import (
    TestVector,
    dump_step_logits,
    make_vector,
    verify_vector,
)
from mostegollm.model import load_model

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"

# Minimal smoke matrix (Modal-only). `hardware` is informational; `device` is the
# torch device; `dtype` is the model cast applied after a deterministic load.
# t4-fp32 sits between cpu-fp32 and t4-fp16 to split device- from dtype-divergence.
CELLS: list[dict] = [
    {"id": "cpu-fp32", "hardware": "cpu", "device": "cpu", "dtype": "float32"},
    {"id": "t4-fp32", "hardware": "T4", "device": "cuda", "dtype": "float32"},
    {"id": "t4-fp16", "hardware": "T4", "device": "cuda", "dtype": "float16"},
]

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model_for_cell(cell: dict):
    """Load the model with mostegollm's deterministic loader, then cast to the cell dtype."""
    model, tokenizer, device = load_model(MODEL_NAME, cell["device"])
    dt = _DTYPES[cell["dtype"]]
    if dt is not torch.float32:
        model = model.to(dt)
    return model, tokenizer, device


def _tag(vector: TestVector, cell: dict) -> TestVector:
    """Stamp the producing cell id + dtype into source_env (frozen-safe)."""
    return dataclasses.replace(
        vector,
        source_env={**vector.source_env, "cell_id": cell["id"], "dtype": cell["dtype"]},
    )


def execute(task: str, cell: dict, payload=None):
    """Run *task* ('generate' | 'verify' | 'dump') for *cell* on this machine."""
    model, tokenizer, device = load_model_for_cell(cell)

    if task == "generate":
        out: list[str] = []
        for prompt in CANONICAL_PROMPTS:
            for payload_bytes in CANONICAL_PAYLOADS.values():
                vector = make_vector(
                    payload_bytes,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    model_name=MODEL_NAME,
                )
                out.append(_tag(vector, cell).to_json())
        return out

    if task == "verify":
        results: list[dict] = []
        for line in payload:
            vector = TestVector.from_json(line)
            result = verify_vector(vector, model=model, tokenizer=tokenizer, device=device)
            results.append(
                {
                    "encoder_cell": vector.source_env.get("cell_id"),
                    "decoder_cell": cell["id"],
                    "prompt": vector.prompt,
                    "payload_hex": vector.payload_hex,
                    "ok": result.ok,
                    "failure_class": (result.failure_class.value if result.failure_class else None),
                    "detail": result.detail,
                }
            )
        return results

    if task == "dump":
        steps = dump_step_logits(
            payload["prompt"],
            payload["token_ids"],
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        return {"cell_id": cell["id"], "top_ids_per_step": [s["top_ids"] for s in steps]}

    raise ValueError(f"unknown task {task!r}")
