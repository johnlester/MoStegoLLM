"""Modal app: run the cross-compatibility smoke matrix in the cloud.

Run from the repo root:

    modal run -m compat.modal_app

Fan-out -> barrier -> fan-out -> synthesize:
  1. generate test vectors on each cell,
  2. pool them locally,
  3. verify the whole pool on each cell (the N×N decode matrix),
  4. dump per-step top-k orderings on each cell for one reference sequence,
  5. render a Markdown report locally into compat/results/.

The actual work lives in the Modal-agnostic compat.cells.execute; the functions
here only choose hardware (CPU vs T4). Model weights are cached in a Volume
(HF_HOME=/cache) so only the first cell downloads SmolLM-135M.
"""

from __future__ import annotations

import json
import pathlib

import modal

app = modal.App("mostegollm-compat-matrix")

# Per the project's Modal rules: deps via pip_install, local code via
# add_local_python_source (never baked weights / Mounts). CUDA torch wheel runs
# on both the T4 cells and the CPU cell.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "numpy",
        "cryptography",
        "python-dotenv",
    )
    .env({"HF_HOME": "/cache"})
    .add_local_python_source("mostegollm", "compat")
)

# HF cache so cells after the first don't re-download the model.
hf_cache = modal.Volume.from_name("mostegollm-hf-cache", create_if_missing=True)

_COMMON = {"image": image, "volumes": {"/cache": hf_cache}, "timeout": 900}


@app.function(**_COMMON)
def run_cpu(task: str, cell: dict, payload=None):
    from compat.cells import execute

    result = execute(task, cell, payload)
    hf_cache.commit()
    return result


@app.function(gpu="T4", **_COMMON)
def run_gpu(task: str, cell: dict, payload=None):
    from compat.cells import execute

    result = execute(task, cell, payload)
    hf_cache.commit()
    return result


def _fn_for(cell: dict):
    return run_gpu if cell["device"] == "cuda" else run_cpu


@app.local_entrypoint()
def main():
    from compat.cells import CELLS
    from compat.report import render_report

    cell_ids = [c["id"] for c in CELLS]
    reference = "cpu-fp32"

    # 1-2. Generate on every cell, pool the vectors.
    per_cell: dict[str, list[str]] = {}
    pooled: list[str] = []
    for cell in CELLS:
        print(f"[generate] {cell['id']} ...")
        vectors = _fn_for(cell).remote("generate", cell, None)
        per_cell[cell["id"]] = vectors
        pooled.extend(vectors)
    print(f"pooled {len(pooled)} vectors from {len(CELLS)} cells")

    # 3. Verify the whole pool on every cell (the N×N matrix).
    results: list[dict] = []
    for cell in CELLS:
        print(f"[verify] {cell['id']} decoding {len(pooled)} vectors ...")
        results.extend(_fn_for(cell).remote("verify", cell, pooled))

    # 4. White-box: replay one reference sequence on every cell.
    ref_vec = json.loads(per_cell[reference][0])
    dump_payload = {"prompt": ref_vec["prompt"], "token_ids": ref_vec["generated_token_ids"]}
    dumps: list[dict] = []
    for cell in CELLS:
        print(f"[dump] {cell['id']} ...")
        dumps.append(_fn_for(cell).remote("dump", cell, dump_payload))

    # 5. Render + persist.
    md = render_report(results, cell_ids, dumps=dumps, reference_cell=reference)
    out_dir = pathlib.Path("compat/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke-matrix.md").write_text(md, encoding="utf-8")
    (out_dir / "smoke-raw.json").write_text(
        json.dumps({"results": results, "dumps": dumps}, indent=1), encoding="utf-8"
    )
    print("\n" + md)
    print(f"\nwrote compat/results/smoke-matrix.md ({len(results)} verify results)")
