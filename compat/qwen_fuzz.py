"""2 streaming Modal workers hunting the intermittent Qwen chunked-encrypted bug.

Run (venv modal so add_local_python_source resolves):

    .venv/bin/modal run -m compat.qwen_fuzz

Each worker reproduces the exact failing path (Qwen2.5-0.5B, password,
chunk_size=20, seed-phrase mode) over a disjoint, deterministic trial range,
printing a heartbeat every few trials so progress is visible (the earlier silent
fan-out only *looked* stalled). On the first captured failure a worker prints +
returns the classification; the returned cover+ids reproduce the drift locally
with just the tokenizer.
"""

from __future__ import annotations

import modal

app = modal.App("mostegollm-qwen-fuzz")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy", "cryptography", "python-dotenv")
    .env({"HF_HOME": "/cache"})
    .add_local_python_source("mostegollm", "compat")
)
hf_cache = modal.Volume.from_name("mostegollm-hf-cache", create_if_missing=True)

ORIGINAL = b"Secret chunked message! " * 6  # ~144 bytes -> ~8 chunks at size 20
CHUNK_SIZE = 20
HEARTBEAT_EVERY = 10  # trials
# Encode-only by default: the per-chunk re-tokenization-drift check IS the direct
# test of the leading hypothesis, and skipping decode ~doubles throughput (vital
# for a rare bug). Flip to True to also catch a pure coder asymmetry.
DECODE_CHECK = False
GPU = "A10G"  # ~2.5x T4; T4 was ~73s/trial (too slow for a rare bug)


def _fuzz_worker(wid: int, start: int, count: int):
    import os
    import random
    import time

    import mostegollm.codec as codec_mod
    from mostegollm import StegoCodec

    t0 = time.time()
    rng = random.Random()
    os.urandom = lambda n: rng.randbytes(n)  # deterministic per-trial encryption

    captures: list[dict] = []
    real_encode = codec_mod._encode

    def wrapped(data, *, model, tokenizer, device, prompt, **kw):
        cover, ids, bits = real_encode(
            data, model=model, tokenizer=tokenizer, device=device, prompt=prompt, **kw
        )
        captures.append({"cover": cover, "ids": list(ids)})
        return cover, ids, bits

    codec_mod._encode = wrapped

    codec = StegoCodec(model_name="Qwen/Qwen2.5-0.5B", device="cuda", password="qwen-test-pw")
    _, tok, _ = codec._ensure_model()
    print(
        f"[w{wid}] loaded in {time.time() - t0:.0f}s; trials {start}..{start + count}", flush=True
    )

    drift_chunks = 0
    for k in range(count):
        trial = start + k
        captures.clear()
        rng.seed(trial)
        covers = codec.encode(ORIGINAL, chunk_size=CHUNK_SIZE)

        # Per-chunk re-tokenization drift check (the hypothesized cause).
        trial_drift = []
        for ci, c in enumerate(captures):
            retok = tok.encode(c["cover"], add_special_tokens=False)
            if retok != c["ids"]:
                trial_drift.append(
                    {"chunk": ci, "cover": c["cover"], "ids": c["ids"], "retok": retok}
                )
        drift_chunks += len(trial_drift)

        # Decode (catches drift-induced failure AND any coder asymmetry).
        decode_err = None
        if DECODE_CHECK:
            try:
                out = codec.decode(covers)
                if out != ORIGINAL:
                    decode_err = "silent mismatch (wrong bytes, no exception)"
            except Exception as exc:  # noqa: BLE001
                decode_err = f"{type(exc).__name__}: {exc}"

        if trial_drift or decode_err:
            verdict = "retok_drift" if trial_drift else "coder_asymmetry"
            print(
                f"[w{wid}] >>> CAPTURED trial {trial}: verdict={verdict} "
                f"decode_err={decode_err} drift_chunks={len(trial_drift)} "
                f"t={time.time() - t0:.0f}s",
                flush=True,
            )
            return {
                "wid": wid,
                "trial": trial,
                "verdict": verdict,
                "decode_err": decode_err,
                "drift": trial_drift,
                "elapsed_s": time.time() - t0,
            }

        if k % HEARTBEAT_EVERY == 0:
            print(
                f"[w{wid}] heartbeat: trial {trial} ({k}/{count}) "
                f"drift_chunks={drift_chunks} t={time.time() - t0:.0f}s",
                flush=True,
            )

    print(f"[w{wid}] DONE {count} trials, no failure (drift_chunks={drift_chunks})", flush=True)
    return None


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache}, timeout=10800)
def fuzz_worker(wid: int, start: int, count: int):
    return _fuzz_worker(wid, start, count)


@app.local_entrypoint()
def main():
    import json
    import pathlib

    budget = 4000  # per worker; bounded by the 7200s timeout in practice
    workers = [(0, 1, budget), (1, 1_000_000, budget)]  # disjoint deterministic ranges
    print(f"launching 2 {GPU} fuzz workers (streaming heartbeats)...", flush=True)

    found = []
    for res in fuzz_worker.starmap(workers, order_outputs=False):
        if res:
            found.append(res)
            print(f">>> w{res['wid']} CAPTURED trial {res['trial']}: {res['verdict']}", flush=True)

    if not found:
        print("both workers finished with no failure", flush=True)
        return

    pathlib.Path("compat/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path("compat/results/qwen-failure.json").write_text(json.dumps(found, indent=1))
    for f in found:
        print(
            f"\nVERDICT w{f['wid']} trial {f['trial']}: {f['verdict']} (decode_err={f['decode_err']})"
        )
        for d in f["drift"]:
            g, r = d["ids"], d["retok"]
            j = next((k for k, (a, b) in enumerate(zip(g, r)) if a != b), min(len(g), len(r)))
            lo = max(0, j - 3)
            print(
                f"  chunk {d['chunk']} drift at pos {j}: gen={g[lo : j + 2]} retok={r[lo : j + 2]}"
            )
            print(f"    cover: {d['cover']!r}")
    print("\nwrote compat/results/qwen-failure.json", flush=True)


# --- One-off extra worker on a beefier GPU (independent of the 2x A10G run) ----
@app.function(image=image, gpu="RTX-PRO-6000", volumes={"/cache": hf_cache}, timeout=3600)
def fuzz_worker_rtx(wid: int, start: int, count: int):
    return _fuzz_worker(wid, start, count)


@app.local_entrypoint()
def one():
    """Run a single RTX-PRO-6000 worker for ~1h on a trial range disjoint from the
    A10G workers (w0: 1.., w1: 1_000_000..). Launch: modal run -m compat.qwen_fuzz::one
    """
    print("launching 1 RTX-PRO-6000 worker (1h, trials 2_000_000..)...", flush=True)
    res = fuzz_worker_rtx.remote(2, 2_000_000, 4000)
    if res:
        print(
            f">>> w2 CAPTURED trial {res['trial']}: {res['verdict']} decode_err={res['decode_err']}"
        )
        import json
        import pathlib

        pathlib.Path("compat/results").mkdir(parents=True, exist_ok=True)
        pathlib.Path("compat/results/qwen-failure-rtx.json").write_text(json.dumps(res, indent=1))
        for d in res["drift"]:
            g, r = d["ids"], d["retok"]
            j = next((k for k, (a, b) in enumerate(zip(g, r)) if a != b), min(len(g), len(r)))
            lo = max(0, j - 3)
            print(
                f"  chunk {d['chunk']} drift at pos {j}: gen={g[lo : j + 2]} retok={r[lo : j + 2]}"
            )
            print(f"    cover: {d['cover']!r}")
    else:
        print("w2 (RTX) finished, no failure", flush=True)
