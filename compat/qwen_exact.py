"""Recreate the EXACT failing test (test_qwen.py::test_chunked_encrypted_roundtrip)
until it fails, with reproducible randomness so the failing input is replayable.

Faithful to the test: Qwen2.5-0.5B, device="cpu", password="qwen-test-pw", the
exact 23-byte message, chunk_size=20, decode-inclusive. Per-trial os.urandom is
seeded by the trial number, so a captured failure replays deterministically.
Classifies a failure as re-tokenization drift vs coder asymmetry.

    .venv/bin/modal run -m compat.qwen_exact            # CPU fan-out hunt
    .venv/bin/python compat/qwen_exact.py <trial>       # replay one trial locally
"""

from __future__ import annotations

import modal

app = modal.App("mostegollm-qwen-exact")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy", "cryptography", "python-dotenv")
    .env({"HF_HOME": "/cache"})
    .add_local_python_source("mostegollm", "compat")
)
hf_cache = modal.Volume.from_name("mostegollm-hf-cache", create_if_missing=True)

ORIGINAL = b"Secret chunked message!"  # EXACT message from the failing test
CHUNK_SIZE = 20
PASSWORD = "qwen-test-pw"
MODEL = "Qwen/Qwen2.5-0.5B"
HEARTBEAT_EVERY = 5


def run_trials(wid: int, start: int, count: int, capture_first: bool = True):
    """Loop the exact failing scenario on CPU; return the first failing trial."""
    import os
    import random
    import time

    import mostegollm.codec as codec_mod
    from mostegollm import StegoCodec

    t0 = time.time()
    rng = random.Random()
    os.urandom = lambda n: rng.randbytes(n)

    captures: list[dict] = []
    real_encode = codec_mod._encode

    def wrapped(data, *, model, tokenizer, device, prompt, **kw):
        cover, ids, bits = real_encode(
            data, model=model, tokenizer=tokenizer, device=device, prompt=prompt, **kw
        )
        captures.append({"cover": cover, "ids": list(ids), "prompt": prompt})
        return cover, ids, bits

    codec_mod._encode = wrapped

    # Mirror the test fixtures exactly: base codec + encrypted codec sharing the model.
    base = StegoCodec(model_name=MODEL, device="cpu")
    model, tok, dev = base._ensure_model()
    codec = StegoCodec(model_name=MODEL, device="cpu", password=PASSWORD)
    codec._model, codec._tokenizer, codec._device = model, tok, dev
    print(
        f"[w{wid}] loaded in {time.time() - t0:.0f}s; trials {start}..{start + count}", flush=True
    )

    for k in range(count):
        trial = start + k
        captures.clear()
        rng.seed(trial)
        covers = codec.encode(ORIGINAL, chunk_size=CHUNK_SIZE)
        decode_err = None
        try:
            out = codec.decode(covers)
            if out != ORIGINAL:
                decode_err = "silent mismatch (wrong bytes)"
        except Exception as exc:  # noqa: BLE001
            decode_err = f"{type(exc).__name__}: {exc}"

        if decode_err:
            drift = []
            for ci, c in enumerate(captures):
                retok = tok.encode(c["cover"], add_special_tokens=False)
                if retok != c["ids"]:
                    drift.append(
                        {"chunk": ci, "cover": c["cover"], "ids": c["ids"], "retok": retok}
                    )
            verdict = "retok_drift" if drift else "coder_asymmetry"
            print(
                f"[w{wid}] >>> CAPTURED trial {trial}: {verdict} :: {decode_err} "
                f"(drift_chunks={len(drift)}, t={time.time() - t0:.0f}s)",
                flush=True,
            )
            result = {
                "wid": wid,
                "trial": trial,
                "verdict": verdict,
                "decode_err": decode_err,
                "n_chunks": len(captures),
                "drift": drift,
            }
            if capture_first:
                return result
        if k % HEARTBEAT_EVERY == 0:
            print(
                f"[w{wid}] heartbeat trial {trial} ({k}/{count}) t={time.time() - t0:.0f}s",
                flush=True,
            )

    print(f"[w{wid}] DONE {count} trials, no failure", flush=True)
    return None


@app.function(image=image, cpu=2.0, volumes={"/cache": hf_cache}, timeout=10800)
def exact_worker(wid: int, start: int, count: int):
    return run_trials(wid, start, count)


@app.local_entrypoint()
def main(workers: int = 16, per: int = 2000):
    import json
    import pathlib

    args = [(i, i * per + 1, per) for i in range(workers)]
    print(
        f"launching {workers} CPU workers x {per} trials = {workers * per} exact trials", flush=True
    )
    found = []
    for res in exact_worker.starmap(args, order_outputs=False):
        if res:
            found.append(res)
            print(f">>> w{res['wid']} CAPTURED trial {res['trial']}: {res['verdict']}", flush=True)
    if not found:
        print("no failure captured across all workers", flush=True)
        return
    pathlib.Path("compat/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path("compat/results/qwen-exact-failure.json").write_text(json.dumps(found, indent=1))
    print(f"\nwrote compat/results/qwen-exact-failure.json ({len(found)} captures)", flush=True)


if __name__ == "__main__":
    # Local run on THIS machine (the one that originally produced the failure).
    #   python compat/qwen_exact.py            # hunt: loop from trial 1
    #   python compat/qwen_exact.py <start> <count>
    #   python compat/qwen_exact.py <trial> 1  # replay one trial
    import sys

    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    print(f"LOCAL hunt: trials {start}..{start + count} (CPU, exact scenario)", flush=True)
    res = run_trials(0, start, count)
    print("RESULT:", res, flush=True)
