"""Deterministic diagnostic harness for the intermittent Qwen chunked-encrypted
decode failure (test_qwen.py::TestQwenChunked::test_chunked_encrypted_roundtrip).

Reproduces the exact failing path (Qwen, password, chunk_size=20, seed-phrase
mode) but makes every trial REPRODUCIBLE by seeding os.urandom per trial. Wraps
codec._encode to capture (prompt, cover_text, generated_ids) per chunk, then on
the first failure classifies it:

  * re-tokenization drift  -> tokenizer.encode(cover) != generated_ids
  * coder asymmetry        -> retok matched everywhere but decode still failed

Run:  .venv/bin/python -u scratch_repro.py [trial_to_reproduce]
"""

from __future__ import annotations

import os
import random
import sys

import torch

import mostegollm.codec as codec_mod
from mostegollm import StegoCodec

ORIGINAL = b"Secret chunked message! " * 6  # ~144 bytes -> ~8 chunks at size 20
CHUNK_SIZE = 20
MAX_TRIALS = 3000

# --- deterministic os.urandom so each trial is reproducible -------------------
_rng = random.Random()
_real_urandom = os.urandom


def _seed_trial(trial: int) -> None:
    _rng.seed(trial)


def _fake_urandom(n: int) -> bytes:
    return _rng.randbytes(n)


os.urandom = _fake_urandom  # patches mostegollm.crypto's os.urandom too

# --- capture every encoder.encode call made through the codec ----------------
_real_encode = codec_mod._encode
_captures: list[dict] = []


def _wrapped_encode(data, *, model, tokenizer, device, prompt, **kw):
    cover_text, ids, bits = _real_encode(
        data, model=model, tokenizer=tokenizer, device=device, prompt=prompt, **kw
    )
    _captures.append({"prompt": prompt, "cover_text": cover_text, "ids": list(ids)})
    return cover_text, ids, bits


codec_mod._encode = _wrapped_encode


def classify(codec: StegoCodec, trial: int, exc: object) -> None:
    tok = codec._tokenizer
    print(f"\n=== FAILURE at trial {trial}: {type(exc).__name__}: {exc} ===", flush=True)
    print(f"=== {len(_captures)} chunks encoded ===", flush=True)
    drift = False
    for i, cap in enumerate(_captures):
        gen = cap["ids"]
        retok = tok.encode(cap["cover_text"], add_special_tokens=False)
        match = retok == gen
        flag = "OK" if match else "DRIFT"
        print(f"chunk {i}: gen={len(gen)} retok={len(retok)} re-tokenization {flag}", flush=True)
        if not match:
            drift = True
            for j, (a, b) in enumerate(zip(gen, retok)):
                if a != b:
                    lo = max(0, j - 3)
                    print(
                        f"   first divergence at pos {j}: gen={a} ({tok.decode([a])!r}) "
                        f"vs retok={b} ({tok.decode([b])!r})",
                        flush=True,
                    )
                    print(
                        f"   gen[{lo}:{j + 2}]={gen[lo : j + 2]} -> {tok.decode(gen[lo : j + 2])!r}",
                        flush=True,
                    )
                    break
            else:
                print(f"   length-only diff (gen {len(gen)} vs retok {len(retok)})", flush=True)
            print(f"   cover repr: {cap['cover_text']!r}", flush=True)
    print(
        "\n>>> VERDICT:",
        "RE-TOKENIZATION DRIFT" if drift else "CODER ASYMMETRY (retok matched, decode failed)",
        flush=True,
    )


def run_trial(codec: StegoCodec, trial: int) -> bool:
    """Return True if the trial reproduced a failure (and classified it)."""
    _captures.clear()
    _seed_trial(trial)
    covers = codec.encode(ORIGINAL, chunk_size=CHUNK_SIZE)
    try:
        out = codec.decode(covers)
        if out != ORIGINAL:
            classify(codec, trial, ValueError("silent mismatch (wrong bytes, no exception)"))
            return True
    except Exception as exc:  # noqa: BLE001 — classify any failure
        classify(codec, trial, exc)
        return True
    return False


def main() -> None:
    codec = StegoCodec(model_name="Qwen/Qwen2.5-0.5B", device="cuda", password="qwen-test-pw")
    codec._ensure_model()
    print("model loaded; fuzzing (deterministic per-trial seed)...", flush=True)

    if len(sys.argv) > 1:  # reproduce a specific trial
        t = int(sys.argv[1])
        print(f"reproducing trial {t}", flush=True)
        print("reproduced!" if run_trial(codec, t) else "trial passed (no failure)", flush=True)
        return

    for trial in range(1, MAX_TRIALS + 1):
        if run_trial(codec, trial):
            print(f"\n(reproduce with: python -u scratch_repro.py {trial})", flush=True)
            return
        if trial % 20 == 0:
            print(f"  {trial} trials, no failure yet", flush=True)
            torch.cuda.empty_cache()
    print(f"\nNo failure in {MAX_TRIALS} trials.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        os.urandom = _real_urandom
