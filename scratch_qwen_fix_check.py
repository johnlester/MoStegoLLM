"""#8 check: run the EXACT Qwen chunked-encrypted scenario under the special-token
fix, on GPU. Confirm 0 decode failures, and instrument the filter to count how
often a Qwen special token was actually in the top-k and got dropped — i.e. how
often the fix mattered along these paths.

Faithful to test_qwen.py::test_chunked_encrypted_roundtrip except device=cuda:
ORIGINAL 23-byte msg, chunk_size=20, password "qwen-test-pw", Qwen2.5-0.5B,
per-trial seeded os.urandom so any capture is replayable.
"""

from __future__ import annotations

import os
import random
import sys
import time

import mostegollm.codec as codec_mod
import mostegollm.encoder as enc_mod
from mostegollm import StegoCodec

ORIGINAL = b"Secret chunked message!"
CHUNK_SIZE = 20
PASSWORD = "qwen-test-pw"
MODEL = "Qwen/Qwen2.5-0.5B"

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def main() -> None:
    rng = random.Random()
    os.urandom = lambda n: rng.randbytes(n)

    base = StegoCodec(model_name=MODEL, device="cuda")
    model, tok, dev = base._ensure_model()
    codec = StegoCodec(model_name=MODEL, device="cuda", password=PASSWORD)
    codec._model, codec._tokenizer, codec._device = model, tok, dev

    special_ids = {s for s in tok.all_special_ids if s is not None}

    # Instrument the shared filter: count steps where a special token was a top-k
    # candidate and got removed (the fix firing), across both encode and decode.
    real_filter = enc_mod._filter_tokens
    stats = {"steps": 0, "special_in_topk_steps": 0, "special_dropped": 0}

    def counting_filter(tokenizer, prev, token_ids, logits, non_rt, merge_cache):
        stats["steps"] += 1
        present = special_ids.intersection(token_ids)
        if present:
            stats["special_in_topk_steps"] += 1
            stats["special_dropped"] += len(present)
        return real_filter(tokenizer, prev, token_ids, logits, non_rt, merge_cache)

    codec_mod._filter_tokens = counting_filter
    enc_mod._filter_tokens = counting_filter
    # decoder imported _filter_tokens by name — patch there too.
    import mostegollm.decoder as dec_mod

    dec_mod._filter_tokens = counting_filter

    t0 = time.time()
    failures = []
    for trial in range(1, N_TRIALS + 1):
        rng.seed(trial)
        covers = codec.encode(ORIGINAL, chunk_size=CHUNK_SIZE)
        try:
            out = codec.decode(covers)
            if out != ORIGINAL:
                failures.append((trial, "silent mismatch"))
        except Exception as exc:  # noqa: BLE001
            failures.append((trial, f"{type(exc).__name__}: {exc}"))
        if trial % 50 == 0:
            print(
                f"  trial {trial}/{N_TRIALS} t={time.time() - t0:.0f}s "
                f"fail={len(failures)} special_steps={stats['special_in_topk_steps']}",
                flush=True,
            )

    print(f"\nspecial_ids({MODEL}): {sorted(special_ids)}")
    print(f"total filter steps:            {stats['steps']}")
    print(
        f"steps w/ special in top-k:     {stats['special_in_topk_steps']}  "
        f"(these are the steps the fix protects)"
    )
    print(f"special tokens dropped:        {stats['special_dropped']}")
    print(f"trials:                        {N_TRIALS}")
    print(f"decode failures:               {len(failures)}")
    if failures:
        for f in failures[:10]:
            print("  FAIL", f)
    print(f"\nRESULT: {'PASS (0 failures)' if not failures else 'FAIL'}")


if __name__ == "__main__":
    main()
