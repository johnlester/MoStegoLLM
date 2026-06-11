"""Baseline benchmark + profile of encode/decode on the default model (SmolLM-135M).

Confirms where time goes before optimizing the BPE filter. Run on GPU so the
model forward is cheap and the CPU-side tokenizer work is the visible cost.

Run:  .venv/bin/python scratch_bench.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

from mostegollm import StegoCodec

PAYLOAD = bytes((i * 73 + 11) & 0xFF for i in range(200))  # fixed 200-byte payload


def main() -> None:
    codec = StegoCodec(device="cuda")
    codec._ensure_model()

    # Warm up: populate the non-roundtrip cache and CUDA kernels.
    cover = codec.encode(PAYLOAD)
    assert codec.decode(cover) == PAYLOAD
    print(f"payload={len(PAYLOAD)}B  cover={len(cover)} chars")

    n = 3
    t0 = time.perf_counter()
    for _ in range(n):
        cover = codec.encode(PAYLOAD)
        assert codec.decode(cover) == PAYLOAD
    dt = (time.perf_counter() - t0) / n
    print(f"\nBASELINE encode+decode: {dt * 1000:.0f} ms/iter (mean of {n})")

    # Profile a single encode+decode and show the top cumulative-time functions.
    pr = cProfile.Profile()
    pr.enable()
    cover = codec.encode(PAYLOAD)
    codec.decode(cover)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
    print(s.getvalue())


if __name__ == "__main__":
    main()
