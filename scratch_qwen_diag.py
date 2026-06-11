"""Diagnose Qwen local stability: trap signals, log RSS + per-iter time.

If it dies, the handler records WHICH signal and when; if it's OOM/SIGKILL the
process just vanishes (uncatchable) and the cgroup memory.events oom_kill counter
would tick. Run detached so it isn't tied to a Bash-tool call:

    setsid bash -c '.venv/bin/python -u scratch_qwen_diag.py' > /tmp/qwen_diag.log 2>&1 &
"""

from __future__ import annotations

import os
import resource
import signal
import sys
import time

T0 = time.time()


def rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def handler(signum, _frame):
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = f"sig{signum}"
    print(f"!!! SIGNAL {signum} ({name}) at t={time.time() - T0:.1f}s rss={rss_mb()}MB", flush=True)
    sys.exit(99)


for s in (
    signal.SIGTERM,
    signal.SIGINT,
    signal.SIGHUP,
    signal.SIGQUIT,
    signal.SIGUSR1,
    signal.SIGUSR2,
    16,  # SIGSTKFLT -> exit 144 if this is the culprit
    signal.SIGXCPU,
    signal.SIGABRT,
):
    try:
        signal.signal(s, handler)
    except Exception:
        pass

print(f"start pid={os.getpid()} sid={os.getsid(0)} t=0", flush=True)
from mostegollm import StegoCodec  # noqa: E402

codec = StegoCodec(model_name="Qwen/Qwen2.5-0.5B", device="cpu")
codec._ensure_model()
print(f"LOADED t={time.time() - T0:.1f}s rss={rss_mb()}MB", flush=True)

payload = b"diagnostic payload"
i = 0
while time.time() - T0 < 600:
    i += 1
    cover = codec.encode(payload)
    assert codec.decode(cover) == payload
    print(f"iter {i} t={time.time() - T0:.1f}s rss={rss_mb()}MB", flush=True)
print(f"DONE {i} iters t={time.time() - T0:.1f}s rss={rss_mb()}MB", flush=True)
