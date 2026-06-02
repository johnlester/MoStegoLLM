"""Reproducible rank-interval coding.

Interval boundaries are fixed integer constants assigned to tokens by rank, so
encode and decode build bit-identical intervals across PyTorch versions,
devices, and BLAS libraries. All constants are generated with integer-only
arithmetic (Python int is arbitrary-precision and platform-independent), so the
schedule itself cannot diverge across platforms.

See docs/superpowers/specs/2026-06-02-cross-platform-reproducible-coding-design.md
"""

from __future__ import annotations

# Arithmetic-coding precision (shared by encoder and decoder).
PRECISION = 32
WHOLE = 1 << PRECISION  # 2^32
HALF = WHOLE >> 1
QUARTER = WHOLE >> 2

# Top-k width considered at each step.
K = 256

# Gap (in logit units) below which adjacent sorted tokens merge into one run.
# Must be >> the cross-platform logit divergence (~3e-5 measured CPU<->GPU);
# 1e-3 is ~35x that margin. Tunable; see the design spec.
GUARD = 1e-3

# Width schedule: geometric decay via an integer-only recurrence. Model-agnostic.
_DECAY_NUM = 7
_DECAY_DEN = 8
_SEED = 1 << 40


def _build_widths(k: int) -> list[int]:
    """Integer geometric-decay widths: w[i] = max(1, w[i-1] * NUM // DEN)."""
    widths: list[int] = []
    w = _SEED
    for _ in range(k):
        widths.append(w)
        w = max(1, (w * _DECAY_NUM) // _DECAY_DEN)
    return widths


def _build_cum(widths: list[int]) -> list[int]:
    """Fixed cumulative boundaries: prefix sums of *widths* scaled to WHOLE.

    Integer multiply + floor-divide only, so the array is identical on every
    platform. Forced strictly increasing (each interval width >= 1).
    """
    total = sum(widths)
    cum = [0]
    running = 0
    for w in widths:
        running += w
        cum.append((running * WHOLE) // total)
    cum[-1] = WHOLE
    # Enforce strictly increasing: for each i, if cum[i] <= cum[i-1],
    # set cum[i] = cum[i-1] + 1. Stop before the last element.
    for i in range(1, len(cum) - 1):
        if cum[i] <= cum[i - 1]:
            cum[i] = cum[i - 1] + 1
    # Find the maximum value in cum[:-1] (all but the last)
    max_val_before_last = max(cum[:-1])
    # If max_val > WHOLE - 1, we need to rescale cum[:-1] to fit in [0, WHOLE-1]
    if max_val_before_last > WHOLE - 1:
        # Linearly rescale: map [cum[0], max_val] to [0, WHOLE-1]
        # Actually, cum[0] = 0, so map [0, max_val] to [0, WHOLE-1]
        for i in range(len(cum) - 1):
            cum[i] = (cum[i] * (WHOLE - 1)) // max_val_before_last
    return cum


_WIDTHS = _build_widths(K)
# CUM is a FIXED constant array of length K+1 (not rescaled per step). A token at
# sorted position i always owns [CUM[i], CUM[i+1]); the final occupied run is
# extended to WHOLE (see step_coding). This makes a low-rank emitted token's
# interval independent of how many tokens survive filtering (m), so top-K cutoff
# instability cannot desync the emitted token.
CUM = _build_cum(_WIDTHS)
