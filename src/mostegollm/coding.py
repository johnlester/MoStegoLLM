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
    """Fixed cumulative boundaries scaled to WHOLE, each interval width >= 1.

    Reserves one unit per position (which guarantees a strictly increasing
    result) and distributes the remaining ``WHOLE - len(widths)`` units by
    weight. Integer-only (multiply + floor-divide), so the array is identical on
    every platform. Constructed so it can never overflow WHOLE — unlike a naive
    scale-then-bump approach, where the many width-1 tail entries would cascade
    the +1 fixups past WHOLE.
    """
    n = len(widths)
    total = sum(widths)
    free = WHOLE - n  # > 0 since n (= K = 256) << WHOLE
    cum = [0]
    acc = 0
    for i, w in enumerate(widths):
        acc += w
        cum.append((i + 1) + (acc * free) // total)
    cum[-1] = WHOLE  # already exact (total divides total*free); set for clarity
    return cum


_WIDTHS = _build_widths(K)
# CUM is a FIXED constant array of length K+1 (not rescaled per step). A token at
# sorted position i always owns [CUM[i], CUM[i+1]); the final occupied run is
# extended to WHOLE (see step_coding). This makes a low-rank emitted token's
# interval independent of how many tokens survive filtering (m), so top-K cutoff
# instability cannot desync the emitted token.
CUM = _build_cum(_WIDTHS)
