"""Reproducible rank-interval coding.

Interval boundaries are fixed integer constants assigned to tokens by rank, so
encode and decode build bit-identical intervals across PyTorch versions,
devices, and BLAS libraries. All constants are generated with integer-only
arithmetic (Python int is arbitrary-precision and platform-independent), so the
schedule itself cannot diverge across platforms.

See docs/superpowers/specs/2026-06-02-cross-platform-reproducible-coding-design.md
"""

from __future__ import annotations

from dataclasses import dataclass

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
# _SEED is intentionally smaller than (8/7)^256; ranks ~196-255 plateau at width
# 1 via the max(1, ...) floor. reserve-one-unit in _build_cum keeps those tail
# entries distinct in CUM, so the plateau is harmless.
_SEED = 1 << 40


def _build_widths(k: int) -> list[int]:
    """Integer geometric-decay widths: widths[0] = _SEED, widths[i] = max(1, widths[i-1] * _DECAY_NUM // _DECAY_DEN)."""
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


@dataclass(frozen=True)
class StepCoding:
    """Per-step coding: a partition of [0, WHOLE) and a token->interval lookup.

    Attributes:
        intervals: list of (lo, hi, representative_token_id), sorted by lo,
            tiling [0, WHOLE). The encoder navigates `value` into one of these
            and emits its representative.
        token_to_run: maps every surviving token_id to its run's (lo, hi). The
            decoder looks up the observed token here.
    """

    intervals: list[tuple[int, int, int]]
    token_to_run: dict[int, tuple[int, int]]


def step_coding(token_ids: list[int], logits: list[float]) -> StepCoding:
    """Build the rank-interval coding for one step.

    Tokens are sorted by (logit DESC, token_id ASC); adjacent tokens whose logit
    gap is < GUARD are merged into a run represented by its lowest token_id.
    Each run is assigned a fixed integer interval from the constant CUM array
    (the final occupied run extends to WHOLE).
    """
    order = sorted(range(len(token_ids)), key=lambda i: (-logits[i], token_ids[i]))
    s_ids = [token_ids[i] for i in order]
    s_log = [logits[i] for i in order]
    m = len(s_ids)

    intervals: list[tuple[int, int, int]] = []
    token_to_run: dict[int, tuple[int, int]] = {}
    i = 0
    while i < m:
        j = i
        while j + 1 < m and (s_log[j] - s_log[j + 1]) < GUARD:
            j += 1
        lo = CUM[i]
        # The final occupied run absorbs up to WHOLE so the partition always
        # covers [0, WHOLE) regardless of how many tokens survived (m).
        hi = WHOLE if j == m - 1 else CUM[j + 1]
        members = s_ids[i : j + 1]
        rep = min(members)
        intervals.append((lo, hi, rep))
        for t in members:
            token_to_run[t] = (lo, hi)
        i = j + 1
    return StepCoding(intervals, token_to_run)
