"""Unit tests for the reproducible rank-interval coding (no model needed)."""

from __future__ import annotations

from mostegollm import coding


def test_constants_are_integers():
    assert coding.WHOLE == 1 << 32
    assert coding.HALF == coding.WHOLE >> 1
    assert coding.QUARTER == coding.WHOLE >> 2
    assert all(isinstance(w, int) for w in coding._WIDTHS)
    assert len(coding._WIDTHS) == coding.K


def test_widths_are_non_increasing_and_positive():
    assert all(w >= 1 for w in coding._WIDTHS)
    assert all(a >= b for a, b in zip(coding._WIDTHS, coding._WIDTHS[1:]))
    assert coding._WIDTHS[0] > coding._WIDTHS[-1]


def test_cum_is_well_formed():
    cum = coding.CUM
    assert len(cum) == coding.K + 1
    assert cum[0] == 0
    assert cum[-1] == coding.WHOLE
    assert all(cum[i] < cum[i + 1] for i in range(coding.K))  # strictly increasing


def test_cum_is_a_fixed_constant():
    # Same object/values every access; never recomputed from floats.
    assert coding.CUM is coding.CUM
    assert coding.CUM == coding._build_cum(coding._WIDTHS)
