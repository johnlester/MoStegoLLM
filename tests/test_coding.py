"""Unit tests for the reproducible rank-interval coding (no model needed)."""

from __future__ import annotations

from mostegollm import coding


def test_constants_are_integers():
    assert coding.WHOLE == 1 << 32
    assert coding.HALF == coding.WHOLE >> 1
    assert coding.QUARTER == coding.WHOLE >> 2
    assert all(isinstance(w, int) for w in coding._WIDTHS)
    assert len(coding._WIDTHS) == coding.K
    assert all(isinstance(v, int) for v in coding.CUM)


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
    # Golden spot-values guard against accidental changes to the schedule
    # constants (_SEED / _DECAY_*); CUM is frozen once chosen.
    assert coding.CUM[1] == 536870881
    assert coding.CUM[128] == 4294967005
    assert coding.CUM[256] == 4294967296
    assert coding.CUM == coding._build_cum(coding._WIDTHS)


def test_step_coding_partitions_whole():
    # well-separated logits (gaps >> GUARD): each token its own run
    ids = [10, 20, 30, 40]
    logits = [5.0, 4.0, 3.0, 2.0]
    sc = coding.step_coding(ids, logits)
    assert len(sc.intervals) == 4
    assert sc.intervals[0][0] == 0
    assert sc.intervals[-1][1] == coding.WHOLE
    # contiguous, non-overlapping
    for (lo, hi, _), (lo2, _, _) in zip(sc.intervals, sc.intervals[1:]):
        assert hi == lo2
        assert lo < hi
    # every token maps to a run interval
    assert set(sc.token_to_run) == {10, 20, 30, 40}


def test_step_coding_is_sort_order_independent():
    a = coding.step_coding([30, 10, 20], [3.0, 5.0, 4.0])
    b = coding.step_coding([10, 20, 30], [5.0, 4.0, 3.0])
    assert a.intervals == b.intervals
    assert a.token_to_run == b.token_to_run


def test_run_merging_groups_near_ties():
    # tokens 1 and 2 are a near-tie (gap < GUARD) -> one run; token 3 separate
    ids = [1, 2, 3]
    logits = [5.0, 5.0 - coding.GUARD / 2, 1.0]
    sc = coding.step_coding(ids, logits)
    assert len(sc.intervals) == 2  # merged pair + singleton
    # the merged run is represented by the lowest token_id (1)
    assert sc.intervals[0][2] == 1
    # both members of the merged run share the same interval
    assert sc.token_to_run[1] == sc.token_to_run[2]
    assert sc.token_to_run[3] != sc.token_to_run[1]


def test_emitted_run_is_stable_under_sub_guard_noise():
    # Perturbing logits by < GUARD must not change any run interval.
    ids = [7, 3, 9, 1]
    base = [5.0, 3.0, 1.0, 0.5]
    noisy = [v + (0.4 * coding.GUARD) * ((-1) ** i) for i, v in enumerate(base)]
    assert coding.step_coding(ids, base).intervals == coding.step_coding(ids, noisy).intervals
