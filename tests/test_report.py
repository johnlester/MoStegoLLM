"""Tests for the cloud-matrix report aggregation (pure, no model)."""

from __future__ import annotations

from compat.report import build_matrix, ordering_agreement, render_report

CELLS = ["cpu-fp32", "t4-fp32", "t4-fp16"]


def _results(make_fail=None):
    """All-pass results for 3 cells × 3 cells × 2 payloads, with optional failures.

    make_fail: optional set of (encoder, decoder) pairs to mark as failed.
    """
    make_fail = make_fail or set()
    out = []
    for e in CELLS:
        for d in CELLS:
            for payload in ("aa", "bb"):
                failed = (e, d) in make_fail
                out.append(
                    {
                        "encoder_cell": e,
                        "decoder_cell": d,
                        "ok": not failed,
                        "failure_class": "logit_divergence" if failed else None,
                        "payload_hex": payload,
                    }
                )
    return out


def test_build_matrix_counts_ok_and_total():
    grid = build_matrix(_results(), CELLS)
    assert len(grid) == 9  # 3×3
    assert grid[("cpu-fp32", "t4-fp16")] == {"ok": 2, "total": 2, "classes": set()}


def test_build_matrix_records_failure_classes():
    grid = build_matrix(_results(make_fail={("t4-fp16", "cpu-fp32")}), CELLS)
    entry = grid[("t4-fp16", "cpu-fp32")]
    assert entry["ok"] == 0 and entry["total"] == 2
    assert entry["classes"] == {"logit_divergence"}


def test_ordering_agreement_vs_reference():
    dumps = [
        {"cell_id": "cpu-fp32", "top_ids_per_step": [[1, 2], [3, 4], [5, 6]]},
        {"cell_id": "t4-fp32", "top_ids_per_step": [[1, 2], [3, 4], [5, 6]]},  # identical
        {"cell_id": "t4-fp16", "top_ids_per_step": [[1, 2], [4, 3], [5, 6]]},  # step 1 reordered
    ]
    agree = ordering_agreement(dumps, "cpu-fp32")
    assert agree["t4-fp32"] == (3, 3)
    assert agree["t4-fp16"] == (2, 3)


def test_render_report_all_pass_contains_checkmarks_and_overall():
    md = render_report(_results(), CELLS, title="T")
    assert "# T" in md
    assert "✓ 2/2" in md
    assert "**Overall: 18/18" in md  # 9 pairs × 2 payloads


def test_render_report_marks_failures_with_class():
    md = render_report(_results(make_fail={("t4-fp16", "cpu-fp32")}), CELLS)
    assert "✗ 0/2 logit_divergence" in md
    assert "**Overall: 16/18" in md


def test_render_report_includes_agreement_table():
    dumps = [
        {"cell_id": "cpu-fp32", "top_ids_per_step": [[1, 2], [3, 4]]},
        {"cell_id": "t4-fp16", "top_ids_per_step": [[1, 2], [4, 3]]},
    ]
    md = render_report(_results(), CELLS, dumps=dumps, reference_cell="cpu-fp32")
    assert "Distribution agreement vs `cpu-fp32`" in md
    assert "1/2 (50.0%)" in md
