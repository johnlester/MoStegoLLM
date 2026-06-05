"""Pure aggregation of cloud-matrix results into a Markdown report.

No Modal, no model — just data in, Markdown out, so it is fully unit-testable.
"""

from __future__ import annotations


def build_matrix(results: list[dict], cell_ids: list[str]) -> dict[tuple[str, str], dict]:
    """Aggregate per-vector verify results into an encoder×decoder grid.

    Args:
        results: dicts with ``encoder_cell``, ``decoder_cell``, ``ok``, and
            (on failure) ``failure_class``.
        cell_ids: all cell ids, used to seed an empty grid.

    Returns:
        ``{(encoder, decoder): {"ok": int, "total": int, "classes": set[str]}}``.
    """
    grid: dict[tuple[str, str], dict] = {
        (e, d): {"ok": 0, "total": 0, "classes": set()} for e in cell_ids for d in cell_ids
    }
    for r in results:
        key = (r["encoder_cell"], r["decoder_cell"])
        entry = grid.setdefault(key, {"ok": 0, "total": 0, "classes": set()})
        entry["total"] += 1
        if r["ok"]:
            entry["ok"] += 1
        elif r.get("failure_class"):
            entry["classes"].add(r["failure_class"])
    return grid


def ordering_agreement(dumps: list[dict], reference_cell: str) -> dict[str, tuple[int, int]]:
    """Per-step top-k ordering agreement of each cell vs the reference cell.

    Args:
        dumps: dicts with ``cell_id`` and ``top_ids_per_step`` (list[list[int]]).
        reference_cell: the cell id to compare against.

    Returns:
        ``{cell_id: (matching_steps, total_reference_steps)}``.
    """
    ref = next((d for d in dumps if d["cell_id"] == reference_cell), None)
    if ref is None:
        return {}
    ref_steps = ref["top_ids_per_step"]
    out: dict[str, tuple[int, int]] = {}
    for d in dumps:
        steps = d["top_ids_per_step"]
        n = min(len(steps), len(ref_steps))
        match = sum(1 for i in range(n) if steps[i] == ref_steps[i])
        out[d["cell_id"]] = (match, len(ref_steps))
    return out


def _cell_str(entry: dict) -> str:
    if entry["total"] == 0:
        return "—"
    if entry["ok"] == entry["total"]:
        return f"✓ {entry['ok']}/{entry['total']}"
    classes = ",".join(sorted(entry["classes"])) or "fail"
    return f"✗ {entry['ok']}/{entry['total']} {classes}"


def render_report(
    results: list[dict],
    cell_ids: list[str],
    dumps: list[dict] | None = None,
    reference_cell: str | None = None,
    *,
    title: str = "Cross-compatibility matrix (Modal smoke)",
) -> str:
    """Render the full Markdown report: N×N decode matrix + ordering agreement."""
    grid = build_matrix(results, cell_ids)
    lines = [f"# {title}", ""]

    lines += ["## Decode matrix (rows = encoder, cols = decoder)", ""]
    lines.append("| enc \\ dec | " + " | ".join(cell_ids) + " |")
    lines.append("|" + "---|" * (len(cell_ids) + 1))
    for e in cell_ids:
        row = [f"**{e}**"] + [_cell_str(grid[(e, d)]) for d in cell_ids]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    total = sum(c["total"] for c in grid.values())
    ok = sum(c["ok"] for c in grid.values())
    lines += [f"**Overall: {ok}/{total} (encoder × decoder × payload) pairs recovered.**", ""]

    if dumps and reference_cell:
        lines += [
            f"## Distribution agreement vs `{reference_cell}` (top-k ordering per step)",
            "",
            "| cell | steps matching reference |",
            "|---|---|",
        ]
        for cid, (m, n) in ordering_agreement(dumps, reference_cell).items():
            pct = f"{100 * m / n:.1f}%" if n else "—"
            lines.append(f"| {cid} | {m}/{n} ({pct}) |")
        lines.append("")

    return "\n".join(lines)
