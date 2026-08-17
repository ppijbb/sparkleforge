"""Anvil roadmap doc (docs/ANVIL_PLAN.md) status sync.

Computing the new status cell ("done" vs "in progress (n/total)") and
rewriting the matching markdown table row is a formatting/judgment decision,
not mechanical extraction, moved verbatim from
sparkleforge-daily-roadmap.yml's "Sync Anvil roadmap doc" step. The
git/commit/push/PR mechanics that followed it in that step now call
`sparkleforge ci publish` instead.
"""

from __future__ import annotations

from pathlib import Path


def compute_status(closed: int, total: int) -> str:
    if total == 0 or closed == 0:
        return ""
    return "✅" if closed == total else f"🔲 진행 중 ({closed}/{total})"


def sync_anvil_doc(plan_path: Path, milestone_number: int, closed: int, total: int) -> bool:
    """Rewrite the matching table row's status cell in place. Returns True if changed."""
    new_status = compute_status(closed, total)
    if not new_status:
        return False

    lines = plan_path.read_text(encoding="utf-8").splitlines()
    marker = f"마일스톤 #{milestone_number}"
    changed = False
    for i, line in enumerate(lines):
        if line.startswith("|") and marker in line:
            cells = line.split("|")
            if len(cells) >= 5 and cells[3].strip() != new_status:
                cells[3] = f" {new_status} "
                lines[i] = "|".join(cells)
                changed = True

    if changed:
        plan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed
