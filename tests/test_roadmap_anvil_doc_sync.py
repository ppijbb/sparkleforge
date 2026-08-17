from src.core.roadmap.anvil_doc_sync import compute_status, sync_anvil_doc

PLAN_MD = """# Anvil Plan

| Phase | Title | Status | Notes |
|---|---|---|---|
| M | Some phase | 🔲 진행 중 (1/3) | details — 마일스톤 #5 |
| N | Another phase | ✅ | details — 마일스톤 #6 |
"""


def test_compute_status_all_closed():
    assert compute_status(3, 3) == "✅"


def test_compute_status_partial():
    assert compute_status(2, 3) == "🔲 진행 중 (2/3)"


def test_compute_status_zero_closed_or_zero_total_is_no_signal():
    assert compute_status(0, 3) == ""
    assert compute_status(0, 0) == ""


def test_sync_rewrites_matching_row(tmp_path):
    plan_path = tmp_path / "ANVIL_PLAN.md"
    plan_path.write_text(PLAN_MD, encoding="utf-8")

    changed = sync_anvil_doc(plan_path, milestone_number=5, closed=3, total=3)

    assert changed is True
    updated = plan_path.read_text(encoding="utf-8")
    assert "| M | Some phase | ✅ | details — 마일스톤 #5 |" in updated
    assert "| N | Another phase | ✅ | details — 마일스톤 #6 |" in updated  # untouched row unchanged


def test_sync_no_change_when_status_already_matches(tmp_path):
    plan_path = tmp_path / "ANVIL_PLAN.md"
    plan_path.write_text(PLAN_MD, encoding="utf-8")

    changed = sync_anvil_doc(plan_path, milestone_number=6, closed=1, total=1)

    assert changed is False
    assert plan_path.read_text(encoding="utf-8") == PLAN_MD


def test_sync_no_change_when_no_sub_issues_tracked(tmp_path):
    plan_path = tmp_path / "ANVIL_PLAN.md"
    plan_path.write_text(PLAN_MD, encoding="utf-8")

    changed = sync_anvil_doc(plan_path, milestone_number=5, closed=0, total=0)

    assert changed is False
