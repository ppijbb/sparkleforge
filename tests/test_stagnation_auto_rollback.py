"""Anvil Phase N-2: stagnation gate auto-rollback of the most recently re-distilled skill."""

import tempfile

from src.core.anvil.skill_repository import SkillRepository
from src.core.ci.stagnation_issue import build_stagnation_issue, maybe_auto_rollback


def test_maybe_auto_rollback_reverts_most_recent_redistilled_skill():
    with tempfile.TemporaryDirectory() as tmp:
        repo = SkillRepository(storage_dir=tmp)
        repo.save_skill("stable", "def run():\n    return 'ok'")  # version 1, never touched again
        repo.save_skill("regressed", "def run():\n    return 'good'")
        repo.save_skill("regressed", "def run():\n    return 'BROKEN'")

        rolled_back = maybe_auto_rollback(storage_dir=repo.storage_dir)

        assert rolled_back is not None
        assert rolled_back.name == "regressed"
        assert "good" in rolled_back.code
        assert rolled_back.version == 3


def test_maybe_auto_rollback_noop_when_nothing_redistilled():
    with tempfile.TemporaryDirectory() as tmp:
        repo = SkillRepository(storage_dir=tmp)
        repo.save_skill("stable", "def run():\n    return 'ok'")

        assert maybe_auto_rollback(storage_dir=repo.storage_dir) is None


def test_build_stagnation_issue_includes_rollback_note():
    from src.core.anvil.skill_repository import Skill

    rolled_back = Skill(
        name="regressed",
        code="def run():\n    return 'good'",
        version=3,
        metadata={"rollback_to_version": 1},
    )
    report = {"stagnation_detected": True, "scenarios": {}}

    issue = build_stagnation_issue(report, history=[], rolled_back=rolled_back)

    assert issue is not None
    assert "Auto-rollback" in issue.body
    assert "regressed" in issue.body


if __name__ == "__main__":
    test_maybe_auto_rollback_reverts_most_recent_redistilled_skill()
    test_maybe_auto_rollback_noop_when_nothing_redistilled()
    test_build_stagnation_issue_includes_rollback_note()
    print("ok")
