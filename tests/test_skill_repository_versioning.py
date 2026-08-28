"""Anvil Phase N-1: SkillRepository version pinning + rollback."""

import tempfile

from src.core.anvil.skill_repository import SkillRepository


def test_resave_increments_version_and_preserves_history():
    with tempfile.TemporaryDirectory() as tmp:
        repo = SkillRepository(storage_dir=tmp)

        repo.save_skill("greet", "def run():\n    return 'v1'")
        repo.save_skill("greet", "def run():\n    return 'v2'")
        repo.save_skill("greet", "def run():\n    return 'v3'")

        assert repo.get_skill("greet").version == 3
        assert repo.list_skill_versions("greet") == [1, 2, 3]

        v1 = repo.get_skill_version("greet", 1)
        assert v1 is not None
        assert "v1" in v1.code


def test_rollback_creates_new_version_with_old_code():
    with tempfile.TemporaryDirectory() as tmp:
        repo = SkillRepository(storage_dir=tmp)

        repo.save_skill("greet", "def run():\n    return 'good'")
        repo.save_skill("greet", "def run():\n    return 'BROKEN'")
        assert repo.get_skill("greet").version == 2

        rolled_back = repo.rollback_skill("greet", 1)

        # rollback is a revert (new version), not a reset -- history stays intact
        assert rolled_back.version == 3
        assert "good" in rolled_back.code
        assert repo.get_skill("greet").code == rolled_back.code
        assert repo.list_skill_versions("greet") == [1, 2, 3]


def test_reload_from_disk_restores_latest_version():
    with tempfile.TemporaryDirectory() as tmp:
        repo1 = SkillRepository(storage_dir=tmp)
        repo1.save_skill("greet", "def run():\n    return 'v1'")
        repo1.save_skill("greet", "def run():\n    return 'v2'")

        repo2 = SkillRepository(storage_dir=tmp)
        assert repo2.get_skill("greet").version == 2
        assert "v2" in repo2.get_skill("greet").code


if __name__ == "__main__":
    test_resave_increments_version_and_preserves_history()
    test_rollback_creates_new_version_with_old_code()
    test_reload_from_disk_restores_latest_version()
    print("ok")
