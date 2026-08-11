"""Issue #1331: guard/memory state modules defaulted to cwd-relative paths
(``os.path.join("data", ...)``), so a coworker session run with a target
repo as cwd would leak SparkleForge's own credential vault key, capability
grants, audit journal, and semantic memory DB into that target repo. The
defaults must resolve to the SparkleForge install location regardless of cwd.
"""
import os
import tempfile

import pytest

from src.core.guard.capability_manager import CapabilityManager
from src.core.guard.action_journal import ActionJournal
from src.core.guard.credential_vault import CredentialVault
from src.core.memory.semantic_memory import SemanticMemory


@pytest.fixture(autouse=True)
def reset_singletons():
    CapabilityManager._instance = None
    ActionJournal._instance = None
    CredentialVault._instance = None
    yield
    CapabilityManager._instance = None
    ActionJournal._instance = None
    CredentialVault._instance = None


def _default_path_from_other_cwd(build_default_path):
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as target_repo:
        os.chdir(target_repo)
        try:
            path = build_default_path()
        finally:
            os.chdir(original_cwd)
        assert not path.startswith(target_repo), (
            f"default path {path!r} leaked into target repo {target_repo!r}"
        )


def test_capability_manager_default_path_ignores_cwd():
    _default_path_from_other_cwd(lambda: CapabilityManager()._state_path)


def test_action_journal_default_path_ignores_cwd():
    _default_path_from_other_cwd(lambda: ActionJournal()._journal_path)


def test_credential_vault_default_path_ignores_cwd():
    _default_path_from_other_cwd(lambda: CredentialVault()._fallback_path)


def test_semantic_memory_default_path_ignores_cwd():
    _default_path_from_other_cwd(lambda: SemanticMemory().db_path)
