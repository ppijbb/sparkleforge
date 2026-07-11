"""
tests/test_verification_gate_scripts.py — Anvil Phase Σ-4 auto-fix verification gate (#511).

Covers the pure logic behind the three new merge-gate scripts:
- scripts/check_import_smoke.py
- scripts/check_no_diff_prefix_paths.py
- scripts/check_issue_scope_overlap.py
"""
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from check_import_smoke import check_imports  # noqa: E402
from check_no_diff_prefix_paths import find_diff_prefix_paths  # noqa: E402
from check_issue_scope_overlap import compute_scope_overlap, extract_mentioned_symbols  # noqa: E402

SCOPE_OVERLAP_SCRIPT = PROJECT_ROOT / "scripts" / "check_issue_scope_overlap.py"


class TestImportSmoke:
    def test_real_core_modules_import_cleanly(self):
        failures = check_imports(["main", "src.core.mcp_integration", "src.core.agent_harness"])
        assert failures == []

    def test_broken_module_is_reported_with_traceback(self):
        failures = check_imports(["this_module_does_not_exist_anywhere"])
        assert len(failures) == 1
        name, tb = failures[0]
        assert name == "this_module_does_not_exist_anywhere"
        assert "ModuleNotFoundError" in tb or "ImportError" in tb


class TestNoDiffPrefixPaths:
    def test_normal_paths_pass(self):
        assert find_diff_prefix_paths(["src/core/mcp/__init__.py", "tests/test_foo.py"]) == []

    def test_stray_b_prefixed_path_is_flagged(self):
        offenders = find_diff_prefix_paths(
            ["src/core/mcp/__init__.py", "b/src/core/mcp/openrouter_client.py"]
        )
        assert offenders == ["b/src/core/mcp/openrouter_client.py"]

    def test_stray_a_prefixed_path_is_flagged(self):
        assert find_diff_prefix_paths(["a/tests/test_foo.py"]) == ["a/tests/test_foo.py"]

    def test_legitimate_paths_starting_with_a_or_b_word_are_not_flagged(self):
        # Only an exact path component of "a" or "b" should trigger, not any
        # directory that happens to start with those letters.
        assert find_diff_prefix_paths(["agents/foo.py", "backend/bar.py"]) == []


class TestIssueScopeOverlap:
    def test_extracts_identifiers_from_a_full_signature_in_one_backtick_span(self):
        # Real issue bodies (e.g. #509) wrap a whole signature in a single
        # backtick span: `delegate_to_agent(role, task, context) -> result`.
        text = "Add `delegate_to_agent(role, task, context) -> result` and wire into `action_journal`."
        symbols = extract_mentioned_symbols(text)
        assert "delegate_to_agent" in symbols
        assert "action_journal" in symbols
        # Plain English words incidentally inside the signature aren't specific
        # enough to count as evidence the issue's scope was addressed.
        assert "role" not in symbols
        assert "task" not in symbols
        assert "context" not in symbols

    def test_drops_generic_short_and_plain_english_tokens(self):
        text = "Uses `self` and `return` a lot, but also `ActionJournal` and `src/core/guard/`."
        symbols = extract_mentioned_symbols(text)
        assert "self" not in symbols
        assert "return" not in symbols
        assert "ActionJournal" in symbols
        assert "src/core/guard/" in symbols

    def test_no_concrete_symbols_means_no_opinion(self):
        result = compute_scope_overlap("Please improve the code quality.", "+some diff content")
        assert result["substantial"] is None

    def test_stub_diff_that_ignores_mentioned_symbols_is_flagged_not_substantial(self):
        issue_text = (
            "Add `delegate_to_agent` in a new module, journaled via `ActionJournal`, "
            "bounded like the existing `overseer_iterations` guard."
        )
        # This mirrors PR #513's real first commit: touches real .py files but
        # implements none of the named symbols.
        stub_diff = (
            "diff --git a/src/core/orchestrator/graph.py b/src/core/orchestrator/graph.py\n"
            "+++ b/src/core/orchestrator/graph.py\n"
            "+import logging\n"
            "diff --git a/src/core/orchestrator/state.py b/src/core/orchestrator/state.py\n"
            "+++ b/src/core/orchestrator/state.py\n"
            "+    delegation_depth: int\n"
            "+    max_delegation_depth: int\n"
        )
        result = compute_scope_overlap(issue_text, stub_diff)
        assert result["substantial"] is False
        assert result["matched"] == []

    def test_real_implementation_diff_is_flagged_substantial(self):
        issue_text = "Add `delegate_to_agent` bounded like `overseer_iterations`, journaled via `ActionJournal`."
        real_diff = (
            "diff --git a/src/core/orchestrator/delegation.py b/src/core/orchestrator/delegation.py\n"
            "+++ b/src/core/orchestrator/delegation.py\n"
            "+async def delegate_to_agent(state, role, task, context=None):\n"
            "+    journal = ActionJournal()\n"
        )
        result = compute_scope_overlap(issue_text, real_diff)
        assert result["substantial"] is True
        assert "delegate_to_agent" in result["matched"]
        assert "ActionJournal" in result["matched"]


class TestScopeOverlapCLIExitCodes:
    """Regression tests for issue #521: exit codes must let the caller tell
    "confirmed not substantial" (2) apart from "the script itself crashed"
    (1), or a broken check silently looks identical to a passed one."""

    @pytest.fixture
    def git_repo(self, tmp_path):
        def run(*args):
            subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

        run("init", "-q", "-b", "main")
        run("config", "user.email", "t@t.com")
        run("config", "user.name", "t")
        (tmp_path / "file.py").write_text("x = 1\n")
        run("add", "-A")
        run("commit", "-q", "-m", "chore: init")
        run("checkout", "-q", "-b", "feature")
        return tmp_path

    def test_exit_2_when_confirmed_not_substantial(self, git_repo):
        (git_repo / "file.py").write_text("x = 1\nimport logging\n")
        subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "fix: stub"], cwd=git_repo, check=True
        )
        issue_file = git_repo / "issue.md"
        issue_file.write_text("Add `delegate_to_agent` journaled via `ActionJournal`.\n- [ ] todo\n")

        result = subprocess.run(
            [sys.executable, str(SCOPE_OVERLAP_SCRIPT), "--issue-file", str(issue_file), "--range", "main...feature"],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, result.stdout + result.stderr

    def test_exit_0_when_substantial(self, git_repo):
        (git_repo / "file.py").write_text("x = 1\n\nasync def delegate_to_agent(): ...\n")
        subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "fix: real"], cwd=git_repo, check=True
        )
        issue_file = git_repo / "issue.md"
        issue_file.write_text("Add `delegate_to_agent`.\n- [ ] todo\n")

        result = subprocess.run(
            [sys.executable, str(SCOPE_OVERLAP_SCRIPT), "--issue-file", str(issue_file), "--range", "main...feature"],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_exit_1_not_2_when_script_crashes(self, git_repo):
        """A missing issue file must crash with exit 1, never exit 2 -- exit 2
        is reserved for a confirmed substantial=false verdict, and the caller
        workflow only trusts exit 2 for that. If a crash also produced exit
        2, a broken check would be indistinguishable from a passed one."""
        missing_issue_file = git_repo / "does_not_exist.md"

        result = subprocess.run(
            [sys.executable, str(SCOPE_OVERLAP_SCRIPT), "--issue-file", str(missing_issue_file), "--range", "main...feature"],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, result.stdout + result.stderr
        assert "substantial=false" not in result.stdout
