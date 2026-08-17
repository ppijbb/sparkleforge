import subprocess

import pytest

from src.core.ci.fix_substantiality import (
    assess_fix_substantiality,
    compute_checklist_heuristic,
    compute_scope_overlap,
    count_unchecked,
    extract_mentioned_symbols,
    gather_diff_stats,
)


class TestExtractMentionedSymbols:
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


class TestComputeScopeOverlap:
    def test_no_concrete_symbols_means_no_opinion(self):
        result = compute_scope_overlap("Please improve the code quality.", "+some diff content")
        assert result["substantial"] is None

    def test_stub_diff_that_ignores_mentioned_symbols_is_flagged_not_substantial(self):
        issue_text = (
            "Add `delegate_to_agent` in a new module, journaled via `ActionJournal`, "
            "bounded like the existing `overseer_iterations` guard."
        )
        # Mirrors PR #513's real first commit: touches real .py files but
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


class TestChecklistHeuristic:
    def test_no_unchecked_items_is_always_substantial(self):
        assert compute_checklist_heuristic(0, ["src/core/foo.py"], 1) is True
        assert compute_checklist_heuristic(0, [".github/workflows/x.yml"], 1) is True

    def test_unchecked_plus_only_trivial_files_is_not_substantial(self):
        assert compute_checklist_heuristic(2, [".github/workflows/x.yml", "docs/y.md"], 50) is False

    def test_unchecked_plus_tiny_diff_is_not_substantial_even_with_real_files(self):
        assert compute_checklist_heuristic(1, ["src/core/foo.py"], 3) is False

    def test_unchecked_plus_substantial_real_file_diff_is_substantial(self):
        assert compute_checklist_heuristic(1, ["src/core/foo.py"], 20) is True

    def test_count_unchecked_counts_only_unchecked_lines(self):
        text = "- [ ] todo one\n- [x] done\n- [ ] todo two\n"
        assert count_unchecked(text) == 2


class TestAssessFixSubstantiality:
    def test_checklist_true_and_no_scope_opinion_stays_substantial(self):
        verdict = assess_fix_substantiality(
            issue_text="No checklist here, no backticks either.",
            diff_text="+real change",
            changed_files=["src/core/foo.py"],
            changed_lines=10,
        )
        assert verdict.substantial is True
        assert verdict.reason == ""

    def test_scope_overlap_can_downgrade_true_to_false(self):
        issue_text = "Add `delegate_to_agent` for real this time."
        verdict = assess_fix_substantiality(
            issue_text=issue_text,
            diff_text="+import logging\n",
            changed_files=["src/core/foo.py"],
            changed_lines=10,
        )
        assert verdict.substantial is False
        assert "never touches" in verdict.reason

    def test_scope_overlap_cannot_upgrade_false_back_to_true(self):
        # Checklist heuristic alone already says not-substantial (trivial
        # file only); scope-overlap has no opinion (no backticked symbols).
        # The combined verdict must stay false, not flip back to true.
        verdict = assess_fix_substantiality(
            issue_text="- [ ] still unresolved",
            diff_text="+trivial workflow tweak",
            changed_files=[".github/workflows/x.yml"],
            changed_lines=1,
        )
        assert verdict.substantial is False
        assert verdict.reason == ""
        assert verdict.unchecked == 1


class TestGatherDiffStats:
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

    def test_gathers_diff_relative_to_cwd(self, git_repo, monkeypatch):
        (git_repo / "file.py").write_text("x = 1\nasync def delegate_to_agent(): ...\n")
        subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "fix: real"], cwd=git_repo, check=True)

        monkeypatch.chdir(git_repo)
        diff_text, changed_files, changed_lines = gather_diff_stats("main...feature")
        assert "delegate_to_agent" in diff_text
        assert changed_files == ["file.py"]
        assert changed_lines == 1
