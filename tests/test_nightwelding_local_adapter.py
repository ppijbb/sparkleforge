from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.nightwelding.local_adapter import LocalGitAdapter
from src.core.nightwelding.models import NightweldingQueue, NightweldingStatus
from src.core.nightwelding.runner import _resolve_adapter, run_nightwelding_issue


def test_local_git_adapter_reads_markdown_file(tmp_path: Path) -> None:
    issue_file = tmp_path / "issue_123.md"
    issue_file.write_text(
        "# Fix KeyError in parser\n\nWhen input is empty, parser raises KeyError.",
        encoding="utf-8",
    )

    adapter = LocalGitAdapter(repo_root=tmp_path)
    context = adapter.fetch_issue_context(str(issue_file))

    assert context.number == str(issue_file)
    assert context.title == "Fix KeyError in parser"
    assert "parser raises KeyError" in context.markdown
    assert context.url.startswith("file://")


def test_local_git_adapter_reads_raw_text(tmp_path: Path) -> None:
    raw_issue = "fix: handle None case in serializer\n\nSerializer fails when payload is None."
    adapter = LocalGitAdapter(repo_root=tmp_path)
    context = adapter.fetch_issue_context(raw_issue)

    assert context.number == raw_issue
    assert context.title == "fix: handle None case in serializer"
    assert context.markdown == raw_issue


def test_local_git_adapter_list_candidate_issues(tmp_path: Path) -> None:
    issues_dir = tmp_path / ".sparkleforge" / "issues"
    issues_dir.mkdir(parents=True)
    (issues_dir / "bug_a.md").write_text("# Bug A\nDesc A", encoding="utf-8")
    (issues_dir / "bug_b.md").write_text("# Bug B\nDesc B", encoding="utf-8")
    (issues_dir / "other.txt").write_text("Not markdown", encoding="utf-8")

    adapter = LocalGitAdapter(issues_dir=issues_dir, repo_root=tmp_path)
    candidates = adapter.list_candidate_issues()

    assert len(candidates) == 2
    assert any("bug_a.md" in c for c in candidates)
    assert any("bug_b.md" in c for c in candidates)


def test_local_git_adapter_list_candidate_issues_respects_declared_labels(tmp_path: Path) -> None:
    """#1407: list_candidate_issues accepted backlog_label/exclude_labels but
    ignored them entirely. Files that opt in via frontmatter `labels:` must
    actually be filtered; undecorated files (tested above) stay unaffected."""
    issues_dir = tmp_path / ".sparkleforge" / "issues"
    issues_dir.mkdir(parents=True)
    (issues_dir / "wanted.md").write_text(
        "---\nlabels: [auto-fix-failed]\n---\n# Wanted\nDesc", encoding="utf-8"
    )
    (issues_dir / "wrong_label.md").write_text(
        "---\nlabels: [enhancement]\n---\n# Wrong label\nDesc", encoding="utf-8"
    )
    (issues_dir / "excluded.md").write_text(
        "---\nlabels: [auto-fix-failed, nightwelding-failed]\n---\n# Excluded\nDesc",
        encoding="utf-8",
    )

    adapter = LocalGitAdapter(issues_dir=issues_dir, repo_root=tmp_path)
    candidates = adapter.list_candidate_issues(
        backlog_label="auto-fix-failed", exclude_labels=["nightwelding-failed"]
    )

    assert len(candidates) == 1
    assert "wanted.md" in candidates[0]


def test_resolve_adapter_prefers_explicit_and_heuristics(tmp_path: Path) -> None:
    # Explicit provider="local"
    adapter = _resolve_adapter("123", repo_root=tmp_path, provider="local")
    assert isinstance(adapter, LocalGitAdapter)

    # Local file path
    dummy_file = tmp_path / "issue.md"
    dummy_file.write_text("# Test", encoding="utf-8")
    adapter2 = _resolve_adapter(str(dummy_file), repo_root=tmp_path)
    assert isinstance(adapter2, LocalGitAdapter)


@pytest.mark.asyncio
async def test_run_nightwelding_issue_with_local_adapter(monkeypatch, tmp_path: Path) -> None:
    issue_file = tmp_path / "bug_repro.md"
    issue_file.write_text(
        "# fix: add fallback for empty string\n\nEmpty string causes crash.",
        encoding="utf-8",
    )

    class FakeRepro:
        success = True
        test_files = ["tests/test_bug.py"]
        red_output = "FAILED tests/test_bug.py"
        reason = ""

    class FakeImplement:
        success = True
        green_output = "PASSED tests/test_bug.py"
        log = "All tests green"
        reason = ""

    from src.core.nightwelding import gate, runner

    async def fake_write_repro(*a, **k):
        return FakeRepro()

    monkeypatch.setattr(gate, "is_reproducible_bug_eligible", lambda md: (True, "Eligible"))
    monkeypatch.setattr(gate, "write_reproduction_test", fake_write_repro)
    monkeypatch.setattr(runner, "implement_until_green", lambda **kw: FakeImplement())

    adapter = LocalGitAdapter(repo_root=tmp_path)
    # Mock git operations
    monkeypatch.setattr(adapter, "create_worktree", lambda r, b, bb: tmp_path)
    monkeypatch.setattr(adapter, "remove_worktree", lambda r, w: None)
    monkeypatch.setattr(
        adapter, "normalize_commit_title", lambda t, r: "fix: add fallback for empty string"
    )

    from src.core.nightwelding import github_adapter
    monkeypatch.setattr(
        github_adapter, "_run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    monkeypatch.setattr(github_adapter, "push_branch", lambda *a, **k: True)

    queue = NightweldingQueue(storage_path=tmp_path / "queue")
    item = await run_nightwelding_issue(
        str(issue_file),
        repo_root=tmp_path,
        queue=queue,
        adapter=adapter,
    )

    assert item.status == NightweldingStatus.DRAFT_OPENED
    assert item.pr_url is not None
    assert "local://" in item.pr_url
