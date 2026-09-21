"""Tests for the Nightwelding issue digest (grouping/dup-flagging, issue #1545)."""

from __future__ import annotations

from pathlib import Path

from src.core.nightwelding import github_adapter
from src.core.nightwelding.models import NightweldingIssue


def test_extract_files_touched_pulls_backtick_paths_in_order():
    body = (
        "Bug in `src/web/live_dashboard.py:31` and again in "
        "`src/utils/supabase_exporter.py`. Also see `src/web/live_dashboard.py:31` again."
    )
    files = github_adapter._extract_files_touched("title has none", body)
    assert files == ["src/web/live_dashboard.py", "src/utils/supabase_exporter.py"]


def test_extract_files_touched_matches_extensionless_and_dotfile_names():
    body = "Broke `Dockerfile`, `docker/Makefile`, `.gitignore`, and `.env` too."
    files = github_adapter._extract_files_touched("", body)
    assert files == ["Dockerfile", "docker/Makefile", ".gitignore", ".env"]


def test_extract_files_touched_ignores_non_path_backticks():
    files = github_adapter._extract_files_touched("`some_var` is unused", "no files here either")
    assert files == []


def _issue(number, files_touched, created_at="2026-01-01T00:00:00Z"):
    return NightweldingIssue(
        number=number,
        title=f"issue {number}",
        url=f"https://example.com/{number}",
        created_at=created_at,
        updated_at=created_at,
        labels=["nightwelding-queue"],
        body="",
        files_touched=files_touched,
    )


def test_build_digest_groups_issues_by_root_file(tmp_path, monkeypatch):
    monkeypatch.setattr(github_adapter, "_file_changed_since", lambda *a, **kw: False)
    issues = [
        _issue(1, ["src/a.py"]),
        _issue(2, ["src/a.py"]),
        _issue(3, ["src/b.py"]),
        _issue(4, []),
    ]

    digest = github_adapter.build_digest(issues, repo_root=tmp_path)

    by_file = {g.root_file: g for g in digest.groups}
    assert by_file["src/a.py"].recurrence_count == 2
    assert by_file["src/b.py"].recurrence_count == 1
    assert by_file["(no file identified)"].recurrence_count == 1
    assert digest.total_issues == 4


def test_build_digest_flags_possibly_fixed_via_git_log(tmp_path, monkeypatch):
    monkeypatch.setattr(github_adapter, "_file_changed_since", lambda root, path, since: path == "src/fixed.py")
    issues = [_issue(1, ["src/fixed.py"]), _issue(2, ["src/still-broken.py"])]

    digest = github_adapter.build_digest(issues, repo_root=tmp_path)

    by_file = {g.root_file: g for g in digest.groups}
    assert by_file["src/fixed.py"].possibly_fixed_issues == [1]
    assert by_file["src/still-broken.py"].possibly_fixed_issues == []


def test_top_recurring_orders_by_recurrence_count():
    digest = github_adapter.build_digest(
        [_issue(1, ["a.py"]), _issue(2, ["b.py"]), _issue(3, ["b.py"]), _issue(4, ["b.py"])],
        repo_root=Path("."),
    )
    top = digest.top_recurring(top_n=1)
    assert len(top) == 1
    assert top[0].root_file == "b.py"


def test_fetch_nightwelding_issues_filters_by_label(monkeypatch):
    import json as json_module

    raw = [
        {
            "number": 1, "title": "keep me", "url": "https://x/1",
            "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z",
            "labels": [{"name": "nightwelding-queue"}], "body": "`src/a.py`",
        },
        {
            "number": 2, "title": "drop me", "url": "https://x/2",
            "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z",
            "labels": [{"name": "some-other-label"}], "body": "",
        },
    ]

    class _Proc:
        stdout = json_module.dumps(raw)

    monkeypatch.setattr(github_adapter, "_run", lambda *a, **kw: _Proc())

    issues = github_adapter.fetch_nightwelding_issues("owner/repo", label="nightwelding-queue")

    assert [i.number for i in issues] == [1]
    assert issues[0].files_touched == ["src/a.py"]


def test_render_digest_markdown_includes_possibly_fixed_warning():
    digest = github_adapter.build_digest([_issue(1, ["a.py"])], repo_root=Path("."))
    digest.groups[0].possibly_fixed_issues = [1]

    markdown = github_adapter.render_digest_markdown(digest)

    assert "#1" in markdown
    assert "possibly already fixed" in markdown
    assert "never auto-closes" in markdown
