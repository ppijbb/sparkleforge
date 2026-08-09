"""git_status must not misparse unstaged-only changes as staged.

Regression: `_run_git_command` used `.strip()` on the whole `git status
--porcelain` stdout blob. Porcelain lines carry meaningful leading
whitespace (" M file" = unstaged-only, "M  file" = staged); stripping the
left side of the blob eats that leading space off the first line whenever
the first reported change is unstaged-only, shifting git_status()'s
`line[3:]` slice by one character -- "src/config.rs" came out as
"rc/config.rs" and was misclassified as staged. That false "already staged"
signal then made git_commit(auto_stage=False) fail with "no changes added
to commit" against a real lfdb checkout.
"""

import subprocess

import pytest

from src.core.git_workflow import GitWorkflow


def _run(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.mark.asyncio
async def test_git_status_reports_unstaged_only_change_correctly(tmp_path):
    _run(tmp_path, "init")
    _run(tmp_path, "config", "user.email", "test@example.com")
    _run(tmp_path, "config", "user.name", "test")
    (tmp_path / "file.txt").write_text("original\n")
    _run(tmp_path, "add", "file.txt")
    _run(tmp_path, "commit", "-m", "initial")

    (tmp_path / "file.txt").write_text("changed\n")

    workflow = GitWorkflow(repo_path=tmp_path)
    status = await workflow.git_status()

    assert status["staged_files"] == []
    assert status["unstaged_files"] == ["file.txt"]
