"""Issue #1330: coworker git tools had no branch-creation step, so a session
starting on main/master could commit and push directly to the base branch
with no PR in between. `git_commit` must auto-branch off a protected branch
before committing; `git_push` must refuse outright if the target branch is
still main/master.
"""

import subprocess

import pytest

from src.core.git_workflow import GitWorkflow


def _run(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _init_repo_on_branch(tmp_path, branch: str):
    _run(tmp_path, "init", "-b", branch)
    _run(tmp_path, "config", "user.email", "test@example.com")
    _run(tmp_path, "config", "user.name", "test")
    (tmp_path / "file.txt").write_text("original\n")
    _run(tmp_path, "add", "file.txt")
    _run(tmp_path, "commit", "-m", "initial")


@pytest.mark.asyncio
async def test_git_commit_on_protected_branch_creates_new_branch(tmp_path):
    _init_repo_on_branch(tmp_path, "main")
    (tmp_path / "file.txt").write_text("changed\n")

    workflow = GitWorkflow(repo_path=tmp_path)
    result = await workflow.git_commit(message="test change")

    assert result["success"] is True
    status = await workflow.git_status()
    assert status["current_branch"] != "main"
    assert status["current_branch"].startswith("agent/")


@pytest.mark.asyncio
async def test_git_commit_on_feature_branch_does_not_switch(tmp_path):
    _init_repo_on_branch(tmp_path, "main")
    _run(tmp_path, "checkout", "-b", "feature/existing-work")
    (tmp_path / "file.txt").write_text("changed\n")

    workflow = GitWorkflow(repo_path=tmp_path)
    result = await workflow.git_commit(message="test change")

    assert result["success"] is True
    status = await workflow.git_status()
    assert status["current_branch"] == "feature/existing-work"


@pytest.mark.asyncio
async def test_git_push_refuses_on_protected_branch(tmp_path):
    _init_repo_on_branch(tmp_path, "main")

    workflow = GitWorkflow(repo_path=tmp_path)
    result = await workflow.git_push(branch="main")

    assert result["success"] is False
    assert "protected branch" in result["error"]


@pytest.mark.asyncio
async def test_git_push_refuses_on_protected_branch_even_with_force(tmp_path):
    _init_repo_on_branch(tmp_path, "main")

    workflow = GitWorkflow(repo_path=tmp_path)
    result = await workflow.git_push(branch="main", force=True)

    assert result["success"] is False
    assert "protected branch" in result["error"]


@pytest.mark.asyncio
async def test_git_push_on_feature_branch_gets_past_the_guard(tmp_path):
    _init_repo_on_branch(tmp_path, "main")
    _run(tmp_path, "checkout", "-b", "feature/existing-work")

    workflow = GitWorkflow(repo_path=tmp_path)
    result = await workflow.git_push()

    # No remote configured in this throwaway repo -- proves the call reached
    # past the protected-branch guard rather than being blocked by it.
    assert result["success"] is False
    assert "protected branch" not in result["error"]
    assert "remote" in result["error"].lower()
