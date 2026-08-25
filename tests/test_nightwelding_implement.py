import subprocess

from src.core.nightwelding import implement


def _proc(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_implement_until_green_invokes_sparkleforge_ci_fix_issue(tmp_path, monkeypatch):
    """implement.py's fix-issue call was retargeted from the (now-deleted)
    scripts/opencode_github_worker.py to `main.py ci fix-issue` -- assert the
    actual subprocess argv, since monkeypatching implement_until_green itself
    (as test_nightwelding_local_adapter.py does) would never catch a bad argv.
    """
    calls = []

    def fake_run(cmd, cwd, timeout=None):
        calls.append(cmd)
        if "fix-issue" in cmd:
            return _proc(returncode=1, stdout="", stderr="no diff")
        return _proc(returncode=0)

    monkeypatch.setattr(implement, "_run", fake_run)
    monkeypatch.setattr(implement.patch_ops, "repository_change_signature", lambda cwd=None: ("M foo.py",))

    result = implement.implement_until_green(
        issue_context="fix the bug",
        repro_test_files=["tests/test_foo.py"],
        repo_root=tmp_path,
        commit_title="fix: something",
        max_iterations=1,
    )

    assert result.success is False
    fix_issue_calls = [c for c in calls if "fix-issue" in c]
    assert len(fix_issue_calls) == 1
    argv = fix_issue_calls[0]
    assert argv[1].endswith("main.py")
    assert argv[2:4] == ["ci", "fix-issue"]
    assert not any("opencode_github_worker.py" in str(part) for part in argv)
