"""Regression test for issue #1757: `src` import must not resolve to an
unrelated project's `src` package via PYTHONPATH."""

import sys

import sparkleforge_bootstrap


def test_prioritize_repo_root_puts_repo_first(monkeypatch):
    fake_unrelated = "/tmp/some-other-project"
    monkeypatch.setattr(sys, "path", [fake_unrelated, "/usr/lib/python3.13"])

    sparkleforge_bootstrap._prioritize_repo_root()

    assert sys.path[0] == sparkleforge_bootstrap.os.path.dirname(
        sparkleforge_bootstrap.os.path.abspath(sparkleforge_bootstrap.__file__)
    )


def test_prioritize_repo_root_dedupes_existing_entry(monkeypatch):
    repo_root = sparkleforge_bootstrap.os.path.dirname(
        sparkleforge_bootstrap.os.path.abspath(sparkleforge_bootstrap.__file__)
    )
    monkeypatch.setattr(sys, "path", ["/tmp/some-other-project", repo_root, "/usr/lib/python3.13"])

    sparkleforge_bootstrap._prioritize_repo_root()

    assert sys.path[0] == repo_root
    assert sys.path.count(repo_root) == 1
