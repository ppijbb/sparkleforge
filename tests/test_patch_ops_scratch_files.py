"""Regression test: SparkleForge's own runtime scratch files must never be
committed into a real target repo.

Covers the incident where issue-context.md/sparkleforge.patch/sparkleforge-extra-
context.md ended up committed into a real PR (qwp0905/lfdb#309) because the
commit-staging logic in the nightwelding adapters and implement.py did a
blanket `git add` of every untracked file.
"""

from __future__ import annotations

from src.core import patch_ops


def test_scratch_files_are_recognized():
    for name in (
        "issue-context.md",
        "sparkleforge.patch",
        "sparkleforge-single.patch",
        "sparkleforge-extra-context.md",
        "sparkleforge-verify.log",
        "sparkleforge-worker-error.log",
        "sparkleforge-self-verify.log",
        "foo.patch.orig",
        "bar.rej",
        "tests/__pycache__/test_readme.cpython-313-pytest-9.0.2.pyc",
        "src/__pycache__/foo.pyc",
        "standalone.pyc",
    ):
        assert patch_ops.is_runtime_scratch_path(name), name


def test_real_repo_files_are_not_scratch():
    for name in ("README.md", "src/table/tests/table_name.rs", "tests/test_foo.py"):
        assert not patch_ops.is_runtime_scratch_path(name), name
