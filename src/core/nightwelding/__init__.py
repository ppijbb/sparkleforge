"""Nightwelding: reproduce-first autonomous issue fixer.

Given an issue (from GitHub or local file/markdown), Nightwelding writes a failing
test that reproduces the described problem, implements a fix until that test passes,
and publishes the result (Draft PR or local branch/patch) for human review.

Entry points: src.core.nightwelding.runner.run_nightwelding_issue /
run_nightwelding_sweep, wired into the CLI via `sparkleforge nightwelding run`.
"""

from src.core.nightwelding.adapter import BaseNightweldingAdapter, IssueContext
from src.core.nightwelding.github_adapter import GitHubAdapter
from src.core.nightwelding.local_adapter import LocalGitAdapter

__all__ = [
    "BaseNightweldingAdapter",
    "IssueContext",
    "GitHubAdapter",
    "LocalGitAdapter",
]
