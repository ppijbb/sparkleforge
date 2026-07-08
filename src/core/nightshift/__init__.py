"""Nightshift: reproduce-first autonomous issue fixer.

Given a GitHub issue, Nightshift writes a failing test that reproduces the
described problem, implements a fix until that test passes, and opens a
Draft-only pull request for human review. It never merges.

Entry points: src.core.nightshift.runner.run_nightshift_issue /
run_nightshift_sweep, wired into the CLI via `sparkleforge nightshift run`.
"""
