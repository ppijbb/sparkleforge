"""Nightwelding: reproduce-first autonomous issue fixer.

Given a GitHub issue, Nightwelding writes a failing test that reproduces the
described problem, implements a fix until that test passes, and opens a
Draft-only pull request for human review. It never merges.

Entry points: src.core.nightwelding.runner.run_nightwelding_issue /
run_nightwelding_sweep, wired into the CLI via `sparkleforge nightwelding run`.
"""

