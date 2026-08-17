"""Autofix: the OpenCode repair-loop that backs `.github/workflows/opencode-auto-fix.yml`.

Retries `sparkleforge ci fix-issue` against an already-checked-out issue
context, gated by an optional self-verify command (aborts immediately on
failure) and a verify command (retries up to max_iterations on failure).
Unlike src/core/nightwelding, this loop has no reproduction-test eligibility
gate and never opens Drafts itself -- it just leaves a committed working tree
for the calling workflow to push and open a PR from.

Entry point: src.core.autofix.runner.run_autofix_repair_loop, wired into the
CLI via `sparkleforge autofix run`.
"""
