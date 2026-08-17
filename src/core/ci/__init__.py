"""CI gate agents: code review, issue triage, merge decision, and single-attempt issue fixing.

Moved out of scripts/opencode_github_worker.py (a standalone GitHub Actions helper
script) so that GitHub Actions is a thin caller of SparkleForge's own CLI
(`sparkleforge ci ...`) rather than the owner of this logic.

Entry points: src.core.ci.code_review.code_review, src.core.ci.issue_triage.issue_triage,
src.core.ci.merge_decision.merge_decision, src.core.ci.fix_issue.fix_issue -- wired into
the CLI via `sparkleforge ci {code-review,issue-triage,merge-decision,fix-issue}`.
"""
