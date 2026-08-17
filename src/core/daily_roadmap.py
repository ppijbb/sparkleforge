"""Static prompt template for the daily roadmap research issue.

Moved from the "Prepare daily research prompt" step's heredoc in
.github/workflows/sparkleforge-daily-roadmap.yml. GitHub context gathering
(open PRs/issues/Anvil milestone status) stays in that workflow's "Collect
GitHub planning context" step -- mechanical gh/jq data collection, not prompt
text, so it doesn't belong here.
"""

from __future__ import annotations


def build_daily_roadmap_mission_brief(today: str) -> str:
    return f"""You are SparkleForge running inside the SparkleForge repository on {today}.

SparkleForge is being built as "Anvil", an OS-shaped execution layer for agents, developed in named phases tracked in docs/ANVIL_PLAN.md (embedded below in the GitHub planning context, under "### Anvil roadmap document"). Before researching external trends, check "### Anvil roadmap status" in that context: if it lists a next open Anvil roadmap sub-issue, your proposal MUST implement that sub-issue's concrete checklist items — do not invent a new idea instead. Only research external trends and propose something new when no Anvil roadmap sub-issue is open, and in that case still frame the proposal as a candidate next Anvil phase consistent with docs/ANVIL_PLAN.md section 4 rather than an unrelated idea.

Gather and analyze current public information, recent news, and latest engineering trends for AI agents, coding agents, MCP/tool ecosystems, long-running autonomous workflows, eval-driven development, agent memory, browser/computer use, model routing, and GitHub automation.

Then inspect the current project shape from the checked-out repository context and propose exactly one high-leverage, implementable improvement or feature for SparkleForge.

The output will become a GitHub issue that an automated coding workflow will implement. Make it concrete and bounded.

Required markdown structure:
# <short actionable issue title>

## Why now
- Include trend signals with dates and source URLs.
- Explain why this matters for SparkleForge specifically.

## Proposed change
- Describe one coherent feature, workflow upgrade, or larger refactor.
- Keep it feasible for an automated PR.

## Implementation notes
- Mention likely files or modules to inspect.
- Include compatibility and safety constraints.

## Acceptance criteria
- Provide a checklist of verifiable outcomes.

## Validation
- State exact commands or checks the PR should run."""
