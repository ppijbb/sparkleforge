"""Fallback roadmap-issue content and issue-body assembly for the daily roadmap.

Moved verbatim (heuristics, thresholds, and text unchanged) from
sparkleforge-daily-roadmap.yml's "Run SparkleForge research" and "Create
GitHub issue" python heredocs. Choosing what today's issue is about when the
primary CLI research path fails, and maintaining the recurring-failure
occurrence log, are judgment calls -- not mechanical extraction.

Known pre-existing oddity, preserved as-is (this is a relocation, not a
rewrite): the final fallback-of-fallbacks heading is the literal string
"Triage issue 13", not a computed value. That was already the workflow's
behavior before this move.
"""

from __future__ import annotations

import re


def _collect(context_md: str, header: str, stop: str) -> list[str]:
    active = False
    rows: list[str] = []
    for raw in context_md.splitlines():
        line = raw.strip()
        if line == header:
            active = True
            continue
        if active and line == stop:
            break
        if active and line.startswith("- #"):
            rows.append(line)
    return rows


def build_fallback_roadmap(
    *,
    context_md: str,
    anvil_target: str,
    rc: str,
    invalid_reason: str,
    output_bytes: str,
    console_bytes: str,
    error_bytes: str,
) -> str:
    """Pick a fallback target (Anvil sub-issue -> open PR -> open issue ->
    generic) and render the same required-section roadmap doc the primary
    CLI path would have produced, so the daily issue always gets updated."""
    prs = _collect(context_md, "### Open pull requests", "### Open issues")
    issues = _collect(context_md, "### Open issues", "### Anvil roadmap status")

    if anvil_target:
        number, _, title = anvil_target.partition("|")
        target_kind = "anvil roadmap sub-issue"
    elif prs:
        target_kind = "pull request"
        match = re.match(r"- #(?P<number>\d+) (?P<title>.*?) \[", prs[0])
        number = match.group("number") if match else ""
        title = match.group("title") if match else "repository backlog"
    elif issues:
        target_kind = "issue"
        match = re.match(r"- #(?P<number>\d+) (?P<title>.*?) \[", issues[0])
        number = match.group("number") if match else ""
        title = match.group("title") if match else "repository backlog"
    else:
        target_kind = "repository backlog"
        number = ""
        title = "repository backlog"

    if target_kind == "anvil roadmap sub-issue" and number:
        heading = f"Advance Anvil roadmap sub-issue {number}"
        target_label = f"Anvil roadmap sub-issue #{number}: {title}"
    elif target_kind == "pull request" and number:
        heading = f"Unblock pull request {number}"
        target_label = f"PR #{number}: {title}"
    elif target_kind == "issue" and number:
        heading = f"Triage issue {number}"
        target_label = f"Issue #{number}: {title}"
    else:
        heading = "Triage issue 13"
        target_label = title

    lines = [
        f"# {heading}",
        "",
        "## Why now",
        "- Daily roadmap generation must update the tracked GitHub issue even when the full SparkleForge research path is slow, unavailable, or returns invalid output.",
        f"- The live GitHub context collection selected {target_label} as the top actionable backlog item.",
        f"- SparkleForge CLI fallback metadata: exit code `{rc}`, output bytes `{output_bytes}`, console bytes `{console_bytes}`, error bytes `{error_bytes}`.",
    ]
    if invalid_reason:
        lines.append(f"- Invalid output reason: {invalid_reason}.")
    lines += [
        "",
        "## Proposed change",
        f"- Use the current repository issue/PR context to move {target_label} toward a mergeable or closable state.",
        "- Keep the scope bounded to the smallest code, workflow, or documentation change needed to unblock that item.",
        "",
        "## Implementation notes",
    ]
    if target_kind == "anvil roadmap sub-issue":
        lines += [
            f"- This is an Anvil roadmap sub-issue (#{number}). Implement exactly the checklist items in that issue's body — do not re-scope it or substitute a different idea.",
            "- See docs/ANVIL_PLAN.md for the phase this sub-issue belongs to and its success criteria.",
        ]
    else:
        lines.append(
            "- Start from the referenced PR/issue, inspect its checks, labels, and latest comments, then patch only the directly related files."
        )
    lines += [
        "- Preserve the daily roadmap workflow contract: collect GitHub context, produce required roadmap sections, and update the existing daily issue instead of creating duplicates.",
        "- If this item is already resolved by the time the automation runs, select the next open Anvil roadmap sub-issue first, then the next open PR, then the next open issue.",
        "",
        "## Acceptance criteria",
        "- [ ] The selected PR/issue has a concrete code or workflow change attached.",
        "- [ ] Required GitHub checks pass for the resulting PR.",
        "- [ ] The daily roadmap issue body is updated with the latest GitHub context rather than skipped.",
        "- [ ] No new duplicate daily-roadmap failure issue is created for the same day.",
        "",
        "## Validation",
        "- `./actionlint .github/workflows/sparkleforge-daily-roadmap.yml .github/workflows/gemini-assistant.yml .github/workflows/opencode-auto-fix.yml .github/workflows/pr-merge-gate.yml`",
        "- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli_entrypoints.py tests/test_open_code_agent_config.py tests/test_cli_result_handling.py`",
        "- `git diff --check`",
    ]
    return "\n".join(lines) + "\n"


def build_issue_body(*, today: str, status: str, roadmap_text: str, previous_body: str) -> str:
    """Assemble the issue body; for the recurring-failure ("fallback") case,
    maintain a deduplicated, most-recent-10 occurrence log instead of losing
    prior occurrence history on every update."""
    lines = [
        f"<!-- sparkleforge-daily-roadmap:{today} -->",
        "",
        "This issue was generated by the daily SparkleForge roadmap workflow.",
        "It is intentionally scoped so the automated OpenCode workflow can create a branch, commit fixes, open a PR, and iterate on verification feedback.",
        "",
    ]

    if status == "fallback":
        occurrences = re.findall(r"^- (\d{4}-\d{2}-\d{2})$", previous_body, flags=re.MULTILINE)
        if today not in occurrences:
            occurrences.append(today)
        occurrences = occurrences[-10:]
        lines.append("## Occurrence log")
        lines.append(
            "Recurring daily-roadmap generation failures are tracked in this single issue instead of one new issue per day."
        )
        for day in occurrences:
            lines.append(f"- {day}")
        lines.append("")

    lines.append(roadmap_text)
    return "\n".join(lines)
