"""Anvil roadmap next-target selection.

Given the open Anvil milestone and each referenced sub-issue's live state
(fetched by the workflow via `gh issue view`, mechanical and left in YAML),
picks the next OPEN sub-issue as today's mandatory roadmap target. This is a
real selection decision -- not extraction -- moved verbatim from
sparkleforge-daily-roadmap.yml's second "Collect GitHub planning context"
heredoc.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnvilTarget:
    number: int | None
    title: str | None

    @property
    def found(self) -> bool:
        return self.number is not None


def select_anvil_target(milestone: dict | None, sub_status: list[dict]) -> AnvilTarget:
    if not milestone:
        return AnvilTarget(None, None)
    for item in sub_status:
        if item.get("state") == "OPEN":
            return AnvilTarget(item["number"], item["title"])
    return AnvilTarget(None, None)


def render_planning_context(milestone: dict | None, sub_status: list[dict]) -> str:
    """Same lines the original heredoc appended to github-planning-context.md."""
    if not milestone:
        return "- No open Anvil phase milestone found."

    lines = [f"- Open milestone: #{milestone['number']} {milestone['title']}"]
    if sub_status:
        for item in sub_status:
            lines.append(f"  - #{item['number']} [{item['state']}] {item['title']}")
    else:
        lines.append("- No sub-issues referenced in the milestone checklist yet.")

    target = select_anvil_target(milestone, sub_status)
    if target.found:
        lines.append(f"- Next open Anvil roadmap sub-issue: #{target.number} {target.title}")
    else:
        lines.append("- All known Anvil roadmap sub-issues are closed (or none tracked).")
    return "\n".join(lines)


def target_file_contents(target: AnvilTarget) -> str:
    """Same format anvil-roadmap-target.md was written in: 'number|title' or empty."""
    if not target.found:
        return ""
    return f"{target.number}|{target.title}"
