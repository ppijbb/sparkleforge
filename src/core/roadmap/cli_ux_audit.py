"""Prompt template for the self-audit that judges SparkleForge's own terminal output.

Mirrors build_daily_roadmap_mission_brief (src/core/daily_roadmap.py): same
required-section shape, so the existing roadmap-issue-body/roadmap-fallback-issue
machinery in sparkleforge-daily-roadmap.yml can render and file this without a
second, parallel issue-assembly path.
"""

from __future__ import annotations

# Representative surface sampled by `sparkleforge report cli-ux-audit-prompt`.
# Cheap, side-effect-free, no network/API calls -- just what a human runs first.
DEFAULT_AUDIT_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("sparkleforge", "--help"),
    ("sparkleforge", "health"),
    ("sparkleforge", "cli", "list"),
)


def build_cli_ux_audit_prompt(today: str, transcripts: dict[str, str]) -> str:
    """transcripts: command string -> pty-captured output, in run order."""
    rendered = "\n\n".join(
        f"### `{command}`\n```\n{output.strip()}\n```" for command, output in transcripts.items()
    )
    return f"""You are SparkleForge running inside the SparkleForge repository on {today}, auditing your own CLI's terminal output.

The transcripts below were captured under a real pty (not a plain pipe), so colors, spinners, and width-based wrapping render exactly as a human would see them in their own terminal. A rendered screenshot of each command's output is also attached as an image, in the same order as the transcripts below -- judge those images too, not just the text. Text-only review has already been proven to miss real bugs here: a missing emoji font made every status glyph render as a blank box on screen while looking completely normal in the text transcript. Look at the actual pixels for unrenderable/garbled glyphs, misalignment, and anything that looks broken rather than just noisy.

CLAUDE.md's CLI UX rule for this repository: "treat noisy/confusing terminal output as an actionable bug, not a side effect. Before adding a log call, ask who it's for: internal debugging belongs in the log file (or DEBUG level), not stdout. Don't add module-level logging side effects that fire before argparse/mode dispatch has decided what the user actually asked for."

## Captured transcripts

{rendered}

Judge these transcripts and the attached screenshots against that rule. Look for: DEBUG/internal logging leaking to stdout, output printed before argparse has resolved the subcommand, stack traces or tracebacks on ordinary non-error paths, inconsistent or missing formatting, unrenderable/garbled glyphs visible in the screenshots, and anything a first-time user would find confusing, broken-looking, or noisy.

If you find nothing actionable, output exactly: NO_ACTIONABLE_CLI_UX_ISSUES

Otherwise the output will become a GitHub issue that an automated coding workflow will implement. Make it concrete and bounded, pointing at the specific command and line from the transcript above.

Required markdown structure:
# <short actionable issue title>

## Why now
- Quote the specific noisy/confusing line(s) and which command produced them.

## Proposed change
- Describe the fix: what should be silenced, moved to DEBUG/log-file, or reordered after dispatch.

## Implementation notes
- Name the likely source file/module (log call site) to inspect.

## Acceptance criteria
- Provide a checklist of verifiable outcomes (e.g. "`sparkleforge --help` prints only the help text, no import-time log lines").

## Validation
- State exact commands to re-run and confirm the transcript is clean."""
