"""Decide whether a code-review finding warrants a new GitHub issue."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.core.ci.config import ensure_config_loaded
from src.core.ci.response_parsing import _parse_triage_response, _strip_fenced_response


async def issue_triage(
    review_path: Path,
    cerebras_path: Path | None = None,
    open_issues_path: Path | None = None,
) -> int:
    if not review_path.exists():
        print(f"Error: Review file {review_path} not found.", file=sys.stderr)
        return 1
    openrouter_review = review_path.read_text(encoding="utf-8")
    cerebras_review = ""
    if cerebras_path and cerebras_path.exists():
        cerebras_review = cerebras_path.read_text(encoding="utf-8")

    combined_review = f"OpenRouter Review:\n{openrouter_review}\n\nCerebras Review:\n{cerebras_review}"

    open_issues_section = "None known."
    if open_issues_path and open_issues_path.exists():
        try:
            open_issues = json.loads(open_issues_path.read_text(encoding="utf-8"))
            if open_issues:
                open_issues_section = "\n".join(
                    f"- #{item['number']}: {item['title']}" for item in open_issues
                )
        except Exception:
            pass

    from src.core.llm_manager import MultiModelOrchestrator, TaskType
    import datetime

    ensure_config_loaded()
    orchestrator = MultiModelOrchestrator()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt = f"""You are an elite architectural and security reviewer for SparkleForge. Based on the review, generate a JSON object with 'should_create_issue', 'title', and 'body' keys.
CRITICAL INSTRUCTION: DO NOT create an issue for stylistic changes, minor nitpicks, or uncertain hallucinations. Set should_create_issue to true ONLY if the review identifies a concrete correctness bug, security vulnerability, workflow failure, or critical architectural debt.
DEDUPLICATION: The following issues are currently OPEN in this repository:
{open_issues_section}
If the finding you would report is already substantially covered by one of the open issues above (even if it would be worded differently), set should_create_issue to false — do not file a near-duplicate.
If should_create_issue is true, the title MUST use the repository's plain Conventional Commit prefix style (e.g. 'fix: ...', 'feat: ...').
The 'body' MUST be a Markdown string that explicitly includes the following structured header at the very top:
> **Date**: {current_date}
> **Issue Type**: [Classify as Parent Issue or Sub-issue (Anvil Phase A)]
> **Related Milestone**: [Mention if related to a known milestone like #13 Anvil, otherwise N/A]
> **Metrics & Justification**: [Provide a concrete, logical proof or metric of why this code must be changed. Do not use simple judgments.]

Below the header, provide a highly specific explanation of what is wrong and an actionable plan to fix it.
If should_create_issue is false, title and body must be empty strings.

Review Content:
{combined_review}"""

    system_message = "You are an elite triage agent. Return only a JSON object."

    result = await orchestrator.execute_with_model(
        prompt=prompt,
        task_type=TaskType.ANALYSIS,
        system_message=system_message,
        use_cascade=False
    )

    try:
        data = _parse_triage_response(result.content)
        raw_clean = _strip_fenced_response(result.content)
        if (
            data == {"should_create_issue": False, "title": "", "body": ""}
            and raw_clean
            and not raw_clean.startswith("{")
        ):
            print("Triage response was not actionable JSON; defaulting to no issue.")
        Path("triage_result.json").write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )
        print("✅ Triage completed and saved to triage_result.json")
        return 0
    except Exception as e:
        print(f"Error: Unexpected error during triage: {e}", file=sys.stderr)
        return 1
