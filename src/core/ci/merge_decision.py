"""Decide whether a reviewed PR should auto-merge into dev/main."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.core.ci.config import ensure_config_loaded


async def merge_decision(pr_meta_path: Path, review_path: Path, cerebras_path: Path | None = None) -> int:
    if not pr_meta_path.exists():
        print(f"Error: PR metadata file {pr_meta_path} not found.", file=sys.stderr)
        return 1
    pr_meta = pr_meta_path.read_text(encoding="utf-8")

    openrouter_review = ""
    if review_path.exists():
        openrouter_review = review_path.read_text(encoding="utf-8")

    cerebras_review = ""
    if cerebras_path and cerebras_path.exists():
        cerebras_review = cerebras_path.read_text(encoding="utf-8")

    from src.core.llm_manager import MultiModelOrchestrator, TaskType

    ensure_config_loaded()
    orchestrator = MultiModelOrchestrator()
    prompt = f"""You are the final merge gate for a dev branch. Return a JSON object with keys should_merge and reason. Set should_merge to true only when the reviews do not identify concrete correctness, security, workflow, packaging, or test failures that require code changes. Set should_merge to false for unresolved risks, failing checks, unclear generated changes, or any concrete issue. Do not block solely because one provider skipped or returned no content if another review is available. Keep reason concise.

PR Metadata:
{pr_meta}

OpenRouter Review:
{openrouter_review}

Cerebras Review:
{cerebras_review}"""

    system_message = "You are a merge decision gate. Return only a JSON object."

    result = await orchestrator.execute_with_model(
        prompt=prompt,
        task_type=TaskType.VERIFICATION,
        system_message=system_message,
        use_cascade=False
    )

    raw_content = result.content.strip()
    if raw_content.startswith("```"):
        lines = raw_content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw_content = "\n".join(lines).strip()

    try:
        json.loads(raw_content)
        Path("merge_decision.json").write_text(raw_content, encoding="utf-8")
        print("✅ Merge decision completed and saved to merge_decision.json")
        return 0
    except Exception as je:
        print(f"Error: Invalid JSON response: {je}", file=sys.stderr)
        print(f"Raw response:\n{result.content}", file=sys.stderr)
        return 1
