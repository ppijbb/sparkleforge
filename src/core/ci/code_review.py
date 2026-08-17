"""PR diff review: summarize a git diff via SparkleForge's own orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path

from src.core.ci.config import ensure_config_loaded


async def code_review(diff_path: Path) -> int:
    if not diff_path.exists():
        print(f"Error: Diff file {diff_path} not found.", file=sys.stderr)
        return 1
    diff = diff_path.read_text(encoding="utf-8")
    if not diff.strip():
        Path("review_result.txt").write_text("No changes or empty diff.", encoding="utf-8")
        return 0

    from src.core.llm_manager import MultiModelOrchestrator, TaskType
    ensure_config_loaded()
    orchestrator = MultiModelOrchestrator()
    prompt = f"You are an expert code reviewer. Read the git diff and summarize key issues, bugs, or style violations briefly.\n\nGit Diff:\n{diff}"
    system_message = "You are an expert code reviewer. If the primary model is unavailable, the system will fallback."

    try:
        result = await orchestrator.execute_with_model(
            prompt=prompt,
            task_type=TaskType.RESEARCH,
            system_message=system_message,
            use_cascade=False
        )
    except Exception as e:
        # All providers being down/rate-limited at once is an external outage,
        # not something this PR's diff caused -- don't fail the whole check
        # over it, or every PR opened during an outage gets stuck forever.
        print(f"::warning::Code review unavailable, all model providers failed: {e}", file=sys.stderr)
        Path("review_result.txt").write_text(
            f"Automated code review unavailable: all model providers failed ({e}).",
            encoding="utf-8",
        )
        return 0

    Path("review_result.txt").write_text(result.content, encoding="utf-8")
    print("✅ Code review completed and saved to review_result.txt")
    return 0
