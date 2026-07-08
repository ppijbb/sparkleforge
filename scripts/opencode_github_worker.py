"""GitHub Actions helper for low-cost OpenCode issue fixing.

This worker reads a GitHub issue context file, asks the local OpenCode agent for
a unified diff, and applies it to the checked-out branch.

Robust patch application strategy:
  1. git apply --3way --ignore-whitespace  (main path)
  2. patch --fuzz=3 -p1                   (fallback: tolerates ±3 lines of offset)
  3. fail with full diagnostics
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.cli_agents.open_code_agent import OpenCodeAgent

from src.core.patch_ops import (
    extract_diff,
    requested_read_paths,
    read_full_file,
    repo_snapshot,
    repository_change_signature,
    _normalize_diff,
    _apply_patch,
    build_prompt,
    run,
)

# CI 잡은 LLM_PROVIDER/LLM_MODEL과 API 키만 주입하므로, 설정 로더가 요구하는
# 나머지 필수 변수에 안전한 기본값을 채운다. 시크릿 성격의 키에는 기본값을 두지 않는다.
_CONFIG_ENV_DEFAULTS = {
    "LLM_TEMPERATURE": "0.2",
    "LLM_MAX_TOKENS": "8192",
    "BUDGET_LIMIT": "5.0",
    "ENABLE_COST_OPTIMIZATION": "true",
}
_MODEL_ROLE_KEYS = (
    "PLANNING_MODEL",
    "REASONING_MODEL",
    "VERIFICATION_MODEL",
    "GENERATION_MODEL",
    "COMPRESSION_MODEL",
)
_CHARS_PER_TOKEN = 3
_PROMPT_SAFETY_TOKENS = 1_000
_MAX_FILE_CONTEXT_CHARS = 200_000


def ensure_config_loaded() -> None:
    """MultiModelOrchestrator 생성 전에 전역 LLM 설정을 1회 로드한다."""
    from src.core import researcher_config

    if researcher_config.config is not None:
        return
    for key, value in _CONFIG_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    default_model = os.getenv("LLM_MODEL")
    if default_model:
        for key in _MODEL_ROLE_KEYS:
            os.environ.setdefault(key, default_model)
    researcher_config.load_config_from_env()


def _infer_relevant_files(issue_context: str, all_files: list[str]) -> list[str]:
    """Heuristically find files most relevant to the issue text."""
    relevant: list[str] = []
    # 1. Exact filename mentions
    for f in all_files:
        basename = Path(f).name
        if basename in issue_context or f in issue_context:
            relevant.append(f)
    # 2. Token-based match (function/class names)
    tokens = {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]{7,}", issue_context)
        if token not in {"github", "actions", "workflow", "unified", "repository"}
    }
    for f in all_files:
        if f in relevant:
            continue
        if not Path(f).is_file():
            continue
        try:
            content = Path(f).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if any(tok in content for tok in sorted(tokens, key=len, reverse=True)[:15]):
            relevant.append(f)
        if len(relevant) >= 8:
            break
    return relevant[:8]


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _available_file_context_chars(
    agent: OpenCodeAgent,
    *,
    snapshot: str,
    status: str,
    issue_context: str,
    file_contents_str: str = "",
    extra_context: str = "",
    tool_context: str = "",
    force_diff: bool = False,
) -> int:
    fixed_prompt = build_prompt(
        snapshot=snapshot,
        status=status,
        issue_context=issue_context,
        file_contents_str=file_contents_str,
        extra_context=extra_context,
        tool_context=tool_context,
        force_diff=force_diff,
    )
    remaining_tokens = (
        agent.prompt_context_budget()
        - _estimate_tokens(fixed_prompt)
        - _PROMPT_SAFETY_TOKENS
    )
    if remaining_tokens <= 0:
        return 0
    return remaining_tokens * _CHARS_PER_TOKEN


def _per_file_context_limit(
    agent: OpenCodeAgent,
    file_count: int,
    *,
    snapshot: str,
    status: str,
    issue_context: str,
    file_contents_str: str = "",
    extra_context: str = "",
    tool_context: str = "",
    force_diff: bool = False,
) -> int:
    if file_count <= 0:
        return 0
    available_chars = _available_file_context_chars(
        agent,
        snapshot=snapshot,
        status=status,
        issue_context=issue_context,
        file_contents_str=file_contents_str,
        extra_context=extra_context,
        tool_context=tool_context,
        force_diff=force_diff,
    )
    if available_chars <= 0:
        return 0
    return max(1, min(_MAX_FILE_CONTEXT_CHARS, available_chars // file_count))


def _prompt_fits_budget(
    agent: OpenCodeAgent,
    *,
    snapshot: str,
    status: str,
    issue_context: str,
    file_contents_str: str = "",
    extra_context: str = "",
    tool_context: str = "",
    force_diff: bool = False,
) -> bool:
    prompt = build_prompt(
        snapshot=snapshot,
        status=status,
        issue_context=issue_context,
        file_contents_str=file_contents_str,
        extra_context=extra_context,
        tool_context=tool_context,
        force_diff=force_diff,
    )
    return (
        _estimate_tokens(prompt) + _PROMPT_SAFETY_TOKENS
        <= agent.prompt_context_budget()
    )


def _read_numbered_files(paths: list[str], per_file_limit: int) -> list[str]:
    if per_file_limit <= 0:
        return []

    contents = []
    for path in paths:
        try:
            content = read_full_file(path, limit=per_file_limit)
            if content:
                contents.append(content)
        except Exception:
            pass
    return contents


def _format_relevant_file_contents(contents: list[str]) -> str:
    joined = "\n\n".join(contents)
    if not joined:
        return ""
    return f"Relevant File Contents (with exact line numbers):\n{joined}\n"


def _format_requested_tool_context(contents: list[str]) -> str:
    joined = "\n\n".join(contents)
    if not joined:
        return ""
    return (
        "Requested file contents are provided below (with exact line numbers). "
        "You must now return only a unified diff, with no tool calls or prose. "
        "Use the exact line numbers shown when writing the diff hunk headers.\n"
        + joined
        + "\n"
    )


def _shrink_limit(limit: int) -> int:
    if limit <= 1:
        return 0
    return max(1, limit // 2)


def _budgeted_relevant_file_contents(
    agent: OpenCodeAgent,
    paths: list[str],
    *,
    snapshot: str,
    status: str,
    issue_context: str,
    extra_context: str = "",
) -> str:
    selected_paths = paths[:5]
    per_file_limit = _per_file_context_limit(
        agent,
        len(selected_paths),
        snapshot=snapshot,
        status=status,
        issue_context=issue_context,
        extra_context=extra_context,
    )
    while per_file_limit > 0:
        file_contents_str = _format_relevant_file_contents(
            _read_numbered_files(selected_paths, per_file_limit)
        )
        if not file_contents_str or _prompt_fits_budget(
            agent,
            snapshot=snapshot,
            status=status,
            issue_context=issue_context,
            file_contents_str=file_contents_str,
            extra_context=extra_context,
        ):
            return file_contents_str
        per_file_limit = _shrink_limit(per_file_limit)
    return ""


def _budgeted_requested_tool_context(
    agent: OpenCodeAgent,
    paths: list[str],
    *,
    snapshot: str,
    status: str,
    issue_context: str,
    file_contents_str: str = "",
    extra_context: str = "",
) -> str:
    selected_paths = paths[:3]
    per_file_limit = _per_file_context_limit(
        agent,
        len(selected_paths),
        snapshot=snapshot,
        status=status,
        issue_context=issue_context,
        file_contents_str=file_contents_str,
        extra_context=extra_context,
    )
    while per_file_limit > 0:
        tool_context = _format_requested_tool_context(
            _read_numbered_files(selected_paths, per_file_limit)
        )
        if not tool_context or _prompt_fits_budget(
            agent,
            snapshot=snapshot,
            status=status,
            issue_context=issue_context,
            file_contents_str=file_contents_str,
            extra_context=extra_context,
            tool_context=tool_context,
        ):
            return tool_context
        per_file_limit = _shrink_limit(per_file_limit)
    return ""


def _strip_fenced_response(raw_content: str) -> str:
    raw_content = raw_content.strip()
    if not raw_content.startswith("```"):
        return raw_content
    lines = raw_content.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_triage_response(raw_content: str) -> dict[str, object]:
    import json

    raw_content = _strip_fenced_response(raw_content)
    candidates = [raw_content]
    start = raw_content.find("{")
    end = raw_content.rfind("}")
    if 0 <= start < end:
        candidates.append(raw_content[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(data, dict)
            and "should_create_issue" in data
            and "title" in data
            and "body" in data
        ):
            return data

    raise ValueError(
        "Malformed triage response: missing required keys "
        "'should_create_issue', 'title', or 'body'. Raw response:\n"
        f"{raw_content}"
    )


async def fix_issue(issue_context_path: Path, extra_context_path: Path | None = None) -> int:
    issue_context = issue_context_path.read_text(encoding="utf-8")
    extra_context = ""
    if extra_context_path and extra_context_path.exists():
        extra_context = extra_context_path.read_text(encoding="utf-8").strip()
    snapshot = repo_snapshot()
    status = run(["git", "status", "--short"]).stdout

    all_files = snapshot.splitlines()
    agent = OpenCodeAgent()

    # Provide relevant files within the active model's prompt budget.
    relevant_files = _infer_relevant_files(issue_context, all_files)
    file_contents_str = _budgeted_relevant_file_contents(
        agent,
        relevant_files,
        snapshot=snapshot,
        status=status,
        issue_context=issue_context,
        extra_context=extra_context,
    )

    tool_context = ""
    response = ""
    diff = ""
    system_message = (
        "You are a careful coding agent working against a real repository. "
        "On each turn, output EITHER a file request using "
        '<parameter name="file_path">path/to/file.py</parameter> '
        "OR a git-apply compatible unified diff — never both, and no other "
        "prose, markdown narration, or tool calls. "
        "The diff context lines must match the file exactly."
    )
    max_llm_attempts = 3
    for llm_attempt in range(max_llm_attempts):
        is_final_attempt = llm_attempt == max_llm_attempts - 1
        prompt = build_prompt(
            snapshot=snapshot,
            status=status,
            issue_context=issue_context,
            file_contents_str=file_contents_str,
            extra_context=extra_context,
            tool_context=tool_context,
            force_diff=is_final_attempt,
        )
        result = await agent.execute_query(prompt, system_message=system_message)
        if not result.get("success"):
            print(
                result.get("response") or result.get("error") or "OpenCode failed", file=sys.stderr
            )
            return 1

        response = result.get("response", "")
        diff = extract_diff(response)
        if diff:
            break

        if is_final_attempt:
            break

        paths = requested_read_paths(response)
        requested_paths = [
            path for path in paths[:3] if path in all_files and Path(path).is_file()
        ]
        tool_context = _budgeted_requested_tool_context(
            agent,
            requested_paths,
            snapshot=snapshot,
            status=status,
            issue_context=issue_context,
            file_contents_str=file_contents_str,
            extra_context=extra_context,
        )
        if not tool_context:
            continue

    if not diff:
        print("OpenCode did not return an applicable diff.", file=sys.stderr)
        return 1
    if not diff.strip():
        raise ValueError("OpenCode produced an empty patch.")

    patch_path = Path("opencode.patch")
    before_signature = repository_change_signature()
    patch_path.write_text(diff, encoding="utf-8")

    success, err = _apply_patch(patch_path)
    if not success:
        print(err, file=sys.stderr)
        print("--- Failed Patch ---", file=sys.stderr)
        print(diff[:4000], file=sys.stderr)
        return 1
    if repository_change_signature() == before_signature:
        print("OpenCode patch applied cleanly but produced no repository changes.", file=sys.stderr)
        print("--- No-op Patch ---", file=sys.stderr)
        print(diff[:4000], file=sys.stderr)
        return 1

    return 0


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
    
    result = await orchestrator.execute_with_model(
        prompt=prompt,
        task_type=TaskType.RESEARCH,
        system_message=system_message,
        use_cascade=False
    )
    
    Path("review_result.txt").write_text(result.content, encoding="utf-8")
    print("✅ Code review completed and saved to review_result.txt")
    return 0


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
            import json as _json

            open_issues = _json.loads(open_issues_path.read_text(encoding="utf-8"))
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
        import json

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
    import json

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fix-issue", "code-review", "issue-triage", "merge-decision"])
    parser.add_argument("--issue-context", default="issue-context.md")
    parser.add_argument("--extra-context", default=None)
    parser.add_argument("--diff", default="diff.txt")
    parser.add_argument("--review-file", default="review_result.txt")
    parser.add_argument("--cerebras-file", default="cerebras_result.txt")
    parser.add_argument("--pr-meta-file", default="pr_meta.json")
    parser.add_argument("--open-issues-file", default=None)
    args = parser.parse_args()

    if args.command == "fix-issue":
        extra_context = Path(args.extra_context) if args.extra_context else None
        return asyncio.run(fix_issue(Path(args.issue_context), extra_context))
    elif args.command == "code-review":
        return asyncio.run(code_review(Path(args.diff)))
    elif args.command == "issue-triage":
        cerebras_file = Path(args.cerebras_file) if args.cerebras_file else None
        open_issues_file = Path(args.open_issues_file) if args.open_issues_file else None
        return asyncio.run(issue_triage(Path(args.review_file), cerebras_file, open_issues_file))
    elif args.command == "merge-decision":
        cerebras_file = Path(args.cerebras_file) if args.cerebras_file else None
        return asyncio.run(merge_decision(Path(args.pr_meta_file), Path(args.review_file), cerebras_file))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
