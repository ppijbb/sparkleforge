"""GitHub Actions helper for low-cost OpenCode issue fixing.

This worker reads a GitHub issue context file, asks the local OpenCode agent for
a unified diff, and applies it to the checked-out branch.

Robust patch application strategy:
  1. git apply --3way --ignore-whitespace  (main path)
  2. patch --fuzz=3 -p1                   (fallback: tolerates ±3 lines of offset)
  3. fail with full diagnostics

code_review/issue_triage/merge_decision moved to src/core/ci/ -- see
src.core.ci.code_review, src.core.ci.issue_triage, src.core.ci.merge_decision,
exposed via `sparkleforge ci {code-review,issue-triage,merge-decision}`.
"""

from __future__ import annotations

import argparse
import asyncio
import re
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
from src.core.ci.response_parsing import _strip_fenced_response

_CHARS_PER_TOKEN = 3
_PROMPT_SAFETY_TOKENS = 1_000
_MAX_FILE_CONTEXT_CHARS = 200_000


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
        "OR a git-apply compatible unified diff, "
        'OR a JSON object with {"action": "decompose", "sub_issues": [{"title": "...", "body": "..."}]} '
        "if the issue is too large for a single patch. "
        "Never output both a diff and a decomposition, and no other prose, markdown narration, or tool calls. "
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
        
        # Check for decomposition request
        import json
        try:
            raw_json = _strip_fenced_response(response)
            if raw_json.startswith("{"):
                data = json.loads(raw_json)
                if data.get("action") == "decompose" and "sub_issues" in data:
                    from src.core.nightwelding.github_adapter import create_subissues
                    # Extract issue number from context file name or path
                    issue_num = re.search(r"#(\d+)", issue_context)
                    if issue_num:
                        await create_subissues(issue_num.group(1), data["sub_issues"])
                        print(f"Successfully decomposed issue #{issue_num.group(1)}")
                        return 0
        except Exception as e:
            print(f"Decomposition parsing failed: {e}", file=sys.stderr)

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fix-issue"])
    parser.add_argument("--issue-context", default="issue-context.md")
    parser.add_argument("--extra-context", default=None)
    args = parser.parse_args()

    if args.command == "fix-issue":
        extra_context = Path(args.extra_context) if args.extra_context else None
        return asyncio.run(fix_issue(Path(args.issue_context), extra_context))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
