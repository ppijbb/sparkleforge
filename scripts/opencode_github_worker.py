"""GitHub Actions helper for low-cost OpenCode issue fixing.

This worker reads a GitHub issue context file, asks the local OpenCode agent for
a unified diff, and applies it to the checked-out branch.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.cli_agents.open_code_agent import OpenCodeAgent


def run(cmd: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def repo_snapshot() -> str:
    files = run(["git", "ls-files"]).stdout.splitlines()
    keep = [
        path
        for path in files
        if path.startswith((".github/", "src/", "scripts/", "tests/"))
        or path in {"pyproject.toml", "package.json", "README.md"}
    ]
    return "\n".join(keep[:600])


def extract_diff(text: str) -> str:
    fenced = re.search(r"```(?:diff|patch)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    idx = candidate.find("diff --git ")
    if idx >= 0:
        return candidate[idx:].strip() + "\n"
    idx = candidate.find("--- ")
    if idx >= 0:
        return candidate[idx:].strip() + "\n"
    return ""


def requested_read_paths(text: str) -> list[str]:
    paths = re.findall(
        r"<parameter\s+name=[\"']file_path[\"']>\s*([^<]+?)\s*</parameter>",
        text,
        re.IGNORECASE,
    )
    return [path.strip() for path in paths if path.strip()]


def _line_snippet(path: str, content: str, start: int, end: int) -> str:
    lines = content.splitlines()
    start = max(start, 1)
    end = min(end, len(lines))
    if start > end:
        return ""
    numbered = [
        f"{line_no:5d}: {lines[line_no - 1]}"
        for line_no in range(start, end + 1)
    ]
    return f"--- {path}:{start}-{end} ---\n" + "\n".join(numbered)


def file_context_for_issue(path: str, issue_context: str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        return ""

    content = file_path.read_text(encoding="utf-8")
    snippets: list[str] = []
    seen: set[tuple[int, int]] = set()

    referenced_lines = [
        int(match.group(1))
        for match in re.finditer(rf"{re.escape(path)}:(\d+)", issue_context)
    ]
    for line_no in referenced_lines:
        start = max(1, line_no - 40)
        end = line_no + 40
        key = (start, end)
        if key not in seen:
            seen.add(key)
            snippets.append(_line_snippet(path, content, start, end))

    tokens = {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]{7,}", issue_context)
        if token not in {"github", "actions", "workflow", "unified"}
    }
    for token in sorted(tokens, key=len, reverse=True)[:20]:
        idx = content.find(token)
        if idx < 0:
            continue
        line_no = content[:idx].count("\n") + 1
        start = max(1, line_no - 25)
        end = line_no + 25
        key = (start, end)
        if key not in seen:
            seen.add(key)
            snippets.append(_line_snippet(path, content, start, end))
        if len(snippets) >= 8:
            break

    if snippets:
        return "\n\n".join(snippets)

    if len(content) > 20000:
        return f"--- {path} ---\n{content[:20000]}\n...[truncated]"
    return f"--- {path} ---\n{content}"


def build_prompt(
    *,
    snapshot: str,
    status: str,
    issue_context: str,
    file_contents_str: str,
    extra_context: str,
    tool_context: str = "",
) -> str:
    return f"""
You are editing the SparkleForge repository in GitHub Actions.

Create a small, focused unified git diff that fixes the issue below.

Rules:
- Output only a unified diff. No prose.
- Do not request tools, emit XML, or describe what you would inspect.
- You cannot call read, grep, shell, or any external tool.
- Use the repository snapshot and file contents already provided in this prompt.
- Prefer editing existing files listed in the repository snapshot.
- Do not change generated artifacts, lockfiles, or unrelated documentation.
- Keep the patch minimal and directly tied to the issue.

Repository snapshot:
{snapshot}

Current git status:
{status}

Issue context:
{issue_context}

{file_contents_str}
{tool_context}
Additional review or verification context:
{extra_context or "None"}
""".strip()


async def fix_issue(issue_context_path: Path, extra_context_path: Path | None = None) -> int:
    issue_context = issue_context_path.read_text(encoding="utf-8")
    extra_context = ""
    if extra_context_path and extra_context_path.exists():
        extra_context = extra_context_path.read_text(encoding="utf-8").strip()
    snapshot = repo_snapshot()
    status = run(["git", "status", "--short"]).stdout

    all_files = snapshot.splitlines()
    relevant_contents = []
    for f in all_files:
        if f in issue_context and Path(f).is_file():
            try:
                relevant_contents.append(file_context_for_issue(f, issue_context))
            except Exception:
                pass

    file_contents_str = "\n\n".join(relevant_contents[:5])
    if file_contents_str:
        file_contents_str = f"Relevant File Contents:\n{file_contents_str}\n"

    agent = OpenCodeAgent()
    tool_context = ""
    response = ""
    diff = ""
    system_message = (
        "You are a careful coding agent. Return only a git-apply compatible "
        "unified diff for the requested fix. Do not use tools, XML tags, markdown "
        "narration, or prose."
    )
    for llm_attempt in range(2):
        prompt = build_prompt(
            snapshot=snapshot,
            status=status,
            issue_context=issue_context,
            file_contents_str=file_contents_str,
            extra_context=extra_context,
            tool_context=tool_context,
        )
        result = await agent.execute_query(prompt, system_message=system_message)
        if not result.get("success"):
            print(result.get("response") or result.get("error") or "OpenCode failed", file=sys.stderr)
            return 1

        response = result.get("response", "")
        diff = extract_diff(response)
        if diff:
            break

        paths = requested_read_paths(response)
        if not paths or llm_attempt == 1:
            break

        requested_context = []
        for path in paths[:3]:
            if path in all_files and Path(path).is_file():
                requested_context.append(file_context_for_issue(path, issue_context + "\n" + response))
        if not requested_context:
            break

        tool_context = (
            "Requested file contents are provided below. You must now return only "
            "a unified diff, with no tool calls or prose.\n"
            + "\n\n".join(requested_context)
            + "\n"
        )

    if not diff:
        print("OpenCode did not return an applicable diff.", file=sys.stderr)
        print(response[:4000], file=sys.stderr)
        return 1

    Path("opencode.patch").write_text(diff, encoding="utf-8")
    # Try git apply first (strict)
    apply_proc = run(["git", "apply", "--check", "opencode.patch"])
    if apply_proc.returncode == 0:
        run(["git", "apply", "opencode.patch"])
        return 0

    # Fallback to patch(1) which is much more lenient with LLM diffs (fuzz, offsets)
    print("git apply failed, falling back to patch...", file=sys.stderr)
    apply_proc = run(["patch", "-p1", "--no-backup-if-mismatch", "--forward", "-i", "opencode.patch"])
    if apply_proc.returncode != 0:
        print(apply_proc.stdout, file=sys.stderr)
        print(apply_proc.stderr, file=sys.stderr)
        print("--- Rejected Patch ---", file=sys.stderr)
        print(diff[:4000], file=sys.stderr)
        return apply_proc.returncode

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
