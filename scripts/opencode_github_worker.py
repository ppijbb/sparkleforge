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


def read_full_file(path: str, limit: int = 200_000) -> str:
    """Return full file contents with line numbers, truncated to limit chars."""
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    content = file_path.read_text(encoding="utf-8")
    if len(content) > limit:
        lines = content[:limit].splitlines()
        numbered = [f"{i+1:5d}: {line}" for i, line in enumerate(lines)]
        return f"--- {path} (truncated to {limit} chars) ---\n" + "\n".join(numbered) + "\n...[truncated]\n"
    lines = content.splitlines()
    numbered = [f"{i+1:5d}: {line}" for i, line in enumerate(lines)]
    return f"--- {path} ---\n" + "\n".join(numbered) + "\n"


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
You are an autonomous coding agent editing the SparkleForge repository.

Your goal is to fix the issue described below.

Step-by-step process:
1. Review the issue context and the repository snapshot.
2. If you need to see the contents of a file, request it using this format:
   <parameter name="file_path">path/to/file.py</parameter>
3. Once you have enough information, output a single unified git diff.

Rules:
- Output ONLY the unified diff or a file request. No prose.
- Do not change generated artifacts or lockfiles.
- Keep the patch minimal.
- IMPORTANT: The file contents below include EXACT line numbers. Your diff MUST use
  the correct line numbers as shown. The context lines in the diff must match the
  actual file content exactly, character-for-character.

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


def _apply_patch(patch_path: Path) -> tuple[bool, str]:
    """Try multiple strategies to apply a patch. Returns (success, stderr)."""

    # Strategy 1: git apply --3way --ignore-whitespace
    proc = run([
        "git", "apply",
        "--3way",
        "--ignore-whitespace",
        str(patch_path),
    ])
    if proc.returncode == 0:
        return True, ""

    git_err = proc.stderr.strip()
    print(f"[git apply --3way] failed:\n{git_err}", file=sys.stderr)

    # Strategy 2: patch --fuzz=3 -p1 (tolerates line number drift up to 3)
    patch_bin = run(["which", "patch"]).stdout.strip()
    if patch_bin:
        proc2 = run(
            ["patch", "--fuzz=3", "-p1", "--batch", "--forward"],
            input_text=patch_path.read_text(encoding="utf-8"),
        )
        if proc2.returncode == 0:
            print("[patch --fuzz=3] succeeded as fallback.", file=sys.stderr)
            return True, ""
        fuzz_err = proc2.stderr.strip() or proc2.stdout.strip()
        print(f"[patch --fuzz=3] also failed:\n{fuzz_err}", file=sys.stderr)
        return False, f"git apply error:\n{git_err}\n\npatch --fuzz=3 error:\n{fuzz_err}"

    return False, git_err


async def fix_issue(issue_context_path: Path, extra_context_path: Path | None = None) -> int:
    issue_context = issue_context_path.read_text(encoding="utf-8")
    extra_context = ""
    if extra_context_path and extra_context_path.exists():
        extra_context = extra_context_path.read_text(encoding="utf-8").strip()
    snapshot = repo_snapshot()
    status = run(["git", "status", "--short"]).stdout

    all_files = snapshot.splitlines()

    # Always provide full file contents for relevant files (not just snippets)
    relevant_files = _infer_relevant_files(issue_context, all_files)
    relevant_contents = []
    for f in relevant_files:
        try:
            content = read_full_file(f)
            if content:
                relevant_contents.append(content)
        except Exception:
            pass

    file_contents_str = "\n\n".join(relevant_contents[:5])
    if file_contents_str:
        file_contents_str = f"Relevant File Contents (with exact line numbers):\n{file_contents_str}\n"

    agent = OpenCodeAgent()
    tool_context = ""
    response = ""
    diff = ""
    system_message = (
        "You are a careful coding agent. Return only a git-apply compatible "
        "unified diff for the requested fix. Do not use tools, XML tags, markdown "
        "narration, or prose. The diff context lines must match the file exactly."
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
                requested_context.append(read_full_file(path))
        if not requested_context:
            break

        tool_context = (
            "Requested file contents are provided below (with exact line numbers). "
            "You must now return only a unified diff, with no tool calls or prose. "
            "Use the exact line numbers shown when writing the diff hunk headers.\n"
            + "\n\n".join(requested_context)
            + "\n"
        )

    if not diff:
        print("OpenCode did not return an applicable diff.", file=sys.stderr)
        print(response[:4000], file=sys.stderr)
        return 1

    patch_path = Path("opencode.patch")
    patch_path.write_text(diff, encoding="utf-8")

    success, err = _apply_patch(patch_path)
    if not success:
        print(err, file=sys.stderr)
        print("--- Failed Patch ---", file=sys.stderr)
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
