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


async def fix_issue(issue_context_path: Path) -> int:
    issue_context = issue_context_path.read_text(encoding="utf-8")
    snapshot = repo_snapshot()
    status = run(["git", "status", "--short"]).stdout

    prompt = f"""
You are editing the SparkleForge repository in GitHub Actions.

Create a small, focused unified git diff that fixes the issue below.

Rules:
- Output only a unified diff. No prose.
- Prefer editing existing files listed in the repository snapshot.
- Do not change generated artifacts, lockfiles, or unrelated documentation.
- Keep the patch minimal and directly tied to the issue.

Repository snapshot:
{snapshot}

Current git status:
{status}

Issue context:
{issue_context}
""".strip()

    agent = OpenCodeAgent()
    result = await agent.execute_query(
        prompt,
        system_message=(
            "You are a careful coding agent. Return only a git-apply compatible "
            "unified diff for the requested fix."
        ),
    )
    if not result.get("success"):
        print(result.get("response") or result.get("error") or "OpenCode failed", file=sys.stderr)
        return 1

    response = result.get("response", "")
    diff = extract_diff(response)
    if not diff:
        print("OpenCode did not return an applicable diff.", file=sys.stderr)
        print(response[:4000], file=sys.stderr)
        return 1

    Path("opencode.patch").write_text(diff, encoding="utf-8")
    check = run(["git", "apply", "--check", "opencode.patch"])
    if check.returncode != 0:
        print(check.stderr, file=sys.stderr)
        print(diff[:4000], file=sys.stderr)
        return check.returncode

    apply = run(["git", "apply", "opencode.patch"])
    if apply.returncode != 0:
        print(apply.stderr, file=sys.stderr)
        return apply.returncode

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fix-issue"])
    parser.add_argument("--issue-context", default="issue-context.md")
    args = parser.parse_args()

    if args.command == "fix-issue":
        return asyncio.run(fix_issue(Path(args.issue_context)))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
