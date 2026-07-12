#!/usr/bin/env python3
"""Check whether a diff actually touches the symbols/paths an issue names.

`.github/workflows/opencode-auto-fix.yml` already refuses to auto-close an
issue when the diff only touches workflow/docs files or is <=3 lines -- but
that heuristic missed issue #509's first auto-fix attempt (PR #513): it
added two unused `TypedDict` fields and a duplicate `import logging` line to
two real `.py` files, which is "non-trivial file, >3 lines changed" by that
rule even though it implemented none of the issue's checklist. This script
adds a second, complementary signal: does the diff's added-line content
actually reference any of the concrete identifiers/paths the issue names in
backticks? If the issue names concrete symbols and the diff hits none of
them, the change is not substantial regardless of file type or line count.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Issue bodies often wrap a whole call signature in one backtick span, e.g.
# `delegate_to_agent(role, task, context) -> result`. Extract identifier-like
# tokens *within* each backtick span rather than requiring the whole span to
# be a bare identifier, or a signature like that would never match at all.
BACKTICK_SPAN_RE = re.compile(r"`([^`]+)`")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./]{3,}")


def _looks_like_identifier(token: str) -> bool:
    """Filter out plain English words (e.g. 'role', 'task') that a backticked
    signature incidentally contains, keeping snake_case/dotted/CamelCase
    tokens and paths that are actually specific to the codebase."""
    if any(ch in token for ch in ("_", ".", "/")):
        return True
    if any(ch.isupper() for ch in token):
        return True
    return len(token) >= 8


def extract_mentioned_symbols(issue_text: str) -> set[str]:
    """Pull concrete identifier/path tokens out of an issue body's backticked spans."""
    found = set()
    for span_match in BACKTICK_SPAN_RE.finditer(issue_text):
        for tok_match in TOKEN_RE.finditer(span_match.group(1)):
            found.add(tok_match.group(0))
    return {s for s in found if _looks_like_identifier(s)}


def added_lines(diff_text: str) -> str:
    """Concatenate only the added-line content of a unified diff."""
    lines = []
    for line in diff_text.splitlines():
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            lines.append(line[1:])
    return "\n".join(lines)


def compute_scope_overlap(issue_text: str, diff_text: str) -> dict:
    """Return overlap stats between issue-mentioned symbols and the diff.

    `substantial` is None (no opinion) when the issue names no concrete
    symbols to check against -- callers should fall back to their other
    heuristics (file type, line count) in that case.
    """
    mentioned = extract_mentioned_symbols(issue_text)
    if not mentioned:
        return {"mentioned": [], "matched": [], "substantial": None}

    haystack = added_lines(diff_text)
    matched = {s for s in mentioned if s in haystack}

    return {
        "mentioned": sorted(mentioned),
        "matched": sorted(matched),
        "substantial": bool(matched),
    }


def main() -> int:
    """Exit codes are load-bearing for the calling workflow (issue #521):

    - 0: success, and either `substantial=true` or no opinion (issue named
      no concrete symbols to check).
    - 2: success, and confirmed `substantial=false` -- the diff does not
      touch anything the issue names.
    - 1: this script itself failed (bad args, unreadable issue file, `git
      diff` failure, or any other exception) -- the caller must NOT treat
      this the same as exit 2, or a crash silently looks like a passed
      check. Python's default unhandled-exception exit code is already 1,
      so uncaught exceptions naturally land here without special handling.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-file", required=True, type=Path, help="Path to issue body markdown")
    parser.add_argument("--range", required=True, help="git diff revision range, e.g. origin/main...HEAD")
    args = parser.parse_args()

    issue_text = args.issue_file.read_text(encoding="utf-8")
    try:
        diff_text = subprocess.run(
            ["git", "diff", args.range],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
            timeout=60,
        ).stdout
    except subprocess.TimeoutExpired:
        print("::error::git diff timed out after 60s while computing scope overlap", file=sys.stderr)
        return 1

    result = compute_scope_overlap(issue_text, diff_text)

    if result["substantial"] is None:
        print("no-opinion: issue names no concrete backticked symbols to check")
        return 0

    if result["substantial"]:
        print(f"substantial=true matched={result['matched']}")
        return 0

    print(f"substantial=false mentioned={result['mentioned']} matched=[]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
