"""Whether an auto-fix diff substantially implements the issue it targets.

Consolidates two signals opencode-auto-fix.yml previously computed
independently -- a bash heuristic, and a standalone script this replaces
(scripts/check_issue_scope_overlap.py):

1. Checklist/file-type/line-count heuristic (issue #511): unresolved
   acceptance-criteria checkboxes plus a diff that only touches trivial
   files (or is tiny) means "probably not done yet".
2. Issue-symbol/diff-overlap check (issue #521): catches diffs that touch
   real .py files with a handful of lines but implement none of the
   concrete symbols the issue names in backticks -- exactly how #509's
   first auto-fix attempt slipped past signal 1 alone.

Signal 2 can only push a "substantial" verdict from signal 1 down to
not-substantial (a confirmed scope mismatch overrides an optimistic
checklist reading); it never overrides a not-substantial verdict back up --
this mirrors the original bash's one-directional combination exactly.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from dataclasses import dataclass

BACKTICK_SPAN_RE = re.compile(r"`([^`]+)`")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./]{3,}")
_TRIVIAL_FILE_PATTERNS = (".github/workflows/*", "*.md", "docs/*")


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


def count_unchecked(issue_text: str) -> int:
    return len(re.findall(r"^- \[ \]", issue_text, flags=re.MULTILINE))


def _is_trivial_file(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in _TRIVIAL_FILE_PATTERNS)


def compute_checklist_heuristic(unchecked: int, changed_files: list[str], changed_lines: int) -> bool:
    """Signal 1: unresolved checklist items + a diff that's only trivial
    files (or <=3 changed lines) means not substantial."""
    non_trivial_file = any(not _is_trivial_file(f) for f in changed_files if f)
    if unchecked > 0 and (not non_trivial_file or changed_lines <= 3):
        return False
    return True


@dataclass
class SubstantialityVerdict:
    substantial: bool
    reason: str
    unchecked: int


def assess_fix_substantiality(
    *, issue_text: str, diff_text: str, changed_files: list[str], changed_lines: int
) -> SubstantialityVerdict:
    unchecked = count_unchecked(issue_text)
    substantial = compute_checklist_heuristic(unchecked, changed_files, changed_lines)

    overlap = compute_scope_overlap(issue_text, diff_text)
    reason = ""
    if overlap["substantial"] is False:
        substantial = False
        reason = " and the diff never touches any of the symbols/paths the issue names"

    return SubstantialityVerdict(substantial=substantial, reason=reason, unchecked=unchecked)


def _run_git_diff(range_spec: str, *extra_args: str) -> str:
    return subprocess.run(
        ["git", "diff", *extra_args, range_spec],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout


def gather_diff_stats(range_spec: str) -> tuple[str, list[str], int]:
    """Returns (diff_text, changed_files, changed_lines) for a git diff range."""
    diff_text = _run_git_diff(range_spec)
    changed_files = [f for f in _run_git_diff(range_spec, "--name-only").splitlines() if f]
    shortstat = _run_git_diff(range_spec, "--shortstat")
    changed_lines = sum(int(n) for n in re.findall(r"(\d+) (?:insertion|deletion)", shortstat))
    return diff_text, changed_files, changed_lines
