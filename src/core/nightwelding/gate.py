"""Reproduce-first gate: write a failing test before any fix is attempted.

Nightwelding's core safety property: an implementation attempt only begins once
a test exists that demonstrably fails against the current repository, and a
PR only opens once that same test demonstrably passes. This module owns the
"must fail first" half; src/core/nightwelding/implement.py owns the "must pass
after" half.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.core import patch_ops
from src.core.cli_agents.open_code_agent import OpenCodeAgent

REPRO_TEST_PATTERN = re.compile(r"^tests/test_[^/]+\.py$")

# Runtime scratch files this module (or patch_ops) writes to the working tree,

# Issue #579: cheap pre-qualification before spending a full reproduce cycle.
# Issue titles starting with these prefixes are design/planning work, not
# reproducible bugs, so nightwelding should skip them and route to a human.
NON_REPRODUCIBLE_TITLE_PREFIXES = ("planning:", "design:", "rfc:", "spike:")

# excluded when scanning `git status` for what the LLM's diff actually touched.
_IGNORED_RUNTIME_FILES = {"opencode.patch"}

_MAX_WRITE_ATTEMPTS = 2
_COLLECT_TIMEOUT_SECONDS = 120
_PYTEST_TIMEOUT_SECONDS = 300


@dataclass
class ReproResult:
    success: bool
    test_files: List[str] = field(default_factory=list)
    red_output: str = ""
    reason: str = ""


def _repro_prompt(issue_context: str, snapshot: str, status: str, extra_context: str = "") -> str:
    return f"""
You are an autonomous coding agent editing the SparkleForge repository.

Your ONLY task is to prove the bug/missing-feature described below is real, by
writing a test that fails today.

Rules:
- Add or modify ONLY test file(s) matching the path pattern `tests/test_*.py`
  (directly under tests/, not a subdirectory).
- Do NOT create, modify, or touch `tests/conftest.py`, anything under
  `tests/benchmark/`, anything under `tests/baselines/`, or any file outside
  `tests/`.
- The test must currently FAIL against this repository as-is, and should PASS
  once the issue below is properly fixed.
- Do NOT implement the fix itself — only prove the problem exists.
- Output ONLY a unified diff. No prose.
- CRITICAL: Always emit diffs in `git diff` format with `a/`/`b/` prefixes:
    diff --git a/path/to/file b/path/to/file
    --- a/path/to/file
    +++ b/path/to/file

Repository snapshot:
{snapshot}

Current git status:
{status}

Issue context:
{issue_context}

Additional context from a previous failed attempt:
{extra_context or "None"}
""".strip()


def _touched_test_files(repo_root: Path) -> tuple[bool, List[str]]:
    """Return (all_paths_are_isolated_repro_tests, touched_paths)."""
    proc = subprocess.run(
        ["git", "status", "--porcelain", "-z", "--untracked-files=all"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    touched: List[str] = []
    all_ok = True
    for entry in proc.stdout.split("\0"):
        if not entry:
            continue
        path = entry[3:] if len(entry) > 3 else ""
        if not path or path in _IGNORED_RUNTIME_FILES:
            continue
        if path.endswith((".orig", ".rej")):
            continue
        touched.append(path)
        if not REPRO_TEST_PATTERN.match(path):
            all_ok = False
    return all_ok, touched


def is_reproducible_bug_eligible(issue_context: str) -> tuple[bool, str]:
    """Cheap pre-qualification check (issue #579).

    Returns (eligible, reason). When not eligible, the caller should skip the
    expensive reproduction-test step and fail fast with `reason`.

    This is intentionally a conservative, low-cost classifier: it catches the
    obvious cases (planning/design titles, explicit non-bug labels) that would
    otherwise waste a full LLM + pytest cycle before being rejected.
    """
    # The issue_context is the issue markdown; the first non-empty line is the
    # "# <title>" header written by github_adapter.fetch_issue_context.
    title = ""
    for line in issue_context.splitlines():
        stripped = line.strip()
        if stripped:
            # Strip a leading "# " heading marker if present.
            if stripped.startswith("# "):
                stripped = stripped[2:].strip()
            title = stripped
            break

    if title:
        lowered = title.lower()
        for prefix in NON_REPRODUCIBLE_TITLE_PREFIXES:
            if lowered.startswith(prefix):
                return (
                    False,
                    f"Issue title {title!r} looks like design/planning work, not a reproducible bug. Skipping the reproduction step and routing to a human.",
                )

    # Heuristic: issues whose body explicitly frames them as design/planning
    # proposals rather than failing-test bugs. Keep this narrow to avoid false
    # positives on ordinary bug reports that mention "design" in passing.
    lowered_body = issue_context.lower()
    if "this is a design" in lowered_body or "this is a planning" in lowered_body:
        return (
            False,
            "Issue is framed as design/planning work, not a failing-test bug. Skipping the reproduction step and routing to a human.",
        )

    return True, ""


async def write_reproduction_test(
    issue_context: str,
    repo_root: Path | None = None,
) -> ReproResult:
    repo_root = repo_root or Path.cwd()
    snapshot = patch_ops.repo_snapshot()
    status = patch_ops.run(["git", "status", "--short"]).stdout

    agent = OpenCodeAgent()
    extra_context = ""
    for attempt in range(_MAX_WRITE_ATTEMPTS):
        prompt = _repro_prompt(issue_context, snapshot, status, extra_context)
        result = await agent.execute_query(
            prompt,
            system_message=(
                "You are a careful coding agent working against a real repository. "
                "Output ONLY a git-apply compatible unified diff that adds or modifies "
                "tests/test_*.py files. No prose, no other files."
            ),
        )
        if not result.get("success"):
            extra_context = result.get("response") or result.get("error") or "OpenCode call failed"
            continue

        diff = patch_ops.extract_diff(result.get("response", ""))
        if not diff.strip():
            extra_context = "Previous response did not contain an applicable diff. Regenerate a minimal diff that only adds/modifies tests/test_*.py files."
            continue

        patch_path = repo_root / "opencode.patch"
        patch_path.write_text(diff, encoding="utf-8")
        applied, err = patch_ops._apply_patch(patch_path)
        if not applied:
            extra_context = f"Previous diff failed to apply:\n{err}\n\nRegenerate a minimal diff that only adds/modifies tests/test_*.py files."
            continue

        break
    else:
        return ReproResult(success=False, reason="OpenCode could not produce an applicable reproduction-test diff after 2 attempts.")

    all_ok, touched = _touched_test_files(repo_root)
    if not touched:
        return ReproResult(success=False, reason="Reproduction diff touched no files.")
    if not all_ok:
        return ReproResult(
            success=False,
            reason=f"Reproduction diff touched files outside the tests/test_*.py pattern: {touched}",
        )

    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "tests/", "-q"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=_COLLECT_TIMEOUT_SECONDS,
        check=False,
    )
    if collect.returncode != 0:
        return ReproResult(
            success=False,
            test_files=touched,
            reason="Reproduction test broke test collection for the whole suite.",
            red_output=(collect.stdout + collect.stderr)[-4000:],
        )

    red = subprocess.run(
        [sys.executable, "-m", "pytest", *touched, "-q"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=_PYTEST_TIMEOUT_SECONDS,
        check=False,
    )
    if red.returncode == 0:
        return ReproResult(
            success=False,
            test_files=touched,
            reason="Reproduction test passed against the current repository — the issue could not be reproduced.",
            red_output=(red.stdout + red.stderr)[-4000:],
        )

    return ReproResult(
        success=True,
        test_files=touched,
        red_output=(red.stdout + red.stderr)[-4000:],
    )
