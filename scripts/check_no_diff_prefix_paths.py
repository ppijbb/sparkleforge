#!/usr/bin/env python3
"""Guard against committed files whose path is literally `a/...`/`b/...`.

Issue #494 was merged with git-diff `a/`/`b/` prefixes left in as literal
repository paths (e.g. `b/src/core/mcp/__init__.py`), which broke `import
main` on main because the real module lived at `src/core/mcp/__init__.py`
and the stray `b/`-prefixed copy shadowed nothing but confused the package
layout. This is the failure mode a correct patch-apply should never produce
(see `src/core/patch_ops.py`'s `_invalid_patch_path_reason`), but nothing
in CI verified the tree stayed clean of it. This script scans tracked files
for that specific pattern and fails the build if it recurs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def tracked_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        print("::error::git ls-files timed out after 60s while scanning tracked files", file=sys.stderr)
        raise
    return [line for line in result.stdout.splitlines() if line.strip()]


def find_diff_prefix_paths(paths: list[str]) -> list[str]:
    """Return paths whose first component is exactly 'a' or 'b'."""
    offenders = []
    for path in paths:
        first_component = path.split("/", 1)[0]
        if first_component in ("a", "b"):
            offenders.append(path)
    return offenders


def main() -> int:
    all_paths = tracked_files()
    offenders = find_diff_prefix_paths(all_paths)
    if offenders:
        print(f"❌ {len(offenders)} tracked path(s) look like stray git-diff a/b prefixes:\n")
        for path in offenders:
            print(f"  {path}")
        print(
            "\nThese are almost certainly diff header prefixes ('a/'/'b/') that leaked "
            "into a committed file path instead of being stripped by the patch tool. "
            "See issue #494/#511."
        )
        return 1

    print(f"✅ no diff-prefix (a/, b/) paths found among {len(all_paths)} tracked files checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
