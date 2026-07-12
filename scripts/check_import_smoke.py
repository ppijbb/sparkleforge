#!/usr/bin/env python3
"""Import smoke test for core entrypoints. CI merge-gate regression guard.

Issue #494 was merged with `b/`-prefixed stub file paths committed literally
(e.g. `b/src/core/mcp/__init__.py`) that broke `import main` on main -- the
PR merge gate had no step that would have caught it. This script imports the
modules that every research/harness run depends on and fails loudly with the
full traceback if any of them can't be imported.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CORE_ENTRYPOINTS = [
    "main",
    "src.core.mcp_integration",
    "src.core.agent_harness",
    "src.core.orchestrator.graph",
]


def check_imports(module_names: list[str]) -> list[tuple[str, str]]:
    """Import each module name, returning (module, traceback) for failures."""
    failures: list[tuple[str, str]] = []
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception:
            failures.append((name, traceback.format_exc()))
    return failures


def main() -> int:
    failures = check_imports(CORE_ENTRYPOINTS)
    if failures:
        print(f"❌ {len(failures)} core module(s) failed to import:\n")
        for name, tb in failures:
            print(f"--- {name} ---")
            print(tb)
        return 1

    print(f"✅ all {len(CORE_ENTRYPOINTS)} core entrypoints imported cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
