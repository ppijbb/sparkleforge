#!/usr/bin/env python3
"""Guard against the InvocationGateway boundary (issue #568) being silently bypassed.

Agent delegation and MCP tool execution each used to skip
IntentGuardrail/CapabilityManager checks simply because nobody remembered to
add them -- the same class of gap #516/#519/#312 each hit independently.
InvocationGateway.authorize() is now the single mandatory choke point for
both, but nothing stops a future edit from quietly removing the call (e.g.
during an unrelated refactor) without anyone noticing, since the code would
still run fine -- it would just stop being authorized/journaled. This script
statically checks that both known enforcement points still call the gateway,
and fails the build if either one doesn't.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (file relative to repo root, function name, required substring)
ENFORCEMENT_POINTS = [
    (
        "src/core/orchestrator/delegation.py",
        "delegate_to_agent",
        "get_invocation_gateway()",
    ),
    (
        "src/core/mcp_integration/hub_mixins/execution.py",
        "execute_tool",
        "get_invocation_gateway()",
    ),
]


def _function_body(source: str, function_name: str) -> str | None:
    """Extract a (possibly nested/method) def's exact source, by AST line range.

    Handles multi-line signatures correctly (unlike naive indentation
    scanning, which mistakes a signature's closing paren at column 0 for a
    dedent out of the function body).
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            return ast.get_source_segment(source, node)
    return None


def main() -> int:
    failures = []

    for rel_path, function_name, required_substring in ENFORCEMENT_POINTS:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            failures.append(f"{rel_path}: file not found")
            continue

        source = path.read_text(encoding="utf-8")
        body = _function_body(source, function_name)
        if body is None:
            failures.append(f"{rel_path}: could not find def {function_name}(...)")
            continue

        if required_substring not in body:
            failures.append(
                f"{rel_path}::{function_name}: no longer calls '{required_substring}' -- "
                "the InvocationGateway boundary (issue #568) appears to have been removed"
            )

    if failures:
        print(f"❌ InvocationGateway wiring check failed ({len(failures)} issue(s)):\n")
        for f in failures:
            print(f"  {f}")
        return 1

    print(f"✅ InvocationGateway is still wired into all {len(ENFORCEMENT_POINTS)} enforcement points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
