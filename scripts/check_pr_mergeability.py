#!/usr/bin/env python3
"""Lightweight CLI wrapper around src.core.ci.mergeability_gate.

Deliberately callable without the full `sparkleforge` bootstrap: `main.py`
runs BootstrapGraph's full init (database, MCP hub, telemetry, automation,
memory) unconditionally for every invocation, including `ci` subcommands.
That's fine for a single LLM judgment call, but gemini-assistant.yml's
auto-merge-ready-fix-prs job calls this mechanical (non-LLM) check once per
matching PR in a loop of up to 100 -- paying a full bootstrap per PR there
risks the job exceeding its 15-minute schedule interval and getting
canceled (cancel-in-progress: true), starving later PRs in the list on
every subsequent run. The actual gate logic lives in
src/core/ci/mergeability_gate.py (plain dataclasses/stdlib, no heavy
imports), reused here and by the CLI-backed merge-decision job alike.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ci.mergeability_gate import check_mechanical_mergeability  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-json-file", required=True, type=Path, help="gh pr view/list JSON output for one PR")
    parser.add_argument("--out", default="mergeability_result.json", type=Path)
    args = parser.parse_args()

    pr = json.loads(args.pr_json_file.read_text(encoding="utf-8"))
    verdict = check_mechanical_mergeability(pr)
    args.out.write_text(json.dumps({"ready": verdict.ready, "reason": verdict.reason}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
