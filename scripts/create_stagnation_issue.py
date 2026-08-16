#!/usr/bin/env python3
"""Create a GitHub issue when scenario-eval detects prolonged stagnation.

Used by .github/workflows/scenario-eval.yml as the CI hard-gate companion to
the stagnation detection in tests/benchmark/run_scenarios.py. When the most
recent N=5 history entries show no meaningful improvement (Δ ≥ 0.03 in at
least 2 of the last 5 runs), this opens an issue naming the lowest-scoring
breakdown item and labels it so opencode-auto-fix.yml does not auto-absorb it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


STAGNATION_LABEL = "scenario-stagnation"
EXEMPT_LABELS = ["no-auto-fix", "nightwelding-queue"]


def _load_history(history_path: Path, n: int = 5) -> list[dict]:
    if not history_path.exists():
        return []
    entries = []
    with history_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries[-n:]


def _count_inconclusive(report: dict) -> tuple[int, int]:
    """Return (inconclusive_count, total_checks) across all scenarios.

    Per Anvil Μ-1, judge-based checks that chronically fall to inconclusive
    are a dead judgment axis and must be surfaced rather than silently
    skipped by the stagnation gate.
    """
    inconclusive = 0
    total = 0
    for scenario in report.get("scenarios", {}).values():
        for check in scenario.get("breakdown", {}).values():
            total += 1
            if check.get("inconclusive"):
                inconclusive += 1
    return inconclusive, total


def _lowest_breakdown(report: dict) -> tuple[str, str, float] | None:
    worst = None
    for scenario_id, scenario in report.get("scenarios", {}).items():
        for check_name, check in scenario.get("breakdown", {}).items():
            if check_name == "loop_engineering_metrics":
                continue
            if check.get("inconclusive"):
                continue
            score = check.get("score", 0.0)
            if worst is None or score < worst[2]:
                worst = (scenario_id, check_name, score)
    return worst


def _gh(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def _ensure_label(repo: str) -> None:
    _gh([
        "gh", "label", "create", STAGNATION_LABEL,
        "--repo", repo,
        "--color", "B60205",
        "--description", "Scenario-eval stagnation gate tripped; requires human follow-up.",
        "--force",
    ], check=False)
    for label in EXEMPT_LABELS:
        _gh([
            "gh", "label", "create", label,
            "--repo", repo,
            "--force",
        ], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Path to scenario_report_ci.json")
    parser.add_argument("--history", required=True, help="Path to scenario_history.jsonl")
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""))
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if report.get("stagnation_detected") is not True:
        return 0

    repo = args.repo
    if not repo:
        print("GITHUB_REPOSITORY not set; cannot create stagnation issue.", file=sys.stderr)
        return 0

    worst = _lowest_breakdown(report)
    inconclusive, total_checks = _count_inconclusive(report)
    worst_str = f"{worst[0]}.{worst[1]} (score={worst[2]:.3f})" if worst else "unknown"
    history_entries = _load_history(Path(args.history))
    trend_lines = "\n".join(
        f"- {e.get('generated_at', '?')}: adjusted={e.get('overall_score_adjusted')}"
        for e in history_entries
    ) or "- (no history)"

    inconclusive_note = ""
    if total_checks and inconclusive / total_checks > 0.5:
        inconclusive_note = (
            f"\n\n### Dead judgment axis warning\n\n"
            f"{inconclusive}/{total_checks} checks are `inconclusive`. "
            "Per Anvil Μ-1, judge-based checks must not chronically fall to "
            "inconclusive — this masks stagnation behind coin-flip judge axes."
        )

    title = "scenario-eval: stagnation detected (Anvil Μ-2)"
    body = (
        "## Scenario-eval stagnation gate\n\n"
        "The last 5 recorded scenario-eval runs show no meaningful improvement "
        "(`overall_score_adjusted` Δ ≥ 0.03 in at least 2 of 5 runs).\n\n"
        f"### Lowest-scoring breakdown item\n\n`{worst_str}`\n\n"
        "### Recent history\n\n"
        f"{trend_lines}\n\n"
        f"{inconclusive_note}\n\n"
        "This issue is intentionally excluded from the opencode-auto-fix.yml "
        "auto-scan/auto-merge pipeline (labeled `no-auto-fix`) per the "
        "CLAUDE.md principle that merges require an explicit in-session human "
        "decision. A human should investigate the named breakdown item and "
        "open a targeted implementation effort."
    )

    _ensure_label(repo)
    proc = _gh([
        "gh", "issue", "create",
        "--repo", repo,
        "--title", title,
        "--body", body,
        "--label", STAGNATION_LABEL,
        "--label", "no-auto-fix",
        "--label", "nightwelding-queue",
    ], check=False)
    if proc.returncode == 0:
        print(f"Created stagnation issue: {proc.stdout.strip()}")
    else:
        print(f"Failed to create stagnation issue: {proc.stderr}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
