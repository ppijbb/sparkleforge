"""Create a GitHub issue when scenario-eval detects prolonged stagnation.

Used by .github/workflows/scenario-eval.yml as the CI hard-gate companion to
the stagnation detection in tests/benchmark/run_scenarios.py. When the most
recent N=5 history entries show no meaningful improvement (Delta >= 0.03 in
at least 2 of the last 5 runs), this opens an issue naming the lowest-scoring
breakdown item and labels it so opencode-auto-fix.yml does not auto-absorb it.

Moved verbatim from scripts/create_stagnation_issue.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

STAGNATION_LABEL = "scenario-stagnation"
EXEMPT_LABELS = ["no-auto-fix", "nightwelding-queue"]


def maybe_auto_rollback(storage_dir: Path = Path("storage/skills")):
    """Anvil N-2: on stagnation, undo the most recently re-distilled skill.

    Best-effort heuristic -- rolls back whichever saved skill has the most
    recent created_at among those with version > 1 (i.e. was re-saved at
    least once). Returns the rolled-back Skill, or None if there was no
    re-distilled skill to blame.
    """
    from src.core.anvil.skill_repository import SkillRepository

    repo = SkillRepository(storage_dir=str(storage_dir))
    name = repo.most_recently_modified_skill()
    if name is None:
        return None
    current = repo.get_skill(name)
    return repo.rollback_skill(name, current.version - 1)


def load_history(history_path: Path, n: int = 5) -> list[dict]:
    if not history_path.exists():
        return []
    entries = []
    with history_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries[-n:]


def count_inconclusive(report: dict) -> tuple[int, int]:
    """Return (inconclusive_count, total_checks) across all scenarios.

    Per Anvil Mu-1, judge-based checks that chronically fall to inconclusive
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


def lowest_breakdown(report: dict) -> tuple[str, str, float] | None:
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


@dataclass
class StagnationIssue:
    title: str
    body: str
    labels: list[str]


def build_stagnation_issue(
    report: dict, history: list[dict], rolled_back=None
) -> StagnationIssue | None:
    """Returns None when the report doesn't confirm stagnation_detected=True.

    `rolled_back`, if given, is the Skill (see skill_repository.Skill) that
    was auto-rolled-back (Anvil N-2) because it was the most recently
    re-distilled skill at the time stagnation was detected.
    """
    if report.get("stagnation_detected") is not True:
        return None

    worst = lowest_breakdown(report)
    inconclusive, total_checks = count_inconclusive(report)
    worst_str = f"{worst[0]}.{worst[1]} (score={worst[2]:.3f})" if worst else "unknown"
    trend_lines = "\n".join(
        f"- {e.get('generated_at', '?')}: adjusted={e.get('overall_score_adjusted')}"
        for e in history
    ) or "- (no history)"

    inconclusive_note = ""
    if total_checks and inconclusive / total_checks > 0.5:
        inconclusive_note = (
            f"\n\n### Dead judgment axis warning\n\n"
            f"{inconclusive}/{total_checks} checks are `inconclusive`. "
            "Per Anvil Mu-1, judge-based checks must not chronically fall to "
            "inconclusive — this masks stagnation behind coin-flip judge axes."
        )

    rollback_note = ""
    if rolled_back is not None:
        rollback_note = (
            f"\n\n### Auto-rollback (Anvil Ν-2)\n\n"
            f"Skill `{rolled_back.name}` was the most recently re-distilled skill, "
            f"so it was auto-rolled-back to v{rolled_back.metadata.get('rollback_to_version')} "
            f"(now recorded as v{rolled_back.version}). If this wasn't the cause of the "
            "stagnation, re-distill it again -- the rollback itself is just another "
            "version and does not delete history."
        )

    title = "scenario-eval: stagnation detected (Anvil Μ-2)"
    body = (
        "## Scenario-eval stagnation gate\n\n"
        "The last 5 recorded scenario-eval runs show no meaningful improvement "
        "(`overall_score_adjusted` Δ ≥ 0.03 in at least 2 of 5 runs).\n\n"
        f"### Lowest-scoring breakdown item\n\n`{worst_str}`\n\n"
        "### Recent history\n\n"
        f"{trend_lines}\n\n"
        f"{inconclusive_note}"
        f"{rollback_note}\n\n"
        "This issue is intentionally excluded from the opencode-auto-fix.yml "
        "auto-scan/auto-merge pipeline (labeled `no-auto-fix`) per the "
        "CLAUDE.md principle that merges require an explicit in-session human "
        "decision. A human should investigate the named breakdown item and "
        "open a targeted implementation effort."
    )

    return StagnationIssue(title=title, body=body, labels=[STAGNATION_LABEL, *EXEMPT_LABELS])


def _gh(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def _ensure_labels(repo: str) -> None:
    _gh([
        "gh", "label", "create", STAGNATION_LABEL,
        "--repo", repo,
        "--color", "B60205",
        "--description", "Scenario-eval stagnation gate tripped; requires human follow-up.",
        "--force",
    ], check=False)
    for label in EXEMPT_LABELS:
        _gh(["gh", "label", "create", label, "--repo", repo, "--force"], check=False)


def create_github_issue(repo: str, issue: StagnationIssue) -> str | None:
    """Ensures labels exist and opens the issue. Returns the issue URL, or
    None if `gh issue create` failed (details already printed to stderr)."""
    _ensure_labels(repo)
    cmd = ["gh", "issue", "create", "--repo", repo, "--title", issue.title, "--body", issue.body]
    for label in issue.labels:
        cmd += ["--label", label]
    proc = _gh(cmd, check=False)
    if proc.returncode == 0:
        url = proc.stdout.strip()
        print(f"Created stagnation issue: {url}")
        return url
    print(f"Failed to create stagnation issue: {proc.stderr}", file=sys.stderr)
    return None
