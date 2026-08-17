#!/usr/bin/env python3
"""Anvil acceptance-scenario eval harness (Ω-1, issue #330, #1107).

Drives the 5 "Anvil v1.0 acceptance scenarios" (#267) end-to-end through the
real natural-language entry point (`python main.py work "<goal>"`), then
scores the resulting filesystem/system state with deterministic checks plus
a capped-weight LLM-judge fallback for subjective quality.

Usage:
    python tests/benchmark/run_scenarios.py                      # run all 5, print + save report
    python tests/benchmark/run_scenarios.py --scenario system_cleanup
    python tests/benchmark/run_scenarios.py --list
    python tests/benchmark/run_scenarios.py --update-baseline    # bump tests/benchmark/baselines/scenario_baseline.json
    python tests/benchmark/run_scenarios.py --compare-to tests/benchmark/baselines/scenario_baseline.json  # regression gate vs a fixed baseline
    python tests/benchmark/run_scenarios.py --compare-to-history tests/benchmark/baselines/scenario_history.jsonl  # regression gate vs the most recent recorded run
    python tests/benchmark/run_scenarios.py --append-history tests/benchmark/baselines/scenario_history.jsonl  # record this run into the trend history
"""

from __future__ import annotations
import enum

import argparse
import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
# Allow running this file directly (`python tests/benchmark/run_scenarios.py`)
# as well as via `python -m tests.benchmark.run_scenarios` — both need the repo
# root on sys.path for the `tests.benchmark.*` and `src.*` imports to resolve.
sys.path.insert(0, str(REPO_ROOT))

class FailureTaxonomy(enum.Enum):
    # OS & Tool Layer
    ToolBindingGap = "ToolBindingGap"
    PermissionDenied = "PermissionDenied"
    SchedulerTimeout = "SchedulerTimeout"
    # Control-Flow Layer
    LoopStagnation = "LoopStagnation"
    AgentPassivity = "AgentPassivity"
    PrematureExit = "PrematureExit"
    # Memory & Context Layer
    ContextBloat = "ContextBloat"
    FactDistortion = "FactDistortion"
    DecoyLeak = "DecoyLeak"
    # Model & Verification Layer
    SpecViolation = "SpecViolation"
    ToolHallucination = "ToolHallucination"
    UnresolvedBug = "UnresolvedBug"

import yaml

# Judge sub-scores call into src.core.llm_manager, which reads the process-global
# config set up by load_config_from_env() — same requirement main.py has at import
# time. Without this, every judge_* check silently scores 0 even with a valid key.
from src.core.researcher_config import load_config_from_env

load_config_from_env()

from tests.benchmark.scenario_grading import weighted_total

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
MAIN_PY = REPO_ROOT / "main.py"

# Judge sub-scores are inherently noisier (LLM-graded) than deterministic ones;
# only they get regression tolerance in --compare-to.
JUDGE_REGRESSION_TOLERANCE = 0.15
DETERMINISTIC_REGRESSION_TOLERANCE = 0.0


def _truncate_excerpt(text: str, limit: int = 1500) -> str:
    """Truncate ``text`` to at most ``limit`` characters on a clean boundary.

    A raw character slice cuts mid-token (e.g. ``...model_regist``), corrupting
    diagnostic excerpts. This trims at the last whitespace boundary before the
    limit so no partial tokens appear in ``stdout_excerpt``/``stderr_excerpt``.
    """
    if not text or len(text) <= limit:
        return text or ""
    candidate = text[:limit]
    boundary = max(candidate.rfind(" "), candidate.rfind("\n"), candidate.rfind("\t"))
    if boundary > 0:
        return candidate[:boundary]
    return candidate


def _llm_provider_keys_present() -> tuple[bool, list[str]]:
    """Return (any_present, present_provider_names) for known LLM API keys."""
    providers = [
        ("google", "GOOGLE_API_KEY"),
        ("google", "GEMINI_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("nvidia", "NVIDIA_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ]
    present = [name for name, env_var in providers if os.environ.get(env_var, "").strip()]
    return (bool(present), present)


def preflight_llm_availability() -> tuple[bool, str]:
    """Fail-fast check: at least one LLM provider API key must be configured."""
    any_present, present = _llm_provider_keys_present()
    if not any_present:
        return False, (
            "No LLM provider API keys configured (checked GOOGLE_API_KEY, "
            "GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, NVIDIA_API_KEY, "
            "OPENROUTER_API_KEY). Aborting scenario run to avoid recording "
            "garbage baselines from trivial rule-based fallbacks."
        )
    return True, f"LLM providers available: {', '.join(sorted(set(present)))}"


def require_openrouter_api_key() -> bool:
    """Return whether judge-dependent scenario scoring can run."""
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return True
    print(
        "[scenario-eval] OPENROUTER_API_KEY is required for judge-dependent scenario scoring.",
        file=sys.stderr,
    )
    return False


def load_scenarios(only_id: str | None = None) -> List[Dict[str, Any]]:
    specs = []
    for path in sorted(SCENARIOS_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        spec["_path"] = str(path)
        if only_id and spec.get("id") != only_id:
            continue
        specs.append(spec)
    return specs


def classify_error(stderr: str) -> str:
    """Classify error into Supabase 12-Taxonomy."""
    if "ToolBindingGap" in stderr: return FailureTaxonomy.ToolBindingGap.value
    if "PermissionDenied" in stderr: return FailureTaxonomy.PermissionDenied.value
    if "timeout" in stderr.lower(): return FailureTaxonomy.SchedulerTimeout.value
    if "LoopStagnation" in stderr: return FailureTaxonomy.LoopStagnation.value
    if "AgentPassivity" in stderr: return FailureTaxonomy.AgentPassivity.value
    if "PrematureExit" in stderr: return FailureTaxonomy.PrematureExit.value
    if "ContextBloat" in stderr: return FailureTaxonomy.ContextBloat.value
    if "FactDistortion" in stderr: return FailureTaxonomy.FactDistortion.value
    if "DecoyLeak" in stderr: return FailureTaxonomy.DecoyLeak.value
    if "SpecViolation" in stderr: return FailureTaxonomy.SpecViolation.value
    if "ToolHallucination" in stderr: return FailureTaxonomy.ToolHallucination.value
    return FailureTaxonomy.UnresolvedBug.value


def run_ablation_matrix(specs: List[Dict[str, Any]]):
    """Run 6-Feature Ablation Suite."""
    # Implementation would iterate through feature flags and run_scenario()
    pass


def run_agent(user_query: str, workspace: Path, timeout_s: int) -> Dict[str, Any]:
    """Invoke the real NL entry point (`python main.py work "<goal>"`) as a subprocess.

    cwd is set to the isolated fixture workspace: src/core/mcp_integration.py's
    `_is_safe_path()` allow-list resolves via `Path.cwd()` at call time, so this
    confines the agent's generic file tools to the fixture without any code change.

    HOME/USERPROFILE are also pinned to the workspace so anything the agent
    writes via `Path.home()` (e.g. src/core/scheduler.py's default
    `~/.sparkleforge/schedules` store) lands inside the fixture instead of the
    real developer/CI-runner home directory. scenario_fixtures/scheduled_summary.py
    reads schedules back from this same workspace-relative location.
    """
    sparkleforge_bin = shutil.which("sparkleforge")
    cmd = [sparkleforge_bin, "work", user_query] if sparkleforge_bin else [sys.executable, str(MAIN_PY), "work", user_query]
    env = os.environ.copy()
    env["HOME"] = str(workspace)
    env["USERPROFILE"] = str(workspace)
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
            "duration_s": time.time() - start,
        }
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        stderr = e.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "returncode": -1,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "timed_out": True,
            "duration_s": time.time() - start,
        }


async def run_scenario(spec: Dict[str, Any]) -> Dict[str, Any]:
    fixture_module = importlib.import_module(f"tests.benchmark.scenario_fixtures.{spec['fixture']}")

    workspace = Path(tempfile.mkdtemp(prefix=f"sparkleforge_scenario_{spec['id']}_"))
    try:
        ctx = await asyncio.to_thread(fixture_module.build, workspace)
        ctx.setdefault("workspace", str(workspace))
        ctx["judge_rubric"] = spec.get("judge_rubric", "")
        user_query = spec["user_query"].format(**ctx)

        exec_result = await asyncio.to_thread(run_agent, user_query, workspace, spec.get("timeout_s", 300))

        if exec_result["timed_out"]:
            scores = {name: (0.0, "scenario timed out") for name in spec["weights"]}
        elif exec_result["returncode"] != 0 and "No available models" in exec_result["stderr"]:
            # Consistent policy: agent failure due to model unavailability marks all checks inconclusive
            scores = {name: (0.0, "agent execution failed: no model available") for name in spec["weights"]}
        else:
            scores = await fixture_module.grade(workspace, ctx, exec_result["stdout"])

        graded = weighted_total(scores, spec["weights"])

        critical_failure = (
            exec_result["returncode"] != 0
            and "No available models" in exec_result["stderr"]
        ) or (
            "All fallback models failed" in exec_result["stdout"]
        )
        if critical_failure:
            # Total LLM infrastructure failure: every check must be marked
            # inconclusive (not conclusive-with-a-score) and zeroed so the
            # run cannot be mistaken for a genuine low-score conclusive run.
            # Restores the pre-regression behavior where fallback judge
            # evaluations propagated inconclusive=True through this path.
            for check in graded["breakdown"].values():
                check["score"] = 0.0
                check["inconclusive"] = True
                if not check["reason"].startswith("agent execution failed"):
                    check["reason"] = (
                        "infrastructure failure: no model available — "
                        + check["reason"]
                    )
            graded["total"] = 0.0
            graded["adjusted_total"] = 0.0

        # Infrastructure failures (model unavailability) must propagate a
        # non-zero returncode so CI gates can distinguish a total model
        # collapse from a genuinely successful run. Use exit code 2 for
        # infrastructure failure, distinct from 1 (agent task failure).
        recorded_returncode = 2 if critical_failure else exec_result["returncode"]
        # The agent subprocess may exit 0 even when every fallback model failed
        # (the failure surfaces only in stdout). Force a non-zero recorded
        # returncode so the run cannot be mistaken for a genuine success.
        if critical_failure and recorded_returncode == 0:
            recorded_returncode = 2

        return {
            "id": spec["id"],
            "name": spec.get("name", spec["id"]),
            "user_query": user_query,
            "total": graded["total"],
            "adjusted_total": graded["adjusted_total"],
            "breakdown": graded["breakdown"],
            "loop_engineering_metrics": ctx.get("loop_stats", {}),
            "inconclusive": critical_failure,
            "critical_failure": critical_failure,
            "returncode": recorded_returncode,
            "timed_out": exec_result["timed_out"],
            "duration_s": round(exec_result["duration_s"], 2),
            "stdout_excerpt": _truncate_excerpt(exec_result["stdout"]),
            "error_taxonomy": classify_error(exec_result["stderr"]) if exec_result["returncode"] != 0 else None,
            "stderr_excerpt": _truncate_excerpt(exec_result["stderr"]),
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


async def run_all(specs: List[Dict[str, Any]], parallel: bool) -> Dict[str, Any]:
    if parallel:
        results = await asyncio.gather(*(run_scenario(spec) for spec in specs))
    else:
        # Sequential by default: each scenario drives a real multi-iteration LLM
        # agent loop, and running several concurrently risks provider rate-limit
        # contention (observed in practice) rather than saving meaningful time.
        results = [await run_scenario(spec) for spec in specs]

    # Exclude inconclusive scenarios from overall_score so infrastructure
    # unavailability (no model available, judge failures) does not depress the
    # score as if it were genuine agent-performance failure. Inconclusive runs
    # are tracked separately via inconclusive_checks and conclusive_checks_count.
    conclusive = [r for r in results if not r.get("inconclusive")]
    overall = round(sum(r["total"] for r in conclusive) / len(conclusive), 4) if conclusive else 0.0
    adjusted = [r["adjusted_total"] for r in conclusive if r["adjusted_total"] is not None]
    overall_adjusted = round(sum(adjusted) / len(adjusted), 4) if adjusted else None
    inconclusive_checks = sum(
        1 for r in results for check in r["breakdown"].values() 
        if check["inconclusive"] or r.get("inconclusive")
    )
    conclusive_checks_count = sum(
        1 for r in results for check in r["breakdown"].values()
        if not check["inconclusive"] and not r.get("inconclusive")
    )
    warning = None
    if inconclusive_checks > 0:
        warning = (
            f"{inconclusive_checks} check(s) were inconclusive (infra/judge unavailable); "
            "overall_score excludes them and has reduced confidence."
        )
        print(f"[scenario-eval] WARNING: {warning}", file=sys.stderr)
    # Post-run sanity check (issue #1290): if every conclusive scenario scored
    # 0.0, the run is degenerate (almost always a total LLM outage that slipped
    # past the per-scenario critical_failure detection). Emit a warning and mark
    # the run critical so the runner exits non-zero and the history append is
    # skipped, preventing a zero-score run from being recorded as a success.
    if conclusive and all(r["total"] == 0.0 for r in conclusive):
        warning = (
            "degenerate run: every conclusive scenario scored 0.0 — likely a "
            "total LLM outage that was not caught by the per-scenario gate."
        )
        print(f"[scenario-eval] WARNING: {warning}", file=sys.stderr)
        for r in results:
            r["critical_failure"] = True
            r["inconclusive"] = True
            r["returncode"] = r.get("returncode") or 2
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_score": overall,
        "overall_score_adjusted": overall_adjusted,
        "inconclusive_checks": inconclusive_checks,
        "conclusive_checks_count": conclusive_checks_count,
        "warning": warning,
        "scenarios": {r["id"]: r for r in results},
    }


def _compare_scenarios(current_scenarios: Dict[str, Any], baseline_scenarios: Dict[str, Any]) -> int:
    """Shared diff logic for both the static-baseline and history-based compare modes.

    Uses adjusted_total (renormalized over checks that actually ran) rather
    than the raw total, and skips any check marked inconclusive on either side
    -- an infra outage (judge unavailable, provider quota exhausted) must never
    register as a capability regression. Returns 0/1 like compare_to_baseline.
    """
    regressions = []
    for scenario_id, current in current_scenarios.items():
        prior = baseline_scenarios.get(scenario_id)
        if prior is None:
            print(f"[scenario-eval] '{scenario_id}': no prior baseline, first run (total={current['total']})")
            continue

        current_adjusted = current.get("adjusted_total")
        prior_adjusted = prior.get("adjusted_total", prior.get("total", 0.0))
        if current_adjusted is None:
            print(f"[scenario-eval] '{scenario_id}': every check was inconclusive this run, skipping comparison")
            continue

        delta = round(current_adjusted - prior_adjusted, 4)
        sign = "+" if delta >= 0 else ""
        print(
            f"[scenario-eval] '{scenario_id}': baseline={prior_adjusted:.3f} current={current_adjusted:.3f} "
            f"({sign}{delta:.3f})"
        )

        for check_name, current_check in current["breakdown"].items():
            if current_check.get("inconclusive"):
                print(f"    {check_name}: SKIPPED (inconclusive this run: {current_check['reason']})")
                continue
            prior_check = prior.get("breakdown", {}).get(check_name)
            if prior_check is None or prior_check.get("inconclusive"):
                continue
            tolerance = (
                JUDGE_REGRESSION_TOLERANCE
                if check_name.startswith("judge_")
                else DETERMINISTIC_REGRESSION_TOLERANCE
            )
            check_drop = prior_check["score"] - current_check["score"]
            if check_drop > tolerance:
                regressions.append(
                    f"{scenario_id}.{check_name}: {prior_check['score']:.3f} -> {current_check['score']:.3f} "
                    f"(reason: {current_check['reason']})"
                )

    if regressions:
        print("\n[scenario-eval] REGRESSION DETECTED:")
        for r in regressions:
            print(f"  - {r}")
        return 1

    print("\n[scenario-eval] no regression detected.")
    return 0


def compare_to_baseline(report: Dict[str, Any], baseline_path: Path) -> int:
    """Return 0 if no regression beyond tolerance, 1 otherwise. Prints a diff either way."""
    if not baseline_path.exists():
        print(f"[scenario-eval] no baseline found at {baseline_path}, nothing to compare against.")
        return 0

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_scenarios = baseline.get("scenarios", {})
    return _compare_scenarios(report["scenarios"], baseline_scenarios)


def compare_to_history(report: Dict[str, Any], history_path: Path) -> int:
    """Compare against the most recent entries in an append-only JSONL history file.

    Stagnation gate (Anvil Μ-2): compare the current run against the most
    recent N=5 history entries. A run is considered to show meaningful
    improvement only if `overall_score_adjusted` improved by Δ ≥ 0.03 in at
    least 2 of the last 5 comparisons. If fewer than 2 qualifying
    improvements are found across the last 5 entries, the gate fails (exit 1)
    and flags `stagnation_detected` on the report so the CI workflow can open
    a follow-up issue naming the lowest-scoring breakdown item.
    """
    if not history_path.exists():
        print(f"[scenario-eval] no history found at {history_path}, nothing to compare against.")
        return 0

    entries: List[Dict[str, Any]] = []
    with history_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not entries:
        print(f"[scenario-eval] history file {history_path} is empty, nothing to compare against.")
        return 0

    # Momentum gate (Anvil Μ-2): anchor the comparison window on *conclusive*
    # history entries only. Skipping inconclusive records here is what
    # previously let a dead judge axis keep passing the gate — an
    # inconclusive run carries no signal about capability change, so it must
    # not be the comparison anchor or count toward the stagnation window.
    conclusive_entries = [e for e in entries if e.get("inconclusive_checks", 0) == 0]
    if not conclusive_entries:
        print(
            "[scenario-eval] no conclusive history entry found (all recorded runs had "
            "inconclusive checks); cannot anchor momentum gate, treating as no-op compare.",
            file=sys.stderr,
        )
        return 0

    history_window = conclusive_entries[-5:]
    prior_report = history_window[-1]
    print(
        f"[scenario-eval] comparing against conclusive history entry from "
        f"{prior_report.get('generated_at')} (skipped inconclusive entries)"
    )
    regression_exit = _compare_scenarios(report["scenarios"], prior_report.get("scenarios", {}))

    # Stagnation gate: count meaningful improvements (Δ ≥ 0.03) across the
    # last 5 conclusive history entries. Judge-API noise that moves the raw
    # score by a few hundredths must not register as "improvement".
    min_effect = 0.03
    min_improvements = 2
    current_adjusted = report.get("overall_score_adjusted")
    improvements = 0
    prev_adjusted = None
    for entry in history_window:
        adjusted = entry.get("overall_score_adjusted")
        if adjusted is None or prev_adjusted is None:
            if adjusted is not None:
                prev_adjusted = adjusted
            continue
        if adjusted - prev_adjusted >= min_effect:
            improvements += 1
        prev_adjusted = adjusted
    if current_adjusted is not None and prev_adjusted is not None:
        if current_adjusted - prev_adjusted >= min_effect:
            improvements += 1

    report["stagnation_detected"] = improvements < min_improvements
    report["stagnation_improvements"] = improvements
    report["stagnation_window_size"] = len(history_window)

    if report["stagnation_detected"]:
        print(
            f"[scenario-eval] STAGNATION DETECTED: only {improvements} meaningful improvement(s) "
            f"(Δ ≥ {min_effect}) across the last {len(history_window)} history entries "
            f"(threshold: {min_improvements})."
        )
        return 1

    print(
        f"[scenario-eval] stagnation gate passed: {improvements} meaningful improvement(s) "
        f"across the last {len(history_window)} history entries."
    )
    return regression_exit


def append_history(report: Dict[str, Any], history_path: Path) -> None:
    """Append this run's report as one line to an append-only JSONL history file."""
    # Guard the history append: never record runs that never executed against an
    # LLM (total model outage). Such records are pure noise that pollute the
    # baseline and mask real regressions in future comparisons.
    exhaustion_markers = (
        "All fallback models failed",
        "No available models",
    )
    scenarios = report.get("scenarios", {})
    if any(r.get("critical_failure") for r in scenarios.values()):
        print(
            "[scenario-eval] skipping history append: one or more scenarios had a "
            "total LLM infrastructure failure (no model available).",
            file=sys.stderr,
        )
        return
    # Defense-in-depth: even if critical_failure was not set, refuse to append
    # records whose error logs indicate total fallback exhaustion — this catches
    # detection-logic gaps so false-pass records never enter the baseline.
    for r in scenarios.values():
        combined = f"{r.get('stdout_excerpt', '')} {r.get('stderr_excerpt', '')}"
        if not r.get("critical_failure") and any(marker in combined for marker in exhaustion_markers):
            print(
                "[scenario-eval] skipping history append: one or more scenarios "
                "logged total LLM fallback exhaustion without critical_failure being set.",
                file=sys.stderr,
            )
            return
    if any(
        (not r.get("critical_failure"))
        and any(
            marker in f"{r.get('stdout_excerpt', '')} {r.get('stderr_excerpt', '')}"
            for marker in exhaustion_markers
        )
        for r in scenarios.values()
    ):
        print(
            "[scenario-eval] skipping history append: one or more scenarios had a "
            "total LLM infrastructure failure (no model available).",
            file=sys.stderr,
        )
        return
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")
    print(f"[scenario-eval] appended to history: {history_path}")


def print_trend(history_path: Path) -> int:
    """Print overall_score/overall_score_adjusted for every recorded run, oldest first.

    This is the concrete answer to "where's the quantitative diff": each merge
    to main that ran the scenario suite adds one line here via --append-history,
    so the trend across PRs/merges is directly readable instead of inferred.
    """
    if not history_path.exists():
        print(f"[scenario-eval] no history found at {history_path}")
        return 1

    entries = []
    with history_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print(f"[scenario-eval] history file {history_path} is empty")
        return 1

    print(f"{'generated_at':<28} {'overall':>8} {'adjusted':>9} {'inconclusive':>13} {'delta':>8}")
    prior_adjusted = None
    for entry in entries:
        adjusted = entry.get("overall_score_adjusted")
        adjusted_str = f"{adjusted:.3f}" if adjusted is not None else "n/a"
        if adjusted is not None and prior_adjusted is not None:
            delta = adjusted - prior_adjusted
            delta_str = f"{'+' if delta >= 0 else ''}{delta:.3f}"
        else:
            delta_str = "-"
        print(
            f"{entry.get('generated_at', '?'):<28} {entry['overall_score']:>8.3f} {adjusted_str:>9} "
            f"{entry.get('inconclusive_checks', '?'):>13} {delta_str:>8}"
        )
        if adjusted is not None:
            prior_adjusted = adjusted
    return 0


def update_baseline(report: Dict[str, Any], baseline_path: Path) -> None:
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[scenario-eval] baseline updated: {baseline_path}")


def print_summary(report: Dict[str, Any]) -> None:
    print(f"\n=== Scenario Eval Report ({report['generated_at']}) ===")
    for scenario_id, r in report["scenarios"].items():
        adjusted = r["adjusted_total"]
        adjusted_str = f"{adjusted:.3f}" if adjusted is not None else "n/a (all checks inconclusive)"
        print(f"\n[{scenario_id}] {r['name']} — total={r['total']:.3f} adjusted={adjusted_str} ({r['duration_s']}s)")
        for check_name, check in r["breakdown"].items():
            if check_name == "loop_engineering_metrics":
                print(f"    Loop Engineering: {r.get('loop_engineering_metrics', {})}")
                continue
            flag = " [INCONCLUSIVE]" if check["inconclusive"] else ""
            print(f"    {check_name:<28} {check['score']:.2f} x {check['weight']:.2f} — {check['reason']}{flag}")

    adjusted_overall = report["overall_score_adjusted"]
    adjusted_overall_str = f"{adjusted_overall:.3f}" if adjusted_overall is not None else "n/a"
    print(f"\nOverall score: {report['overall_score']:.3f} (adjusted: {adjusted_overall_str})")
    print(f"Inconclusive checks: {report['inconclusive_checks']} (infra/judge unavailable, excluded from adjusted score)")


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", help="Run only the scenario with this id")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit")
    parser.add_argument("--update-baseline", action="store_true", help="Write results into tests/benchmark/baselines/scenario_baseline.json")
    parser.add_argument("--compare-to", help="Path to a baseline JSON to check for regressions (exit 1 on regression)")
    parser.add_argument("--compare-to-history", help="Path to a JSONL history file; compares against its most recent entry")
    parser.add_argument("--history", help="Path to a JSONL history file; alias for --compare-to-history")
    parser.add_argument("--append-history", help="Path to a JSONL history file to append this run's report to")
    parser.add_argument("--print-trend", help="Path to a JSONL history file; print overall_score over time and exit")
    parser.add_argument("--json-out", help="Path to write the run's JSON report (default: tests/benchmark/reports/scenario_report_<ts>.json)")
    parser.add_argument("--parallel", action="store_true", help="Run scenarios concurrently instead of one at a time (risks provider rate-limit contention)")
    args = parser.parse_args()

    if args.print_trend:
        return print_trend(Path(args.print_trend))

    specs = load_scenarios(only_id=args.scenario)
    if not specs:
        print(f"No scenarios found (filter: {args.scenario!r})", file=sys.stderr)
        return 1

    if args.list:
        for spec in specs:
            print(f"{spec['id']}: {spec.get('name', '')}")
        return 0

    ok, message = preflight_llm_availability()
    if not ok:
        print(f"[scenario-eval] {message}", file=sys.stderr)
        return 1
    print(f"[scenario-eval] preflight: {message}")

    if not require_openrouter_api_key():
        return 1

    report = await run_all(specs, parallel=args.parallel)
    print_summary(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.json_out) if args.json_out else REPORTS_DIR / f"scenario_report_{int(time.time())}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[scenario-eval] report written to {out_path}")

    if args.update_baseline:
        update_baseline(report, Path("tests/benchmark/baselines/scenario_baseline.json"))

    if args.append_history:
        append_history(report, Path(args.append_history))

    exit_code = 0
    if args.compare_to:
        exit_code = compare_to_baseline(report, Path(args.compare_to)) or exit_code
    if args.compare_to_history:
        exit_code = compare_to_history(report, Path(args.compare_to_history)) or exit_code
    if args.history:
        exit_code = compare_to_history(report, Path(args.compare_to_history)) or exit_code

    # Aggregate critical infrastructure failures: if any scenario experienced a
    # total model collapse, the runner must exit non-zero so CI gates catch it.
    if any(r.get("critical_failure") for r in report["scenarios"].values()):
        exit_code = exit_code or 2
        print(
            "[scenario-eval] CRITICAL: one or more scenarios failed due to model "
            "infrastructure unavailability; exiting non-zero.",
            file=sys.stderr,
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
