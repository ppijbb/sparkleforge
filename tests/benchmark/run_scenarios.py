#!/usr/bin/env python3
"""Anvil acceptance-scenario eval harness (Ω-1, issue #330).

Drives the 5 "Anvil v1.0 acceptance scenarios" (#267) end-to-end through the
real natural-language entry point (`python main.py work "<goal>"`), then
scores the resulting filesystem/system state with deterministic checks plus
a capped-weight LLM-judge fallback for subjective quality.

Usage:
    python tests/benchmark/run_scenarios.py                      # run all 5, print + save report
    python tests/benchmark/run_scenarios.py --scenario system_cleanup
    python tests/benchmark/run_scenarios.py --list
    python tests/benchmark/run_scenarios.py --update-baseline    # bump docs/benchmark_baseline.json
    python tests/benchmark/run_scenarios.py --compare-to docs/benchmark_baseline.json  # CI regression gate
"""

from __future__ import annotations

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
    cmd = [sys.executable, str(MAIN_PY), "work", user_query]
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
        return {
            "returncode": -1,
            "stdout": (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            "stderr": (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or ""),
            "timed_out": True,
            "duration_s": time.time() - start,
        }


async def run_scenario(spec: Dict[str, Any]) -> Dict[str, Any]:
    fixture_module = importlib.import_module(f"tests.benchmark.scenario_fixtures.{spec['fixture']}")

    workspace = Path(tempfile.mkdtemp(prefix=f"sparkleforge_scenario_{spec['id']}_"))
    try:
        ctx = await asyncio.to_thread(fixture_module.build, workspace)
        ctx.setdefault("workspace", str(workspace))
        user_query = spec["user_query"].format(**ctx)

        exec_result = await asyncio.to_thread(run_agent, user_query, workspace, spec.get("timeout_s", 300))

        if exec_result["timed_out"]:
            scores = {name: (0.0, "scenario timed out") for name in spec["weights"]}
        else:
            scores = await fixture_module.grade(workspace, ctx, exec_result["stdout"])

        graded = weighted_total(scores, spec["weights"])

        return {
            "id": spec["id"],
            "name": spec.get("name", spec["id"]),
            "user_query": user_query,
            "total": graded["total"],
            "breakdown": graded["breakdown"],
            "returncode": exec_result["returncode"],
            "timed_out": exec_result["timed_out"],
            "duration_s": round(exec_result["duration_s"], 2),
            "stdout_excerpt": exec_result["stdout"][:1500],
            "stderr_excerpt": exec_result["stderr"][:1500],
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

    overall = round(sum(r["total"] for r in results) / len(results), 4) if results else 0.0
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_score": overall,
        "scenarios": {r["id"]: r for r in results},
    }


def compare_to_baseline(report: Dict[str, Any], baseline_path: Path) -> int:
    """Return 0 if no regression beyond tolerance, 1 otherwise. Prints a diff either way."""
    if not baseline_path.exists():
        print(f"[scenario-eval] no baseline found at {baseline_path}, nothing to compare against.")
        return 0

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_scenarios = baseline.get("capability_scenarios", {}).get("scenarios", {})

    regressions = []
    for scenario_id, current in report["scenarios"].items():
        prior = baseline_scenarios.get(scenario_id)
        if prior is None:
            print(f"[scenario-eval] '{scenario_id}': no prior baseline, first run (total={current['total']})")
            continue

        prior_total = prior.get("total", 0.0)
        delta = round(current["total"] - prior_total, 4)
        sign = "+" if delta >= 0 else ""
        print(
            f"[scenario-eval] '{scenario_id}': baseline={prior_total:.3f} current={current['total']:.3f} "
            f"({sign}{delta:.3f})"
        )

        for check_name, current_check in current["breakdown"].items():
            prior_check = prior.get("breakdown", {}).get(check_name)
            if prior_check is None:
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


def update_baseline(report: Dict[str, Any], baseline_path: Path) -> None:
    baseline: Dict[str, Any] = {}
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["capability_scenarios"] = report
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[scenario-eval] baseline updated: {baseline_path}")


def print_summary(report: Dict[str, Any]) -> None:
    print(f"\n=== Scenario Eval Report ({report['generated_at']}) ===")
    for scenario_id, r in report["scenarios"].items():
        print(f"\n[{scenario_id}] {r['name']} — total={r['total']:.3f} ({r['duration_s']}s)")
        for check_name, check in r["breakdown"].items():
            print(f"    {check_name:<28} {check['score']:.2f} x {check['weight']:.2f} — {check['reason']}")
    print(f"\nOverall score: {report['overall_score']:.3f}")


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", help="Run only the scenario with this id")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit")
    parser.add_argument("--update-baseline", action="store_true", help="Write results into docs/benchmark_baseline.json")
    parser.add_argument("--compare-to", help="Path to a baseline JSON to check for regressions (exit 1 on regression)")
    parser.add_argument("--json-out", help="Path to write the run's JSON report (default: tests/benchmark/reports/scenario_report_<ts>.json)")
    parser.add_argument("--parallel", action="store_true", help="Run scenarios concurrently instead of one at a time (risks provider rate-limit contention)")
    args = parser.parse_args()

    specs = load_scenarios(only_id=args.scenario)
    if not specs:
        print(f"No scenarios found (filter: {args.scenario!r})", file=sys.stderr)
        return 1

    if args.list:
        for spec in specs:
            print(f"{spec['id']}: {spec.get('name', '')}")
        return 0

    report = await run_all(specs, parallel=args.parallel)
    print_summary(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.json_out) if args.json_out else REPORTS_DIR / f"scenario_report_{int(time.time())}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[scenario-eval] report written to {out_path}")

    if args.update_baseline:
        update_baseline(report, Path("docs/benchmark_baseline.json"))

    if args.compare_to:
        return compare_to_baseline(report, Path(args.compare_to))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
