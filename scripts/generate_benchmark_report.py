"""Regenerate the sourced-metrics block in docs/BENCHMARK_REPORT.md.

Anvil Phase Mu-4 (issue #1220): docs/BENCHMARK_REPORT.md previously quoted a
hand-typed "Research Pass Rate 100.0% (Score: 0.775)" that had no basis in
tests/benchmark/baselines/scenario_history.jsonl (real measured value: far
lower). Every number between the markers below must trace to one of the two
files this script reads -- no hand-typed figures. `--check` exits 1 if the
checked-in file doesn't match what this script would write, so a manual edit
inside the generated block is caught by CI instead of silently drifting from
the data again (docs/ANVIL_PLAN.md SS5.4 Mu-4).

Anvil Phase Mu-5 (issue #1221): a single overall_score scalar hides Campbell's
law -- an internal metric that keeps "improving" while the externally-defined
axis (SWE-bench Lite) or cost (duration_s) moves the other way means the
internal metric became the target, not a proxy for real capability. This
script also tracks three momentum axes (internal score, external resolve
rate, cost) between the two most recent comparable data points and prints an
explicit warning when internal improves while either of the other two does
not.

Usage:
    python scripts/generate_benchmark_report.py             # regenerate in place
    python scripts/generate_benchmark_report.py --check     # exit 1 if stale, don't write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

REPORT_PATH = Path("docs/BENCHMARK_REPORT.md")
HISTORY_PATH = Path("tests/benchmark/baselines/scenario_history.jsonl")
SWEBENCH_PATH = Path("docs/SWEBENCH_REPORT.md")

BEGIN_MARKER = "<!-- BEGIN GENERATED: scripts/generate_benchmark_report.py -->"
END_MARKER = "<!-- END GENERATED -->"

SWEBENCH_RUN_RE = re.compile(
    r"^## (?P<date>\d{4}-\d{2}-\d{2}) — run `(?P<run_id>[^`]+)`\n"
    r"\n"
    r"- Resolved: \*\*(?P<resolved>\d+) / (?P<submitted>\d+)\*\* submitted",
    re.MULTILINE,
)


def load_conclusive_scenario_entries(path: Path) -> list[dict[str, Any]]:
    """All fully-conclusive runs, oldest first, so a judge/infra outage never becomes a published number."""
    if not path.exists():
        return []
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    conclusive = [e for e in entries if e.get("inconclusive_checks", 0) == 0]
    return conclusive or entries


def load_latest_scenario_entry(path: Path) -> Optional[dict[str, Any]]:
    entries = load_conclusive_scenario_entries(path)
    return entries[-1] if entries else None


def load_swebench_runs(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return [m.groupdict() for m in SWEBENCH_RUN_RE.finditer(path.read_text(encoding="utf-8"))]


def total_duration_s(entry: dict[str, Any]) -> float:
    return sum((s.get("duration_s") or 0) for s in entry.get("scenarios", {}).values())


def _trend(delta: Optional[float], higher_is_better: bool, eps: float = 1e-9) -> Optional[str]:
    if delta is None:
        return None
    if abs(delta) < eps:
        return "flat"
    improving = (delta > 0) if higher_is_better else (delta < 0)
    return "improving" if improving else "declining"


def compute_momentum(
    scenario_entries: list[dict[str, Any]], swebench_runs: list[dict[str, str]]
) -> dict[str, Any]:
    """Compare the two most recent comparable data points on each of three axes.

    internal: overall_score_adjusted (scenario_history.jsonl) -- higher is better.
    external: SWE-bench Lite resolve rate (docs/SWEBENCH_REPORT.md) -- higher is better.
    cost: total scenario duration_s (scenario_history.jsonl) -- lower is better.

    `divergence` is true only when internal is improving while external or cost
    is not (the Campbell's-law signature Mu-5 exists to catch) -- an internal
    regression alone, or all three moving together, is not divergence.
    """
    axes: dict[str, dict[str, Any]] = {
        "internal": {"trend": None, "delta": None},
        "external": {"trend": None, "delta": None},
        "cost": {"trend": None, "delta": None},
    }

    if len(scenario_entries) >= 2:
        prev, latest = scenario_entries[-2], scenario_entries[-1]
        prev_score, latest_score = prev.get("overall_score_adjusted"), latest.get("overall_score_adjusted")
        if prev_score is not None and latest_score is not None:
            delta = round(latest_score - prev_score, 4)
            axes["internal"] = {"trend": _trend(delta, higher_is_better=True), "delta": delta}

        delta_cost = round(total_duration_s(latest) - total_duration_s(prev), 2)
        axes["cost"] = {"trend": _trend(delta_cost, higher_is_better=False), "delta": delta_cost}

    if len(swebench_runs) >= 2:
        def resolve_rate(run: dict[str, str]) -> Optional[float]:
            submitted = int(run["submitted"])
            return (int(run["resolved"]) / submitted) if submitted else None

        prev_rate, latest_rate = resolve_rate(swebench_runs[-2]), resolve_rate(swebench_runs[-1])
        if prev_rate is not None and latest_rate is not None:
            delta = round(latest_rate - prev_rate, 4)
            axes["external"] = {"trend": _trend(delta, higher_is_better=True), "delta": delta}

    opposing = [
        axis
        for axis in ("external", "cost")
        if axes[axis]["trend"] in ("flat", "declining")
    ]
    divergence = axes["internal"]["trend"] == "improving" and bool(opposing)

    return {"axes": axes, "divergence": divergence, "opposing_axes": opposing}


AXIS_LABELS = {
    "internal": "Internal score (scenario_history.jsonl)",
    "external": "External resolve rate (SWE-bench Lite)",
    "cost": "Cost (total scenario duration_s)",
}


def render_momentum(momentum: dict[str, Any]) -> list[str]:
    lines = ["", "#### Momentum Divergence Check (Anvil Mu-5)", ""]
    lines.append("| Axis | Trend | Delta |")
    lines.append("| :--- | :---: | ---: |")
    for axis, label in AXIS_LABELS.items():
        info = momentum["axes"][axis]
        trend, delta = info["trend"], info["delta"]
        trend_str = trend or "not enough data"
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        lines.append(f"| {label} | {trend_str} | {delta_str} |")
    lines.append("")
    if momentum["divergence"]:
        opposing = ", ".join(AXIS_LABELS[a] for a in momentum["opposing_axes"])
        lines.append(
            f"⚠️ **DIVERGENCE DETECTED**: internal score is improving while {opposing} "
            "is not -- this is the Campbell's-law pattern (an internal metric becoming the "
            "target instead of a proxy for real capability). Do not treat the internal-score "
            "improvement alone as evidence of progress."
        )
    else:
        lines.append("No divergence detected across the three tracked axes.")
    return lines


def render_block(
    scenario_entry: Optional[dict[str, Any]],
    swebench_runs: list[dict[str, str]],
    momentum: Optional[dict[str, Any]] = None,
) -> str:
    lines = [BEGIN_MARKER, ""]
    lines.append(
        "_Generated by `scripts/generate_benchmark_report.py` from "
        "`tests/benchmark/baselines/scenario_history.jsonl` and `docs/SWEBENCH_REPORT.md` "
        "-- editing between the markers above/below gets overwritten on the next run._"
    )
    lines.append("")

    if scenario_entry is None:
        lines.append(
            "**Research Pass Rate (Scenario Eval)**: no recorded runs yet in "
            f"`{HISTORY_PATH.as_posix()}`."
        )
    else:
        adjusted = scenario_entry.get("overall_score_adjusted")
        adjusted_pct = f"{adjusted * 100:.1f}%" if adjusted is not None else "n/a"
        lines.append(
            f"**Research Pass Rate (Scenario Eval)**: **{adjusted_pct}** "
            f"(adjusted score: {adjusted if adjusted is not None else 'n/a'}) -- "
            f"{scenario_entry.get('generated_at', '?')}, "
            f"{scenario_entry.get('inconclusive_checks', 0)} inconclusive check(s)"
        )
        lines.append("")
        lines.append("| Scenario | Total | Adjusted |")
        lines.append("| :--- | :---: | :---: |")
        for scenario_id, result in scenario_entry.get("scenarios", {}).items():
            adj = result.get("adjusted_total")
            adj_str = f"{adj:.3f}" if adj is not None else "n/a"
            lines.append(f"| {scenario_id} | {result.get('total', 0):.3f} | {adj_str} |")

    lines.append("")
    if not swebench_runs:
        lines.append(f"**SWE-bench Lite**: no recorded runs yet in `{SWEBENCH_PATH.as_posix()}`.")
    else:
        latest = swebench_runs[-1]
        lines.append(
            f"**SWE-bench Lite (latest, {latest['date']})**: **{latest['resolved']} / {latest['submitted']}** "
            f"resolved, run `{latest['run_id']}` -- see [SWEBENCH_REPORT.md](SWEBENCH_REPORT.md) for the full weekly trend."
        )

    if momentum is not None:
        lines.extend(render_momentum(momentum))

    lines.append("")
    lines.append(END_MARKER)
    return "\n".join(lines)


def splice(existing: str, block: str) -> str:
    if BEGIN_MARKER not in existing or END_MARKER not in existing:
        raise SystemExit(
            f"{REPORT_PATH} is missing the generated-block markers "
            f"({BEGIN_MARKER!r} / {END_MARKER!r}) -- add them once by hand "
            "where the generated metrics block should live."
        )
    before, _, rest = existing.partition(BEGIN_MARKER)
    _, _, after = rest.partition(END_MARKER)
    return before + block + after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Exit 1 if the file would change instead of writing it")
    args = parser.parse_args()

    scenario_entries = load_conclusive_scenario_entries(HISTORY_PATH)
    scenario_entry = scenario_entries[-1] if scenario_entries else None
    swebench_runs = load_swebench_runs(SWEBENCH_PATH)
    momentum = compute_momentum(scenario_entries, swebench_runs)
    block = render_block(scenario_entry, swebench_runs, momentum)

    existing = REPORT_PATH.read_text(encoding="utf-8")
    updated = splice(existing, block)

    if args.check:
        if updated != existing:
            print(f"{REPORT_PATH} is stale -- run `python {sys.argv[0]}` to regenerate.", file=sys.stderr)
            return 1
        print(f"{REPORT_PATH} matches generated content.")
        return 0

    REPORT_PATH.write_text(updated, encoding="utf-8")
    print(f"Regenerated {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
