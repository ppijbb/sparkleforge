"""Infra-outage vs. real-regression classification for scenario-eval.yml.

Moved verbatim from the record-history job's "Baseline validity gate"
heredoc: a run where (nearly) every scenario failed due to a provider/model
outage rather than agent capability must not be recorded into scenario
history, or it pollutes the benchmark dataset and makes regression
detection impossible.
"""

from __future__ import annotations

from dataclasses import dataclass

_INFRA_FAILURE_MARKERS = ("No available models.", "All fallback models failed")


@dataclass
class ScenarioOutcome:
    overall_score: float
    infra_failed: int
    total: int
    infra_ratio: float
    is_infra_outage: bool


def classify_scenario_outcome(report: dict) -> ScenarioOutcome:
    overall = float(report.get("overall_score", 0.0) or 0.0)
    scenarios = report.get("scenarios", {}) or {}
    infra_failed = 0
    total = 0
    for scenario in scenarios.values():
        total += 1
        stdout = scenario.get("stdout_excerpt") or ""
        if any(marker in stdout for marker in _INFRA_FAILURE_MARKERS):
            infra_failed += 1
    infra_ratio = (infra_failed / total) if total else 1.0
    is_infra_outage = overall < 0.3 or infra_ratio > 0.9
    return ScenarioOutcome(
        overall_score=overall,
        infra_failed=infra_failed,
        total=total,
        infra_ratio=infra_ratio,
        is_infra_outage=is_infra_outage,
    )
