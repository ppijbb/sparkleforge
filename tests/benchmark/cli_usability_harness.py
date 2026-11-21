#!/usr/bin/env python3
"""SparkleForge CLI usability and orchestration QA evaluation harness.

Measures and scores (0.0-1.0 per dimension + overall):
1. Latency: cold-start / command duration (ms) vs threshold.
2. Output Hygiene: absence of Loading weights, Batches:, leaked [WARN]/Traceback, and \\ufffd.
3. Request Info Processing: query routing, flag handling (--model/--depth/--autopilot/--format),
   and structured output validity.
4. Response Quality: non-empty coherent output without raw 402/404/429 provider dump spam.
5. Orchestration Monitoring: task/subagent lifecycle visibility, step latency breakdown,
   and clean returncode == 0.

Supports CLI execution in offline/mock and live subprocess modes, with --json-out export.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DIMENSIONS: tuple[str, ...] = (
    "latency",
    "output_hygiene",
    "request_info_processing",
    "response_quality",
    "orchestration_monitoring",
)

DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "latency": 0.20,
    "output_hygiene": 0.20,
    "request_info_processing": 0.20,
    "response_quality": 0.20,
    "orchestration_monitoring": 0.20,
}

REQUIRED_RUN_FLAGS: tuple[str, ...] = (
    "--model",
    "--depth",
    "--autopilot",
    "--format",
)

VALID_DEPTH_CHOICES = frozenset({"quick", "medium", "deep", "auto"})
VALID_FORMAT_CHOICES = frozenset({"json", "yaml", "txt", "markdown", "md"})
VALID_AUTOPILOT_CHOICES = frozenset({"true", "false", "yes", "no", "1", "0", "on", "off"})

DEFAULT_LATENCY_THRESHOLD_MS = 5000.0
DEFAULT_LIVE_TIMEOUT_S = 30.0
DEFAULT_LOW_COST_MODEL = "google/gemini-3.1-flash-lite-preview"

_STAGE_TIMING_RE = re.compile(
    r"\[(?:OK|FAIL)\]\s+([A-Za-z0-9_.-]+)\s+\(([0-9]+(?:\.[0-9]+)?)\s*ms\)"
)
_LEAKED_WARN_RE = re.compile(r"\[WARN\]|(?:^|\n)\s*(?:WARNING|WARN)\s*[:\-]")
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):|(?:^|\n)\s*Traceback\b")
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_]{2,}")
_PARSE_ERROR_RE = re.compile(
    r"(?:unrecognized arguments|invalid choice|the following arguments are required|Unknown command)",
    re.IGNORECASE,
)
_PROVIDER_DUMP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:http\s*|status(?:_code)?[=:\s]*|error\s*|code[=:\s]*|\()\b(?:402|404|429)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:402|404|429)\b[^\n]{0,100}"
        r"(?:payment required|not found|too many requests|rate[- ]?limit|decommissioned|"
        r"openrouter|groq|gemini|nvidia|cerebras|fallback|provider)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:openrouter|groq|gemini|nvidia|cerebras|provider|fallback|upstream|api)"
        r"[^\n]{0,100}\b(?:402|404|429)\b",
        re.IGNORECASE,
    ),
    re.compile(r"all fallback models failed", re.IGNORECASE),
    re.compile(r"<!DOCTYPE html>", re.IGNORECASE),
)


@dataclass
class ProbeObservation:
    """Captured execution metrics and output for a single CLI probe."""

    name: str
    command: list[str]
    returncode: int
    duration_ms: float
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    query: str | None = None
    routed_target: str | None = None
    flags: dict[str, Any] = field(default_factory=dict)
    supported_flags: list[str] | None = None
    applied_flags: dict[str, Any] = field(default_factory=dict)
    expected_format: str | None = None
    output_file_content: str | None = None
    structured_output: Any | None = None
    orchestration: dict[str, Any] = field(default_factory=dict)
    latency_threshold_ms: float | None = None
    cold_start_ms: float | None = None


@dataclass
class DimensionScore:
    """Evaluation result for one QA dimension on a probe."""

    name: str
    score: float
    passed: bool
    findings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeEvaluation:
    """5-dimension evaluation result for a single probe."""

    name: str
    command: list[str]
    returncode: int
    duration_ms: float
    overall_score: float
    dimensions: dict[str, DimensionScore]


@dataclass
class UsabilityReport:
    """Aggregated CLI usability & orchestration QA report."""

    mode: str
    timestamp: str
    overall_score: float
    dimension_scores: dict[str, float]
    passed: bool
    probes: list[ProbeEvaluation]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report into a JSON-compatible dictionary."""
        return {
            "mode": self.mode,
            "timestamp": self.timestamp,
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "passed": self.passed,
            "summary": self.summary,
            "probes": [
                {
                    "name": probe.name,
                    "command": probe.command,
                    "returncode": probe.returncode,
                    "duration_ms": probe.duration_ms,
                    "overall_score": probe.overall_score,
                    "dimensions": {
                        dim_name: asdict(dim_score)
                        for dim_name, dim_score in probe.dimensions.items()
                    },
                }
                for probe in self.probes
            ],
        }


def _clamp_score(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def _normalize_bool_token(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "yes", "1", "on"}:
        return True
    if token in {"false", "no", "0", "off"}:
        return False
    return None


def _try_parse_json(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None


def _is_failure_only(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    return all(
        line.startswith("[FAIL]")
        or line.startswith("Error:")
        or line.startswith("Fatal error")
        for line in lines
    )


def detect_provider_dump_spam(text: str) -> list[str]:
    """Return matching provider HTTP dump spam snippets found in CLI output."""
    hits: list[str] = []
    for pattern in _PROVIDER_DUMP_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(match.group(0).strip()[:120])
    return hits


def extract_step_latencies_ms(
    orchestration: dict[str, Any],
    structured_output: Any,
    combined_text: str,
) -> dict[str, float]:
    """Extract per-step/stage latencies in milliseconds from telemetry or stdout."""
    latencies: dict[str, float] = {}

    sources: list[dict[str, Any]] = []
    if isinstance(orchestration, dict):
        sources.append(orchestration)
    if isinstance(structured_output, dict):
        sources.append(structured_output)
        if isinstance(structured_output.get("metadata"), dict):
            sources.append(structured_output["metadata"])

    for src in sources:
        for key in ("step_latencies_ms", "stage_timings_ms"):
            raw = src.get(key)
            if isinstance(raw, dict):
                for step_name, val in raw.items():
                    if isinstance(val, (int, float)) and val >= 0:
                        latencies[str(step_name)] = float(val)
            elif isinstance(raw, list):
                for idx, item in enumerate(raw):
                    if isinstance(item, dict):
                        name = str(item.get("name") or item.get("step") or f"step_{idx}")
                        val = item.get("duration_ms", item.get("latency_ms"))
                        if isinstance(val, (int, float)) and val >= 0:
                            latencies[name] = float(val)

        for list_key in ("stages", "tasks", "steps"):
            items = src.get(list_key)
            if isinstance(items, list):
                for idx, item in enumerate(items):
                    if isinstance(item, dict):
                        name = str(
                            item.get("name")
                            or item.get("id")
                            or item.get("step")
                            or f"{list_key}_{idx}"
                        )
                        if isinstance(item.get("duration_ms"), (int, float)):
                            val = float(item["duration_ms"])
                            if val >= 0:
                                latencies[name] = val
                        elif isinstance(item.get("latency_ms"), (int, float)):
                            val = float(item["latency_ms"])
                            if val >= 0:
                                latencies[name] = val
                        elif isinstance(item.get("execution_time"), (int, float)):
                            val = float(item["execution_time"]) * 1000.0
                            if val >= 0:
                                latencies[name] = val

    for match in _STAGE_TIMING_RE.finditer(combined_text):
        stage_name, ms_str = match.group(1), match.group(2)
        latencies.setdefault(stage_name, float(ms_str))

    return latencies


def evaluate_latency(
    obs: ProbeObservation,
    default_threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS,
) -> DimensionScore:
    """Score command / cold-start duration against the configured ms threshold."""
    threshold_ms = (
        float(obs.latency_threshold_ms)
        if obs.latency_threshold_ms is not None
        else float(default_threshold_ms)
    )
    measured_ms = (
        float(obs.duration_ms)
        if obs.duration_ms > 0 or obs.cold_start_ms is None
        else float(obs.cold_start_ms)
    )
    findings: list[str] = []

    if obs.timed_out or measured_ms < 0:
        score = 0.0
        findings.append("Command timed out or produced negative duration")
    elif measured_ms <= threshold_ms:
        score = 1.0
    else:
        over_ratio = (measured_ms - threshold_ms) / max(threshold_ms, 1.0)
        score = _clamp_score(1.0 - over_ratio)
        findings.append(
            f"Duration {measured_ms:.1f}ms exceeded threshold {threshold_ms:.1f}ms"
        )

    passed = (not obs.timed_out) and 0 <= measured_ms <= threshold_ms
    return DimensionScore(
        name="latency",
        score=score,
        passed=passed,
        findings=findings,
        metrics={
            "duration_ms": round(measured_ms, 2),
            "cold_start_ms": (
                round(float(obs.cold_start_ms), 2)
                if obs.cold_start_ms is not None
                else round(measured_ms, 2)
            ),
            "threshold_ms": round(threshold_ms, 2),
            "timed_out": obs.timed_out,
        },
    )


def evaluate_output_hygiene(obs: ProbeObservation) -> DimensionScore:
    """Score absence of progress bars, leaked [WARN], Traceback, and garbled glyphs."""
    combined = f"{obs.stdout}\n{obs.stderr}"
    checks: dict[str, bool] = {
        "no_loading_weights": "loading weights" not in combined.lower(),
        "no_batches_progress": "batches:" not in combined.lower(),
        "no_leaked_warn": not bool(_LEAKED_WARN_RE.search(combined)),
        "no_traceback": not bool(_TRACEBACK_RE.search(combined)),
        "no_garbled_glyphs": "\ufffd" not in combined,
    }
    findings = [
        f"Output hygiene violation: {name.removeprefix('no_').replace('_', ' ')}"
        for name, ok in checks.items()
        if not ok
    ]
    score = _clamp_score(sum(1.0 for ok in checks.values() if ok) / len(checks))
    return DimensionScore(
        name="output_hygiene",
        score=score,
        passed=all(checks.values()),
        findings=findings,
        metrics=checks,
    )


def _score_flag_handling(obs: ProbeObservation, combined: str) -> tuple[float, list[str]]:
    findings: list[str] = []
    if _PARSE_ERROR_RE.search(combined):
        findings.append("CLI argument parser rejected flags or command arguments")
        return 0.0, findings

    if obs.supported_flags is not None:
        supported = set(obs.supported_flags)
        matched = [flag for flag in REQUIRED_RUN_FLAGS if flag in supported]
        missing = [flag for flag in REQUIRED_RUN_FLAGS if flag not in supported]
        if missing:
            findings.append(f"Missing supported run flags: {', '.join(missing)}")
        return len(matched) / len(REQUIRED_RUN_FLAGS), findings

    if obs.flags:
        valid_count = 0
        for key, raw_val in obs.flags.items():
            norm_key = key.lstrip("-").lower()
            valid = True
            if norm_key == "depth":
                valid = str(raw_val).strip().lower() in VALID_DEPTH_CHOICES
            elif norm_key == "format":
                valid = str(raw_val).strip().lower() in VALID_FORMAT_CHOICES
            elif norm_key == "autopilot":
                valid = _normalize_bool_token(raw_val) is not None
            elif norm_key == "model":
                valid = bool(str(raw_val).strip())

            if valid and obs.applied_flags and norm_key in obs.applied_flags:
                applied_val = obs.applied_flags[norm_key]
                if norm_key == "autopilot":
                    valid = _normalize_bool_token(applied_val) == _normalize_bool_token(
                        raw_val
                    )
                else:
                    valid = str(applied_val).strip() == str(raw_val).strip()

            if valid:
                valid_count += 1
            else:
                findings.append(f"Flag handling mismatch or invalid value for --{norm_key}")
        return valid_count / len(obs.flags), findings

    return 1.0, findings


def evaluate_request_info_processing(obs: ProbeObservation) -> DimensionScore:
    """Score query routing, CLI flag handling, and structured output validity."""
    combined = f"{obs.stdout}\n{obs.stderr}"
    findings: list[str] = []

    parse_error = bool(_PARSE_ERROR_RE.search(combined))
    if obs.query is not None:
        query_ok = (
            bool(obs.query.strip())
            and not parse_error
            and (obs.routed_target is None or bool(obs.routed_target.strip()))
        )
        if not query_ok:
            findings.append("Query routing check failed or empty query")
    else:
        query_ok = obs.returncode != 2 and not parse_error
        if not query_ok:
            findings.append("Command routing failed with parser exit code")
    routing_score = 1.0 if query_ok else 0.0

    flag_score, flag_findings = _score_flag_handling(obs, combined)
    findings.extend(flag_findings)

    fmt = (obs.expected_format or str(obs.flags.get("format") or "")).strip().lower()
    payload = obs.output_file_content if obs.output_file_content is not None else obs.stdout

    if obs.structured_output is not None:
        structured_ok = (
            isinstance(obs.structured_output, (dict, list))
            and len(obs.structured_output) > 0
        )
    elif fmt == "json":
        parsed = _try_parse_json(payload)
        structured_ok = isinstance(parsed, (dict, list)) and len(parsed) > 0
    elif fmt == "yaml":
        try:
            parsed_yaml = yaml.safe_load(payload)
        except yaml.YAMLError:
            parsed_yaml = None
        structured_ok = isinstance(parsed_yaml, (dict, list)) and len(parsed_yaml) > 0
    elif fmt in {"markdown", "md", "txt"}:
        structured_ok = bool(payload.strip()) and not bool(_TRACEBACK_RE.search(payload))
    else:
        structured_ok = bool(payload.strip()) and obs.returncode == 0

    if not structured_ok:
        findings.append(
            f"Structured output invalid or empty for format '{fmt or 'default'}'"
        )
    structured_score = 1.0 if structured_ok else 0.0

    score = _clamp_score((routing_score + flag_score + structured_score) / 3.0)
    passed = routing_score == 1.0 and flag_score == 1.0 and structured_score == 1.0
    return DimensionScore(
        name="request_info_processing",
        score=score,
        passed=passed,
        findings=findings,
        metrics={
            "query_routing_score": round(routing_score, 4),
            "flag_handling_score": round(flag_score, 4),
            "structured_output_score": round(structured_score, 4),
            "format": fmt or "default",
        },
    )


def evaluate_response_quality(obs: ProbeObservation) -> DimensionScore:
    """Score non-empty coherent output and absence of 402/404/429 provider dump spam."""
    response_text = (
        obs.output_file_content if obs.output_file_content is not None else obs.stdout
    ).strip()
    if not response_text and obs.structured_output is not None:
        response_text = json.dumps(obs.structured_output, ensure_ascii=False).strip()

    combined = f"{obs.stdout}\n{obs.stderr}"
    spam_hits = detect_provider_dump_spam(combined)

    if not response_text:
        return DimensionScore(
            name="response_quality",
            score=0.0,
            passed=False,
            findings=["Empty response output"],
            metrics={
                "non_empty": False,
                "coherent": False,
                "no_provider_spam": not bool(spam_hits),
                "no_failure_only": False,
                "provider_spam_hits": spam_hits,
                "length": 0,
            },
        )

    word_tokens = _WORD_TOKEN_RE.findall(response_text)
    failure_only = _is_failure_only(response_text)
    has_traceback = bool(_TRACEBACK_RE.search(response_text))
    coherent = len(word_tokens) >= 3 and not failure_only and not has_traceback
    no_provider_spam = len(spam_hits) == 0
    no_failure_only = not failure_only

    score = 0.30
    if coherent:
        score += 0.30
    if no_provider_spam:
        score += 0.25
    if no_failure_only:
        score += 0.15

    findings: list[str] = []
    if not coherent:
        findings.append("Response lacks coherent non-error content")
    if not no_provider_spam:
        findings.append(
            f"Raw 402/404/429 provider dump spam detected: {', '.join(spam_hits[:3])}"
        )
    if not no_failure_only:
        findings.append("Response contains only terminal failure lines")

    final_score = _clamp_score(score)
    return DimensionScore(
        name="response_quality",
        score=final_score,
        passed=coherent and no_provider_spam and no_failure_only,
        findings=findings,
        metrics={
            "non_empty": True,
            "coherent": coherent,
            "no_provider_spam": no_provider_spam,
            "no_failure_only": no_failure_only,
            "provider_spam_hits": spam_hits,
            "length": len(response_text),
            "word_count": len(word_tokens),
        },
    )


def evaluate_orchestration_monitoring(obs: ProbeObservation) -> DimensionScore:
    """Score task/subagent lifecycle visibility, step latency breakdown, and clean exit."""
    combined = f"{obs.stdout}\n{obs.stderr}"
    orch = obs.orchestration if isinstance(obs.orchestration, dict) else {}
    structured = (
        obs.structured_output
        if isinstance(obs.structured_output, dict)
        else _try_parse_json(obs.output_file_content or obs.stdout)
    )
    if not isinstance(structured, dict):
        structured = {}

    clean_returncode = obs.returncode == 0 and not obs.timed_out

    lifecycle_items = (
        list(orch.get("tasks") or [])
        + list(orch.get("subagents") or [])
        + list(orch.get("stages") or [])
        + list(orch.get("lifecycle_events") or [])
        + list(structured.get("tasks") or [])
        + list(structured.get("stages") or [])
        + list(structured.get("subagents") or [])
    )
    stage_matches = _STAGE_TIMING_RE.findall(combined)
    has_lifecycle_markers = bool(
        re.search(
            r"\b(?:subagent|handoff|lifecycle|agent_states|workflow_summary)\b",
            combined,
            re.IGNORECASE,
        )
    )
    lifecycle_visible = bool(lifecycle_items or stage_matches or has_lifecycle_markers)

    step_latencies = extract_step_latencies_ms(orch, structured, combined)
    has_step_latency = len(step_latencies) > 0

    score = 0.0
    if clean_returncode:
        score += 0.40
    if lifecycle_visible:
        score += 0.30
    if has_step_latency:
        score += 0.30

    findings: list[str] = []
    if not clean_returncode:
        findings.append(f"Non-zero returncode: {obs.returncode}")
    if not lifecycle_visible:
        findings.append("No task/subagent lifecycle visibility observed")
    if not has_step_latency:
        findings.append("No step latency breakdown observed")

    return DimensionScore(
        name="orchestration_monitoring",
        score=_clamp_score(score),
        passed=clean_returncode and lifecycle_visible and has_step_latency,
        findings=findings,
        metrics={
            "clean_returncode": clean_returncode,
            "returncode": obs.returncode,
            "lifecycle_visible": lifecycle_visible,
            "lifecycle_count": len(lifecycle_items) + len(stage_matches),
            "step_latency_count": len(step_latencies),
            "step_latencies_ms": step_latencies,
        },
    )


def evaluate_probe(
    obs: ProbeObservation,
    latency_threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS,
    weights: dict[str, float] | None = None,
) -> ProbeEvaluation:
    """Evaluate one CLI probe across all 5 UX & orchestration dimensions."""
    dim_weights = weights or DEFAULT_DIMENSION_WEIGHTS
    dimensions = {
        "latency": evaluate_latency(obs, default_threshold_ms=latency_threshold_ms),
        "output_hygiene": evaluate_output_hygiene(obs),
        "request_info_processing": evaluate_request_info_processing(obs),
        "response_quality": evaluate_response_quality(obs),
        "orchestration_monitoring": evaluate_orchestration_monitoring(obs),
    }

    total_weight = sum(dim_weights.get(d, 0.0) for d in DIMENSIONS) or 1.0
    overall = _clamp_score(
        sum(dimensions[d].score * dim_weights.get(d, 0.0) for d in DIMENSIONS)
        / total_weight
    )
    return ProbeEvaluation(
        name=obs.name,
        command=obs.command,
        returncode=obs.returncode,
        duration_ms=round(obs.duration_ms, 2),
        overall_score=overall,
        dimensions=dimensions,
    )


def evaluate_suite(
    observations: list[ProbeObservation],
    mode: str,
    latency_threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS,
    weights: dict[str, float] | None = None,
    min_score: float = 0.0,
) -> UsabilityReport:
    """Evaluate a list of CLI probes and produce an aggregated UsabilityReport."""
    dim_weights = weights or DEFAULT_DIMENSION_WEIGHTS
    evaluations = [
        evaluate_probe(obs, latency_threshold_ms=latency_threshold_ms, weights=dim_weights)
        for obs in observations
    ]

    dimension_scores: dict[str, float] = {}
    for dim in DIMENSIONS:
        if evaluations:
            dimension_scores[dim] = _clamp_score(
                sum(ev.dimensions[dim].score for ev in evaluations) / len(evaluations)
            )
        else:
            dimension_scores[dim] = 0.0

    total_weight = sum(dim_weights.get(d, 0.0) for d in DIMENSIONS) or 1.0
    overall_score = _clamp_score(
        sum(dimension_scores[d] * dim_weights.get(d, 0.0) for d in DIMENSIONS)
        / total_weight
    )

    passed = bool(evaluations) and overall_score >= min_score
    return UsabilityReport(
        mode=mode,
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_score=overall_score,
        dimension_scores=dimension_scores,
        passed=passed,
        probes=evaluations,
        summary={
            "probe_count": len(evaluations),
            "clean_returncodes": sum(1 for ev in evaluations if ev.returncode == 0),
            "min_score_gate": min_score,
            "latency_threshold_ms": latency_threshold_ms,
        },
    )


def build_offline_mock_observations() -> list[ProbeObservation]:
    """Build deterministic offline mock probes covering all 5 UX & orchestration dimensions."""
    run_payload = {
        "success": True,
        "session_id": "mock_session_001",
        "routing": {"mode": "research", "query": "summarize cli architecture"},
        "flags": {
            "model": DEFAULT_LOW_COST_MODEL,
            "depth": "quick",
            "autopilot": True,
            "format": "json",
        },
        "tasks": [
            {"id": "classify", "status": "completed", "duration_ms": 18.4},
            {"id": "planner", "status": "completed", "duration_ms": 42.1},
            {"id": "synthesize", "status": "completed", "duration_ms": 95.0},
        ],
        "subagents": [
            {"name": "cli_inspector", "status": "completed", "duration_ms": 63.5}
        ],
        "metadata": {
            "step_latencies_ms": {
                "bootstrap": 12.2,
                "classify": 18.4,
                "planner": 42.1,
                "subagent_delegate": 63.5,
                "synthesize": 95.0,
            }
        },
        "final_synthesis": {
            "content": "SparkleForge CLI routes requests via src.cli.entry with quiet startup logging."
        },
    }

    return [
        ProbeObservation(
            name="cold_start_help",
            command=["sparkleforge", "--help"],
            returncode=0,
            duration_ms=145.0,
            cold_start_ms=145.0,
            stdout=(
                "usage: entry.py [-h] {run,work,session,health,tools,cli,report} ...\n"
                "SparkleForge - Autonomous Multi-Agent Research System\n"
                "[OK] cli_bootstrap (12.4 ms)\n"
            ),
            stderr="",
            supported_flags=list(REQUIRED_RUN_FLAGS),
            orchestration={
                "stages": [{"name": "cli_bootstrap", "status": "ok", "duration_ms": 12.4}],
                "step_latencies_ms": {"cli_bootstrap": 12.4},
            },
        ),
        ProbeObservation(
            name="quiet_boot_health",
            command=["sparkleforge", "health"],
            returncode=0,
            duration_ms=320.0,
            stdout=(
                "Running system health check...\n"
                "[OK] docker: ok\n"
                "[OK] sandbox_write: ok\n"
                "[OK] System is healthy\n"
                "[OK] config (1.4 ms)\n"
                "[OK] database (3.2 ms)\n"
            ),
            stderr="",
            orchestration={
                "stages": [
                    {"name": "config", "status": "ok", "duration_ms": 1.4},
                    {"name": "database", "status": "ok", "duration_ms": 3.2},
                ],
                "step_latencies_ms": {"config": 1.4, "database": 3.2},
            },
        ),
        ProbeObservation(
            name="run_flag_routing_json",
            command=[
                "sparkleforge",
                "run",
                "summarize cli architecture",
                "--model",
                DEFAULT_LOW_COST_MODEL,
                "--depth",
                "quick",
                "--autopilot",
                "true",
                "--format",
                "json",
            ],
            returncode=0,
            duration_ms=480.0,
            stdout=json.dumps(run_payload, indent=2),
            stderr="",
            query="summarize cli architecture",
            routed_target="research",
            flags={
                "model": DEFAULT_LOW_COST_MODEL,
                "depth": "quick",
                "autopilot": "true",
                "format": "json",
            },
            applied_flags={
                "model": DEFAULT_LOW_COST_MODEL,
                "depth": "quick",
                "autopilot": True,
                "format": "json",
            },
            expected_format="json",
            structured_output=run_payload,
            orchestration={
                "tasks": run_payload["tasks"],
                "subagents": run_payload["subagents"],
                "step_latencies_ms": run_payload["metadata"]["step_latencies_ms"],
            },
        ),
    ]


def _extract_flags_from_help(help_text: str) -> list[str]:
    found: list[str] = []
    for flag in REQUIRED_RUN_FLAGS:
        if flag in help_text:
            found.append(flag)
    return found


def _run_subprocess_probe(
    name: str,
    args: list[str],
    *,
    timeout_s: float = DEFAULT_LIVE_TIMEOUT_S,
    env_overrides: dict[str, str] | None = None,
    query: str | None = None,
    flags: dict[str, Any] | None = None,
    expected_format: str | None = None,
    output_file: Path | None = None,
    extract_run_flags: bool = False,
) -> ProbeObservation:
    cmd = [sys.executable, "-m", "src.cli.entry", *args]
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(REPO_ROOT)
    if env_overrides:
        env.update(env_overrides)

    start = time.perf_counter()
    timed_out = False
    stdout = ""
    stderr = ""
    returncode = 0

    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else str(exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else str(exc.stderr or "")
        )

    duration_ms = (time.perf_counter() - start) * 1000.0
    output_file_content: str | None = None
    structured_output: Any | None = None
    if output_file is not None and output_file.exists():
        output_file_content = output_file.read_text(encoding="utf-8", errors="replace")
        if (expected_format or "").lower() == "json":
            structured_output = _try_parse_json(output_file_content)

    supported_flags = _extract_flags_from_help(stdout) if extract_run_flags else None
    step_latencies = extract_step_latencies_ms({}, structured_output, f"{stdout}\n{stderr}")
    stages = [
        {"name": stage_name, "status": "ok", "duration_ms": ms}
        for stage_name, ms in step_latencies.items()
    ]

    return ProbeObservation(
        name=name,
        command=cmd,
        returncode=returncode,
        duration_ms=duration_ms,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        query=query,
        flags=flags or {},
        supported_flags=supported_flags,
        expected_format=expected_format,
        output_file_content=output_file_content,
        structured_output=structured_output,
        orchestration={"stages": stages, "step_latencies_ms": step_latencies},
    )


def collect_live_observations(
    *,
    timeout_s: float = DEFAULT_LIVE_TIMEOUT_S,
    live_query: str | None = None,
    model: str = DEFAULT_LOW_COST_MODEL,
) -> list[ProbeObservation]:
    """Execute real SparkleForge CLI subprocess probes and collect telemetry."""
    observations = [
        _run_subprocess_probe(
            "cli_help",
            ["--help"],
            timeout_s=timeout_s,
        ),
        _run_subprocess_probe(
            "cli_run_help_flags",
            ["run", "--help"],
            timeout_s=timeout_s,
            extract_run_flags=True,
        ),
        _run_subprocess_probe(
            "cli_health_quiet_boot",
            ["health"],
            timeout_s=timeout_s,
            env_overrides={
                "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY") or "probe_google_key",
                "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY") or "probe_gemini_key",
            },
        ),
        _run_subprocess_probe(
            "cli_debug_bootstrap",
            ["--debug-bootstrap"],
            timeout_s=timeout_s,
        ),
    ]

    if live_query:
        with tempfile.TemporaryDirectory(prefix="cli_ux_qa_") as tmpdir:
            out_path = Path(tmpdir) / "live_result.json"
            observations.append(
                _run_subprocess_probe(
                    "cli_live_run_query",
                    [
                        "run",
                        live_query,
                        "--model",
                        model,
                        "--depth",
                        "quick",
                        "--autopilot",
                        "true",
                        "--format",
                        "json",
                        "--output",
                        str(out_path),
                    ],
                    timeout_s=max(timeout_s, 60.0),
                    query=live_query,
                    flags={
                        "model": model,
                        "depth": "quick",
                        "autopilot": "true",
                        "format": "json",
                    },
                    expected_format="json",
                    output_file=out_path,
                )
            )

    return observations


def run_usability_qa(
    *,
    mode: str = "offline",
    latency_threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS,
    min_score: float = 0.0,
    timeout_s: float = DEFAULT_LIVE_TIMEOUT_S,
    live_query: str | None = None,
    model: str = DEFAULT_LOW_COST_MODEL,
) -> UsabilityReport:
    """Run CLI usability & orchestration QA evaluation in offline/mock or live mode."""
    normalized_mode = "offline" if mode in ("offline", "mock") else "live"
    if normalized_mode == "offline":
        observations = build_offline_mock_observations()
    else:
        observations = collect_live_observations(
            timeout_s=timeout_s,
            live_query=live_query,
            model=model,
        )

    return evaluate_suite(
        observations,
        mode=normalized_mode,
        latency_threshold_ms=latency_threshold_ms,
        min_score=min_score,
    )


def _format_text_summary(report: UsabilityReport) -> str:
    lines = [
        f"SparkleForge CLI Usability & Orchestration QA ({report.mode})",
        f"Overall Score: {report.overall_score:.3f} (passed={report.passed})",
        "Dimension Scores:",
    ]
    for dim in DIMENSIONS:
        lines.append(f"  - {dim}: {report.dimension_scores.get(dim, 0.0):.3f}")
    lines.append("Probes:")
    for probe in report.probes:
        lines.append(
            f"  - {probe.name}: score={probe.overall_score:.3f}, "
            f"rc={probe.returncode}, duration={probe.duration_ms:.1f}ms"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the usability & orchestration QA harness."""
    parser = argparse.ArgumentParser(
        description="Evaluate SparkleForge CLI usability and orchestration quality."
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "mock", "live"],
        default="offline",
        help="Execution mode: offline/mock (deterministic) or live (real subprocesses).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full JSON scorecard report.",
    )
    parser.add_argument(
        "--latency-threshold-ms",
        type=float,
        default=DEFAULT_LATENCY_THRESHOLD_MS,
        help="Latency pass threshold in milliseconds (default: 5000).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum required overall score in [0.0, 1.0] (default: 0.0).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_LIVE_TIMEOUT_S,
        help="Per-probe subprocess timeout in seconds for live mode (default: 30).",
    )
    parser.add_argument(
        "--live-query",
        default=None,
        help="Optional query to execute in live mode via `run` with --model/--depth/--autopilot/--format.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LOW_COST_MODEL,
        help="Low-cost model identifier used when --live-query is supplied.",
    )
    args = parser.parse_args(argv)

    report = run_usability_qa(
        mode=args.mode,
        latency_threshold_ms=args.latency_threshold_ms,
        min_score=args.min_score,
        timeout_s=args.timeout_s,
        live_query=args.live_query,
        model=args.model,
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(_format_text_summary(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
