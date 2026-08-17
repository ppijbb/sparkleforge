"""Shared grading primitives for the scenario eval harness (run_scenarios.py).

Each scenario fixture module (tests/benchmark/scenario_fixtures/*.py) implements
its own `grade()` using these primitives, since "did it actually do the thing"
is scenario-specific. This module only holds the reusable, generic pieces:
directory snapshots/diffs, text-content checks, and the capped LLM-judge call.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from src.core.anvil.dynamic_checklist_generator import Checklist, ChecklistItem

logger = logging.getLogger(__name__)

# check name -> (score 0..1, human-readable reason)
GradeResult = Tuple[float, str]


def rubric_from_context(ctx: Dict[str, Any], fallback: str) -> str:
    """Return the scenario YAML rubric when present, otherwise a fixture fallback."""
    rubric = str(ctx.get("judge_rubric") or "").strip()
    return rubric or fallback


def snapshot_tree(root: Path) -> Dict[str, str]:
    """Map every regular file under root (relative path -> sha256 hex digest)."""
    snapshot: Dict[str, str] = {}
    if not root.exists():
        return snapshot
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        snapshot[path.relative_to(root).as_posix()] = digest
    return snapshot


# The agent runtime itself (PersistentMemory, audit logs, etc.) writes cwd-relative
# artifacts like "data/semantic_memory.db" as a side effect of just running — these
# have nothing to do with any scenario's task and would otherwise be miscounted as
# "new files the agent produced" in every scenario that diffs the workspace tree.
RUNTIME_ARTIFACT_PREFIXES = ("data/", "logs/", "storage/", ".sparkleforge/")


def is_runtime_artifact(relpath: str) -> bool:
    return relpath.startswith(RUNTIME_ARTIFACT_PREFIXES)


def new_files(before: Dict[str, str], after: Dict[str, str]) -> list[str]:
    """Relative paths present after but not before (created or overwritten-new).

    Excludes known agent-runtime artifacts (see RUNTIME_ARTIFACT_PREFIXES) that
    are a side effect of the process running, not of the scenario being solved.
    """
    return [p for p in after if p not in before and not is_runtime_artifact(p)]


def removed_files(before: Dict[str, str], after: Dict[str, str]) -> list[str]:
    """Relative paths present before but gone after."""
    return [p for p in before if p not in after]


def modified_files(before: Dict[str, str], after: Dict[str, str]) -> list[str]:
    """Relative paths present in both snapshots whose content hash changed.

    Excludes known agent-runtime artifacts (see RUNTIME_ARTIFACT_PREFIXES) so
    side-effect files from the process running aren't miscounted as scenario
    work. Detecting modifications is required for scenarios that expect an
    existing file to be edited in place rather than created or deleted.
    """
    common = before.keys() & after.keys()
    return [
        p for p in common if before[p] != after[p] and not is_runtime_artifact(p)
    ]


def checklist_to_weights(checklist: Checklist) -> Dict[str, float]:
    """Convert an approved Anvil Checklist (src/core/anvil) into a scenario weights dict.

    Part of the Mu-3 wiring: RequestAnalyzer/DynamicChecklistGenerator output
    (see src.core.anvil.checklist_proposal) becomes the same weights shape
    run_scenario() already expects from scenario YAML.
    """
    return {item.item_id: item.weight for item in checklist.items}


def checklist_item_check(item: ChecklistItem, before: Dict[str, str], after: Dict[str, str]) -> GradeResult:
    """Quantitative file-diff/existence check for one agent-proposed ChecklistItem.

    ChecklistItem carries no structured target path, so this is the generic
    signal a diff-based check can give without re-parsing free text back into a
    path: did the workspace tree change at all while this item was open,
    including files that were modified in place (not just created/removed).
    Scenarios that need a stronger check still write a scenario-specific one.
    """
    added = new_files(before, after)
    removed = removed_files(before, after)
    modified = modified_files(before, after)
    changed = bool(added or removed or modified)
    reason = (
        f"workspace changed: +{len(added)}/-{len(removed)}/~{len(modified)} file(s)"
        if changed
        else "no file changes observed for this checklist item"
    )
    return (1.0 if changed else 0.0), reason


def unchanged(before: Dict[str, str], after: Dict[str, str], relpath: str) -> bool:
    """True if relpath exists in both snapshots with an identical hash."""
    return relpath in before and relpath in after and before[relpath] == after[relpath]


def read_text_safe(path: Path, max_bytes: int = 200_000) -> str:
    """Best-effort text read; returns '' for missing/binary/oversized files."""
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def concat_new_file_text(workspace: Path, relpaths: Iterable[str]) -> str:
    """Concatenate the text content of a set of newly-created files, for keyword checks."""
    chunks = []
    for rel in relpaths:
        text = read_text_safe(workspace / rel)
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def keyword_hit(text: str, keywords: Iterable[str]) -> bool:
    """Case-insensitive substring match against any keyword."""
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in keywords)


JUDGE_TIMEOUT_S = 45.0

# Deterministic rule-based fallback used when every LLM provider is unavailable
# (no API key, network timeout, quota exhaustion). This keeps CI scenario grading
# conclusive instead of marking every judge_* check as inconclusive and skipping
# the run. The heuristic is intentionally conservative: it only rewards evidence
# that the agent actually addressed the rubric, so a no-op transcript still
# scores 0.0 with a concrete reason rather than an inconclusive skip.
RULE_BASED_JUDGE_KEYWORDS = {
    "report": ["report", "summary", "정리", "리포트", "요약", "cleanup", "removed", "deleted"],
    "summary": ["summary", "요약", "recap", "worklog", "작업", "로그"],
    "explanation": ["receipt", "영수증", "2025", "trip", "출장", "collected", "정리"],
    "quality": ["setup", "세팅", "build", "환경", "install", "dependency", "의존"],
    "risk": ["risk", "위험", "secret", "비밀", "malicious", "악성", "quarantine", "격리", "mitigat"],
}


def _rule_based_judge(rubric: str, transcript: str, context: str = "") -> GradeResult:
    """Deterministic fallback judge used when all LLM providers are unavailable.

    Scores on simple evidence signals so CI grading stays conclusive without an
    LLM. Returns a genuine (non-inconclusive) 0.0-1.0 score with a reason that
    explains what evidence was found, never an INCONCLUSIVE_MARKER reason.
    """
    text = f"{transcript}\n{context}".lower()
    rubric_lower = rubric.lower()
    hits: list[str] = []
    matched_groups = 0
    for label, keywords in RULE_BASED_JUDGE_KEYWORDS.items():
        if label in rubric_lower or any(kw.lower() in rubric_lower for kw in keywords):
            if any(kw.lower() in text for kw in keywords):
                matched_groups += 1
                hits.append(label)
    if not transcript.strip():
        return 0.0, "rule-based fallback: empty transcript, nothing to judge"
    if matched_groups == 0:
        return 0.0, "rule-based fallback: no rubric-relevant evidence found in agent output"
    # Partial credit scaled by how many rubric-relevant signal groups matched.
    score = min(1.0, round(0.25 * matched_groups + 0.25, 4))
    return score, f"rule-based fallback judge: matched {matched_groups} signal group(s) ({', '.join(hits)})"


# Marker prefix for judge_score() reasons that mean "the judge itself never
# ran" (no model available, timeout) as opposed to "the judge ran and scored
# the agent's output low". weighted_total() strips this and tracks affected
# checks as inconclusive rather than failed, since a 0.0 from provider quota
# exhaustion was previously silently indistinguishable from a 0.0 the agent
# actually earned, corrupting the overall_score signal.
INCONCLUSIVE_MARKER = "__INCONCLUSIVE__"


# Subprocess failure signatures that indicate the agent runtime failed to
# execute the scenario at all (e.g. all fallback models exhausted). When these
# appear in stdout/stderr, the recorded returncode must be treated as a failure
# even if the subprocess wrapper exited 0, and the run must not be ingested
# into the baseline JSONL as a valid data point.
# Exit code mapping (see issue #879):
#   0 — scenario executed, agent produced output, scoring completed
#   1 — scenario execution failed due to unrecoverable error (model unavailable, retry exhaustion)
#   2 — scenario timed out
#   3 — scenario skipped due to pre-condition failure
# normalize_returncode() maps a wrapper-exited-0 run with an execution-failure
# signature to 1 (unrecoverable execution failure), not a timeout.
EXECUTION_FAILURE_SIGNATURES = (
    "Execution failed",
    "All fallback models failed",
    "No available models",
)


def detect_execution_failure(stdout: str, stderr: str = "") -> bool:
    """Return True if a known execution-failure signature appears in subprocess output."""
    combined = f"{stdout}\n{stderr}"
    return any(sig in combined for sig in EXECUTION_FAILURE_SIGNATURES)


def normalize_returncode(returncode: int, stdout: str, stderr: str = "") -> int:
    """Override a zero returncode to non-zero when an execution failure signature is present."""
    if returncode == 0 and detect_execution_failure(stdout, stderr):
        return 1
    return returncode


async def judge_score(rubric: str, transcript: str, context: str = "") -> GradeResult:
    """Capped-weight LLM-judge fallback for subjective quality checks.

    Never raises and never blocks past JUDGE_TIMEOUT_S. Distinguishes two
    failure classes in the returned reason:
    - the judge never ran at all (no model available, timeout) -> marked
      inconclusive, so callers don't count it as a real 0.0 in aggregates.
    - the judge ran but had nothing worth judging (no rubric configured,
      empty transcript) -> a genuine 0.0, since the agent produced nothing.
    """
    if not rubric:
        return 0.0, "no judge_rubric configured"
    if not transcript.strip():
        return 0.0, "Agent produced no output for the judge to evaluate."

    try:
        return await asyncio.wait_for(_call_judge(rubric, transcript, context), timeout=JUDGE_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("[ScenarioGrading] LLM judge timed out after %ss", JUDGE_TIMEOUT_S)
        score, reason = _rule_based_judge(rubric, transcript, context)
        return score, f"{INCONCLUSIVE_MARKER}{reason} (LLM judge timed out after {JUDGE_TIMEOUT_S}s)"
    except Exception as e:  # noqa: BLE001 - judge must never crash grading
        logger.warning("[ScenarioGrading] LLM judge unavailable: %s", e)
        score, reason = _rule_based_judge(rubric, transcript, context)
        return score, f"{INCONCLUSIVE_MARKER}{reason} (LLM judge unavailable: {e})"


async def _call_judge(rubric: str, transcript: str, context: str) -> GradeResult:
    from src.core.llm_manager import MultiModelOrchestrator, TaskType

    orchestrator = MultiModelOrchestrator()  # type: ignore[call-arg]
    prompt = (
        "You are grading whether an autonomous agent's output satisfies a rubric.\n"
        f"Rubric: {rubric}\n\n"
        f"Context: {context}\n\n"
        f"Agent output/transcript:\n{transcript[:4000]}\n\n"
        "Respond with a single line: SCORE=<0.0-1.0> REASON=<short reason>"
    )
    result = await orchestrator.execute_with_model(prompt=prompt, task_type=TaskType.RESEARCH, max_tokens=256)
    content = getattr(result, "content", "") or ""

    score = 0.0
    reason = content.strip()[:300] or "judge returned no content"
    for line in content.splitlines():
        if "SCORE=" in line:
            try:
                raw = line.split("SCORE=", 1)[1].strip().split()[0]
                score = max(0.0, min(1.0, float(raw)))
            except (ValueError, IndexError):
                pass
            if "REASON=" in line:
                reason = line.split("REASON=", 1)[1].strip()[:300]
    return score, reason


# judge_* checks are the only subjective axis (docs/ANVIL_PLAN.md SS5.3): left
# uncapped, a single judge-API hiccup or harsh score can dominate an otherwise
# passing scenario. Cap their combined share of the total weight.
JUDGE_WEIGHT_CAP_RATIO = 0.3


def _cap_judge_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    judge_names = [n for n in weights if n.startswith("judge_")]
    judge_total = sum(weights[n] for n in judge_names)
    cap = JUDGE_WEIGHT_CAP_RATIO * total
    if not judge_names or total <= 0 or judge_total <= cap:
        return weights
    scale = cap / judge_total
    return {n: (w * scale if n in judge_names else w) for n, w in weights.items()}


def weighted_total(scores: Dict[str, GradeResult], weights: Dict[str, float]) -> Dict[str, Any]:
    """Combine named sub-scores with their configured weights into a scenario total.

    `total` keeps the original, conservative semantics: an inconclusive check
    (see INCONCLUSIVE_MARKER) still contributes 0 to it, same as before this
    distinction existed. `adjusted_total` renormalizes weights over only the
    checks that actually ran, so a scenario isn't penalized for infra outages
    it had no control over -- that's the number that should drive trend
    tracking and regression comparisons. `adjusted_total` is None only when
    every single check in the scenario was inconclusive.

    Weights for any check named `judge_*` are capped at JUDGE_WEIGHT_CAP_RATIO
    of the total before scoring (see _cap_judge_weights).
    """
    weights = _cap_judge_weights(weights)
    total = 0.0
    conclusive_weight = 0.0
    conclusive_contribution = 0.0
    breakdown = {}
    for name, weight in weights.items():
        score, reason = scores.get(name, (0.0, f"check '{name}' did not run"))
        inconclusive = reason.startswith(INCONCLUSIVE_MARKER)
        if inconclusive:
            reason = reason[len(INCONCLUSIVE_MARKER):]
        else:
            conclusive_weight += weight
            conclusive_contribution += score * weight
        total += score * weight
        breakdown[name] = {
            "score": score,
            "weight": weight,
            "reason": reason,
            "inconclusive": inconclusive,
        }
    adjusted_total = round(conclusive_contribution / conclusive_weight, 4) if conclusive_weight > 0 else None
    return {"total": round(total, 4), "adjusted_total": adjusted_total, "breakdown": breakdown}
