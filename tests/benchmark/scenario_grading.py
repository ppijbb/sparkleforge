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
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

logger = logging.getLogger(__name__)

# check name -> (score 0..1, human-readable reason)
GradeResult = Tuple[float, str]


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


async def judge_score(rubric: str, transcript: str, context: str = "") -> GradeResult:
    """Capped-weight LLM-judge fallback for subjective quality checks.

    Never raises and never blocks past JUDGE_TIMEOUT_S: on any failure (no API
    key, model error, timeout) this returns (0.0, reason) so a missing/slow
    judge call degrades the score instead of hanging or crashing the whole
    scenario run.
    """
    if not rubric:
        return 0.0, "no judge_rubric configured"
    if not transcript.strip():
        return 0.0, "empty transcript, nothing to judge"

    try:
        return await asyncio.wait_for(_call_judge(rubric, transcript, context), timeout=JUDGE_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("[ScenarioGrading] LLM judge timed out after %ss", JUDGE_TIMEOUT_S)
        return 0.0, f"judge timed out after {JUDGE_TIMEOUT_S}s"
    except Exception as e:  # noqa: BLE001 - judge must never crash grading
        logger.warning("[ScenarioGrading] LLM judge unavailable: %s", e)
        return 0.0, f"judge unavailable: {e}"


async def _call_judge(rubric: str, transcript: str, context: str) -> GradeResult:
    from src.core.llm_manager import MultiModelOrchestrator, TaskType

    orchestrator = MultiModelOrchestrator()
    prompt = (
        "You are grading whether an autonomous agent's output satisfies a rubric.\n"
        f"Rubric: {rubric}\n\n"
        f"Context: {context}\n\n"
        f"Agent output/transcript:\n{transcript[:4000]}\n\n"
        "Respond with a single line: SCORE=<0.0-1.0> REASON=<short reason>"
    )
    result = await orchestrator.execute_with_model(prompt=prompt, task_type=TaskType.RESEARCH)
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


def weighted_total(scores: Dict[str, GradeResult], weights: Dict[str, float]) -> Dict[str, Any]:
    """Combine named sub-scores with their configured weights into a scenario total."""
    total = 0.0
    breakdown = {}
    for name, weight in weights.items():
        score, reason = scores.get(name, (0.0, f"check '{name}' did not run"))
        contribution = score * weight
        total += contribution
        breakdown[name] = {"score": score, "weight": weight, "reason": reason}
    return {"total": round(total, 4), "breakdown": breakdown}
