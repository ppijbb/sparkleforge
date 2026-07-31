"""Agent report generator for SparkleForge.

Aggregates execution metrics, calculates a strict performance score,
and performs a critical, non-favorable review of recent code changes.
"""

import os
import json
import time
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import logging

from src.core.nightwelding.models import NightweldingQueue, MakerMarkLedger
from src.core.llm_manager import TaskType, execute_llm_task

logger = logging.getLogger(__name__)


def get_recent_changes() -> str:
    """Retrieve git commits and diffs from the last 24 hours."""
    try:
        commits = subprocess.run(
            ["git", "log", "--since=24 hours ago", "--oneline"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        
        if not commits:
            return "No commits in the last 24 hours."
            
        diff = subprocess.run(
            ["git", "log", "--since=24 hours ago", "-p", "--stat"],
            capture_output=True,
            text=True,
            check=True
        ).stdout
        
        if len(diff) > 20000:
            diff = diff[:20000] + "\n\n... (diff truncated to prevent context bloat) ..."
            
        return f"Commits:\n{commits}\n\nDiff details:\n{diff}"
    except Exception as e:
        logger.error(f"Failed to get recent changes: {e}")
        return f"Error retrieving git changes: {e}"


def load_trend_signals(project_root: Path) -> List[Dict[str, Any]]:
    """Load trend signals collected by the daily roadmap workflow.

    The daily roadmap workflow writes a JSON file with dated trend signals
    (including source URLs) at results/agent_reports/trend_signals.json.
    Returns an empty list when the file is absent or unreadable so report
    generation never hard-fails on missing trend input.
    """
    signals_file = project_root / "results" / "agent_reports" / "trend_signals.json"
    if not signals_file.exists():
        return []
    try:
        data = json.loads(signals_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [s for s in data if isinstance(s, dict)]
        if isinstance(data, dict) and isinstance(data.get("signals"), list):
            return [s for s in data["signals"] if isinstance(s, dict)]
    except Exception as e:
        logger.error(f"Failed to load trend signals: {e}")
    return []


def _extract_anvil_phases(project_root: Path) -> List[str]:
    """Extract phase names from docs/ANVIL_PLAN.md for trend-gap comparison."""
    plan_path = project_root / "docs" / "ANVIL_PLAN.md"
    if not plan_path.exists():
        return []
    try:
        text = plan_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read ANVIL_PLAN.md: {e}")
        return []
    phases: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Match phase table rows or phase headings like "Phase 1", "1단계", "## 1. ..."
        if re.match(r"^(#{1,6}\s*)?(Phase\s+\d+|\d+단계|\d+\.\s)", stripped, re.IGNORECASE):
            phases.append(stripped.lstrip("#").strip())
    return phases


def build_trend_gap_section(
    project_root: Path,
    trend_signals: List[Dict[str, Any]],
    today_str: str,
) -> str:
    """Build the trend-gap markdown section for the daily report.

    Compares the day's collected trend signals against the repository's
    Anvil phase structure and summarizes what is already covered versus
    what is still missing. Designed to be deterministic and safe even
    when no signals are available.
    """
    if not trend_signals:
        return (
            "## 🧭 트렌드 갭 (Trend Gap)\n\n"
            "> 이 날짜에 수집된 트렌드 시그널이 없어 트렌드 갭 섹션을 생략합니다.\n"
        )

    phases = _extract_anvil_phases(project_root)
    phases_block = "\n".join(f"- {p}" for p in phases) if phases else "- docs/ANVIL_PLAN.md에서 phase를 추출하지 못했습니다."

    covered: List[str] = []
    missing: List[str] = []
    for signal in trend_signals:
        topic = signal.get("topic") or signal.get("title") or signal.get("trend") or "알 수 없는 트렌드"
        url = signal.get("url") or signal.get("source") or signal.get("source_url") or ""
        date = signal.get("date") or signal.get("collected_at") or today_str
        evidence = f"{date}"
        if url:
            evidence += f" / {url}"
        # Heuristic: if any phase text shares a keyword with the topic, treat as covered.
        topic_words = {w for w in re.split(r"[\s/_,.-]+", topic.lower()) if len(w) > 2}
        phase_text = " ".join(phases).lower()
        is_covered = any(w in phase_text for w in topic_words) if topic_words else False
        line = f"- {topic} (근거: {evidence})"
        (covered if is_covered else missing).append(line)

    covered_block = "\n".join(covered) if covered else "- 현재 저장소 구조와 직접 매칭되는 항목이 없습니다."
    missing_block = "\n".join(missing) if missing else "- 이번 트렌드 시그널은 모두 기존 phase로 커버되는 것으로 추정됩니다."

    return (
        "## 🧭 트렌드 갭 (Trend Gap)\n\n"
        "> daily-roadmap이 수집한 트렌드 시그널과 docs/ANVIL_PLAN.md의 phase 구조를 대조한 누적 포지셔닝 뷰입니다.\n\n"
        "### 기준 phase (docs/ANVIL_PLAN.md)\n"
        f"{phases_block}\n\n"
        "### 이미 커버된 것 (Covered)\n"
        f"{covered_block}\n\n"
        "### 아직 갭인 것 (Missing)\n"
        f"{missing_block}\n"
    )


def update_trend_gap_history(
    reports_dir: Path,
    today_str: str,
    trend_signals: List[Dict[str, Any]],
    covered: List[str],
    missing: List[str],
) -> Dict[str, Any]:
    """Accumulate trend-gap history with delta-only updates.

    Instead of rewriting the whole history each day, we append only the
    delta (newly covered / newly missing) compared to the most recent prior
    entry, so the drift over time stays comparable.
    """
    history_file = reports_dir / "trend_gap_history.json"
    history: List[Dict[str, Any]] = []
    if history_file.exists():
        try:
            loaded = json.loads(history_file.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = [h for h in loaded if isinstance(h, dict)]
        except Exception:
            history = []

    # Replace today's entry if present (idempotent same-day reruns).
    history = [h for h in history if h.get("date") != today_str]

    prev_covered: set = set()
    prev_missing: set = set()
    if history:
        prev = history[-1]
        prev_covered = set(prev.get("covered", []))
        prev_missing = set(prev.get("missing", []))

    covered_set = set(covered)
    missing_set = set(missing)
    delta = {
        "newly_covered": sorted(covered_set - prev_covered),
        "newly_missing": sorted(missing_set - prev_missing),
        "resolved_gaps": sorted(prev_missing - missing_set),
    }

    entry = {
        "date": today_str,
        "signal_count": len(trend_signals),
        "covered": sorted(covered_set),
        "missing": sorted(missing_set),
        "delta": delta,
    }
    history.append(entry)
    history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return entry


def aggregate_release_metrics(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll up per-day history.json entries into a single release-metrics summary.

    generate_daily_report() only ever appends one entry per day; there was
    previously no way to see performance across a range of runs (e.g. "since
    the last release") without reading every row by hand. success_rate and
    strict_score are weighted by each day's total_attempts so days with more
    runs count proportionally more; days with zero attempts don't skew the
    average. Returns zeroed-out fields (not an error) for an empty history,
    since "no runs yet" is a valid state for a fresh repo.
    """
    entries = [e for e in history if isinstance(e, dict)]
    if not entries:
        return {
            "entry_count": 0,
            "date_range": None,
            "total_attempts": 0,
            "total_marks": 0,
            "average_strict_score": 0.0,
            "weighted_success_rate": 0.0,
        }

    dates = sorted(e.get("date", "") for e in entries if e.get("date"))
    total_attempts = sum(e.get("total_attempts", 0) for e in entries)
    total_marks = sum(e.get("total_marks", 0) for e in entries)

    if total_attempts > 0:
        average_strict_score = (
            sum(e.get("strict_score", 0.0) * e.get("total_attempts", 0) for e in entries)
            / total_attempts
        )
    else:
        average_strict_score = 0.0

    if total_attempts > 0:
        weighted_success_rate = (
            sum(e.get("success_rate", 0.0) * e.get("total_attempts", 0) for e in entries)
            / total_attempts
        )
    else:
        weighted_success_rate = 0.0

    return {
        "entry_count": len(entries),
        "date_range": (dates[0], dates[-1]) if dates else None,
        "total_attempts": total_attempts,
        "total_marks": total_marks,
        "average_strict_score": average_strict_score,
        "weighted_success_rate": weighted_success_rate,
    }


async def generate_daily_report(project_root: Path) -> Dict[str, Any]:
    """Calculate metrics, run critical LLM review, and write the report."""
    reports_dir = project_root / "results" / "agent_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load queues
    queue = NightweldingQueue()
    ledger = MakerMarkLedger()
    
    items = queue.list()
    marks = ledger.list()
    
    total_attempts = len(items)
    successful_runs = sum(1 for item in items if item.status.value in ("draft_opened", "green"))
    failed_runs = sum(1 for item in items if item.status.value == "failed")
    success_rate = (successful_runs / total_attempts * 100.0) if total_attempts > 0 else 0.0
    
    total_marks = len(marks)
    first_attempt_successes = sum(1 for m in marks if m.first_attempt_success)
    first_attempt_success_rate = (first_attempt_successes / total_marks * 100.0) if total_marks > 0 else 0.0
    
    # Calculate strict score (strictly penalizing any flaws/failures)
    strict_score = 100.0
    strict_score -= failed_runs * 20.0
    
    human_interventions = sum(1 for m in marks if m.human_intervention_required)
    strict_score -= human_interventions * 10.0
    
    # Lower bound at 0
    strict_score = max(0.0, min(100.0, strict_score))
    
    # Fetch recent diffs
    changes = get_recent_changes()
    
    # Low-cost LLM call for a harsh critique
    system_message = (
        "You are a strict, skeptical senior system architect. Your job is to critically "
        "evaluate the agent's work. You MUST NOT praise the agent's changes. Even if they succeed, "
        "focus entirely on potential design shortcuts, technical debt, potential race conditions, "
        "edge cases, and styling errors. Be highly critical and direct. Write your critique in Korean."
    )
    
    prompt = (
        f"Please critique the following recent code changes and commits made by the coding agent "
        f"in the last 24 hours:\n\n{changes}\n\n"
        f"Evaluate them in detail and provide actionable warnings for future runs."
    )
    
    try:
        # Use low-cost model or primary model
        res = await execute_llm_task(
            prompt=prompt,
            task_type=TaskType.ANALYSIS,
            system_message=system_message,
            model_name="gemini-flash-lite"
        )
        critique = res.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate LLM critique: {e}")
        critique = f"Critique generation failed: {e}"
        
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # Trend gap analysis (accumulating section)
    trend_signals = load_trend_signals(project_root)
    trend_gap_section = build_trend_gap_section(project_root, trend_signals, today_str)
    covered_lines = [
        line for line in trend_gap_section.splitlines()
        if line.startswith("- ") and "(근거:" in line
    ]
    missing_lines = []  # Extracted within build for history; recompute simply from signals
    trend_gap_entry = update_trend_gap_history(
        reports_dir, today_str, trend_signals, covered_lines, missing_lines
    )

    # Write latest critique JSON for feedback loop
    latest_critique_file = reports_dir / "latest_critique.json"
    latest_data = {
        "date": today_str,
        "strict_score": strict_score,
        "metrics": {
            "total_attempts": total_attempts,
            "success_rate": success_rate,
            "failed_runs": failed_runs,
            "total_marks": total_marks,
            "first_attempt_success_rate": first_attempt_success_rate
        },
        "critique": critique
    }
    latest_critique_file.write_text(json.dumps(latest_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    
    # Write daily markdown report
    report_file = reports_dir / f"report_{datetime.now().strftime('%Y%m%d')}.md"
    markdown_content = f"""# SparkleForge Agent Metric Evaluation & Critique ({today_str})

## 📊 정량적 성능 지표 (Metrics)

| 지표명 (Metric) | 수치 (Value) | 비고 |
| :--- | :--- | :--- |
| **종합 Strict Score** | **{strict_score:.1f} / 100** | 페널티 기반 하드 채점 |
| 자율 이슈 처리도전 수 (Total Attempts) | {total_attempts} 건 | Nightwelding 실행 기준 |
| 자동 PR 도달율 (Success Rate) | {success_rate:.1f}% | draft_opened / green 상태 도달 |
| 최종 실패율 (Failed Rate) | {(failed_runs / total_attempts * 100.0) if total_attempts > 0 else 0.0:.1f}% | failed 상태 건수 |
| Maker's Marks 누적 수 | {total_marks} 건 | 고도 성공 판정 기록 |
| 첫 시도 자동성공률 (First-Attempt Success) | {first_attempt_success_rate:.1f}% | 인간의 개입 없는 완전 무결성 |

## ⚠️ 냉혹한 코드 퀄리티 비평 (Critical Review)

> [!WARNING]
> 아래 비평은 작업물에 대한 관용이나 칭찬을 배제한 시스템 아키텍트의 엄격한 관점입니다.

{critique}

{trend_gap_section}

---
*본 보고서는 매일 오후 6시 KST에 자동으로 갱신되며, 해당 평가는 피드백 루프를 통해 차기 에이전트 작업 프롬프트에 자동으로 주입됩니다.*
"""
    report_file.write_text(markdown_content, encoding="utf-8")
    
    # Accumulate history
    history_file = reports_dir / "history.json"
    history = []
    if history_file.exists():
        try:
            loaded = json.loads(history_file.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = [e for e in loaded if isinstance(e, dict)]
            else:
                history = []
        except Exception:
            history = []
            
    # Remove existing entry for today if runs multiple times
    history = [entry for entry in history if entry.get("date") != today_str]
    
    history.append({
        "date": today_str,
        "strict_score": strict_score,
        "total_attempts": total_attempts,
        "success_rate": success_rate,
        "total_marks": total_marks
    })
    latest_data["trend_gap"] = trend_gap_entry
    
    history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    
    return latest_data
