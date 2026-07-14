"""Agent report generator for SparkleForge.

Aggregates execution metrics, calculates a strict performance score,
and performs a critical, non-favorable review of recent code changes.
"""

import os
import json
import time
import subprocess
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
    latest_critique_file.write_text(json.dumps(latest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    
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

---
*본 보고서는 매일 오후 6시 KST에 자동으로 갱신되며, 해당 평가는 피드백 루프를 통해 차기 에이전트 작업 프롬프트에 자동으로 주입됩니다.*
"""
    report_file.write_text(markdown_content, encoding="utf-8")
    
    # Accumulate history
    history_file = reports_dir / "history.json"
    history = []
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))
        except Exception:
            pass
            
    # Remove existing entry for today if runs multiple times
    history = [entry for entry in history if entry.get("date") != today_str]
    
    history.append({
        "date": today_str,
        "strict_score": strict_score,
        "total_attempts": total_attempts,
        "success_rate": success_rate,
        "total_marks": total_marks
    })
    
    history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    
    return latest_data
