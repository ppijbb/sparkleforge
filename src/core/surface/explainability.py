"""
explainability.py — "Why did the AI do this?" view based on the action audit journal.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExplanationReport:
    entry_id: str
    action: str
    agent_id: str
    timestamp: float
    risk_level: str
    outcome: str
    reasoning: str           # Human-readable explanation
    evidence: List[str]      # Supporting facts / journal trail
    recommendations: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"Action:      {self.action}",
            f"Agent:       {self.agent_id}",
            f"Timestamp:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            f"Risk Level:  {self.risk_level}",
            f"Outcome:     {self.outcome}",
            "",
            "Why did the AI do this?",
            f"  {self.reasoning}",
            "",
            "Evidence:",
        ]
        for e in self.evidence:
            lines.append(f"  • {e}")
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for r in self.recommendations:
                lines.append(f"  → {r}")
        return "\n".join(lines)


# Risk-level to plain-text explanation templates
_RISK_EXPLANATIONS: Dict[str, str] = {
    "low":      "This action was classified as low-risk (read-only, no system changes) and executed automatically.",
    "medium":   "This action was classified as medium-risk (local, reversible) and executed after internal validation.",
    "high":     "This action was classified as high-risk and required explicit approval before execution.",
    "critical": "This action was classified as critical-risk. Explicit human approval was required and the action was journaled for audit purposes.",
}

_OUTCOME_RECOMMENDATIONS: Dict[str, List[str]] = {
    "failure":      ["Review the error log for details.", "Consider retrying with elevated permissions if appropriate."],
    "rolled_back":  ["The system reverted to its prior state.", "Review what triggered the rollback before retrying."],
    "pending":      ["This action is still awaiting completion.", "Check the task dashboard for progress updates."],
}


class ExplainabilityEngine:
    """
    Generates human-readable explanations for agent actions using the action journal.
    """

    def __init__(self, journal: Optional[Any] = None) -> None:
        """
        journal: ActionJournal instance. If None, imports singleton lazily.
        """
        self._journal = journal

    def _get_journal(self) -> Any:
        if self._journal is not None:
            return self._journal
        from src.core.guard.action_journal import ActionJournal
        return ActionJournal()

    def explain(self, entry_id: str) -> Optional[ExplanationReport]:
        """Generate a human-readable explanation for a specific journal entry."""
        journal = self._get_journal()
        entries = journal.recent(limit=1000)
        entry = next((e for e in entries if e.entry_id == entry_id), None)
        if not entry:
            logger.warning("No journal entry found for ID: %s", entry_id)
            return None

        return self._build_report(entry)

    def explain_recent(self, agent_id: Optional[str] = None, limit: int = 5) -> List[ExplanationReport]:
        """Explain the most recent actions for an agent."""
        journal = self._get_journal()
        entries = journal.recent(limit=limit, agent_id=agent_id)
        return [self._build_report(e) for e in entries]

    def _build_report(self, entry: Any) -> ExplanationReport:
        risk = entry.risk_level or "low"
        reasoning = _RISK_EXPLANATIONS.get(risk, "This action was executed as part of the agent's task plan.")

        evidence = [
            f"Action recorded at {time.strftime('%H:%M:%S', time.localtime(entry.timestamp))}",
            f"Triggered by agent: {entry.agent_id}",
            f"Command: {entry.action}",
            f"Description: {entry.description}",
        ]

        if entry.metadata:
            for k, v in entry.metadata.items():
                evidence.append(f"Context — {k}: {v}")

        if entry.snapshot_id:
            evidence.append(f"Pre-action snapshot taken (ID: {entry.snapshot_id[:8]}…) — rollback available")

        recommendations = _OUTCOME_RECOMMENDATIONS.get(entry.outcome, [])

        return ExplanationReport(
            entry_id=entry.entry_id,
            action=entry.action,
            agent_id=entry.agent_id,
            timestamp=entry.timestamp,
            risk_level=risk,
            outcome=entry.outcome,
            reasoning=reasoning,
            evidence=evidence,
            recommendations=recommendations,
        )

    def why(self, entry_id: str) -> str:
        """Convenience method — returns human-readable text explanation."""
        report = self.explain(entry_id)
        if report:
            return report.to_text()
        return f"No explanation available for entry ID: {entry_id}"
