"""Distill successful workflow traces into reusable Anvil skills."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, List

from .skill_repository import Skill, SkillRepository


def _clip(value: Any, limit: int = 500) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 15].rstrip() + "... [truncated]"


def _slug(text: str, fallback: str = "workflow") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:48] or fallback


@dataclass
class WorkflowStep:
    """One normalized step from a tool trace or workflow audit entry."""

    name: str
    tool_name: str = ""
    input_summary: str = ""
    output_summary: str = ""
    success: bool = True
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowTrace:
    """Normalized successful workflow trace for skill distillation."""

    goal: str
    steps: List[WorkflowStep]
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tool_traces(
        cls,
        goal: str,
        traces: Iterable[Any],
        *,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> "WorkflowTrace":
        steps = []
        for index, trace in enumerate(traces, start=1):
            tool_name = (
                getattr(trace, "mcp_tool_name", None)
                or getattr(trace, "tool_type", "")
                or getattr(trace, "tool_id", "")
            )
            steps.append(
                WorkflowStep(
                    name=f"step_{index}",
                    tool_name=tool_name,
                    input_summary=_clip(getattr(trace, "query", "")),
                    output_summary=_clip(
                        getattr(trace, "summary", "") or getattr(trace, "raw_answer", "")
                    ),
                    success=True,
                    metadata={
                        "tool_id": getattr(trace, "tool_id", ""),
                        "citation_id": getattr(trace, "citation_id", ""),
                    },
                )
            )
        return cls(goal=goal, steps=steps, success=success, metadata=metadata or {})

    @classmethod
    def from_workflow_audit(
        cls,
        goal: str,
        entries: Iterable[dict[str, Any]],
        *,
        success: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "WorkflowTrace":
        steps = []
        observed_success = True
        for index, entry in enumerate(entries, start=1):
            if entry.get("type") != "tool_execution":
                continue
            step_success = bool(entry.get("success", False))
            observed_success = observed_success and step_success
            steps.append(
                WorkflowStep(
                    name=f"step_{index}",
                    tool_name=str(entry.get("tool_name", "")),
                    input_summary=_clip(entry.get("parameters", {})),
                    output_summary="",
                    success=step_success,
                    duration_ms=float(entry.get("duration", 0.0)) * 1000.0,
                    metadata={"agent_id": entry.get("agent_id", "")},
                )
            )
        return cls(
            goal=goal,
            steps=steps,
            success=observed_success if success is None else success,
            metadata=metadata or {},
        )

    def successful_steps(self) -> list[WorkflowStep]:
        return [step for step in self.steps if step.success]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "success": self.success,
            "metadata": self.metadata,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class SkillDraft:
    """A generated skill that has not necessarily passed the quality gate."""

    name: str
    code: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistilledSkillMatch:
    """Similarity match between a new task and a distilled skill."""

    skill_name: str
    score: float
    reasons: list[str] = field(default_factory=list)


class SkillDistiller:
    """Create, validate, register, and match skills distilled from successful traces."""

    def distill(self, trace: WorkflowTrace) -> SkillDraft:
        if not trace.success:
            raise ValueError("Only successful workflow traces can be distilled.")
        steps = trace.successful_steps()
        if not steps:
            raise ValueError("Cannot distill a workflow trace with no successful steps.")

        trace_payload = trace.to_dict()
        digest = hashlib.sha1(json.dumps(trace_payload, sort_keys=True).encode()).hexdigest()[:8]
        name = f"distilled_{_slug(trace.goal)}_{digest}"
        procedure = [
            {
                "name": step.name,
                "tool": step.tool_name,
                "input": step.input_summary,
                "expected_output": step.output_summary,
            }
            for step in steps
        ]
        code = self._build_skill_code(trace.goal, procedure)
        return SkillDraft(
            name=name,
            code=code,
            description=f"Distilled workflow skill for: {trace.goal}",
            metadata={
                "source": "workflow_trace_distillation",
                "goal": trace.goal,
                "step_count": len(steps),
                "trace_metadata": trace.metadata,
                "quality_gate": "draft",
            },
        )

    def validate_draft(self, draft: SkillDraft) -> bool:
        ast.parse(draft.code)
        with tempfile.TemporaryDirectory() as tmp:
            repo = SkillRepository(storage_dir=tmp)
            repo.save_skill(
                draft.name,
                draft.code,
                description=draft.description,
                metadata=draft.metadata,
            )
            result = repo.execute_skill(draft.name, context={"goal": draft.metadata.get("goal")})
        return (
            isinstance(result, dict)
            and isinstance(result.get("steps"), list)
            and len(result["steps"]) > 0
            and result.get("source_goal") == draft.metadata.get("goal")
        )

    def distill_and_register(
        self,
        trace: WorkflowTrace,
        repository: SkillRepository,
        *,
        skill_tree: Any | None = None,
        performance_tracker: Any | None = None,
    ) -> Skill:
        draft = self.distill(trace)
        if not self.validate_draft(draft):
            raise ValueError(f"Generated skill draft failed validation: {draft.name}")

        metadata = dict(draft.metadata)
        metadata["quality_gate"] = "validated_replay"
        skill = repository.save_skill(
            draft.name,
            draft.code,
            description=draft.description,
            metadata=metadata,
        )
        if skill_tree is not None:
            skill_tree.add_skill(skill.name, ["distilled", "workflow"])
        if performance_tracker is not None:
            performance_tracker.record(skill.name, success=True, latency_ms=0.0, quality_score=1.0)
        return skill

    def distill_and_export(
        self,
        trace: WorkflowTrace,
        marketplace: Any,
        *,
        repository: SkillRepository | None = None,
        dependencies: Iterable[str] | None = None,
    ) -> Any:
        """Distill a trace and export the resulting draft as a shareable bundle.

        The draft is validated and (when a repository is provided) registered
        locally, then immediately published to the supplied skill marketplace
        so other SparkleForge instances can import it.
        """
        draft = self.distill(trace)
        if not self.validate_draft(draft):
            raise ValueError(f"Generated skill draft failed validation: {draft.name}")

        if repository is not None:
            metadata = dict(draft.metadata)
            metadata["quality_gate"] = "validated_replay"
            repository.save_skill(
                draft.name,
                draft.code,
                description=draft.description,
                metadata=metadata,
            )
        return marketplace.export_draft(draft, dependencies=dependencies)

    def match_distilled_skills(
        self,
        query: str,
        repository: SkillRepository,
        *,
        max_skills: int = 5,
        min_score: float = 0.1,
    ) -> list[DistilledSkillMatch]:
        query_terms = _terms(query)
        matches: list[DistilledSkillMatch] = []
        for name in repository.list_skills():
            skill = repository.get_skill(name)
            if skill is None:
                continue
            metadata = skill.metadata or {}
            if metadata.get("source") != "workflow_trace_distillation":
                continue
            haystack = " ".join(
                [
                    skill.name,
                    skill.description,
                    str(metadata.get("goal", "")),
                    json.dumps(metadata.get("trace_metadata", {}), ensure_ascii=False),
                ]
            )
            skill_terms = _terms(haystack)
            if not skill_terms:
                continue
            overlap = query_terms & skill_terms
            score = len(overlap) / max(len(query_terms), 1)
            if score >= min_score:
                matches.append(
                    DistilledSkillMatch(
                        skill_name=skill.name,
                        score=score,
                        reasons=[f"matched terms: {', '.join(sorted(overlap)[:5])}"],
                    )
                )
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:max_skills]

    @staticmethod
    def record_skill_reuse(
        skill_name: str,
        *,
        success: bool,
        latency_ms: float = 0.0,
        quality_score: float = 1.0,
        skill_tree: Any | None = None,
        performance_tracker: Any | None = None,
    ) -> None:
        if skill_tree is not None:
            skill_tree.update_performance(skill_name, success, latency_ms, quality_score)
        if performance_tracker is not None:
            performance_tracker.record(skill_name, success, latency_ms, quality_score)

    @staticmethod
    def _build_skill_code(goal: str, procedure: list[dict[str, str]]) -> str:
        goal_literal = repr(goal)
        procedure_json = json.dumps(procedure, ensure_ascii=False, indent=2)
        return f'''"""Auto-distilled workflow skill.

This skill returns a parameterized replay plan distilled from a successful
workflow trace. Callers may execute the returned steps with their own tool
adapters and verify outputs against the expected summaries.
"""

SOURCE_GOAL = {goal_literal}
PROCEDURE = {procedure_json}


def run(context=None, **kwargs):
    context = context or {{}}
    goal = context.get("goal") or kwargs.get("goal") or SOURCE_GOAL
    return {{
        "goal": goal,
        "source_goal": SOURCE_GOAL,
        "steps": PROCEDURE,
        "verification": [
            {{
                "type": "expected_step_outputs",
                "description": "Replay each step with current inputs and compare outputs to the distilled summaries.",
            }}
        ],
    }}
'''


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9_]{3,}", text.lower())}
