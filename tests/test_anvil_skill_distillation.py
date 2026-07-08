import pytest

from src.core.anvil.skill_distillation import SkillDistiller, WorkflowTrace
from src.core.anvil.skill_repository import SkillRepository
from src.core.skill_tree import SkillPerformanceTracker, SkillTree
from src.core.tool_trace import ToolTrace


def sample_trace() -> WorkflowTrace:
    traces = [
        ToolTrace(
            tool_id="tool_1",
            citation_id="CIT-1",
            tool_type="web_search",
            query="find invoice reconciliation steps",
            raw_answer="Use ledger export, compare invoice ids, report missing payments.",
            summary="Compare ledger invoice ids against payment records.",
        ),
        ToolTrace(
            tool_id="tool_2",
            citation_id="CIT-2",
            tool_type="python",
            query="group missing invoices by customer",
            raw_answer="customer A has 2 missing payments",
            summary="Group mismatches by customer and count missing payments.",
        ),
    ]
    return WorkflowTrace.from_tool_traces(
        "reconcile invoices",
        traces,
        metadata={"workflow": "finance_reconciliation"},
    )


def test_workflow_trace_from_tool_traces_normalizes_steps() -> None:
    trace = sample_trace()

    assert trace.goal == "reconcile invoices"
    assert trace.success
    assert [step.tool_name for step in trace.steps] == ["web_search", "python"]
    assert trace.steps[0].metadata["citation_id"] == "CIT-1"


def test_distills_and_registers_validated_skill(tmp_path) -> None:
    repo = SkillRepository(storage_dir=str(tmp_path / "skills"))
    tree = SkillTree(agent_id="agent_a")
    tracker = SkillPerformanceTracker(store_path=tmp_path / "skill_performance.json")

    skill = SkillDistiller().distill_and_register(
        sample_trace(),
        repo,
        skill_tree=tree,
        performance_tracker=tracker,
    )

    saved = repo.get_skill(skill.name)
    assert saved is not None
    assert saved.metadata["quality_gate"] == "validated_replay"
    assert tree.get_skills_by_category("workflow") == [skill.name]
    assert tracker.get_top_skills(top_k=1) == [skill.name]

    result = repo.execute_skill(skill.name, context={"goal": "reconcile July invoices"})

    assert result["goal"] == "reconcile July invoices"
    assert result["source_goal"] == "reconcile invoices"
    assert result["steps"][0]["tool"] == "web_search"


def test_matches_distilled_skills_for_similar_tasks(tmp_path) -> None:
    repo = SkillRepository(storage_dir=str(tmp_path / "skills"))
    distiller = SkillDistiller()
    skill = distiller.distill_and_register(sample_trace(), repo)

    matches = distiller.match_distilled_skills(
        "repeat the invoice reconciliation workflow",
        repo,
    )

    assert matches
    assert matches[0].skill_name == skill.name
    assert matches[0].score > 0


def test_rejects_unsuccessful_trace(tmp_path) -> None:
    repo = SkillRepository(storage_dir=str(tmp_path / "skills"))
    trace = sample_trace()
    trace.success = False

    with pytest.raises(ValueError, match="Only successful"):
        SkillDistiller().distill_and_register(trace, repo)
