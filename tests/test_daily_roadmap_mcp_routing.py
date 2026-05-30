from types import SimpleNamespace

import pytest

from src.core.orchestrator.planning import PlanningNode


@pytest.mark.asyncio
async def test_preliminary_research_uses_resilient_academic_tool(monkeypatch):
    calls = []

    async def fake_execute_tool(tool_name, parameters):
        calls.append((tool_name, parameters))
        if tool_name == "arxiv":
            return {"success": True, "data": {"result": {"title": "paper"}}}
        return {"success": True, "data": {"results": []}}

    monkeypatch.setattr(
        "src.core.orchestrator.planning.execute_tool",
        fake_execute_tool,
    )

    node = PlanningNode(
        context_manager=None,
        context_loader=None,
        research_depth="standard",
        hybrid_storage=None,
        streaming_manager=None,
    )

    result = await node._conduct_preliminary_research(
        {
            "analyzed_objectives": [
                {"description": "current mcp agent automation research"}
            ],
            "domain_analysis": {"fields": ["github automation"]},
        }
    )

    called_tools = [tool_name for tool_name, _ in calls]
    assert "arxiv" in called_tools
    assert "semantic_scholar::papers-search-basic" not in called_tools
    assert result["academic_results"][0]["tool"] == "arxiv"


@pytest.mark.asyncio
async def test_autopilot_planning_does_not_wait_for_clarification(monkeypatch):
    class ContextManager:
        def __init__(self):
            self.context = None

        def get_current_context(self):
            return self.context

        def push_context(self, context_data, depth):
            self.context = SimpleNamespace(context_id="ctx")
            return "ctx"

        def extend_context(self, context_id, analysis_context, metadata):
            return SimpleNamespace(context_id=context_id)

    class ContextLoader:
        async def load_context(self):
            return None

    class ResearchDepth:
        def determine_depth(self, user_request, preset=None, context=None):
            return SimpleNamespace(
                preset=SimpleNamespace(value="standard"),
                planning={"decompose": {"mode": "fixed", "initial_subtopics": 1}},
                researching={},
                reporting={},
                complexity_score=1.0,
            )

    class ClarificationHandler:
        async def detect_ambiguities(self, request, context):
            return [{"type": "scope", "description": "ambiguous but non-blocking"}]

        async def generate_question(self, ambiguity, context):
            raise AssertionError("autopilot mode must not generate user questions")

    monkeypatch.setattr(
        "src.core.human_clarification_handler.get_clarification_handler",
        lambda: ClarificationHandler(),
    )

    node = PlanningNode(
        context_manager=ContextManager(),
        context_loader=ContextLoader(),
        research_depth=ResearchDepth(),
        hybrid_storage=None,
        streaming_manager=None,
    )

    async def fake_preliminary_research(state):
        return {"keywords": ["daily"], "search_results": [], "academic_results": []}

    async def fake_decompose(state, preliminary_research, depth_config):
        return [
            {
                "task_id": "task_1",
                "name": "Draft roadmap",
                "description": "Create a bounded daily roadmap issue",
                "estimated_complexity": 1,
                "dependencies": [],
                "required_tools": [],
            }
        ]

    async def fake_assign(tasks, state):
        return {"task_1": ["technical_researcher"]}

    async def fake_plan(tasks, assignments):
        return {"strategy": "sequential", "task_count": len(tasks)}

    monkeypatch.setattr(node, "_conduct_preliminary_research", fake_preliminary_research)
    monkeypatch.setattr(node, "_decompose_into_tasks", fake_decompose)
    monkeypatch.setattr(node, "_assign_agents_dynamically", fake_assign)
    monkeypatch.setattr(node, "_create_execution_plan", fake_plan)

    state = {
        "user_request": "Generate the daily SparkleForge roadmap issue",
        "context": {},
        "objective_id": "test",
        "analyzed_objectives": [{"description": "daily roadmap"}],
        "domain_analysis": {},
        "scope_analysis": {},
        "complexity_score": 1.0,
        "autopilot_mode": True,
    }

    result = await node.planning_agent(state)

    assert result["current_step"] == "verify_plan"
    assert result["waiting_for_user"] is False
    assert result["pending_questions"] == []
    assert result["autopilot_assumptions"]
