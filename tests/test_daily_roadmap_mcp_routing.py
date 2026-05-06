import pytest

from src.core.orchestrator.planning import PlanningNode


@pytest.mark.asyncio
async def test_preliminary_research_uses_resilient_academic_tool(monkeypatch):
    calls = []

    async def fake_execute_tool(tool_name, parameters):
        calls.append((tool_name, parameters))
        if tool_name == "arxiv":
            return {"success": True, "data": {"results": [{"title": "paper"}]}}
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
