"""Issue #1174: preliminary research sent the exact same joined keyword
phrase to every search tool (no distribution). A prior attempt at fixing
that regressed to distributing single decontextualized tokens ("current",
"agent", "automation") instead of phrases, which real search engines return
near-zero results for, and also assigned queries to tavily/exa even when
their API keys aren't configured (those silently return zero results
instead of erroring, wasting the query).
"""

import pytest

from src.core.orchestrator.planning import PlanningNode


def _make_node():
    return PlanningNode(
        context_manager=None,
        context_loader=None,
        research_depth="standard",
        hybrid_storage=None,
        streaming_manager=None,
    )


@pytest.mark.asyncio
async def test_search_queries_are_multi_word_phrases_not_single_tokens(monkeypatch):
    queries = []

    async def fake_execute_tool(tool_name, parameters):
        queries.append(parameters["query"])
        return {"success": True, "data": {"results": []}}

    monkeypatch.setattr("src.core.orchestrator.planning.execute_tool", fake_execute_tool)
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    monkeypatch.setenv("EXA_API_KEY", "x")

    node = _make_node()
    await node._conduct_preliminary_research(
        {
            "analyzed_objectives": [
                {"description": "current mcp agent automation research"}
            ],
            "domain_analysis": {"fields": ["github automation"]},
        }
    )

    assert queries, "expected at least one search query"
    assert all(" " in q for q in queries), (
        f"every query must be a multi-word phrase, not a decontextualized single "
        f"token: {queries!r}"
    )


@pytest.mark.asyncio
async def test_queries_are_distributed_across_tools_not_repeated_identically(monkeypatch):
    calls = []

    async def fake_execute_tool(tool_name, parameters):
        calls.append((tool_name, parameters["query"]))
        return {"success": True, "data": {"results": []}}

    monkeypatch.setattr("src.core.orchestrator.planning.execute_tool", fake_execute_tool)
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    monkeypatch.setenv("EXA_API_KEY", "x")

    node = _make_node()
    await node._conduct_preliminary_research(
        {
            "analyzed_objectives": [
                {"description": "current mcp agent automation research"}
            ],
            "domain_analysis": {"fields": ["github automation"]},
        }
    )

    tools_used = {tool_name for tool_name, _ in calls}
    assert len(tools_used) > 1, f"expected more than one distinct tool used, got {calls!r}"


@pytest.mark.asyncio
async def test_unconfigured_providers_are_skipped_entirely(monkeypatch):
    calls = []

    async def fake_execute_tool(tool_name, parameters):
        calls.append(tool_name)
        return {"success": True, "data": {"results": []}}

    monkeypatch.setattr("src.core.orchestrator.planning.execute_tool", fake_execute_tool)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    node = _make_node()
    await node._conduct_preliminary_research(
        {
            "analyzed_objectives": [
                {"description": "current mcp agent automation research"}
            ],
            "domain_analysis": {"fields": ["github automation"]},
        }
    )

    web_search_calls = [tool_name for tool_name in calls if tool_name != "arxiv"]
    assert web_search_calls, "expected g-search to still be tried"
    assert set(web_search_calls) == {"g-search"}, (
        f"tavily/exa have no API key configured and must not be assigned a query: {calls!r}"
    )
