"""End-to-end coverage for ResearchAgent analysis/synthesis/validation pipelines.

These tasks previously crashed with AttributeError because self.config,
self.adaptive_strategies, self.analysis_methods, and
_calculate_reliability_score were never initialized/defined.
"""

import asyncio

import pytest

from src.agents.research_agent import ResearchAgent


class _FakeLLMResult:
    def __init__(self, content):
        self.content = content
        self.model_used = "fake"
        self.confidence = 0.9


async def _fake_execute_llm_task(prompt, task_type, system_message=None, **kwargs):
    payload = prompt
    if "JSON array of strings" in payload:
        return _FakeLLMResult('["insight one", "insight two"]')
    if "JSON format with keys: integrated_findings" in payload:
        return _FakeLLMResult('{"integrated_findings": ["a"], "themes": ["b"]}')
    if "JSON format with keys: overall_score" in payload:
        return _FakeLLMResult('{"overall_score": 0.9, "issues": []}')
    if "Select from" in payload:
        return _FakeLLMResult("mixed")
    if "comprehensive analysis report" in payload:
        return _FakeLLMResult('{"executive_summary": "ok"}')
    if "synthesis report" in payload.lower():
        return _FakeLLMResult('{"summary": "ok"}')
    if "validation report" in payload.lower():
        return _FakeLLMResult('{"validation_score": 0.9}')
    return _FakeLLMResult('{"results": "ok"}')


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    instance = ResearchAgent()
    instance.llm = None
    return instance


def test_init_initializes_required_attributes(agent):
    assert hasattr(agent, "config")
    assert hasattr(agent.config, "prompts")
    assert "search_execution" in agent.config.prompts or "analysis" in agent.config.prompts
    assert agent.adaptive_strategies["search_depth"] >= 1
    assert isinstance(agent.analysis_methods, dict)
    assert callable(agent._calculate_reliability_score)


def test_calculate_reliability_score(agent):
    assert agent._calculate_reliability_score({"overall_score": 0.8}) == 0.8
    assert agent._calculate_reliability_score({"issues": ["a", "b"]}) == 0.8
    assert agent._calculate_reliability_score({}) == 0.8
    assert agent._calculate_reliability_score(None) == 0.5


def test_execute_analysis_task_end_to_end(agent, monkeypatch):
    monkeypatch.setattr(
        "src.core.llm_manager.execute_llm_task",
        _fake_execute_llm_task,
    )

    async def run():
        return await agent._execute_analysis_task(
            {"task_id": "a1", "description": "analyze data", "task_type": "analysis"},
            "obj-1",
        )

    result = asyncio.run(run())
    assert "analysis_result" in result
    assert "insights" in result
    assert "analysis_report" in result


def test_execute_synthesis_task_end_to_end(agent, monkeypatch):
    monkeypatch.setattr(
        "src.core.llm_manager.execute_llm_task",
        _fake_execute_llm_task,
    )

    async def run():
        return await agent._execute_synthesis_task(
            {"task_id": "s1", "description": "synthesize findings", "task_type": "synthesis"},
            "obj-1",
        )

    result = asyncio.run(run())
    assert "synthesis_result" in result
    assert "recommendations" in result
    assert "synthesis_report" in result


def test_execute_validation_task_end_to_end(agent, monkeypatch):
    monkeypatch.setattr(
        "src.core.llm_manager.execute_llm_task",
        _fake_execute_llm_task,
    )

    async def run():
        return await agent._execute_validation_task(
            {"task_id": "v1", "description": "validate results", "task_type": "validation"},
            "obj-1",
        )

    result = asyncio.run(run())
    assert "validation_result" in result
    assert "validation_report" in result
    assert "metadata" in result
    assert "reliability_score" in result["metadata"]


def test_update_capabilities_does_not_crash(agent):
    async def run():
        await agent.update_capabilities(
            {"feedback": ["insufficient_data"], "quality_metrics": {"overall_score": 0.9}},
            iteration=1,
        )

    asyncio.run(run())
    assert agent.adaptive_strategies["search_depth"] >= 1
