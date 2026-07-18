"""Regression tests for issue #790: ResearchAgent's analysis/synthesis/
validation pipeline crashed on missing self.config/self.adaptive_strategies/
self.analysis_methods, an undefined _calculate_reliability_score, and a
nonexistent execute_research_task method called by ResearchOperator.
"""

import os
from unittest.mock import patch

import pytest


@pytest.fixture
def research_agent():
    from src.core.researcher_config import load_config_from_env

    with patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "sk-or-test_key",
            "GOOGLE_API_KEY": "test_google_key",
            "LLM_PROVIDER": "openrouter",
            "LLM_MODEL": "google/gemini-3.1-flash-lite-preview",
            "LLM_TEMPERATURE": "0.1",
            "LLM_MAX_TOKENS": "4000",
            "PLANNING_MODEL": "google/gemini-3.1-flash-lite-preview",
            "REASONING_MODEL": "google/gemini-3.1-flash-lite-preview",
            "VERIFICATION_MODEL": "google/gemini-3.1-flash-lite-preview",
            "GENERATION_MODEL": "google/gemini-3.1-flash-lite-preview",
            "COMPRESSION_MODEL": "google/gemini-3.1-flash-lite-preview",
            "BUDGET_LIMIT": "0.0",
            "ENABLE_COST_OPTIMIZATION": "true",
            "MCP_ENABLED": "true",
            "ENABLE_AUTO_FALLBACK": "false",
        },
    ):
        load_config_from_env()
        from src.agents.research_agent import ResearchAgent

        yield ResearchAgent()


def test_config_prompts_cover_pipeline_keys(research_agent):
    """task_pipelines.py reads these exact keys off self.config.prompts."""
    for key in (
        "search_execution",
        "content_analysis",
        "synthesis_report",
        "recommendation_generation",
        "quality_validation",
        "final_assessment",
    ):
        assert "system_message" in research_agent.config.prompts[key]


def test_adaptive_strategies_and_analysis_methods_initialized(research_agent):
    assert research_agent.adaptive_strategies == {
        "search_depth": 3,
        "content_length": 5000,
        "source_diversity": 0.7,
    }
    assert set(research_agent.analysis_methods.keys()) == {
        "quantitative",
        "qualitative",
        "mixed_methods",
    }


def test_calculate_reliability_score(research_agent):
    assert research_agent._calculate_reliability_score({}) == 0.5
    assert research_agent._calculate_reliability_score(
        {"overall_score": 0.9, "issues": ["a", "b"]}
    ) == pytest.approx(0.7)
    assert research_agent._calculate_reliability_score({"overall_score": 0.3}) == 0.3


@pytest.mark.asyncio
async def test_execute_research_task_delegates_to_execute_task(research_agent):
    """ResearchOperator calls execute_research_task(), not execute_task()."""
    with patch.object(research_agent, "execute_task") as mock_execute:
        mock_execute.return_value = {"status": "completed"}
        result = await research_agent.execute_research_task(
            task={"task_id": "t1"}, objective_id="obj1"
        )
        mock_execute.assert_called_once_with({"task_id": "t1"}, "obj1", False, None)
        assert result == {"status": "completed"}
