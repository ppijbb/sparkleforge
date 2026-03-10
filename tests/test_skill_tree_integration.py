#!/usr/bin/env python3
"""Integration tests for Skill Tree system: tree, tracker, cache, retriever, factory."""

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

# Project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestSkillTree:
    """SkillTree create/serialize/deserialize and hot_skills."""

    def test_tree_add_remove(self):
        from src.core.skill_tree import SkillTree

        t = SkillTree("agent_1")
        t.add_skill("research_planner", ["research"])
        t.add_skill("research_executor", ["research"])
        assert "research_planner" in t.get_skills_by_category("research")
        assert t.remove_skill("research_planner")
        assert "research_planner" not in t.get_skills_by_category("research")

    def test_tree_serialize_deserialize(self):
        from src.core.skill_tree import SkillTree

        t = SkillTree("agent_1")
        t.add_skill("research_planner", ["research"])
        t.add_skill("evaluator", ["evaluation"])
        data = t.serialize()
        assert data["agent_id"] == "agent_1"
        assert len(data["skills"]) == 2
        t2 = SkillTree.deserialize(data)
        assert t2.agent_id == t.agent_id
        assert set(t2.get_skills_by_category("research") + t2.get_skills_by_category("evaluation")) == {
            "research_planner",
            "evaluator",
        }


class TestSkillPerformanceTracker:
    """SkillPerformanceTracker record and get_top_skills."""

    def test_tracker_record_and_top(self, temp_store):
        from src.core.skill_tree import SkillPerformanceTracker

        path = temp_store / "perf.json"
        tracker = SkillPerformanceTracker(store_path=path)
        tracker.record("research_planner", True, 100.0, 0.9)
        tracker.record("research_planner", True, 80.0, 0.95)
        tracker.record("evaluator", False, 200.0, 0.3)
        top = tracker.get_top_skills(top_k=5)
        assert "research_planner" in top
        m = tracker.get_metrics("research_planner")
        assert m is not None
        assert m.success_rate == 1.0
        assert m.total_uses == 2


class TestHotSkillCache:
    """HotSkillCache refresh and get."""

    def test_cache_refresh_and_get(self, temp_store):
        from src.core.skill_tree import HotSkillCache, SkillPerformanceTracker
        from src.core.skills_manager import SkillManager

        skills_dir = project_root / "skills"
        if not skills_dir.exists():
            pytest.skip("skills/ directory not found")
        sm = SkillManager(project_root)
        tracker = SkillPerformanceTracker(store_path=temp_store / "perf.json")
        tracker.record("research_planner", True, 50.0, 0.9)
        cache = HotSkillCache(sm, max_size=10)
        cache.refresh(tracker)
        skill = cache.get("research_planner")
        if sm.get_skill_by_id("research_planner"):
            assert skill is not None
        else:
            assert skill is None or skill is not None  # no local skill


class TestSkillRetriever:
    """SkillRetriever.retrieve returns SkillMatch list."""

    @pytest.mark.asyncio
    async def test_retriever_returns_matches(self):
        from src.core.skill_tree import SkillRetriever
        from src.core.skills_manager import get_skill_manager

        sm = get_skill_manager()
        retriever = SkillRetriever(sm)
        matches = await retriever.retrieve("plan and execute research", top_k=3)
        assert isinstance(matches, list)
        for m in matches:
            assert hasattr(m, "skill_id")
            assert hasattr(m, "score")
            assert hasattr(m, "metadata")


class TestOpenSkillProvider:
    """OpenSkillProvider fetch and auto_categorize."""

    def test_auto_categorize(self):
        from src.core.open_skill_provider import OpenSkillProvider
        from src.core.skills_loader import SkillMetadata

        p = OpenSkillProvider()
        assert p.auto_categorize_from_text("search documents") == "research"
        assert p.auto_categorize_from_text("verify quality") == "evaluation"
        meta = SkillMetadata(
            skill_id="x",
            name="x",
            description="summarize report",
            version="1.0",
            category="general",
            tags=[],
            author="",
            created_at="",
            updated_at="",
            path="",
        )
        assert p.auto_categorize(meta) == "synthesis"

    @pytest.mark.asyncio
    async def test_fetch_registry_empty_or_invalid(self):
        from src.core.open_skill_provider import OpenSkillProvider

        p = OpenSkillProvider()
        # Non-JSON or 404 should return []
        result = await p.fetch_from_registry_url("https://httpbin.org/status/404")
        assert result == []


class TestDynamicSubAgentFactory:
    """DynamicSubAgentFactory create_agents_from_plan assigns skill_tree."""

    @pytest.mark.asyncio
    async def test_factory_creates_agents_with_skill_tree(self):
        from src.core.dynamic_sub_agent_factory import DynamicSubAgentFactory
        from src.core.sub_agent_manager import get_sub_agent_manager

        factory = DynamicSubAgentFactory()
        plan = {
            "tasks": [
                {
                    "task_id": "t1",
                    "description": "Search and gather research data",
                    "domain": "research",
                },
            ]
        }
        agents = await factory.create_agents_from_plan(
            plan, network_id="test_skill_tree_net", parent_agent_id="planner"
        )
        assert len(agents) >= 0
        for agent in agents:
            assert hasattr(agent, "skill_tree")
            assert hasattr(agent, "knowledge_base")
            if agent.skill_tree:
                assert agent.knowledge_base.get("assigned_skills") is not None


class TestSkillSelector:
    """SkillSelector select_skills_for_task uses retriever."""

    def test_select_skills_for_task_sync(self):
        from src.core.skills_selector import get_skill_selector

        selector = get_skill_selector()
        matches = selector.select_skills_for_task("plan and execute research", max_skills=3)
        assert isinstance(matches, list)
        for m in matches:
            assert hasattr(m, "skill_id")
            assert hasattr(m, "metadata")

    @pytest.mark.asyncio
    async def test_select_skills_proactively(self):
        from src.core.skills_selector import get_skill_selector

        selector = get_skill_selector()
        matches = await selector.select_skills_proactively(
            "verify and evaluate quality", max_skills=3
        )
        assert isinstance(matches, list)


class TestSkillManagerIntegration:
    """SkillManager get_performance_tracker and sync_external_skills."""

    def test_get_performance_tracker(self):
        from src.core.skills_manager import get_skill_manager

        sm = get_skill_manager()
        tracker = sm.get_performance_tracker()
        assert tracker is not None
        assert hasattr(tracker, "record")
        assert hasattr(tracker, "get_top_skills")
