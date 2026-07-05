"""Tests for the Anvil Workflow Engine and Skill Repository.

Milestone 2 핵심 모듈의 단위 테스트:
- AnvilWorkflowEngine: DAG 정렬, 태스크 추가/삭제/삽입, 실행, 재시도
- SkillRepository: 스킬 저장/조회/삭제/실행, 디스크 영속성
"""

import asyncio
import os
import shutil
import sys
import tempfile

import pytest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.anvil.engine import AnvilTask, AnvilWorkflowEngine
from src.core.anvil.skill_repository import Skill, SkillRepository

# =============================================================================
# AnvilWorkflowEngine Tests
# =============================================================================


class TestAnvilTask:
    """AnvilTask dataclass 테스트."""

    def test_default_values(self):
        task = AnvilTask(task_id="t1", name="Test", handler="handler_fn")
        assert task.status == "pending"
        assert task.dependencies == []
        assert task.retry_count == 0
        assert task.max_retries == 2
        assert task.result is None
        assert task.error is None

    def test_with_dependencies(self):
        task = AnvilTask(
            task_id="t2",
            name="Dependent",
            handler="fn",
            dependencies=["t1"],
        )
        assert task.dependencies == ["t1"]


class TestAnvilWorkflowEngine:
    """AnvilWorkflowEngine 테스트."""

    def test_add_task(self):
        engine = AnvilWorkflowEngine()
        task = AnvilTask(task_id="t1", name="Test", handler="fn")
        engine.add_task(task)
        assert "t1" in engine.tasks
        assert engine.tasks["t1"].name == "Test"

    def test_remove_task(self):
        engine = AnvilWorkflowEngine()
        engine.add_task(AnvilTask(task_id="t1", name="A", handler="fn"))
        engine.add_task(
            AnvilTask(task_id="t2", name="B", handler="fn", dependencies=["t1"])
        )
        engine.remove_task("t1")
        assert "t1" not in engine.tasks
        # t2의 의존성에서 t1 제거 확인
        assert "t1" not in engine.tasks["t2"].dependencies

    def test_insert_task_after(self):
        engine = AnvilWorkflowEngine()
        engine.add_task(AnvilTask(task_id="t1", name="A", handler="fn"))
        new_task = AnvilTask(task_id="t2", name="B", handler="fn")
        engine.insert_task_after("t1", new_task)
        assert "t2" in engine.tasks
        assert "t1" in engine.tasks["t2"].dependencies

    def test_insert_task_after_invalid_raises(self):
        engine = AnvilWorkflowEngine()
        new_task = AnvilTask(task_id="t2", name="B", handler="fn")
        with pytest.raises(ValueError, match="not found in DAG"):
            engine.insert_task_after("nonexistent", new_task)

    def test_topological_sort_linear(self):
        engine = AnvilWorkflowEngine()
        engine.add_task(AnvilTask(task_id="t1", name="A", handler="fn"))
        engine.add_task(
            AnvilTask(task_id="t2", name="B", handler="fn", dependencies=["t1"])
        )
        engine.add_task(
            AnvilTask(task_id="t3", name="C", handler="fn", dependencies=["t2"])
        )
        levels = engine._topological_sort()
        assert levels == [["t1"], ["t2"], ["t3"]]

    def test_topological_sort_parallel(self):
        engine = AnvilWorkflowEngine()
        engine.add_task(AnvilTask(task_id="t1", name="A", handler="fn"))
        engine.add_task(
            AnvilTask(task_id="t2", name="B", handler="fn", dependencies=["t1"])
        )
        engine.add_task(
            AnvilTask(task_id="t3", name="C", handler="fn", dependencies=["t1"])
        )
        levels = engine._topological_sort()
        assert levels[0] == ["t1"]
        assert set(levels[1]) == {"t2", "t3"}

    def test_topological_sort_circular_dependency(self):
        engine = AnvilWorkflowEngine()
        engine.add_task(
            AnvilTask(task_id="t1", name="A", handler="fn", dependencies=["t2"])
        )
        engine.add_task(
            AnvilTask(task_id="t2", name="B", handler="fn", dependencies=["t1"])
        )
        with pytest.raises(ValueError, match="Circular dependency"):
            engine._topological_sort()

    @pytest.mark.asyncio
    async def test_execute_empty(self):
        engine = AnvilWorkflowEngine()
        result = await engine.execute(context={})
        assert result["status"] == "empty"

    @pytest.mark.asyncio
    async def test_execute_with_handlers(self):
        engine = AnvilWorkflowEngine()

        async def handler_a(ctx):
            return {"value": "a_done"}

        async def handler_b(ctx):
            return {"value": "b_done"}

        engine.register_handler("fn_a", handler_a)
        engine.register_handler("fn_b", handler_b)
        engine.add_task(AnvilTask(task_id="t1", name="A", handler="fn_a"))
        engine.add_task(
            AnvilTask(
                task_id="t2", name="B", handler="fn_b", dependencies=["t1"]
            )
        )

        result = await engine.execute(context={})
        assert result["status"] == "completed"
        assert result["results"]["t1"] == {"value": "a_done"}
        assert result["results"]["t2"] == {"value": "b_done"}
        assert engine.tasks["t1"].status == "completed"
        assert engine.tasks["t2"].status == "completed"

    @pytest.mark.asyncio
    async def test_execute_handler_not_found_skips(self):
        engine = AnvilWorkflowEngine()
        engine.add_task(
            AnvilTask(task_id="t1", name="Missing", handler="nonexistent")
        )
        result = await engine.execute(context={})
        assert engine.tasks["t1"].status == "skipped"

    @pytest.mark.asyncio
    async def test_execute_with_retry(self):
        call_count = 0

        async def flaky_handler(ctx):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Transient error")
            return "success"

        engine = AnvilWorkflowEngine()
        engine.register_handler("flaky", flaky_handler)
        engine.add_task(
            AnvilTask(task_id="t1", name="Flaky", handler="flaky", max_retries=2)
        )

        result = await engine.execute(context={})
        assert result["status"] == "completed"
        assert engine.tasks["t1"].result == "success"

    @pytest.mark.asyncio
    async def test_execute_exhausted_retries(self):
        async def always_fail(ctx):
            raise RuntimeError("Permanent error")

        engine = AnvilWorkflowEngine()
        engine.register_handler("fail", always_fail)
        engine.add_task(
            AnvilTask(task_id="t1", name="Failing", handler="fail", max_retries=1)
        )

        result = await engine.execute(context={})
        assert result["status"] == "partial_failure"
        assert result["failed"] == 1
        assert engine.tasks["t1"].status == "failed"

    @pytest.mark.asyncio
    async def test_execute_sync_handler(self):
        def sync_handler(ctx):
            return "sync_result"

        engine = AnvilWorkflowEngine()
        engine.register_handler("sync", sync_handler)
        engine.add_task(AnvilTask(task_id="t1", name="Sync", handler="sync"))

        result = await engine.execute(context={})
        assert result["status"] == "completed"
        assert result["results"]["t1"] == "sync_result"

    def test_reset(self):
        engine = AnvilWorkflowEngine()
        engine.add_task(AnvilTask(task_id="t1", name="A", handler="fn"))
        engine.reset()
        assert len(engine.tasks) == 0

    def test_get_status(self):
        engine = AnvilWorkflowEngine()
        engine.add_task(AnvilTask(task_id="t1", name="A", handler="fn"))
        engine.add_task(
            AnvilTask(task_id="t2", name="B", handler="fn", status="completed")
        )
        status = engine.get_status()
        assert status == {"t1": "pending", "t2": "completed"}


# =============================================================================
# SkillRepository Tests
# =============================================================================


class TestSkillRepository:
    """SkillRepository 테스트."""

    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리 생성/정리."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_save_and_get_skill(self, temp_dir):
        repo = SkillRepository(storage_dir=temp_dir)
        skill = repo.save_skill(
            name="test_skill",
            code="def run(): return 42",
            description="Test skill",
        )
        assert skill.name == "test_skill"
        retrieved = repo.get_skill("test_skill")
        assert retrieved is not None
        assert retrieved.code == "def run(): return 42"

    def test_list_skills(self, temp_dir):
        repo = SkillRepository(storage_dir=temp_dir)
        repo.save_skill("skill_a", "code_a")
        repo.save_skill("skill_b", "code_b")
        names = repo.list_skills()
        assert set(names) == {"skill_a", "skill_b"}

    def test_delete_skill(self, temp_dir):
        repo = SkillRepository(storage_dir=temp_dir)
        repo.save_skill("to_delete", "code")
        assert repo.delete_skill("to_delete") is True
        assert repo.get_skill("to_delete") is None
        assert repo.delete_skill("nonexistent") is False

    def test_execute_skill(self, temp_dir):
        repo = SkillRepository(storage_dir=temp_dir)
        repo.save_skill(
            name="adder",
            code="def run(*args, **kwargs): return sum(args)",
        )
        result = repo.execute_skill("adder", 1, 2, 3)
        assert result == 6

    def test_execute_skill_not_found(self, temp_dir):
        repo = SkillRepository(storage_dir=temp_dir)
        with pytest.raises(ValueError, match="not found"):
            repo.execute_skill("missing")

    def test_disk_persistence(self, temp_dir):
        # 첫 번째 리포지토리에서 스킬 저장
        repo1 = SkillRepository(storage_dir=temp_dir)
        repo1.save_skill("persistent", "def run(): return 'hello'", description="persistent")

        # 새 리포지토리 인스턴스에서 디스크 로드 확인
        repo2 = SkillRepository(storage_dir=temp_dir)
        loaded = repo2.get_skill("persistent")
        assert loaded is not None
        assert loaded.name == "persistent"
        assert loaded.code == "def run(): return 'hello'"

    def test_skill_metadata(self, temp_dir):
        repo = SkillRepository(storage_dir=temp_dir)
        repo.save_skill(
            "meta_skill",
            "def run(): pass",
            metadata={"version": "1.0", "author": "test"},
        )
        skill = repo.get_skill("meta_skill")
        assert skill.metadata["version"] == "1.0"
        assert skill.metadata["author"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
