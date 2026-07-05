"""Anvil M5 - 동적 방법 탐색기 및 실행 모드 컨트롤러 테스트."""

import shutil
import tempfile

import pytest

from src.core.anvil.method_resolver import MethodResolver, ResolutionStrategy
from src.core.anvil.mode_controller import ExecutionMode, ModeController
from src.core.anvil.skill_repository import SkillRepository


@pytest.fixture
def temp_skill_repo():
    tmp = tempfile.mkdtemp()
    yield SkillRepository(storage_dir=tmp)
    shutil.rmtree(tmp, ignore_errors=True)


class TestMethodResolver:
    """MethodResolver 탐색 체인 테스트."""

    @pytest.mark.asyncio
    async def test_resolves_from_registry_first(self, temp_skill_repo):
        def handler():
            return "from_registry"

        resolver = MethodResolver(
            handler_registry={"fetch_data": handler},
            skill_repository=temp_skill_repo,
        )
        method = await resolver.resolve("fetch_data")
        assert method.resolved
        assert method.strategy == ResolutionStrategy.REGISTERED_HANDLER
        assert method.handler() == "from_registry"

    @pytest.mark.asyncio
    async def test_resolves_from_skill_repository(self, temp_skill_repo):
        temp_skill_repo.save_skill(
            "parse_csv", "def run(*args, **kwargs): return 'from_skill'"
        )
        resolver = MethodResolver(skill_repository=temp_skill_repo)
        method = await resolver.resolve("parse_csv")
        assert method.resolved
        assert method.strategy == ResolutionStrategy.SKILL_REPOSITORY
        assert method.handler() == "from_skill"

    @pytest.mark.asyncio
    async def test_builder_code_is_saved_as_skill_and_reused(self, temp_skill_repo):
        def builder(capability, context):
            return "def run(*args, **kwargs): return 'built'"

        resolver = MethodResolver(
            skill_repository=temp_skill_repo, tool_builder=builder
        )
        method = await resolver.resolve("convert_units")
        assert method.strategy == ResolutionStrategy.TOOL_BUILDER
        assert method.handler() == "built"

        # 제작된 도구는 스킬로 보존되어 다음 탐색에서 2단계로 해결된다
        method2 = await resolver.resolve("convert_units")
        assert method2.strategy == ResolutionStrategy.SKILL_REPOSITORY

    @pytest.mark.asyncio
    async def test_async_builder_returning_callable(self, temp_skill_repo):
        async def builder(capability, context):
            return lambda: "async_built"

        resolver = MethodResolver(
            skill_repository=temp_skill_repo, tool_builder=builder
        )
        method = await resolver.resolve("scrape_page")
        assert method.strategy == ResolutionStrategy.TOOL_BUILDER
        assert method.handler() == "async_built"

    @pytest.mark.asyncio
    async def test_alternative_process_as_last_resort(self, temp_skill_repo):
        resolver = MethodResolver(skill_repository=temp_skill_repo)
        resolver.register_alternative("send_fax", lambda: "email_instead")
        method = await resolver.resolve("send_fax")
        assert method.strategy == ResolutionStrategy.ALTERNATIVE_PROCESS
        assert method.handler() == "email_instead"

    @pytest.mark.asyncio
    async def test_unresolved_records_all_attempts(self, temp_skill_repo):
        resolver = MethodResolver(skill_repository=temp_skill_repo)
        method = await resolver.resolve("teleport")
        assert not method.resolved
        assert method.strategy == ResolutionStrategy.UNRESOLVED
        attempted = {a.strategy for a in method.attempts}
        assert ResolutionStrategy.REGISTERED_HANDLER in attempted
        assert ResolutionStrategy.SKILL_REPOSITORY in attempted
        assert all(not a.succeeded for a in method.attempts)

    @pytest.mark.asyncio
    async def test_builder_failure_falls_through_to_alternative(self, temp_skill_repo):
        def failing_builder(capability, context):
            raise RuntimeError("builder unavailable")

        resolver = MethodResolver(
            skill_repository=temp_skill_repo, tool_builder=failing_builder
        )
        resolver.register_alternative("summarize", lambda: "manual_process")
        method = await resolver.resolve("summarize")
        assert method.strategy == ResolutionStrategy.ALTERNATIVE_PROCESS
        builder_attempt = next(
            a for a in method.attempts if a.strategy == ResolutionStrategy.TOOL_BUILDER
        )
        assert not builder_attempt.succeeded
        assert "builder unavailable" in builder_attempt.detail


class TestModeController:
    """ModeController 모드 전환 테스트."""

    def test_starts_autonomous(self):
        controller = ModeController()
        assert controller.is_autonomous()

    def test_consecutive_failures_switch_to_hitl(self):
        controller = ModeController(failure_threshold=3)
        controller.record_failure()
        controller.record_failure()
        assert controller.is_autonomous()
        mode = controller.record_failure()
        assert mode == ExecutionMode.HITL_COLLABORATIVE
        assert len(controller.transitions) == 1

    def test_success_resets_failure_streak(self):
        controller = ModeController(failure_threshold=2)
        controller.record_failure()
        controller.record_success()
        controller.record_failure()
        assert controller.is_autonomous()

    def test_recovery_switches_back_to_autonomous(self):
        controller = ModeController(failure_threshold=1, recovery_threshold=2)
        controller.record_failure()
        assert not controller.is_autonomous()
        controller.record_success()
        assert not controller.is_autonomous()
        controller.record_success()
        assert controller.is_autonomous()
        assert len(controller.transitions) == 2

    def test_intent_review_forces_hitl(self):
        controller = ModeController()
        mode = controller.on_intent_review_needed()
        assert mode == ExecutionMode.HITL_COLLABORATIVE

    def test_checkpoint_revise_forces_hitl(self):
        controller = ModeController()
        assert controller.on_checkpoint_decision("approve") == ExecutionMode.AUTONOMOUS
        assert (
            controller.on_checkpoint_decision("revise")
            == ExecutionMode.HITL_COLLABORATIVE
        )

    def test_unresolved_capability_forces_hitl(self):
        controller = ModeController()
        mode = controller.on_unresolved_capability("teleport")
        assert mode == ExecutionMode.HITL_COLLABORATIVE
        assert "teleport" in controller.transitions[0].reason

    def test_summary_reflects_state(self):
        controller = ModeController(failure_threshold=1)
        controller.record_failure()
        summary = controller.summary()
        assert summary["mode"] == ExecutionMode.HITL_COLLABORATIVE.value
        assert len(summary["transitions"]) == 1
        assert summary["transitions"][0]["from"] == ExecutionMode.AUTONOMOUS.value
