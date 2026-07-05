"""Anvil M4 - HITL 체크포인트 및 의도 가드레일 테스트."""

import pytest

from src.core.anvil.dynamic_checklist_generator import DynamicChecklistGenerator
from src.core.anvil.hitl_checkpoint import (
    CheckpointDecision,
    CheckpointStage,
    HITLCheckpointManager,
)
from src.core.anvil.intent_guardrail import IntentGuardrail, _tokenize
from src.core.anvil.request_analyzer import RequestAnalyzer


class TestIntentGuardrail:
    """IntentGuardrail 자가 진단 테스트."""

    def _analysis(self, request: str):
        return RequestAnalyzer().analyze(request)

    def test_aligned_summary_passes(self):
        analysis = self._analysis("파이썬 코드에서 로그인 버그를 수정하고 테스트를 추가해줘")
        guardrail = IntentGuardrail(analysis)
        assessment = guardrail.evaluate("로그인 버그 원인을 파악해 코드 수정 후 테스트 추가 진행 중")
        assert assessment.aligned
        assert assessment.drift_score <= guardrail.drift_threshold

    def test_unrelated_summary_detects_drift(self):
        analysis = self._analysis("파이썬 코드에서 로그인 버그를 수정하고 테스트를 추가해줘")
        guardrail = IntentGuardrail(analysis)
        assessment = guardrail.evaluate("weather forecast dashboard styling and font colors")
        assert not assessment.aligned
        assert assessment.drift_score > guardrail.drift_threshold
        assert assessment.reasons

    def test_empty_summary_is_full_drift(self):
        analysis = self._analysis("로그 파일을 정리해줘")
        guardrail = IntentGuardrail(analysis)
        assessment = guardrail.evaluate("")
        assert not assessment.aligned
        assert assessment.drift_score == 1.0

    def test_constraint_violation_flagged(self):
        analysis = self._analysis("서버 설정을 정리해줘. production 데이터베이스 삭제는 금지")
        guardrail = IntentGuardrail(analysis)
        assessment = guardrail.evaluate("production 데이터베이스 삭제 작업과 서버 설정 정리 진행")
        assert assessment.violated_constraints
        assert not assessment.aligned

    def test_should_check_interval(self):
        analysis = self._analysis("데이터 분석해줘")
        guardrail = IntentGuardrail(analysis, check_interval=3)
        assert not guardrail.should_check(0)
        assert not guardrail.should_check(2)
        assert guardrail.should_check(3)
        assert guardrail.should_check(6)

    def test_needs_human_review_after_streak(self):
        analysis = self._analysis("파이썬 코드 버그 수정해줘")
        guardrail = IntentGuardrail(analysis, escalation_streak=2)
        guardrail.evaluate("unrelated topic entirely different")
        assert not guardrail.needs_human_review()
        guardrail.evaluate("another unrelated activity happening")
        assert guardrail.needs_human_review()

    def test_aligned_result_resets_streak(self):
        analysis = self._analysis("파이썬 코드 버그 수정해줘")
        guardrail = IntentGuardrail(analysis, escalation_streak=2)
        guardrail.evaluate("unrelated topic entirely different")
        guardrail.evaluate("파이썬 코드 버그 수정 진행 중")
        assert not guardrail.needs_human_review()

    def test_tokenize_filters_stopwords(self):
        tokens = _tokenize("the code and 그리고 버그")
        assert "code" in tokens
        assert "버그" in tokens
        assert "the" not in tokens
        assert "그리고" not in tokens


class TestHITLCheckpointManager:
    """HITLCheckpointManager 체크포인트 및 피드백 반영 테스트."""

    @pytest.mark.asyncio
    async def test_headless_auto_approve(self):
        manager = HITLCheckpointManager()
        result = await manager.checkpoint(CheckpointStage.AFTER_PLANNING)
        assert result.decision == CheckpointDecision.APPROVE
        assert result.auto_resolved
        assert len(manager.history) == 1

    @pytest.mark.asyncio
    async def test_sync_provider(self):
        def provider(stage, context):
            return CheckpointDecision.REVISE, "출처를 반드시 포함해줘"

        manager = HITLCheckpointManager(feedback_provider=provider)
        result = await manager.checkpoint(CheckpointStage.AFTER_DATA_COLLECTION)
        assert result.decision == CheckpointDecision.REVISE
        assert result.feedback == "출처를 반드시 포함해줘"
        assert not result.auto_resolved

    @pytest.mark.asyncio
    async def test_async_provider(self):
        async def provider(stage, context):
            return CheckpointDecision.ABORT, "방향이 잘못됨"

        manager = HITLCheckpointManager(feedback_provider=provider)
        result = await manager.checkpoint(CheckpointStage.INTENT_DRIFT)
        assert result.decision == CheckpointDecision.ABORT
        assert manager.aborted()

    @pytest.mark.asyncio
    async def test_provider_failure_defaults_to_approve(self):
        def provider(stage, context):
            raise RuntimeError("channel unavailable")

        manager = HITLCheckpointManager(feedback_provider=provider)
        result = await manager.checkpoint(CheckpointStage.BEFORE_FINAL_REPORT)
        assert result.decision == CheckpointDecision.APPROVE

    @pytest.mark.asyncio
    async def test_apply_feedback_adds_checklist_items(self):
        analysis = RequestAnalyzer().analyze("최신 LLM 동향을 조사해줘")
        checklist = DynamicChecklistGenerator().generate(analysis)
        before = len(checklist.items)

        def provider(stage, context):
            return CheckpointDecision.REVISE, "2025년 이후 자료만 사용하고 한국어로 요약해줘"

        manager = HITLCheckpointManager(feedback_provider=provider)
        result = await manager.checkpoint(CheckpointStage.AFTER_PLANNING)
        added = manager.apply_feedback(checklist, result)

        assert added
        assert len(checklist.items) == before + len(added)
        assert all(item.description.startswith("[피드백") for item in added)

    @pytest.mark.asyncio
    async def test_apply_feedback_ignores_approve(self):
        analysis = RequestAnalyzer().analyze("최신 LLM 동향을 조사해줘")
        checklist = DynamicChecklistGenerator().generate(analysis)
        before = len(checklist.items)

        manager = HITLCheckpointManager()
        result = await manager.checkpoint(CheckpointStage.AFTER_PLANNING)
        added = manager.apply_feedback(checklist, result)

        assert added == []
        assert len(checklist.items) == before

    @pytest.mark.asyncio
    async def test_summary_reflects_history(self):
        manager = HITLCheckpointManager()
        await manager.checkpoint(CheckpointStage.AFTER_PLANNING)
        await manager.checkpoint(CheckpointStage.BEFORE_FINAL_REPORT)
        summary = manager.summary()
        assert len(summary) == 2
        assert summary[0]["stage"] == CheckpointStage.AFTER_PLANNING.value
        assert summary[1]["decision"] == CheckpointDecision.APPROVE.value
