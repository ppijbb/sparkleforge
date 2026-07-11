"""Anvil M4 - HITL 체크포인트 및 의도 가드레일 테스트."""

import asyncio
import json

import pytest

from src.core.anvil.dynamic_checklist_generator import DynamicChecklistGenerator
from src.core.anvil.hitl_checkpoint import (
    CheckpointDecision,
    CheckpointStage,
    HITLCheckpointManager,
    HITLProviderError,
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
    async def test_provider_returning_none_suspends_and_saves_state(self, tmp_path):
        def provider(stage, context):
            return None

        manager = HITLCheckpointManager(
            feedback_provider=provider, checkpoint_dir=str(tmp_path)
        )
        result = await manager.checkpoint(CheckpointStage.AFTER_PLANNING)

        assert result.decision == CheckpointDecision.ABORT
        assert manager.aborted()
        state_file = tmp_path / f"{result.checkpoint_id}.json"
        assert state_file.exists()

    @pytest.mark.asyncio
    async def test_suspension_state_save_does_not_block_event_loop(self, tmp_path):
        def provider(stage, context):
            return None

        manager = HITLCheckpointManager(
            feedback_provider=provider, checkpoint_dir=str(tmp_path)
        )

        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0)
                ticks += 1

        ticker_task = asyncio.ensure_future(ticker())
        await manager.checkpoint(CheckpointStage.AFTER_PLANNING)
        ticker_task.cancel()

        # If the state save blocked the event loop synchronously, the ticker
        # coroutine would never have had a chance to run concurrently.
        assert ticks > 0

    @pytest.mark.asyncio
    async def test_malformed_provider_response_raises_instead_of_approving(self):
        def provider(stage, context):
            return "not a tuple"

        manager = HITLCheckpointManager(feedback_provider=provider)
        with pytest.raises(HITLProviderError):
            await manager.checkpoint(CheckpointStage.AFTER_PLANNING)

    @pytest.mark.asyncio
    async def test_string_decision_value_is_normalized(self):
        def provider(stage, context):
            return "revise", "짧게 요약해줘"

        manager = HITLCheckpointManager(feedback_provider=provider)
        result = await manager.checkpoint(CheckpointStage.AFTER_PLANNING)
        assert result.decision == CheckpointDecision.REVISE

    @pytest.mark.asyncio
    async def test_invalid_decision_string_raises(self):
        def provider(stage, context):
            return "maybe", "..."

        manager = HITLCheckpointManager(feedback_provider=provider)
        with pytest.raises(HITLProviderError):
            await manager.checkpoint(CheckpointStage.AFTER_PLANNING)

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

    @pytest.mark.asyncio
    async def test_timeout_resolution(self, tmp_path):
        manager = HITLCheckpointManager(checkpoint_dir=str(tmp_path), timeout_seconds=1)
        # Manually create a timed-out checkpoint file
        cp_id = "test_timeout"
        with open(tmp_path / f"{cp_id}.json", "w") as f:
            import json, time
            json.dump({"created_at": time.time() - 10}, f)
        
        decision = manager.resolve_timeout(cp_id)
        assert decision == CheckpointDecision.ABORT

    @pytest.mark.asyncio
    async def test_state_write_is_atomic_on_crash_mid_write(self, tmp_path, monkeypatch):
        """A crash mid-write must never leave a corrupt checkpoint at the
        canonical resume path — only the previous valid state or the new one."""
        from src.core.anvil import hitl_checkpoint

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        state_file = checkpoint_dir / "cp_crash.json"

        # Seed a valid previous checkpoint state.
        previous_payload = {"stage": "after_planning", "previous": True}
        with open(state_file, "w") as f:
            json.dump(previous_payload, f)

        # Simulate a crash mid-write by making json.dump raise after it has
        # opened the temp file but before it finishes writing.
        def crashing_dump(data, f, **kwargs):
            raise RuntimeError("simulated crash mid-write")

        monkeypatch.setattr(hitl_checkpoint.json, "dump", crashing_dump)

        with pytest.raises(RuntimeError):
            hitl_checkpoint.HITLCheckpointManager._write_state_file(
                str(state_file), {"stage": "after_planning", "new": True}
            )

        # The canonical checkpoint path must still hold the previous valid
        # state — never a partial/corrupt write.
        with open(state_file) as f:
            loaded = json.load(f)
        assert loaded == previous_payload

        # No leftover temp file should pollute the checkpoint directory.
        leftovers = [
            p.name for p in checkpoint_dir.iterdir() if p.name.endswith(".tmp")
        ]
        assert leftovers == []

    @pytest.mark.asyncio
    async def test_state_write_persists_valid_payload(self, tmp_path):
        from src.core.anvil.hitl_checkpoint import HITLCheckpointManager

        state_file = tmp_path / "cp_ok.json"
        payload = {"stage": "after_planning", "context": {"k": "v"}}
        HITLCheckpointManager._write_state_file(str(state_file), payload)

        with open(state_file) as f:
            assert json.load(f) == payload
        assert not (tmp_path / "cp_ok.json.tmp").exists()
