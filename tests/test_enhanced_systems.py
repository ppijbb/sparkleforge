#!/usr/bin/env python3
"""
Enhanced Systems Test Suite

새로 추가된 시스템 컴포넌트들의 기능을 테스트하는 통합 테스트 스위트
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.output_manager import (
    UserCenteredOutputManager,
    get_output_manager,
    OutputLevel,
    OutputFormat,
    ToolExecutionResult,
    AgentCommunicationInfo
)
from src.core.error_handler import ErrorHandler, get_error_handler, ErrorCategory, ErrorContext
from src.core.progress_tracker import ProgressTracker, get_progress_tracker, WorkflowStage, AgentStatus
from src.utils.logger import setup_enhanced_logger, get_enhanced_logger


async def test_output_manager():
    """출력 매니저 테스트."""
    print("🧪 Testing Output Manager...")

    # 출력 매니저 초기화
    output_manager = UserCenteredOutputManager(
        output_level=OutputLevel.USER,
        enable_colors=True,
        stream_output=True
    )

    # 기본 출력 테스트
    await output_manager.output("테스트 메시지", level=OutputLevel.USER)

    # 도구 실행 결과 테스트
    tool_result = ToolExecutionResult(
        tool_name="test_tool",
        success=True,
        execution_time=1.5,
        result_summary="테스트 성공",
        confidence=0.95
    )
    await output_manager.output_tool_execution(tool_result)

    # 에이전트 통신 테스트
    comm_info = AgentCommunicationInfo(
        agent_id="test_agent",
        action="result_shared",
        shared_results_count=5
    )
    await output_manager.output_agent_communication(comm_info)

    # 진행 상황 테스트
    await output_manager.start_progress("테스트 단계", 100, "진행 중...")
    for i in range(0, 101, 25):
        await output_manager.update_progress(i, f"{i}% 완료")
        await asyncio.sleep(0.1)
    await output_manager.complete_progress(True)

    print("✅ Output Manager tests passed")


async def test_error_handler():
    """에러 핸들러 테스트."""
    print("🧪 Testing Error Handler...")

    error_handler = ErrorHandler(log_errors=False, enable_recovery=True)

    # 일반 에러 테스트
    try:
        raise ValueError("테스트 에러")
    except Exception as e:
        error_info = await error_handler.handle_error(
            e,
            category=ErrorCategory.VALIDATION,
            context=ErrorContext(
                component="test_component",
                operation="test_operation"
            )
        )
        assert error_info.category == ErrorCategory.VALIDATION
        assert error_info.severity.name == "MEDIUM"
        assert len(error_info.recovery_suggestions) > 0

    # 복구 제안 확인
    assert len(error_info.recovery_suggestions) > 0
    print(f"💡 Recovery suggestions: {error_info.recovery_suggestions[:2]}")

    print("✅ Error Handler tests passed")


async def test_progress_tracker():
    """진행 상황 추적기 테스트."""
    print("🧪 Testing Progress Tracker...")

    tracker = ProgressTracker("test_session", enable_real_time_updates=False)

    # 에이전트 등록
    agent1 = tracker.register_agent("agent_1", "executor")
    agent2 = tracker.register_agent("agent_2", "verifier")

    assert len(tracker.workflow_progress.agents) == 2

    # 진행 상황 업데이트
    tracker.update_agent_status("agent_1", AgentStatus.RUNNING, "작업 시작")
    tracker.update_agent_progress("agent_1", 0.5)

    agent1_progress = tracker.get_agent_summary("agent_1")
    assert agent1_progress['status'] == 'running'
    assert agent1_progress['progress'] == 0.5

    # 워크플로우 단계 변경
    tracker.set_workflow_stage(WorkflowStage.EXECUTING)

    summary = tracker.get_workflow_summary()
    assert summary['current_stage'] == 'executing'
    assert summary['total_agents'] == 2

    # 완료
    tracker.update_agent_status("agent_1", AgentStatus.COMPLETED)
    tracker.update_agent_status("agent_2", AgentStatus.COMPLETED)

    print("✅ Progress Tracker tests passed")


async def test_enhanced_logger():
    """향상된 로거 테스트."""
    print("🧪 Testing Enhanced Logger...")

    logger = setup_enhanced_logger("test_logger", log_level="INFO", console_output=False)

    # 기본 로깅
    logger.info("테스트 로그 메시지")

    # 도구 실행 로깅
    from src.utils.logger import log_tool_execution
    log_tool_execution(
        logger,
        tool_name="test_tool",
        execution_time=2.1,
        success=True,
        confidence=0.88
    )

    # 에이전트 통신 로깅
    from src.utils.logger import log_agent_communication
    log_agent_communication(
        logger,
        from_agent="agent_1",
        action="shared_results",
        result_count=3
    )

    # 컨텍스트 매니저
    async with logger.agent_context("test_agent", "test_session"):
        logger.info("컨텍스트 내 로깅")

    print("✅ Enhanced Logger tests passed")


async def test_integration():
    """통합 테스트."""
    print("🧪 Testing System Integration...")

    # 시스템 초기화
    output_manager = UserCenteredOutputManager()
    error_handler = ErrorHandler()
    progress_tracker = ProgressTracker("integration_test")

    # 진행 상황 추적 시작
    await progress_tracker.start_tracking()

    # 콜백 설정
    async def progress_callback(workflow_progress):
        progress_pct = int(workflow_progress.overall_progress * 100)
        await output_manager.output(
            f"진행률: {progress_pct}% - {workflow_progress.current_stage.value}",
            level=output_manager.OutputLevel.SERVICE
        )

    progress_tracker.add_progress_callback(progress_callback)

    # 워크플로우 시뮬레이션
    progress_tracker.set_workflow_stage(WorkflowStage.PLANNING)

    agent = progress_tracker.register_agent("integration_agent", "executor")
    progress_tracker.update_agent_status("integration_agent", AgentStatus.RUNNING)

    # 진행률 업데이트
    for progress in [0.2, 0.5, 0.8, 1.0]:
        progress_tracker.update_agent_progress("integration_agent", progress)
        await asyncio.sleep(0.1)

    progress_tracker.update_agent_status("integration_agent", AgentStatus.COMPLETED)
    progress_tracker.set_workflow_stage(WorkflowStage.COMPLETED)

    # 완료 요약
    await output_manager.output_workflow_summary()

    await progress_tracker.stop_tracking()

    print("✅ Integration tests passed")


async def run_performance_test():
    """성능 테스트."""
    print("🧪 Running Performance Tests...")

    start_time = time.time()

    # 출력 매니저 성능 테스트
    output_manager = UserCenteredOutputManager()
    for i in range(100):
        await output_manager.output(f"테스트 메시지 {i}", level=OutputLevel.DEBUG)

    # 에러 핸들러 성능 테스트
    error_handler = ErrorHandler()
    for i in range(50):
        try:
            if i % 2 == 0:
                raise ValueError(f"테스트 에러 {i}")
        except Exception as e:
            await error_handler.handle_error(e, category=ErrorCategory.VALIDATION)

    # 진행 추적기 성능 테스트
    tracker = ProgressTracker("perf_test")
    for i in range(20):
        agent = tracker.register_agent(f"agent_{i}", "executor")
        tracker.update_agent_progress(f"agent_{i}", 1.0)

    elapsed = time.time() - start_time
    print(f"⚡ Performance test completed in {elapsed:.2f}s")

    # 성능 검증
    assert elapsed < 5.0, f"Performance test took too long: {elapsed:.2f}s"
    print("✅ Performance tests passed")


async def main():
    """메인 테스트 함수."""
    print("🚀 Starting Enhanced Systems Test Suite")
    print("=" * 60)

    try:
        # 개별 컴포넌트 테스트
        await test_output_manager()
        await test_error_handler()
        await test_progress_tracker()
        await test_enhanced_logger()

        # 통합 테스트
        await test_integration()

        # 성능 테스트
        await run_performance_test()

        print("=" * 60)
        print("🎉 All tests passed successfully!")
        print("✅ Enhanced systems are working correctly")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
