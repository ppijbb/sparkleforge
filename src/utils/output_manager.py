"""통합 출력 시스템 - 사용자 중심 출력 관리

gemini-cli 수준의 실시간 스트리밍 출력, 도구 실행 결과 포맷팅,
진행 상황 표시, 색상 지원을 제공하는 통합 출력 시스템
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List


class OutputLevel(Enum):
    """출력 레벨 정의."""

    DEBUG = "debug"
    SERVICE = "service"
    USER = "user"


class OutputFormat(Enum):
    """출력 형식 정의."""

    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass
class ProgressInfo:
    """진행 상황 정보."""

    stage: str
    current: int
    total: int
    message: str = ""
    estimated_time_remaining: float | None = None
    start_time: float | None = None


@dataclass
class ToolExecutionResult:
    """도구 실행 결과."""

    tool_name: str
    success: bool
    execution_time: float
    result_summary: str
    error_message: str | None = None
    confidence: float = 0.0


@dataclass
class AgentCommunicationInfo:
    """에이전트 통신 정보."""

    agent_id: str
    action: str
    shared_results_count: int = 0
    discussion_topics: List[str] = None

    def __post_init__(self):
        if self.discussion_topics is None:
            self.discussion_topics = []


class UserCenteredOutputManager:
    """사용자 중심 출력 매니저.

    사용자가 봐야 할 정보만 표시하고, 불필요한 디버그 정보는 로그 파일로만 기록.
    실시간 스트리밍 출력, 도구 실행 결과 포맷팅, 진행 상황 표시 제공.
    """

    def __init__(
        self,
        output_level: OutputLevel = OutputLevel.USER,
        output_format: OutputFormat = OutputFormat.TEXT,
        enable_colors: bool = True,
        stream_output: bool = True,
        show_progress: bool = True,
        log_file: str | None = None,
    ):
        """초기화."""
        self.output_level = output_level
        self.output_format = output_format
        self.enable_colors = enable_colors
        self.stream_output = stream_output
        self.show_progress = show_progress

        # 색상 코드 정의 (기존 ColoredFormatter 확장)
        self.colors = {
            "reset": "\033[0m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "bright_red": "\033[91m",
            "bright_green": "\033[92m",
            "bright_yellow": "\033[93m",
            "bright_blue": "\033[94m",
            "bright_magenta": "\033[95m",
            "bright_cyan": "\033[96m",
            "bright_white": "\033[97m",
        }

        # 에이전트별 색상 (기존 AGENT_COLORS 확장)
        self.agent_colors = {
            "planner": "bright_blue",
            "executor": "bright_green",
            "verifier": "bright_yellow",
            "generator": "bright_magenta",
            "orchestrator": "bright_cyan",
            "parallel_executor": "green",
            "parallel_verifier": "yellow",
        }

        # 상태별 색상
        self.status_colors = {
            "success": "bright_green",
            "error": "bright_red",
            "warning": "bright_yellow",
            "info": "bright_blue",
            "progress": "bright_cyan",
            "tool_success": "green",
            "tool_error": "red",
        }

        # 진행 상황 추적
        self.current_progress: ProgressInfo | None = None
        self.progress_start_time: float | None = None

        # 통계
        self.stats = {
            "tools_executed": 0,
            "tools_successful": 0,
            "agents_communicated": 0,
            "results_shared": 0,
        }

        # 스트림 출력 설정
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        # 로깅 설정 (디버그 정보용)
        self.debug_logger = None
        if log_file:
            self.debug_logger = self._setup_debug_logger(log_file)

    def _setup_debug_logger(self, log_file: str) -> logging.Logger:
        """디버그 로거 설정."""
        logger = logging.getLogger("output_manager_debug")
        logger.setLevel(logging.DEBUG)

        # 파일 핸들러
        from pathlib import Path

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _colorize(self, text: str, color: str) -> str:
        """텍스트에 색상 적용."""
        if not self.enable_colors or color not in self.colors:
            return text
        return f"{self.colors[color]}{text}{self.colors['reset']}"

    def _format_agent_name(self, agent_name: str) -> str:
        """에이전트 이름 포맷팅."""
        color = self.agent_colors.get(agent_name.lower(), "bright_white")
        return self._colorize(f"[{agent_name.upper()}]", color)

    def _format_status(self, status: str, status_type: str = "info") -> str:
        """상태 텍스트 포맷팅."""
        color = self.status_colors.get(status_type, "bright_white")
        return self._colorize(status, color)

    def _should_output(self, level: OutputLevel) -> bool:
        """출력 레벨에 따라 출력 여부 결정."""
        level_hierarchy = {
            OutputLevel.DEBUG: 0,
            OutputLevel.SERVICE: 1,
            OutputLevel.USER: 2,
        }
        return level_hierarchy[level] >= level_hierarchy[self.output_level]

    async def output(
        self,
        message: str,
        level: OutputLevel = OutputLevel.USER,
        agent_name: str | None = None,
        status_type: str | None = None,
        **kwargs,
    ):
        """메시지 출력."""
        if not self._should_output(level):
            # 디버그 레벨은 로거에 기록
            if level == OutputLevel.DEBUG and self.debug_logger:
                self.debug_logger.debug(message)
            return

        # 메시지 포맷팅
        formatted_message = message

        if agent_name:
            formatted_message = f"{self._format_agent_name(agent_name)} {formatted_message}"

        if status_type:
            formatted_message = self._format_status(formatted_message, status_type)

        # 타임스탬프 추가 (서비스 레벨 이상)
        if level.value in ["service", "user"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {formatted_message}"

        # 출력
        if self.stream_output:
            print(formatted_message, flush=True)

    async def output_tool_execution(self, tool_result: ToolExecutionResult):
        """도구 실행 결과 출력."""
        if not self._should_output(OutputLevel.USER):
            return

        # 아이콘 선택
        icon = "✅" if tool_result.success else "❌"

        # 실행 시간 포맷팅
        exec_time = f"{tool_result.execution_time:.2f}s"

        # 신뢰도 표시 (성공시에만)
        confidence = ""
        if tool_result.success and tool_result.confidence > 0:
            confidence = f" (신뢰도: {tool_result.confidence:.1%})"

        # 결과 요약
        result_preview = tool_result.result_summary[:100]
        if len(tool_result.result_summary) > 100:
            result_preview += "..."

        # 메시지 구성
        message = f"{icon} 도구 '{tool_result.tool_name}' 실행 완료 ({exec_time}){confidence}"
        if result_preview:
            message += f"\n    결과: {result_preview}"

        if not tool_result.success and tool_result.error_message:
            message += f"\n    오류: {tool_result.error_message}"

        await self.output(
            message,
            level=OutputLevel.USER,
            status_type="tool_success" if tool_result.success else "tool_error",
        )

        # 통계 업데이트
        self.stats["tools_executed"] += 1
        if tool_result.success:
            self.stats["tools_successful"] += 1

    async def output_agent_communication(self, comm_info: AgentCommunicationInfo):
        """에이전트 통신 정보 출력."""
        if not self._should_output(OutputLevel.SERVICE):
            return

        message = f"🤝 {comm_info.agent_id}: {comm_info.action}"

        if comm_info.shared_results_count > 0:
            message += f" ({comm_info.shared_results_count}개 결과 공유)"

        if comm_info.discussion_topics:
            topics = ", ".join(comm_info.discussion_topics)
            message += f" - 토론 주제: {topics}"

        await self.output(message, level=OutputLevel.SERVICE, agent_name=comm_info.agent_id)

        # 통계 업데이트
        self.stats["agents_communicated"] += 1
        self.stats["results_shared"] += comm_info.shared_results_count

    async def start_progress(
        self,
        stage: str,
        total: int,
        message: str = "",
        estimated_time: float | None = None,
    ):
        """진행 상황 시작."""
        if not self.show_progress or not self._should_output(OutputLevel.USER):
            return

        self.current_progress = ProgressInfo(
            stage=stage,
            current=0,
            total=total,
            message=message,
            estimated_time_remaining=estimated_time,
            start_time=time.time(),
        )
        self.progress_start_time = time.time()

        await self._display_progress()

    async def update_progress(
        self,
        current: int | None = None,
        message: str | None = None,
        increment: bool = False,
    ):
        """진행 상황 업데이트."""
        if not self.current_progress or not self.show_progress:
            return

        if current is not None:
            self.current_progress.current = current
        elif increment:
            self.current_progress.current += 1

        if message is not None:
            self.current_progress.message = message

        # 예상 남은 시간 계산
        if self.progress_start_time and self.current_progress.total > 0:
            elapsed = time.time() - self.progress_start_time
            progress_ratio = self.current_progress.current / self.current_progress.total
            if progress_ratio > 0:
                estimated_total = elapsed / progress_ratio
                remaining = estimated_total - elapsed
                self.current_progress.estimated_time_remaining = max(0, remaining)

        await self._display_progress()

    async def complete_progress(self, success: bool = True):
        """진행 상황 완료."""
        if not self.current_progress or not self.show_progress:
            return

        self.current_progress.current = self.current_progress.total

        # 완료 메시지
        status_icon = "✅" if success else "❌"
        status_text = "완료" if success else "실패"
        message = f"{status_icon} {self.current_progress.stage} {status_text}"

        if self.progress_start_time:
            total_time = time.time() - self.progress_start_time
            message += f" (총 {total_time:.1f}초)"

        await self.output(
            message,
            level=OutputLevel.USER,
            status_type="success" if success else "error",
        )

        self.current_progress = None
        self.progress_start_time = None

    async def _display_progress(self):
        """진행률 표시."""
        if not self.current_progress:
            return

        progress = self.current_progress
        percentage = (progress.current / progress.total * 100) if progress.total > 0 else 0

        # 진행률 바
        bar_width = 40
        filled = int(bar_width * progress.current / progress.total) if progress.total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        # 예상 시간
        eta = ""
        if progress.estimated_time_remaining and progress.estimated_time_remaining > 0:
            eta = f" (예상 {progress.estimated_time_remaining:.0f}초 남음)"

        message = f"📊 {progress.stage}: [{bar}] {percentage:.1f}% ({progress.current}/{progress.total}){eta}"

        if progress.message:
            message += f" - {progress.message}"

        # 이전 라인 지우고 새로 쓰기 (같은 줄에 업데이트)
        if self.stream_output:
            # ANSI escape code로 줄 끝까지 지우기
            import sys

            sys.stdout.write(f"\r\033[K{message}")
            sys.stdout.flush()

            if progress.current >= progress.total:
                sys.stdout.write("\n")  # 완료 시에만 줄바꿈
                sys.stdout.flush()

    async def output_workflow_summary(self):
        """워크플로우 요약 출력."""
        if not self._should_output(OutputLevel.USER):
            return

        await self.output("\n" + "=" * 80, level=OutputLevel.USER)
        await self.output("📋 워크플로우 실행 요약", level=OutputLevel.USER)

        # 통계 출력
        await self.output(
            f"🔧 실행된 도구: {self.stats['tools_executed']}개", level=OutputLevel.USER
        )
        await self.output(
            f"✅ 성공한 도구: {self.stats['tools_successful']}개",
            level=OutputLevel.USER,
        )

        if self.stats["tools_executed"] > 0:
            success_rate = self.stats["tools_successful"] / self.stats["tools_executed"] * 100
            await self.output(f"📈 성공률: {success_rate:.1f}%", level=OutputLevel.USER)

        await self.output(
            f"🤝 에이전트 통신: {self.stats['agents_communicated']}회",
            level=OutputLevel.USER,
        )
        await self.output(
            f"📤 공유된 결과: {self.stats['results_shared']}개", level=OutputLevel.USER
        )

        await self.output("=" * 80, level=OutputLevel.USER)

    async def output_error(
        self,
        error: Exception,
        context: str = "",
        agent_name: str | None = None,
        show_traceback: bool = False,
    ):
        """에러 출력."""
        error_message = str(error)

        if context:
            error_message = f"{context}: {error_message}"

        await self.output(
            f"❌ 오류 발생: {error_message}",
            level=OutputLevel.USER,
            agent_name=agent_name,
            status_type="error",
        )

        # 트레이스백 출력 (디버그 모드에서)
        if show_traceback and self._should_output(OutputLevel.DEBUG):
            import traceback

            tb = traceback.format_exc()
            if self.debug_logger:
                self.debug_logger.error(f"Traceback for error: {error_message}\n{tb}")

    async def output_success(
        self,
        message: str,
        agent_name: str | None = None,
        details: Dict[str, Any] | None = None,
    ):
        """성공 메시지 출력."""
        full_message = message

        if details:
            # 중요한 세부 정보만 표시
            important_details = []
            if "count" in details:
                important_details.append(f"개수: {details['count']}")
            if "time" in details:
                important_details.append(f"시간: {details['time']:.2f}초")
            if "quality" in details and details["quality"] > 0:
                important_details.append(f"품질: {details['quality']:.1%}")

            if important_details:
                full_message += f" ({', '.join(important_details)})"

        await self.output(
            full_message,
            level=OutputLevel.USER,
            agent_name=agent_name,
            status_type="success",
        )

    def set_output_level(self, level: OutputLevel):
        """출력 레벨 설정."""
        self.output_level = level

    def set_output_format(self, format: OutputFormat):
        """출력 형식 설정."""
        self.output_format = format

    @asynccontextmanager
    async def session_context(self):
        """세션 컨텍스트 매니저."""
        try:
            await self.output("🚀 Local Researcher 세션 시작", level=OutputLevel.USER)
            yield self
        finally:
            await self.output_workflow_summary()
            await self.output("🏁 Local Researcher 세션 종료", level=OutputLevel.USER)


# 전역 인스턴스
_output_manager = None


def get_output_manager() -> UserCenteredOutputManager:
    """전역 출력 매니저 인스턴스 반환."""
    global _output_manager
    if _output_manager is None:
        _output_manager = UserCenteredOutputManager()
    return _output_manager


def set_output_manager(manager: UserCenteredOutputManager):
    """전역 출력 매니저 설정."""
    global _output_manager
    _output_manager = manager
