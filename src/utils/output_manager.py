"""통합 출력 시스템 - 사용자 중심 출력 관리

rich 기반의 실시간 스트리밍 출력, 도구 실행 결과 포맷팅, 진행 상황 표시를
제공하는 통합 출력 시스템. 공개 API(`UserCenteredOutputManager`,
`get_output_manager`/`set_output_manager`, `output_tool_execution` 등)는
`src/core/mcp_integration/hub_mixins/execution.py`(도구 실행 결과 표시),
`src/core/error_handler.py`, `src/core/progress_tracker.py`에서 그대로
사용하므로 시그니처는 유지하고, 내부 렌더링만 손으로 짠 ANSI 코드/텍스트
진행률 바에서 rich(+공유 테마)로 교체했다.
"""

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from src.cli.ui import theme


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


_STATUS_STYLE = {
    "success": theme.STYLE_SUCCESS,
    "error": theme.STYLE_ERROR,
    "warning": theme.STYLE_WARNING,
    "info": theme.STYLE_INFO,
    "tool_success": theme.STYLE_SUCCESS,
    "tool_error": theme.STYLE_ERROR,
    "progress": theme.STYLE_INFO,
}

_AGENT_STYLE = {
    "planner": "bright_blue",
    "executor": "bright_green",
    "verifier": "bright_yellow",
    "generator": "bright_magenta",
    "orchestrator": "bright_cyan",
    "parallel_executor": "green",
    "parallel_verifier": "yellow",
}


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

        self.console = Console(no_color=not enable_colors, soft_wrap=True)

        # 진행 상황 추적 (rich Progress는 start_progress에서 지연 생성)
        self.current_progress: ProgressInfo | None = None
        self.progress_start_time: float | None = None
        self._progress: Progress | None = None
        self._progress_task_id: int | None = None

        # 통계
        self.stats = {
            "tools_executed": 0,
            "tools_successful": 0,
            "agents_communicated": 0,
            "results_shared": 0,
        }

        # 로깅 설정 (디버그 정보용)
        self.debug_logger = None
        if log_file:
            self.debug_logger = self._setup_debug_logger(log_file)

    def _setup_debug_logger(self, log_file: str) -> logging.Logger:
        """디버그 로거 설정."""
        logger = logging.getLogger("output_manager_debug")
        logger.setLevel(logging.DEBUG)

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

    def _format_agent_name(self, agent_name: str) -> str:
        """에이전트 이름 포맷팅 (rich markup)."""
        style = _AGENT_STYLE.get(agent_name.lower(), "bright_white")
        return f"[{style}][{escape(agent_name.upper())}][/{style}]"

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
            if level == OutputLevel.DEBUG and self.debug_logger:
                self.debug_logger.debug(message)
            return

        style = _STATUS_STYLE.get(status_type) if status_type else None
        # theme.markup_for() escapes `message` before embedding it; take that
        # path (with a neutral default style) even with no status_type, since
        # otherwise raw dynamic text flows straight into the f-strings below.
        rendered = theme.markup_for(message, style or "default")

        if agent_name:
            rendered = f"{self._format_agent_name(agent_name)} {rendered}"

        if level.value in ("service", "user"):
            timestamp = datetime.now().strftime("%H:%M:%S")
            rendered = f"[dim]{timestamp}[/dim] {rendered}"

        if self.stream_output:
            self.console.print(rendered, highlight=False)

    async def output_tool_execution(self, tool_result: ToolExecutionResult):
        """도구 실행 결과 출력."""
        if not self._should_output(OutputLevel.USER):
            return

        icon = theme.ICON_SUCCESS if tool_result.success else theme.ICON_ERROR
        exec_time = f"{tool_result.execution_time:.2f}s"

        confidence = ""
        if tool_result.success and tool_result.confidence > 0:
            confidence = f" (신뢰도: {tool_result.confidence:.1%})"

        result_preview = tool_result.result_summary[:100]
        if len(tool_result.result_summary) > 100:
            result_preview += "..."

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

        self.stats["agents_communicated"] += 1
        self.stats["results_shared"] += comm_info.shared_results_count

    def _ensure_progress_started(self) -> None:
        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console,
            )
            self._progress.start()
            self._progress_task_id = self._progress.add_task("", total=100)

    def _progress_description(self) -> str:
        # rich's Progress TextColumn renders {task.description} through its own
        # markup parser, so dynamic stage/message text needs the same escaping
        # as everything routed through theme.markup_for.
        if not self.current_progress:
            return ""
        if self.current_progress.message:
            return escape(f"{self.current_progress.stage} - {self.current_progress.message}")
        return escape(self.current_progress.stage)

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

        self._ensure_progress_started()
        self._progress.reset(
            self._progress_task_id, total=total, description=self._progress_description()
        )

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

        if self.progress_start_time and self.current_progress.total > 0:
            elapsed = time.time() - self.progress_start_time
            progress_ratio = self.current_progress.current / self.current_progress.total
            if progress_ratio > 0:
                estimated_total = elapsed / progress_ratio
                remaining = estimated_total - elapsed
                self.current_progress.estimated_time_remaining = max(0, remaining)

        if self._progress is not None:
            self._progress.update(
                self._progress_task_id,
                completed=self.current_progress.current,
                total=self.current_progress.total,
                description=self._progress_description(),
            )

    async def complete_progress(self, success: bool = True):
        """진행 상황 완료."""
        if not self.current_progress or not self.show_progress:
            return

        stage = self.current_progress.stage
        elapsed = time.time() - self.progress_start_time if self.progress_start_time else None

        if self._progress is not None:
            self._progress.update(self._progress_task_id, completed=self.current_progress.total)
            self._progress.stop()
            self._progress = None
            self._progress_task_id = None

        status_icon = theme.ICON_SUCCESS if success else theme.ICON_ERROR
        status_text = "완료" if success else "실패"
        message = f"{status_icon} {stage} {status_text}"
        if elapsed is not None:
            message += f" (총 {elapsed:.1f}초)"

        await self.output(
            message,
            level=OutputLevel.USER,
            status_type="success" if success else "error",
        )

        self.current_progress = None
        self.progress_start_time = None

    async def output_workflow_summary(self):
        """워크플로우 요약 출력."""
        if not self._should_output(OutputLevel.USER):
            return

        await self.output("\n" + "=" * 80, level=OutputLevel.USER)
        await self.output("📋 워크플로우 실행 요약", level=OutputLevel.USER)

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
