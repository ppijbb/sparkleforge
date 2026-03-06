"""구조화된 에러 처리 시스템

에러 분류, 상세한 에러 메시지 및 스택 트레이스, 복구 제안,
에러 로깅 및 리포트를 제공하는 통합 에러 처리 시스템
"""

import asyncio
import logging
import sys
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List


class ErrorSeverity(Enum):
    """에러 심각도 레벨."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """에러 카테고리."""

    NETWORK = "network"
    API = "api"
    LLM = "llm"
    TOOL = "tool"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """에러 발생 컨텍스트."""

    component: str
    operation: str
    agent_id: str | None = None
    session_id: str | None = None
    workflow_stage: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorInfo:
    """구조화된 에러 정보."""

    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_error: Exception | None = None
    stack_trace: str | None = None
    context: ErrorContext | None = None
    recovery_suggestions: List[str] = field(default_factory=list)
    related_errors: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if not self.error_id:
            self.error_id = f"err_{int(self.timestamp.timestamp())}_{hash(self.message) % 10000:04d}"

        # 스택 트레이스 자동 추출
        if self.original_error and not self.stack_trace:
            self.stack_trace = self._extract_stack_trace()

        # 복구 제안 자동 생성
        if not self.recovery_suggestions:
            self.recovery_suggestions = self._generate_recovery_suggestions()

    def _extract_stack_trace(self) -> str:
        """예외에서 스택 트레이스 추출."""
        if not self.original_error:
            return ""

        # 현재 스택 트레이스 가져오기
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_tb:
            return "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            # 예외 객체에서 스택 트레이스 추출 시도
            try:
                return "".join(
                    traceback.format_exception(
                        type(self.original_error),
                        self.original_error,
                        self.original_error.__traceback__,
                    )
                )
            except:
                return str(self.original_error)

    def _generate_recovery_suggestions(self) -> List[str]:
        """에러 카테고리에 따른 복구 제안 생성."""
        suggestions = []

        if self.category == ErrorCategory.NETWORK:
            suggestions.extend(
                [
                    "네트워크 연결 상태를 확인하세요",
                    "프록시 설정을 검토하세요",
                    "잠시 후 다시 시도하세요",
                ]
            )

        elif self.category == ErrorCategory.API:
            suggestions.extend(
                [
                    "API 키가 유효한지 확인하세요",
                    "API 할당량을 확인하세요",
                    "API 엔드포인트가 올바른지 검토하세요",
                ]
            )

        elif self.category == ErrorCategory.LLM:
            suggestions.extend(
                [
                    "LLM 모델이 사용 가능한지 확인하세요",
                    "토큰 제한을 초과하지 않았는지 확인하세요",
                    "다른 LLM 모델로 전환을 고려하세요",
                ]
            )

        elif self.category == ErrorCategory.TOOL:
            suggestions.extend(
                [
                    "도구가 올바르게 설치되었는지 확인하세요",
                    "도구 파라미터가 유효한지 검토하세요",
                    "도구 버전을 업데이트하세요",
                ]
            )

        elif self.category == ErrorCategory.TIMEOUT:
            suggestions.extend(
                [
                    "타임아웃 시간을 늘리세요",
                    "네트워크 연결을 개선하세요",
                    "작업을 더 작은 단위로 분할하세요",
                ]
            )

        elif self.category == ErrorCategory.PERMISSION:
            suggestions.extend(
                [
                    "필요한 권한이 있는지 확인하세요",
                    "보안 설정을 검토하세요",
                    "관리자에게 문의하세요",
                ]
            )

        elif self.category == ErrorCategory.CONFIGURATION:
            suggestions.extend(
                [
                    "환경 변수 설정을 확인하세요",
                    "설정 파일을 검토하세요",
                    "기본 설정으로 재설정하세요",
                ]
            )

        # 일반적인 제안
        suggestions.extend(
            [
                "로그 파일에서 자세한 정보를 확인하세요",
                "문제를 재현할 수 있는 최소 예제를 만들어 보세요",
                "최신 버전으로 업데이트하세요",
            ]
        )

        return suggestions[:5]  # 최대 5개 제안


class ErrorHandler:
    """통합 에러 처리 시스템.

    에러 분류, 로깅, 복구 제안, 재시도 로직을 제공.
    """

    def __init__(self, log_errors: bool = True, enable_recovery: bool = True):
        """초기화."""
        self.log_errors = log_errors
        self.enable_recovery = enable_recovery
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.logger = self._setup_error_logger()

        # 기본 복구 전략 등록
        self._register_default_recovery_strategies()

    def _setup_error_logger(self) -> logging.Logger:
        """에러 전용 로거 설정."""
        logger = logging.getLogger("error_handler")
        logger.setLevel(logging.ERROR)

        # 중복 핸들러 방지
        if logger.handlers:
            return logger

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter(
            "%(asctime)s - ERROR - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 파일 핸들러 (선택적)
        try:
            from pathlib import Path

            error_log_path = Path("logs/errors.log")
            error_log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(error_log_path, encoding="utf-8")
            file_handler.setLevel(logging.ERROR)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s\n%(exc_info)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to setup error file logging: {e}")

        logger.propagate = False
        return logger

    def _register_default_recovery_strategies(self):
        """기본 복구 전략 등록."""
        # 네트워크 에러 복구
        self.recovery_strategies[ErrorCategory.NETWORK] = [
            self._retry_with_backoff,
            self._check_network_connectivity,
        ]

        # API 에러 복구
        self.recovery_strategies[ErrorCategory.API] = [
            self._retry_with_backoff,
            self._validate_api_key,
        ]

        # 타임아웃 복구
        self.recovery_strategies[ErrorCategory.TIMEOUT] = [
            self._increase_timeout,
            self._retry_with_backoff,
        ]

    async def handle_error(
        self,
        error: Exception,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: ErrorContext | None = None,
        custom_message: str | None = None,
    ) -> ErrorInfo:
        """에러 처리 및 구조화.

        Args:
            error: 발생한 예외
            category: 에러 카테고리
            severity: 에러 심각도
            context: 에러 발생 컨텍스트
            custom_message: 사용자 정의 메시지

        Returns:
            구조화된 에러 정보
        """
        # 에러 정보 생성
        error_info = ErrorInfo(
            error_id="",
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            message=custom_message or str(error),
            original_error=error,
            context=context,
        )

        # 에러 히스토리 저장
        self.error_history.append(error_info)
        self.error_counts[str(category)] = self.error_counts.get(str(category), 0) + 1

        # 로깅
        if self.log_errors:
            await self._log_error(error_info)

        # 출력 매니저를 통한 사용자 표시
        await self._display_error(error_info)

        return error_info

    async def _log_error(self, error_info: ErrorInfo):
        """에러 로깅."""
        log_message = f"[{error_info.category.value.upper()}] {error_info.message}"

        if error_info.context:
            context_parts = []
            if error_info.context.component:
                context_parts.append(f"component={error_info.context.component}")
            if error_info.context.operation:
                context_parts.append(f"operation={error_info.context.operation}")
            if error_info.context.agent_id:
                context_parts.append(f"agent={error_info.context.agent_id}")
            if context_parts:
                log_message += f" ({', '.join(context_parts)})"

        self.logger.error(log_message)

        # 심각한 에러의 경우 스택 트레이스도 로깅
        if (
            error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
            and error_info.stack_trace
        ):
            self.logger.error(
                f"Stack trace for {error_info.error_id}:\n{error_info.stack_trace}"
            )

    async def _display_error(self, error_info: ErrorInfo):
        """출력 매니저를 통한 에러 표시."""
        try:
            from src.utils.output_manager import OutputLevel, get_output_manager

            output_manager = get_output_manager()

            # 심각도에 따른 출력 레벨 결정
            if error_info.severity == ErrorSeverity.CRITICAL:
                level = OutputLevel.USER
            elif error_info.severity == ErrorSeverity.HIGH:
                level = OutputLevel.USER
            else:
                level = OutputLevel.SERVICE

            # 에러 메시지 구성
            error_message = (
                f"[{error_info.category.value.upper()}] {error_info.message}"
            )

            if error_info.context and error_info.context.agent_id:
                error_message = f"{output_manager._format_agent_name(error_info.context.agent_id)} {error_message}"

            # UNKNOWN 범주의 일반론적 복구 제안은 CLI 출력에서 제외해 노이즈를 줄임
            if (
                error_info.recovery_suggestions
                and error_info.category != ErrorCategory.UNKNOWN
            ):
                error_message += "\n💡 복구 제안:"
                for i, suggestion in enumerate(error_info.recovery_suggestions[:3], 1):
                    error_message += f"\n  {i}. {suggestion}"

            await output_manager.output(error_message, level=level, status_type="error")

        except Exception as display_error:
            # 출력 매니저 실패 시 기본 로깅
            self.logger.error(f"Failed to display error: {display_error}")

    async def attempt_recovery(
        self, error_info: ErrorInfo, recovery_func: Callable | None = None
    ) -> bool:
        """에러 복구 시도.

        Args:
            error_info: 에러 정보
            recovery_func: 사용자 정의 복구 함수

        Returns:
            복구 성공 여부
        """
        if not self.enable_recovery:
            return False

        try:
            # 사용자 정의 복구 함수 우선 사용
            if recovery_func:
                result = await recovery_func(error_info)
                if result:
                    self.logger.info(
                        f"Custom recovery succeeded for {error_info.error_id}"
                    )
                    return True

            # 자동 복구 전략 적용
            strategies = self.recovery_strategies.get(error_info.category, [])
            for strategy in strategies:
                try:
                    result = await strategy(error_info)
                    if result:
                        self.logger.info(
                            f"Auto recovery succeeded for {error_info.error_id} using {strategy.__name__}"
                        )
                        return True
                except Exception as recovery_error:
                    self.logger.warning(
                        f"Recovery strategy {strategy.__name__} failed: {recovery_error}"
                    )

            return False

        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return False

    async def retry_operation(
        self,
        operation: Callable,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        error_category: ErrorCategory = ErrorCategory.UNKNOWN,
        *args,
        **kwargs,
    ):
        """재시도 로직이 포함된 작업 실행.

        Args:
            operation: 실행할 작업 함수
            max_retries: 최대 재시도 횟수
            backoff_factor: 백오프 계수
            error_category: 에러 카테고리
            *args, **kwargs: 작업 함수에 전달할 인자

        Returns:
            작업 결과 또는 마지막 에러
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e

                if attempt < max_retries:
                    # 재시도 전 대기
                    wait_time = backoff_factor**attempt
                    self.logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time:.1f}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # 마지막 시도 실패
                    await self.handle_error(
                        e,
                        category=error_category,
                        severity=ErrorSeverity.MEDIUM,
                        custom_message=f"Operation failed after {max_retries + 1} attempts: {str(e)}",
                    )

        return last_error

    # 기본 복구 전략들
    async def _retry_with_backoff(self, error_info: ErrorInfo) -> bool:
        """백오프를 사용한 재시도."""
        if error_info.retry_count >= error_info.max_retries:
            return False

        wait_time = 2**error_info.retry_count  # 지수 백오프
        await asyncio.sleep(min(wait_time, 30))  # 최대 30초

        error_info.retry_count += 1
        return True  # 재시도 신호

    async def _check_network_connectivity(self, error_info: ErrorInfo) -> bool:
        """네트워크 연결성 확인."""
        try:
            import socket

            # 간단한 연결 테스트
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return True
        except:
            return False

    async def _validate_api_key(self, error_info: ErrorInfo) -> bool:
        """API 키 유효성 검증 (플레이스홀더)."""
        # 실제 구현에서는 API 키 검증 로직 추가
        return False

    async def _increase_timeout(self, error_info: ErrorInfo) -> bool:
        """타임아웃 증가 (플레이스홀더)."""
        # 실제 구현에서는 타임아웃 설정 조정
        return False

    def get_error_summary(self) -> Dict[str, Any]:
        """에러 통계 요약."""
        return {
            "total_errors": len(self.error_history),
            "error_counts_by_category": self.error_counts.copy(),
            "recent_errors": [
                {
                    "id": e.error_id,
                    "category": e.category.value,
                    "severity": e.severity.value,
                    "message": e.message[:100] + "..."
                    if len(e.message) > 100
                    else e.message,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in self.error_history[-10:]  # 최근 10개
            ],
        }

    @asynccontextmanager
    async def error_context(
        self,
        component: str,
        operation: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ):
        """에러 컨텍스트 매니저.

        자동으로 에러 컨텍스트를 설정하고 에러 발생 시 처리.
        """
        context = ErrorContext(
            component=component,
            operation=operation,
            agent_id=agent_id,
            session_id=session_id,
        )

        try:
            yield context
        except Exception as e:
            await self.handle_error(
                e, context=context, custom_message=f"{operation} failed in {component}"
            )
            raise


# 전역 에러 핸들러 인스턴스
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """전역 에러 핸들러 인스턴스 반환."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def set_error_handler(handler: ErrorHandler):
    """전역 에러 핸들러 설정."""
    global _error_handler
    _error_handler = handler


# 편의 함수들
async def handle_error_async(
    error: Exception,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: ErrorContext | None = None,
    custom_message: str | None = None,
) -> ErrorInfo:
    """비동기 에러 처리 헬퍼 함수."""
    handler = get_error_handler()
    return await handler.handle_error(
        error, category, severity, context, custom_message
    )


def handle_error_sync(
    error: Exception,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: ErrorContext | None = None,
    custom_message: str | None = None,
) -> None:
    """동기 에러 처리 헬퍼 함수."""
    handler = get_error_handler()
    # 동기 컨텍스트에서 비동기 함수 실행
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 이미 실행 중인 이벤트 루프에서는 태스크 생성
            loop.create_task(
                handler.handle_error(error, category, severity, context, custom_message)
            )
        else:
            # 이벤트 루프 실행
            loop.run_until_complete(
                handler.handle_error(error, category, severity, context, custom_message)
            )
    except RuntimeError:
        # 이벤트 루프가 없는 경우 (메인 스레드 외부)
        import threading

        def run_async():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(
                    handler.handle_error(
                        error, category, severity, context, custom_message
                    )
                )
            finally:
                new_loop.close()

        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
