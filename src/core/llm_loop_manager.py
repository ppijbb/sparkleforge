"""
상태 없는 LLM 반복 호출 루프 관리

LLM은 상태가 없으며, 대화의 연속성은 사용자가 관리하는 문자열 배열(context)로 유지
상태 없는 LLM과의 반복 호출 루프 구현, 다중 회차 대화 연속성 보장
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class LoopState(Enum):
    """루프 상태."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LoopAction(Enum):
    """루프 액션."""
    CONTINUE = "continue"      # 계속 진행
    RETRY = "retry"           # 재시도
    TERMINATE = "terminate"   # 종료
    PAUSE = "pause"          # 일시 정지
    RESUME = "resume"        # 재개


@dataclass
class ConversationMessage:
    """대화 메시지."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self):
        self.timestamp = self.timestamp or time.time()
        self.token_count = self._estimate_tokens()

    def _estimate_tokens(self) -> int:
        """토큰 수 추정."""
        word_count = len(self.content.split())
        return int(word_count * 1.3)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'token_count': self.token_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """딕셔너리로부터 생성."""
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=data.get('timestamp', time.time()),
            metadata=data.get('metadata', {}),
            token_count=data.get('token_count', 0)
        )


@dataclass
class LoopContext:
    """루프 컨텍스트."""
    conversation_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    max_iterations: int = 10
    current_iteration: int = 0
    state: LoopState = LoopState.INITIALIZING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        """총 토큰 수."""
        return sum(msg.token_count for msg in self.messages)

    @property
    def message_count(self) -> int:
        """메시지 수."""
        return len(self.messages)

    def add_message(self, message: ConversationMessage):
        """메시지 추가."""
        self.messages.append(message)
        self.last_updated = time.time()

    def get_recent_messages(self, limit: int = 10) -> List[ConversationMessage]:
        """최근 메시지 가져오기."""
        return self.messages[-limit:] if limit > 0 else self.messages

    def compress_context(self, max_tokens: int) -> List[ConversationMessage]:
        """컨텍스트 압축."""
        if self.total_tokens <= max_tokens:
            return self.messages

        # 시스템 메시지와 최근 메시지 우선 유지
        system_messages = [msg for msg in self.messages if msg.role == 'system']
        other_messages = [msg for msg in self.messages if msg.role != 'system']

        # 최근 메시지부터 선택
        compressed = []
        current_tokens = 0
        target_messages = other_messages[::-1]  # 역순으로 (최근 것부터)

        for msg in target_messages:
            if current_tokens + msg.token_count <= max_tokens:
                compressed.append(msg)
                current_tokens += msg.token_count
            else:
                break

        # 시스템 메시지 추가 (공간이 있으면)
        for sys_msg in system_messages:
            if current_tokens + sys_msg.token_count <= max_tokens:
                compressed.insert(0, sys_msg)  # 시스템 메시지는 앞에
                current_tokens += sys_msg.token_count

        return compressed[::-1]  # 원래 순서로


@dataclass
class LoopResult:
    """루프 결과."""
    conversation_id: str
    final_state: LoopState
    total_iterations: int
    messages: List[ConversationMessage]
    final_response: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMCallHandler:
    """LLM 호출 핸들러."""

    def __init__(self, llm_call_func: Callable):
        """
        초기화.

        Args:
            llm_call_func: LLM 호출 함수 (messages, **kwargs) -> response
        """
        self.llm_call_func = llm_call_func

    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        LLM 호출.

        Args:
            messages: 메시지 리스트
            **kwargs: 추가 파라미터

        Returns:
            LLM 응답
        """
        try:
            response = await self.llm_call_func(messages, **kwargs)
            return {
                'success': True,
                'response': response,
                'error': None
            }
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {
                'success': False,
                'response': None,
                'error': str(e)
            }


class LLMLoopManager:
    """
    상태 없는 LLM 반복 호출 루프 관리자.

    LLM은 상태가 없으며, 대화의 연속성은 사용자가 관리하는 문자열 배열(context)로 유지.
    """

    def __init__(
        self,
        llm_handler: LLMCallHandler,
        max_iterations: int = 10,
        context_window_size: int = 8000,
        enable_context_compression: bool = True
    ):
        """
        초기화.

        Args:
            llm_handler: LLM 호출 핸들러
            max_iterations: 최대 반복 횟수
            context_window_size: 컨텍스트 윈도우 크기 (토큰)
            enable_context_compression: 컨텍스트 압축 활성화
        """
        self.llm_handler = llm_handler
        self.max_iterations = max_iterations
        self.context_window_size = context_window_size
        self.enable_context_compression = enable_context_compression

        self.active_loops: Dict[str, LoopContext] = {}
        self.completed_loops: Dict[str, LoopResult] = {}

        # 콜백 함수들
        self.loop_callbacks: List[Callable] = []
        self.iteration_callbacks: List[Callable] = []

        logger.info(f"LLMLoopManager initialized: max_iterations={max_iterations}, "
                   f"context_window={context_window_size}")

    async def start_conversation_loop(
        self,
        conversation_id: str,
        initial_messages: List[ConversationMessage],
        termination_condition: Optional[Callable[[LoopContext], LoopAction]] = None,
        iteration_processor: Optional[Callable[[LoopContext, Dict[str, Any]], Dict[str, Any]]] = None,
        **llm_kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        대화 루프 시작.

        Args:
            conversation_id: 대화 ID
            initial_messages: 초기 메시지들
            termination_condition: 종료 조건 함수
            iteration_processor: 반복 처리 함수
            **llm_kwargs: LLM 호출 추가 파라미터

        Yields:
            각 반복의 결과
        """
        # 루프 컨텍스트 생성
        loop_context = LoopContext(
            conversation_id=conversation_id,
            messages=initial_messages.copy(),
            max_iterations=self.max_iterations,
            state=LoopState.RUNNING
        )

        self.active_loops[conversation_id] = loop_context

        start_time = time.time()
        logger.info(f"Started conversation loop: {conversation_id}")

        try:
            # 콜백 호출
            await self._trigger_callbacks(self.loop_callbacks, 'started', loop_context)

            while loop_context.current_iteration < loop_context.max_iterations:
                loop_context.current_iteration += 1

                # 반복 시작 콜백
                await self._trigger_callbacks(self.iteration_callbacks, 'iteration_start', loop_context)

                # 컨텍스트 압축
                if self.enable_context_compression:
                    compressed_messages = loop_context.compress_context(self.context_window_size)
                    if len(compressed_messages) != len(loop_context.messages):
                        logger.debug(f"Context compressed: {len(loop_context.messages)} -> {len(compressed_messages)} messages")

                # LLM 메시지 포맷 변환
                llm_messages = [msg.to_dict() for msg in loop_context.messages]

                # LLM 호출
                llm_result = await self.llm_handler.call_llm(llm_messages, **llm_kwargs)

                if not llm_result['success']:
                    # LLM 호출 실패
                    error_msg = ConversationMessage(
                        role="system",
                        content=f"LLM call failed: {llm_result['error']}",
                        timestamp=time.time(),
                        metadata={'error': True, 'iteration': loop_context.current_iteration}
                    )
                    loop_context.add_message(error_msg)
                    loop_context.state = LoopState.FAILED

                    yield {
                        'iteration': loop_context.current_iteration,
                        'success': False,
                        'error': llm_result['error'],
                        'context': loop_context
                    }
                    break

                # LLM 응답 처리
                llm_response = llm_result['response']
                response_msg = ConversationMessage(
                    role="assistant",
                    content=llm_response,
                    timestamp=time.time(),
                    metadata={'iteration': loop_context.current_iteration}
                )
                loop_context.add_message(response_msg)

                # 반복 처리
                if iteration_processor:
                    processed_result = await iteration_processor(loop_context, llm_result)
                    if processed_result:
                        # 처리 결과에 따른 추가 메시지
                        if 'additional_messages' in processed_result:
                            for msg_data in processed_result['additional_messages']:
                                additional_msg = ConversationMessage(**msg_data)
                                loop_context.add_message(additional_msg)

                # 종료 조건 확인
                action = LoopAction.CONTINUE
                if termination_condition:
                    action = await termination_condition(loop_context)

                # 반복 종료 콜백
                await self._trigger_callbacks(self.iteration_callbacks, 'iteration_end', loop_context, action)

                # 결과 생성
                iteration_result = {
                    'iteration': loop_context.current_iteration,
                    'success': True,
                    'llm_response': llm_response,
                    'action': action.value,
                    'context': loop_context,
                    'total_tokens': loop_context.total_tokens
                }

                yield iteration_result

                # 액션 처리
                if action == LoopAction.TERMINATE:
                    loop_context.state = LoopState.COMPLETED
                    logger.info(f"Conversation loop terminated: {conversation_id}")
                    break
                elif action == LoopAction.PAUSE:
                    loop_context.state = LoopState.WAITING
                    logger.info(f"Conversation loop paused: {conversation_id}")
                    break
                elif action == LoopAction.RETRY:
                    # 재시도는 다음 반복에서 처리
                    continue

            else:
                # 최대 반복 횟수 초과
                loop_context.state = LoopState.COMPLETED
                logger.warning(f"Conversation loop reached max iterations: {conversation_id}")

        except Exception as e:
            loop_context.state = LoopState.FAILED
            logger.error(f"Conversation loop failed: {conversation_id}, error: {e}")

            yield {
                'iteration': loop_context.current_iteration,
                'success': False,
                'error': str(e),
                'context': loop_context
            }

        finally:
            # 루프 완료 처리
            execution_time = time.time() - start_time

            result = LoopResult(
                conversation_id=conversation_id,
                final_state=loop_context.state,
                total_iterations=loop_context.current_iteration,
                messages=loop_context.messages,
                final_response=loop_context.messages[-1].content if loop_context.messages else None,
                execution_time=execution_time
            )

            # 완료된 루프 저장
            self.completed_loops[conversation_id] = result

            # 활성 루프에서 제거
            if conversation_id in self.active_loops:
                del self.active_loops[conversation_id]

            # 콜백 호출
            await self._trigger_callbacks(self.loop_callbacks, 'completed', result)

            logger.info(f"Conversation loop completed: {conversation_id}, "
                       f"iterations={result.total_iterations}, "
                       f"state={result.final_state.value}, "
                       f"time={execution_time:.2f}s")

    async def resume_conversation_loop(
        self,
        conversation_id: str,
        additional_messages: Optional[List[ConversationMessage]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        대화 루프 재개.

        Args:
            conversation_id: 대화 ID
            additional_messages: 추가 메시지들
            **kwargs: 추가 파라미터
        """
        if conversation_id not in self.completed_loops:
            raise ValueError(f"Conversation {conversation_id} not found")

        result = self.completed_loops[conversation_id]

        # 새 컨텍스트 생성
        loop_context = LoopContext(
            conversation_id=conversation_id,
            messages=result.messages.copy(),
            max_iterations=self.max_iterations,
            current_iteration=result.total_iterations,
            state=LoopState.RUNNING
        )

        if additional_messages:
            for msg in additional_messages:
                loop_context.add_message(msg)

        self.active_loops[conversation_id] = loop_context

        # 재개된 루프 실행
        async for result in self.start_conversation_loop(
            conversation_id=conversation_id,
            initial_messages=[],  # 이미 컨텍스트에 있음
            **kwargs
        ):
            # 컨텍스트 업데이트
            if 'context' in result:
                self.active_loops[conversation_id] = result['context']
            yield result

    async def _trigger_callbacks(self, callbacks: List[Callable], event: str, *args, **kwargs):
        """콜백 함수들 호출."""
        for callback in callbacks:
            try:
                await callback(event, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback failed: {e}")

    def add_loop_callback(self, callback: Callable):
        """루프 콜백 등록."""
        self.loop_callbacks.append(callback)

    def add_iteration_callback(self, callback: Callable):
        """반복 콜백 등록."""
        self.iteration_callbacks.append(callback)

    def get_active_loops(self) -> Dict[str, LoopContext]:
        """활성 루프들 반환."""
        return self.active_loops.copy()

    def get_completed_loops(self) -> Dict[str, LoopResult]:
        """완료된 루프들 반환."""
        return self.completed_loops.copy()

    def get_loop_stats(self) -> Dict[str, Any]:
        """루프 통계."""
        active_count = len(self.active_loops)
        completed_count = len(self.completed_loops)

        total_iterations = sum(result.total_iterations for result in self.completed_loops.values())
        total_execution_time = sum(result.execution_time for result in self.completed_loops.values())

        avg_iterations = total_iterations / completed_count if completed_count > 0 else 0
        avg_execution_time = total_execution_time / completed_count if completed_count > 0 else 0

        return {
            'active_loops': active_count,
            'completed_loops': completed_count,
            'total_iterations': total_iterations,
            'total_execution_time': total_execution_time,
            'avg_iterations_per_loop': avg_iterations,
            'avg_execution_time_per_loop': avg_execution_time
        }


# 헬퍼 함수들
def create_simple_termination_condition(max_iterations: int = 5) -> Callable:
    """단순 종료 조건 생성."""
    async def termination_condition(context: LoopContext) -> LoopAction:
        if context.current_iteration >= max_iterations:
            return LoopAction.TERMINATE

        # 마지막 메시지가 "종료"나 "완료"를 포함하면 종료
        if context.messages:
            last_message = context.messages[-1]
            if last_message.role == "assistant":
                content_lower = last_message.content.lower()
                if any(keyword in content_lower for keyword in ["종료", "완료", "끝", "finished", "done"]):
                    return LoopAction.TERMINATE

        return LoopAction.CONTINUE

    return termination_condition


def create_keyword_termination_condition(keywords: List[str]) -> Callable:
    """키워드 기반 종료 조건 생성."""
    async def termination_condition(context: LoopContext) -> LoopAction:
        if context.messages:
            last_message = context.messages[-1]
            if last_message.role == "assistant":
                content_lower = last_message.content.lower()
                if any(keyword.lower() in content_lower for keyword in keywords):
                    return LoopAction.TERMINATE

        return LoopAction.CONTINUE

    return termination_condition


# 전역 LLM 루프 매니저 인스턴스
_llm_loop_manager = None

def get_llm_loop_manager() -> Optional[LLMLoopManager]:
    """전역 LLM 루프 매니저 인스턴스 반환."""
    return _llm_loop_manager

def set_llm_loop_manager(manager: LLMLoopManager):
    """전역 LLM 루프 매니저 설정."""
    global _llm_loop_manager
    _llm_loop_manager = manager


@asynccontextmanager
async def conversation_context(conversation_id: str, llm_handler: LLMCallHandler):
    """
    대화 컨텍스트 매니저.

    자동으로 LLM 루프 매니저를 생성하고 정리.
    """
    manager = LLMLoopManager(llm_handler)
    previous_manager = get_llm_loop_manager()
    set_llm_loop_manager(manager)

    try:
        yield manager
    finally:
        # 정리
        stats = manager.get_loop_stats()
        logger.info(f"Conversation context cleaned up: {stats}")

        # 이전 매니저 복원
        set_llm_loop_manager(previous_manager)
