"""
Background Memory Generation Service

백서 요구사항: 메모리 생성은 비동기 백그라운드 프로세스
- 이벤트 기반 메모리 생성 (세션 종료 시 트리거)
- 내부 큐 관리 및 재시도 메커니즘
- Dead-letter queue 지원
- 사용자 경험 차단 없음 (non-blocking)
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json

from src.core.memory_extraction import get_memory_extractor, get_memory_consolidator
from src.core.memory_types import BaseMemory
from src.core.memory_provenance import get_provenance_tracker
from src.core.memory_realtime_processor import (
    RealtimeMemoryProcessor,
    get_realtime_memory_processor,
)

logger = logging.getLogger(__name__)


class MemoryTaskStatus(Enum):
    """메모리 작업 상태."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class MemoryGenerationTask:
    """메모리 생성 작업."""
    task_id: str
    session_id: str
    user_id: str
    conversation_history: List[Dict[str, Any]]
    status: MemoryTaskStatus = MemoryTaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "error": self.error,
            "metadata": self.metadata
        }


class BackgroundMemoryService:
    """
    백그라운드 메모리 생성 서비스.
    
    백서 요구사항:
    - 비동기 처리 (사용자 경험 차단 없음)
    - 이벤트 기반 (세션 종료 시 트리거)
    - 내부 큐 관리
    - 재시도 메커니즘 (exponential backoff)
    - Dead-letter queue
    
    Phase 2: 실시간 업데이트 모드 추가
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 5,
        retry_delays: List[float] = None,
        inactivity_delay: float = 30.0,  # 30초 비활성 후 처리
        update_mode: str = "background"  # "background" | "realtime" (기본값: background, 기존 동작 유지)
    ):
        """
        초기화.
        
        Args:
            max_concurrent_tasks: 최대 동시 처리 작업 수
            retry_delays: 재시도 지연 시간 (초) 리스트
            inactivity_delay: 비활성 지연 시간 (초)
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.retry_delays = retry_delays or [1.0, 5.0, 15.0]  # Exponential backoff
        self.inactivity_delay = inactivity_delay
        self.update_mode = update_mode  # "background" | "realtime"
        
        # 작업 큐 (백그라운드 모드용)
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.processing_tasks: Dict[str, MemoryGenerationTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)  # 최근 1000개만 유지
        self.dead_letter_queue: List[MemoryGenerationTask] = []
        
        # 작업 저장소 (영구 저장용)
        self.task_storage: Dict[str, MemoryGenerationTask] = {}
        
        # 서비스 상태
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        # 메모리 추출기/통합기
        self.extractor = get_memory_extractor()
        self.consolidator = get_memory_consolidator()
        self.provenance_tracker = get_provenance_tracker()
        
        # 실시간 프로세서 (실시간 모드용)
        self.realtime_processor: Optional[RealtimeMemoryProcessor] = None
        if self.update_mode == "realtime":
            self.realtime_processor = get_realtime_memory_processor()
        
        # 콜백 함수 (메모리 생성 완료 시 호출)
        self.on_memory_generated: Optional[Callable[[str, List[BaseMemory]], None]] = None
        
        logger.info(
            f"BackgroundMemoryService initialized: "
            f"max_concurrent={max_concurrent_tasks}, mode={update_mode}"
        )
    
    async def start(self):
        """서비스 시작."""
        if self.is_running:
            logger.warning("Service is already running")
            return
        
        self.is_running = True
        
        # 워커 태스크 시작
        for i in range(self.max_concurrent_tasks):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        logger.info(f"BackgroundMemoryService started with {self.max_concurrent_tasks} workers")
    
    async def stop(self):
        """서비스 중지."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 워커 태스크 종료 대기
        for task in self.worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        logger.info("BackgroundMemoryService stopped")
    
    async def submit_memory_generation(
        self,
        session_id: str,
        user_id: str,
        conversation_history: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        메모리 생성 작업 제출 (non-blocking).
        
        Args:
            session_id: 세션 ID
            user_id: 사용자 ID
            conversation_history: 대화 히스토리
            metadata: 추가 메타데이터
            
        Returns:
            작업 ID
        """
        # 실시간 모드: 즉시 처리
        if self.update_mode == "realtime" and self.realtime_processor:
            try:
                # 마지막 메시지만 실시간 처리 (증분 업데이트)
                if conversation_history:
                    last_message = conversation_history[-1]
                    await self.realtime_processor.process_message(
                        session_id, user_id, last_message
                    )
                    logger.debug(f"Realtime memory processed for session {session_id}")
            except Exception as e:
                logger.error(f"Realtime memory processing failed: {e}")
                # 실패 시 백그라운드 모드로 폴백
        
        # 백그라운드 모드 또는 실시간 실패 시: 큐에 추가
        task_id = f"mem_task_{uuid.uuid4().hex[:12]}"
        
        task = MemoryGenerationTask(
            task_id=task_id,
            session_id=session_id,
            user_id=user_id,
            conversation_history=conversation_history,
            metadata=metadata or {}
        )
        
        # 큐에 추가
        await self.task_queue.put(task)
        self.task_storage[task_id] = task
        
        logger.info(f"Memory generation task submitted: {task_id} for session {session_id}")
        return task_id
    
    async def on_message_added(
        self,
        session_id: str,
        user_id: str,
        message: Dict[str, Any]
    ) -> None:
        """
        메시지 추가 이벤트 핸들러 (Phase 2: 실시간 업데이트).
        
        Args:
            session_id: 세션 ID
            user_id: 사용자 ID
            message: 추가된 메시지
        """
        if self.update_mode == "realtime" and self.realtime_processor:
            try:
                await self.realtime_processor.process_message(
                    session_id, user_id, message
                )
                logger.debug(f"Realtime memory updated for message in session {session_id}")
            except Exception as e:
                logger.error(f"Realtime memory update failed: {e}")
                # 실패 시 백그라운드 큐에 추가
                await self.submit_memory_generation(
                    session_id, user_id, [message]
                )
    
    def set_update_mode(self, mode: str) -> None:
        """
        업데이트 모드 변경.
        
        Args:
            mode: "background" | "realtime"
        """
        if mode not in ["background", "realtime"]:
            raise ValueError(f"Invalid update mode: {mode}. Must be 'background' or 'realtime'")
        
        self.update_mode = mode
        
        if mode == "realtime" and self.realtime_processor is None:
            self.realtime_processor = get_realtime_memory_processor()
        elif mode == "background":
            # 실시간 프로세서는 유지하되 사용하지 않음
            pass
        
        logger.info(f"Update mode changed to: {mode}")
    
    async def _worker(self, worker_name: str):
        """워커 루프 (작업 처리)."""
        logger.info(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                # 큐에서 작업 가져오기 (타임아웃: 1초)
                try:
                    task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 처리 중 상태로 변경
                task.status = MemoryTaskStatus.PROCESSING
                task.started_at = datetime.now()
                self.processing_tasks[task.task_id] = task
                
                try:
                    # 메모리 생성 실행
                    await self._process_memory_generation(task)
                    
                    # 완료 상태
                    task.status = MemoryTaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    self.completed_tasks.append(task)
                    
                    logger.info(f"Memory generation completed: {task.task_id}")
                    
                except Exception as e:
                    # 실패 처리
                    await self._handle_task_failure(task, e)
                
                finally:
                    # 처리 중 목록에서 제거
                    self.processing_tasks.pop(task.task_id, None)
                    self.task_queue.task_done()
                    
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}", exc_info=True)
                await asyncio.sleep(1)  # 에러 후 잠시 대기
    
    async def _process_memory_generation(self, task: MemoryGenerationTask):
        """메모리 생성 처리."""
        try:
            # 1. Extraction: 대화에서 메모리 추출
            extracted_memories = await self.extractor.extract_memories(
                conversation_history=task.conversation_history,
                session_id=task.session_id,
                user_id=task.user_id,
                turn=None  # 전체 세션에서 추출
            )
            
            if not extracted_memories:
                logger.debug(f"No memories extracted for task {task.task_id}")
                return
            
            # 2. 기존 메모리 로드 (통합을 위해)
            existing_memories = await self._load_existing_memories(task.user_id)
            
            # 3. Consolidation: 기존 메모리와 통합
            consolidation_result = await self.consolidator.consolidate_memories(
                new_memories=extracted_memories,
                existing_memories=existing_memories,
                user_id=task.user_id
            )
            
            # 4. 통합된 메모리 저장
            consolidated = consolidation_result["consolidated"]
            await self._save_memories(consolidated, task.user_id)
            
            # 5. 콜백 호출 (메모리 저장소에 저장 등)
            if self.on_memory_generated:
                try:
                    self.on_memory_generated(task.session_id, consolidated)
                except Exception as e:
                    logger.error(f"Memory generation callback failed: {e}")
            
            logger.info(f"Generated {len(consolidated)} memories for user {task.user_id}")
            
        except Exception as e:
            logger.error(f"Memory generation failed for task {task.task_id}: {e}", exc_info=True)
            raise
    
    async def _handle_task_failure(
        self,
        task: MemoryGenerationTask,
        error: Exception
    ):
        """작업 실패 처리 (재시도 또는 Dead-letter)."""
        task.retry_count += 1
        task.error = str(error)
        
        if task.retry_count <= task.max_retries:
            # 재시도
            delay = self.retry_delays[min(task.retry_count - 1, len(self.retry_delays) - 1)]
            task.status = MemoryTaskStatus.PENDING
            
            logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries}) after {delay}s")
            
            # 지연 후 재큐
            await asyncio.sleep(delay)
            await self.task_queue.put(task)
        else:
            # Dead-letter queue로 이동
            task.status = MemoryTaskStatus.DEAD_LETTER
            self.dead_letter_queue.append(task)
            
            logger.error(f"Task {task.task_id} moved to dead-letter queue after {task.max_retries} retries")
    
    async def _load_existing_memories(self, user_id: str) -> List[BaseMemory]:
        """기존 메모리 로드."""
        try:
            from src.core.adaptive_memory import get_adaptive_memory
            adaptive_memory = get_adaptive_memory()
            return adaptive_memory.get_all_memories(user_id=user_id)
        except Exception as e:
            logger.debug(f"Failed to load existing memories: {e}")
            return []
    
    async def _save_memories(self, memories: List[BaseMemory], user_id: str):
        """메모리 저장 (검증 후 저장)."""
        try:
            from src.core.adaptive_memory import get_adaptive_memory
            from src.core.memory_validation import get_memory_validator
            
            adaptive_memory = get_adaptive_memory()
            validator = get_memory_validator()
            
            saved_count = 0
            for memory in memories:
                # 검증
                validation_result = await validator.validate_memory(memory, user_id)
                
                if validation_result.is_valid:
                    # 검증된 메모리만 저장
                    if validation_result.sanitized_content:
                        memory.content = validation_result.sanitized_content
                    
                    adaptive_memory.store_memory(memory)
                    saved_count += 1
                else:
                    logger.warning(f"Memory {memory.memory_id} failed validation: {validation_result.issues}")
            
            logger.info(f"Saved {saved_count}/{len(memories)} validated memories for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회."""
        task = self.task_storage.get(task_id)
        if task:
            return task.to_dict()
        return None
    
    def get_dead_letter_tasks(self) -> List[Dict[str, Any]]:
        """Dead-letter queue 작업 목록."""
        return [task.to_dict() for task in self.dead_letter_queue]
    
    def get_service_stats(self) -> Dict[str, Any]:
        """서비스 통계."""
        return {
            "is_running": self.is_running,
            "queue_size": self.task_queue.qsize(),
            "processing_count": len(self.processing_tasks),
            "completed_count": len(self.completed_tasks),
            "dead_letter_count": len(self.dead_letter_queue),
            "total_tasks": len(self.task_storage)
        }


# 전역 인스턴스
_background_memory_service: Optional[BackgroundMemoryService] = None


def get_background_memory_service(
    update_mode: Optional[str] = None  # None이면 환경변수 또는 기본값 사용
) -> BackgroundMemoryService:
    """
    전역 백그라운드 메모리 서비스 인스턴스 반환.
    
    Args:
        update_mode: 업데이트 모드 ("background" | "realtime"), None이면 환경변수 또는 기본값 사용
        
    Returns:
        BackgroundMemoryService 인스턴스
    """
    global _background_memory_service
    if _background_memory_service is None:
        import os
        # 환경변수에서 모드 읽기 (기본값: background, 기존 동작 유지)
        env_mode = os.getenv("MEMORY_UPDATE_MODE", "background")
        mode = update_mode or env_mode
        _background_memory_service = BackgroundMemoryService(update_mode=mode)
    return _background_memory_service

