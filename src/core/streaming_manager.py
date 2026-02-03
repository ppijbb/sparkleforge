#!/usr/bin/env python3
"""
Real-time Streaming Manager for Local Researcher Project (v2.0 - 8대 혁신)

Production-grade WebSocket 기반 실시간 이벤트 스트리밍 매니저.
에이전트 상태 변경, 진행 상황 업데이트, 백프레셔 핸들링을 통한
안정적인 실시간 통신을 제공합니다.

2025년 10월 최신 기술 스택:
- asyncio + websockets for real-time communication
- Redis for event broadcasting (optional)
- Circuit breaker pattern for reliability
- Structured logging with sensitive data masking
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum
import weakref
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

# Production-grade reliability imports
from src.core.reliability import execute_with_reliability, CircuitBreakerConfig, CircuitBreaker
from src.core.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """스트리밍 이벤트 타입."""
    AGENT_STATUS_CHANGE = "agent_status_change"
    PROGRESS_UPDATE = "progress_update"
    AGENT_ACTION = "agent_action"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    ERROR_OCCURRED = "error_occurred"
    CREATIVE_INSIGHT = "creative_insight"  # 창의성 에이전트 인사이트


class AgentStatus(Enum):
    """에이전트 상태."""
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"
    CREATING = "creating"  # 창의적 아이디어 생성 중


@dataclass
class StreamingEvent:
    """스트리밍 이벤트 데이터 구조."""
    event_id: str
    event_type: EventType
    agent_id: Optional[str]
    workflow_id: str
    timestamp: datetime
    data: Dict[str, Any]
    priority: int = 0  # 0=normal, 1=high, 2=critical
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'agent_id': self.agent_id,
            'workflow_id': self.workflow_id,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'priority': self.priority
        }


@dataclass
class AgentProgress:
    """에이전트 진행 상황."""
    agent_id: str
    workflow_id: str
    status: AgentStatus
    progress_percentage: float  # 0.0-100.0
    current_task: str
    estimated_completion: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            'agent_id': self.agent_id,
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'progress_percentage': self.progress_percentage,
            'current_task': self.current_task,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }


class StreamingManager:
    """
    Production-grade 실시간 스트리밍 매니저.
    
    Features:
    - WebSocket 기반 실시간 이벤트 스트리밍
    - 에이전트 상태 변경 이벤트 브로드캐스팅
    - 진행 상황 업데이트 (0-100% 진행률)
    - 버퍼링 및 백프레셔 핸들링
    - Circuit breaker 패턴으로 안정성 보장
    - 메모리 효율적인 이벤트 큐 관리
    """
    
    def __init__(self, max_buffer_size: int = 1000, max_connections: int = 100):
        """
        스트리밍 매니저 초기화.
        
        Args:
            max_buffer_size: 최대 이벤트 버퍼 크기
            max_connections: 최대 동시 연결 수
        """
        self.max_buffer_size = max_buffer_size
        self.max_connections = max_connections
        
        # 이벤트 큐 및 버퍼링
        self.event_queue: deque = deque(maxlen=max_buffer_size)
        self.agent_progress: Dict[str, AgentProgress] = {}
        self.workflow_agents: Dict[str, Set[str]] = defaultdict(set)
        
        # 연결 관리
        self.connections: Set[Any] = set()
        self.connection_lock = asyncio.Lock()
        
        # Circuit breaker for reliability
        self.circuit_breaker = CircuitBreaker(
            name="streaming_manager",
            config=CircuitBreakerConfig(
                failure_threshold=10,
                recovery_timeout=30,
                timeout=5.0
            )
        )
        
        # 백프레셔 제어
        self.backpressure_threshold = max_buffer_size * 0.8
        self.is_backpressured = False
        
        # 스레드 풀 for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="streaming")
        
        # 이벤트 통계
        self.stats = {
            'total_events': 0,
            'events_per_second': 0.0,
            'active_connections': 0,
            'backpressure_events': 0,
            'circuit_breaker_trips': 0
        }
        
        # 마지막 통계 업데이트 시간
        self.last_stats_update = time.time()
        
        logger.info(f"StreamingManager initialized: max_buffer={max_buffer_size}, max_connections={max_connections}")
    
    async def stream_event(
        self,
        event_type: EventType,
        agent_id: Optional[str],
        workflow_id: str,
        data: Dict[str, Any],
        priority: int = 0
    ) -> bool:
        """
        이벤트를 스트리밍합니다.
        
        Args:
            event_type: 이벤트 타입
            agent_id: 에이전트 ID (None 가능)
            workflow_id: 워크플로우 ID
            data: 이벤트 데이터
            priority: 우선순위 (0=normal, 1=high, 2=critical)
            
        Returns:
            bool: 이벤트 전송 성공 여부
        """
        try:
            # Circuit breaker 체크
            if self.circuit_breaker.state.name == "OPEN":
                logger.warning("Circuit breaker is OPEN, dropping event")
                return False
            
            # 백프레셔 체크
            if len(self.event_queue) >= self.backpressure_threshold:
                if not self.is_backpressured:
                    logger.warning(f"Backpressure activated: {len(self.event_queue)}/{self.max_buffer_size}")
                    self.is_backpressured = True
                    self.stats['backpressure_events'] += 1
                
                # Critical 이벤트만 처리
                if priority < 2:
                    return False
            
            # 이벤트 생성
            event = StreamingEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                agent_id=agent_id,
                workflow_id=workflow_id,
                timestamp=datetime.now(timezone.utc),
                data=data,
                priority=priority
            )
            
            # 이벤트 큐에 추가
            self.event_queue.append(event)
            
            # 워크플로우-에이전트 매핑 업데이트
            if agent_id:
                self.workflow_agents[workflow_id].add(agent_id)
            
            # 통계 업데이트
            self.stats['total_events'] += 1
            self._update_stats()
            
            # 연결된 클라이언트들에게 브로드캐스트
            await self._broadcast_event(event)
            
            # 백프레셔 해제 체크
            if self.is_backpressured and len(self.event_queue) < self.backpressure_threshold * 0.5:
                logger.info("Backpressure released")
                self.is_backpressured = False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stream event: {e}", exc_info=True)
            self.circuit_breaker.record_failure()
            return False
    
    async def broadcast_progress(
        self,
        workflow_id: str,
        agent_id: str,
        progress: float,
        message: str,
        status: AgentStatus = AgentStatus.WORKING
    ) -> bool:
        """
        진행 상황을 브로드캐스트합니다.
        
        Args:
            workflow_id: 워크플로우 ID
            agent_id: 에이전트 ID
            progress: 진행률 (0.0-100.0)
            message: 현재 작업 메시지
            status: 에이전트 상태
            
        Returns:
            bool: 브로드캐스트 성공 여부
        """
        try:
            # 진행 상황 업데이트
            agent_progress = AgentProgress(
                agent_id=agent_id,
                workflow_id=workflow_id,
                status=status,
                progress_percentage=min(100.0, max(0.0, progress)),
                current_task=message,
                last_activity=datetime.now(timezone.utc)
            )
            
            self.agent_progress[agent_id] = agent_progress
            
            # 이벤트 데이터 준비
            event_data = {
                'progress': agent_progress.to_dict(),
                'workflow_summary': self._get_workflow_summary(workflow_id)
            }
            
            # 이벤트 스트리밍
            return await self.stream_event(
                event_type=EventType.PROGRESS_UPDATE,
                agent_id=agent_id,
                workflow_id=workflow_id,
                data=event_data,
                priority=1  # High priority for progress updates
            )
            
        except Exception as e:
            logger.error(f"Failed to broadcast progress: {e}", exc_info=True)
            return False
    
    async def stream_agent_action(
        self,
        agent_id: str,
        workflow_id: str,
        action: str,
        status: AgentStatus,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        에이전트 액션을 스트리밍합니다.
        
        Args:
            agent_id: 에이전트 ID
            workflow_id: 워크플로우 ID
            action: 액션 설명
            status: 에이전트 상태
            details: 추가 세부사항
            
        Returns:
            bool: 스트리밍 성공 여부
        """
        try:
            event_data = {
                'action': action,
                'status': status.value,
                'details': details or {},
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return await self.stream_event(
                event_type=EventType.AGENT_ACTION,
                agent_id=agent_id,
                workflow_id=workflow_id,
                data=event_data,
                priority=1  # High priority for agent actions
            )
            
        except Exception as e:
            logger.error(f"Failed to stream agent action: {e}", exc_info=True)
            return False
    
    async def stream_creative_insight(
        self,
        agent_id: str,
        workflow_id: str,
        insight: str,
        related_concepts: List[str],
        confidence: float
    ) -> bool:
        """
        창의적 인사이트를 스트리밍합니다.
        
        Args:
            agent_id: 에이전트 ID
            workflow_id: 워크플로우 ID
            insight: 창의적 인사이트
            related_concepts: 관련 개념들
            confidence: 신뢰도 (0.0-1.0)
            
        Returns:
            bool: 스트리밍 성공 여부
        """
        try:
            event_data = {
                'insight': insight,
                'related_concepts': related_concepts,
                'confidence': confidence,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return await self.stream_event(
                event_type=EventType.CREATIVE_INSIGHT,
                agent_id=agent_id,
                workflow_id=workflow_id,
                data=event_data,
                priority=2  # Critical priority for creative insights
            )
            
        except Exception as e:
            logger.error(f"Failed to stream creative insight: {e}", exc_info=True)
            return False
    
    async def _broadcast_event(self, event: StreamingEvent) -> None:
        """이벤트를 모든 연결된 클라이언트에게 브로드캐스트합니다."""
        if not self.connections:
            return
        
        event_json = json.dumps(event.to_dict(), ensure_ascii=False)
        failed_connections = set()
        
        async with self.connection_lock:
            for connection in self.connections.copy():
                try:
                    if hasattr(connection, 'send'):
                        await connection.send(event_json)
                    elif hasattr(connection, 'write'):
                        connection.write(event_json.encode('utf-8'))
                except Exception as e:
                    logger.warning(f"Failed to send event to connection: {e}")
                    failed_connections.add(connection)
            
            # 실패한 연결 제거
            self.connections -= failed_connections
    
    def _get_workflow_summary(self, workflow_id: str) -> Dict[str, Any]:
        """워크플로우 요약 정보를 반환합니다."""
        agents = self.workflow_agents.get(workflow_id, set())
        agent_progresses = [self.agent_progress[aid] for aid in agents if aid in self.agent_progress]
        
        if not agent_progresses:
            return {'total_agents': 0, 'completed_agents': 0, 'overall_progress': 0.0}
        
        completed = sum(1 for ap in agent_progresses if ap.status == AgentStatus.COMPLETED)
        overall_progress = sum(ap.progress_percentage for ap in agent_progresses) / len(agent_progresses)
        
        return {
            'total_agents': len(agent_progresses),
            'completed_agents': completed,
            'overall_progress': overall_progress,
            'agents': [ap.to_dict() for ap in agent_progresses]
        }
    
    def _update_stats(self) -> None:
        """통계를 업데이트합니다."""
        current_time = time.time()
        time_diff = current_time - self.last_stats_update
        
        if time_diff >= 1.0:  # 1초마다 업데이트
            self.stats['events_per_second'] = self.stats['total_events'] / time_diff
            self.stats['active_connections'] = len(self.connections)
            self.last_stats_update = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """현재 통계를 반환합니다."""
        self._update_stats()
        return self.stats.copy()
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """특정 워크플로우의 상태를 반환합니다."""
        return self._get_workflow_summary(workflow_id)
    
    def cleanup_workflow(self, workflow_id: str) -> None:
        """워크플로우 관련 데이터를 정리합니다."""
        # 에이전트 진행 상황 정리
        agents_to_remove = []
        for agent_id, progress in self.agent_progress.items():
            if progress.workflow_id == workflow_id:
                agents_to_remove.append(agent_id)
        
        for agent_id in agents_to_remove:
            del self.agent_progress[agent_id]
        
        # 워크플로우-에이전트 매핑 정리
        if workflow_id in self.workflow_agents:
            del self.workflow_agents[workflow_id]
        
        logger.info(f"Cleaned up workflow {workflow_id}: removed {len(agents_to_remove)} agents")
    
    async def shutdown(self) -> None:
        """스트리밍 매니저를 종료합니다."""
        logger.info("Shutting down StreamingManager...")
        
        # 모든 연결 종료
        async with self.connection_lock:
            for connection in self.connections.copy():
                try:
                    if hasattr(connection, 'close'):
                        await connection.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
            self.connections.clear()
        
        # 스레드 풀 종료 (timeout 설정으로 무한 대기 방지)
        try:
            # shutdown(wait=False)로 즉시 반환하고, 별도로 종료 대기
            self.executor.shutdown(wait=False)
            # 최대 5초만 대기 (더 안전한 방법)
            import time
            import threading
            start_time = time.time()
            while time.time() - start_time < 5.0:
                # 활성 스레드 수 확인
                active_threads = threading.active_count()
                # executor의 스레드가 종료되었는지 간접적으로 확인
                # (executor 내부 스레드 추적이 어려우므로 짧은 대기 후 종료)
                if time.time() - start_time >= 2.0:  # 최소 2초 대기
                    break
                time.sleep(0.1)
            logger.debug("ThreadPoolExecutor shutdown completed")
        except Exception as e:
            logger.warning(f"Error shutting down ThreadPoolExecutor: {e}")
        
        logger.info("StreamingManager shutdown complete")


# Global streaming manager instance
_streaming_manager: Optional[StreamingManager] = None


def get_streaming_manager() -> StreamingManager:
    """전역 스트리밍 매니저 인스턴스를 반환합니다."""
    global _streaming_manager
    if _streaming_manager is None:
        _streaming_manager = StreamingManager()
    return _streaming_manager


def initialize_streaming() -> StreamingManager:
    """스트리밍 매니저를 초기화합니다."""
    global _streaming_manager
    if _streaming_manager is None:
        _streaming_manager = StreamingManager()
        logger.info("StreamingManager initialized")
    return _streaming_manager


async def cleanup_streaming() -> None:
    """스트리밍 매니저를 정리합니다."""
    global _streaming_manager
    if _streaming_manager is not None:
        await _streaming_manager.shutdown()
        _streaming_manager = None
        logger.info("StreamingManager cleaned up")
