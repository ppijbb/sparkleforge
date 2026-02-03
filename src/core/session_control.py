"""
Session Control & Backoffice (24/7 연중무휴 + 세션 관리)

세션 검색, 제어(중지/재개/삭제), 연속성 보장, 세션 단위 작업 제어 기능 제공.
백오피스 수준의 세션 관리 시스템.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import uuid

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """세션 상태."""
    ACTIVE = "active"  # 실행 중
    PAUSED = "paused"  # 일시 중지
    COMPLETED = "completed"  # 완료
    FAILED = "failed"  # 실패
    CANCELLED = "cancelled"  # 취소됨
    WAITING = "waiting"  # 대기 중 (사용자 응답 등)


class TaskStatus(Enum):
    """작업 상태."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class SessionInfo:
    """세션 정보."""
    session_id: str
    status: SessionStatus
    created_at: datetime
    last_activity: datetime
    user_query: Optional[str] = None
    current_task: Optional[str] = None
    progress_percentage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0


@dataclass
class TaskInfo:
    """작업 정보."""
    task_id: str
    session_id: str
    status: TaskStatus
    task_type: str
    description: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionControl:
    """세션 제어 및 백오피스 시스템."""
    
    def __init__(self):
        """초기화."""
        from src.core.session_manager import get_session_manager
        from src.core.checkpoint_manager import CheckpointManager
        
        self.session_manager = get_session_manager()
        self.checkpoint_manager = CheckpointManager()
        
        # 활성 세션 추적
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_tasks: Dict[str, Dict[str, TaskInfo]] = {}  # session_id -> {task_id -> TaskInfo}
        self.session_controls: Dict[str, asyncio.Event] = {}  # session_id -> control event
        
        logger.info("SessionControl initialized")
    
    async def search_sessions(
        self,
        query: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        tags: Optional[List[str]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[SessionInfo]:
        """
        세션 검색.
        
        Args:
            query: 검색 쿼리 (user_query, session_id 등에서 검색)
            status: 상태 필터
            tags: 태그 필터
            created_after: 생성일 이후
            created_before: 생성일 이전
            limit: 최대 결과 수
            offset: 오프셋
        
        Returns:
            세션 정보 리스트
        """
        # 저장된 세션 목록 가져오기
        all_sessions = self.session_manager.list_sessions(limit=limit * 2, offset=offset)
        
        results = []
        
        for session_meta in all_sessions:
            session_id = session_meta.session_id
            
            # 활성 세션 정보와 병합
            active_info = self.active_sessions.get(session_id, {})
            
            # 상태 결정
            if session_id in self.active_sessions:
                status_val = active_info.get('status', SessionStatus.ACTIVE)
            else:
                status_val = SessionStatus.COMPLETED
            
            # 필터 적용
            if status and status_val != status:
                continue
            
            if tags and not any(tag in session_meta.tags for tag in tags):
                continue
            
            if created_after and session_meta.created_at < created_after:
                continue
            
            if created_before and session_meta.created_at > created_before:
                continue
            
            if query:
                query_lower = query.lower()
                if (query_lower not in session_id.lower() and
                    query_lower not in (session_meta.description or "").lower() and
                    query_lower not in (active_info.get('user_query', "") or "").lower()):
                    continue
            
            # SessionInfo 생성
            session_info = SessionInfo(
                session_id=session_id,
                status=status_val,
                created_at=session_meta.created_at,
                last_activity=session_meta.last_accessed,
                user_query=active_info.get('user_query'),
                current_task=active_info.get('current_task'),
                progress_percentage=active_info.get('progress', 0.0),
                metadata=active_info.get('metadata', {}),
                tags=session_meta.tags,
                error_count=active_info.get('error_count', 0),
                warning_count=active_info.get('warning_count', 0)
            )
            
            results.append(session_info)
        
        # 정렬 (최근 활동 순)
        results.sort(key=lambda x: x.last_activity, reverse=True)
        
        return results[:limit]
    
    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        세션 정보 조회.
        
        Args:
            session_id: 세션 ID
        
        Returns:
            SessionInfo 또는 None
        """
        # 활성 세션 확인
        if session_id in self.active_sessions:
            active_info = self.active_sessions[session_id]
            return SessionInfo(
                session_id=session_id,
                status=active_info.get('status', SessionStatus.ACTIVE),
                created_at=active_info.get('created_at', datetime.now()),
                last_activity=active_info.get('last_activity', datetime.now()),
                user_query=active_info.get('user_query'),
                current_task=active_info.get('current_task'),
                progress_percentage=active_info.get('progress', 0.0),
                metadata=active_info.get('metadata', {}),
                tags=active_info.get('tags', []),
                error_count=active_info.get('error_count', 0),
                warning_count=active_info.get('warning_count', 0)
            )
        
        # 저장된 세션 확인
        session_state = self.session_manager.load_session(session_id)
        if session_state:
            return SessionInfo(
                session_id=session_id,
                status=SessionStatus.COMPLETED,
                created_at=datetime.fromisoformat(session_state.metadata.get('created_at', datetime.now().isoformat())),
                last_activity=datetime.fromisoformat(session_state.metadata.get('last_accessed', datetime.now().isoformat())),
                metadata=session_state.metadata,
                tags=session_state.metadata.get('tags', [])
            )
        
        return None
    
    async def pause_session(self, session_id: str) -> bool:
        """
        세션 일시 중지.
        
        Args:
            session_id: 세션 ID
        
        Returns:
            성공 여부
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session not active: {session_id}")
            return False
        
        # 제어 이벤트 설정
        if session_id not in self.session_controls:
            self.session_controls[session_id] = asyncio.Event()
        
        self.session_controls[session_id].clear()  # pause 신호
        
        # 상태 업데이트
        self.active_sessions[session_id]['status'] = SessionStatus.PAUSED
        self.active_sessions[session_id]['paused_at'] = datetime.now()
        
        logger.info(f"Session paused: {session_id}")
        return True
    
    async def resume_session(self, session_id: str) -> bool:
        """
        세션 재개.
        
        Args:
            session_id: 세션 ID
        
        Returns:
            성공 여부
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session not active: {session_id}")
            return False
        
        # 제어 이벤트 설정
        if session_id in self.session_controls:
            self.session_controls[session_id].set()  # resume 신호
        
        # 상태 업데이트
        self.active_sessions[session_id]['status'] = SessionStatus.ACTIVE
        if 'paused_at' in self.active_sessions[session_id]:
            paused_at = self.active_sessions[session_id]['paused_at']
            pause_duration = (datetime.now() - paused_at).total_seconds()
            self.active_sessions[session_id]['total_pause_time'] = \
                self.active_sessions[session_id].get('total_pause_time', 0) + pause_duration
            del self.active_sessions[session_id]['paused_at']
        
        logger.info(f"Session resumed: {session_id}")
        return True
    
    async def cancel_session(self, session_id: str) -> bool:
        """
        세션 취소.
        
        Args:
            session_id: 세션 ID
        
        Returns:
            성공 여부
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session not active: {session_id}")
            return False
        
        # 상태 업데이트
        self.active_sessions[session_id]['status'] = SessionStatus.CANCELLED
        self.active_sessions[session_id]['cancelled_at'] = datetime.now()
        
        # 제어 이벤트 설정 (취소 신호)
        if session_id in self.session_controls:
            self.session_controls[session_id].clear()
        
        # 모든 작업 취소
        if session_id in self.session_tasks:
            for task_info in self.session_tasks[session_id].values():
                if task_info.status == TaskStatus.RUNNING:
                    task_info.status = TaskStatus.CANCELLED
                    task_info.completed_at = datetime.now()
        
        logger.info(f"Session cancelled: {session_id}")
        return True
    
    async def delete_session(self, session_id: str, delete_storage: bool = True) -> bool:
        """
        세션 삭제.
        
        Args:
            session_id: 세션 ID
            delete_storage: 저장소에서도 삭제할지 여부
        
        Returns:
            성공 여부
        """
        # 활성 세션에서 제거
        if session_id in self.active_sessions:
            # 먼저 취소
            await self.cancel_session(session_id)
            del self.active_sessions[session_id]
        
        # 제어 이벤트 제거
        if session_id in self.session_controls:
            del self.session_controls[session_id]
        
        # 작업 정보 제거
        if session_id in self.session_tasks:
            del self.session_tasks[session_id]
        
        # 저장소에서 삭제
        if delete_storage:
            success = self.session_manager.delete_session(session_id)
            if success:
                logger.info(f"Session deleted from storage: {session_id}")
            else:
                logger.warning(f"Failed to delete session from storage: {session_id}")
        
        logger.info(f"Session deleted: {session_id}")
        return True
    
    async def restore_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        세션 복원 (연속성 보장).
        
        Args:
            session_id: 세션 ID
        
        Returns:
            복원된 상태 또는 None
        """
        # 저장된 세션 로드
        session_state = self.session_manager.load_session(session_id)
        if not session_state:
            logger.warning(f"Session not found: {session_id}")
            return None
        
        # 활성 세션으로 등록
        self.active_sessions[session_id] = {
            'status': SessionStatus.ACTIVE,
            'created_at': datetime.fromisoformat(session_state.metadata.get('created_at', datetime.now().isoformat())),
            'last_activity': datetime.now(),
            'user_query': session_state.agent_state.get('user_query'),
            'metadata': session_state.metadata,
            'tags': session_state.metadata.get('tags', []),
            'restored': True,
            'restored_at': datetime.now().isoformat()
        }
        
        # 제어 이벤트 생성
        if session_id not in self.session_controls:
            self.session_controls[session_id] = asyncio.Event()
            self.session_controls[session_id].set()  # 기본적으로 활성
        
        logger.info(f"Session restored: {session_id}")
        return session_state.agent_state
    
    def register_active_session(
        self,
        session_id: str,
        user_query: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        활성 세션 등록.
        
        Args:
            session_id: 세션 ID
            user_query: 사용자 쿼리
            metadata: 추가 메타데이터
        """
        self.active_sessions[session_id] = {
            'status': SessionStatus.ACTIVE,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'user_query': user_query,
            'current_task': None,
            'progress': 0.0,
            'metadata': metadata or {},
            'tags': [],
            'error_count': 0,
            'warning_count': 0
        }
        
        # 제어 이벤트 생성
        if session_id not in self.session_controls:
            self.session_controls[session_id] = asyncio.Event()
            self.session_controls[session_id].set()  # 기본적으로 활성
        
        # 작업 딕셔너리 초기화
        if session_id not in self.session_tasks:
            self.session_tasks[session_id] = {}
        
        logger.info(f"Active session registered: {session_id}")
    
    def update_session_progress(
        self,
        session_id: str,
        current_task: Optional[str] = None,
        progress: Optional[float] = None
    ):
        """
        세션 진행 상황 업데이트.
        
        Args:
            session_id: 세션 ID
            current_task: 현재 작업
            progress: 진행률 (0.0-100.0)
        """
        if session_id not in self.active_sessions:
            return
        
        self.active_sessions[session_id]['last_activity'] = datetime.now()
        
        if current_task is not None:
            self.active_sessions[session_id]['current_task'] = current_task
        
        if progress is not None:
            self.active_sessions[session_id]['progress'] = progress
    
    def register_task(
        self,
        session_id: str,
        task_id: str,
        task_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskInfo:
        """
        작업 등록.
        
        Args:
            session_id: 세션 ID
            task_id: 작업 ID
            task_type: 작업 타입
            description: 작업 설명
            metadata: 추가 메타데이터
        
        Returns:
            TaskInfo
        """
        if session_id not in self.session_tasks:
            self.session_tasks[session_id] = {}
        
        task_info = TaskInfo(
            task_id=task_id,
            session_id=session_id,
            status=TaskStatus.PENDING,
            task_type=task_type,
            description=description,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        self.session_tasks[session_id][task_id] = task_info
        return task_info
    
    def update_task_status(
        self,
        session_id: str,
        task_id: str,
        status: TaskStatus,
        progress: Optional[float] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """
        작업 상태 업데이트.
        
        Args:
            session_id: 세션 ID
            task_id: 작업 ID
            status: 상태
            progress: 진행률
            result: 결과
            error: 에러 메시지
        """
        if session_id not in self.session_tasks:
            return
        
        if task_id not in self.session_tasks[session_id]:
            return
        
        task_info = self.session_tasks[session_id][task_id]
        task_info.status = status
        
        if status == TaskStatus.RUNNING and not task_info.started_at:
            task_info.started_at = datetime.now()
        
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task_info.completed_at = datetime.now()
        
        if progress is not None:
            task_info.progress = progress
        
        if result is not None:
            task_info.result = result
        
        if error is not None:
            task_info.error = error
    
    async def pause_task(self, session_id: str, task_id: str) -> bool:
        """
        작업 일시 중지.
        
        Args:
            session_id: 세션 ID
            task_id: 작업 ID
        
        Returns:
            성공 여부
        """
        if session_id not in self.session_tasks:
            return False
        
        if task_id not in self.session_tasks[session_id]:
            return False
        
        task_info = self.session_tasks[session_id][task_id]
        if task_info.status == TaskStatus.RUNNING:
            task_info.status = TaskStatus.PAUSED
            logger.info(f"Task paused: {session_id}/{task_id}")
            return True
        
        return False
    
    async def resume_task(self, session_id: str, task_id: str) -> bool:
        """
        작업 재개.
        
        Args:
            session_id: 세션 ID
            task_id: 작업 ID
        
        Returns:
            성공 여부
        """
        if session_id not in self.session_tasks:
            return False
        
        if task_id not in self.session_tasks[session_id]:
            return False
        
        task_info = self.session_tasks[session_id][task_id]
        if task_info.status == TaskStatus.PAUSED:
            task_info.status = TaskStatus.RUNNING
            logger.info(f"Task resumed: {session_id}/{task_id}")
            return True
        
        return False
    
    async def cancel_task(self, session_id: str, task_id: str) -> bool:
        """
        작업 취소.
        
        Args:
            session_id: 세션 ID
            task_id: 작업 ID
        
        Returns:
            성공 여부
        """
        if session_id not in self.session_tasks:
            return False
        
        if task_id not in self.session_tasks[session_id]:
            return False
        
        task_info = self.session_tasks[session_id][task_id]
        if task_info.status in [TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.PENDING]:
            task_info.status = TaskStatus.CANCELLED
            task_info.completed_at = datetime.now()
            logger.info(f"Task cancelled: {session_id}/{task_id}")
            return True
        
        return False
    
    def get_session_tasks(self, session_id: str) -> List[TaskInfo]:
        """
        세션의 모든 작업 조회.
        
        Args:
            session_id: 세션 ID
        
        Returns:
            작업 정보 리스트
        """
        if session_id not in self.session_tasks:
            return []
        
        return list(self.session_tasks[session_id].values())
    
    def get_task(self, session_id: str, task_id: str) -> Optional[TaskInfo]:
        """
        작업 정보 조회.
        
        Args:
            session_id: 세션 ID
            task_id: 작업 ID
        
        Returns:
            TaskInfo 또는 None
        """
        if session_id not in self.session_tasks:
            return None
        
        return self.session_tasks[session_id].get(task_id)
    
    def check_session_control(self, session_id: str) -> bool:
        """
        세션 제어 상태 확인 (pause/resume 체크).
        
        Args:
            session_id: 세션 ID
        
        Returns:
            True면 계속 진행, False면 일시 중지
        """
        if session_id not in self.session_controls:
            return True
        
        event = self.session_controls[session_id]
        return event.is_set()
    
    async def wait_for_resume(self, session_id: str, timeout: Optional[float] = None):
        """
        세션 재개 대기.
        
        Args:
            session_id: 세션 ID
            timeout: 타임아웃 (초, None이면 무제한)
        """
        if session_id not in self.session_controls:
            return
        
        event = self.session_controls[session_id]
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
    
    def get_active_sessions(self) -> List[SessionInfo]:
        """
        활성 세션 목록 조회.
        
        Returns:
            활성 세션 정보 리스트
        """
        sessions = []
        for session_id, active_info in self.active_sessions.items():
            sessions.append(SessionInfo(
                session_id=session_id,
                status=active_info.get('status', SessionStatus.ACTIVE),
                created_at=active_info.get('created_at', datetime.now()),
                last_activity=active_info.get('last_activity', datetime.now()),
                user_query=active_info.get('user_query'),
                current_task=active_info.get('current_task'),
                progress_percentage=active_info.get('progress', 0.0),
                metadata=active_info.get('metadata', {}),
                tags=active_info.get('tags', []),
                error_count=active_info.get('error_count', 0),
                warning_count=active_info.get('warning_count', 0)
            ))
        
        return sessions
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        세션 통계 조회.
        
        Returns:
            통계 딕셔너리
        """
        active_count = len(self.active_sessions)
        total_tasks = sum(len(tasks) for tasks in self.session_tasks.values())
        
        status_counts = {}
        for active_info in self.active_sessions.values():
            status = active_info.get('status', SessionStatus.ACTIVE).value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'active_sessions': active_count,
            'total_tasks': total_tasks,
            'status_distribution': status_counts,
            'sessions_by_status': status_counts
        }


# 전역 인스턴스
_session_control: Optional[SessionControl] = None


def get_session_control() -> SessionControl:
    """전역 세션 제어 인스턴스 반환."""
    global _session_control
    if _session_control is None:
        _session_control = SessionControl()
    return _session_control

