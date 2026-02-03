#!/usr/bin/env python3
"""
Task Queue System for Parallel Agent Execution

작업 큐 및 우선순위 관리 시스템.
병렬 실행 가능한 작업 그룹을 식별하고, 의존성을 고려하여 작업을 분배합니다.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class TopicStatus(Enum):
    """Topic block 상태 열거형 (9대 혁신: Task Queue 구조화)."""
    PENDING = "pending"  # 대기 중
    RESEARCHING = "researching"  # 연구 중
    COMPLETED = "completed"  # 완료
    FAILED = "failed"  # 실패


@dataclass
class ToolTrace:
    """
    도구 호출 추적 정보 (9대 혁신: ToolTrace 추적 시스템).
    
    도구 호출의 완전한 루프를 기록합니다.
    """
    tool_id: str  # 고유 식별자 (e.g., "tool_1", "tool_2")
    citation_id: str  # Citation ID (e.g., "CIT-1-01")
    tool_type: str  # 도구 타입 (rag_naive, web_search 등)
    query: str  # 발행된 쿼리
    raw_answer: str  # 도구가 반환한 원시 상세 결과 (50KB 제한)
    summary: str  # Note Agent가 생성한 핵심 요약
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_answer_truncated: bool = False  # raw_answer가 잘렸는지 여부
    raw_answer_original_size: int = 0  # 잘리기 전 원본 크기
    
    def __post_init__(self):
        """초기화 후 처리: raw_answer 크기 제한."""
        if self.raw_answer_original_size == 0:
            self.raw_answer_original_size = len(self.raw_answer)
        
        # 50KB 제한 적용
        max_size = 50 * 1024
        if len(self.raw_answer) > max_size:
            self.raw_answer = self._truncate_raw_answer(self.raw_answer, max_size)
            self.raw_answer_truncated = True
    
    @staticmethod
    def _truncate_raw_answer(raw_answer: str, max_size: int) -> str:
        """
        raw_answer를 스마트하게 잘라냅니다 (유효한 JSON 구조 보존 시도).
        
        Args:
            raw_answer: 원본 raw answer 문자열
            max_size: 최대 크기 (바이트)
        
        Returns:
            잘린 문자열
        """
        if len(raw_answer) <= max_size:
            return raw_answer
        
        # JSON으로 파싱 시도하고 지능적으로 잘라내기
        try:
            data = json.loads(raw_answer)
            
            # dict인 경우 일반적인 RAG 응답 필드를 잘라냄
            if isinstance(data, dict):
                content_fields = ["answer", "content", "text", "chunks", "documents"]
                for field_name in content_fields:
                    if field_name in data:
                        if isinstance(data[field_name], str) and len(data[field_name]) > max_size // 2:
                            data[field_name] = data[field_name][:max_size // 2] + "... [truncated]"
                        elif isinstance(data[field_name], list):
                            # 처음 몇 개 항목만 유지
                            data[field_name] = data[field_name][:3]
                            if data[field_name]:
                                data[field_name].append({"note": "... additional items truncated"})
                
                truncated = json.dumps(data, ensure_ascii=False)
                if len(truncated) <= max_size:
                    return truncated
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Fallback: 간단한 잘라내기
        truncation_marker = f"\n... [content truncated, original size: {len(raw_answer)} bytes]"
        return raw_answer[:max_size - len(truncation_marker)] + truncation_marker
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolTrace":
        """딕셔너리에서 생성."""
        # 하위 호환성: old data에는 새 필드가 없을 수 있음
        if "raw_answer_truncated" not in data:
            data["raw_answer_truncated"] = False
        if "raw_answer_original_size" not in data:
            data["raw_answer_original_size"] = len(data.get("raw_answer", ""))
        return cls(**data)


@dataclass
class TopicBlock:
    """
    Topic Block - 큐의 최소 스케줄링 단위 (9대 혁신: Task Queue 구조화).
    
    기존 TaskQueueItem을 확장하여 상태 관리와 도구 추적 기능을 추가합니다.
    """
    block_id: str  # 고유 식별자 (e.g., "block_1", "block_2")
    sub_topic: str  # 하위 주제 이름
    overview: str  # 주제 개요/배경
    status: TopicStatus = TopicStatus.PENDING  # 주제 상태
    tool_traces: List[ToolTrace] = field(default_factory=list)  # 도구 호출 추적 리스트
    iteration_count: int = 0  # 현재 반복 횟수
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터
    
    # 기존 TaskQueueItem 호환성
    task: Optional[Dict[str, Any]] = None
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    
    def add_tool_trace(self, trace: ToolTrace) -> None:
        """도구 추적 추가."""
        self.tool_traces.append(trace)
        self.updated_at = datetime.now().isoformat()
    
    def get_latest_trace(self) -> Optional[ToolTrace]:
        """최신 도구 추적 반환."""
        return self.tool_traces[-1] if self.tool_traces else None
    
    def get_all_summaries(self) -> str:
        """모든 도구 추적의 요약을 연결하여 반환."""
        if not self.tool_traces:
            return ""
        return "\n".join([f"[{trace.tool_type}] {trace.summary}" for trace in self.tool_traces])
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        data = asdict(self)
        data["status"] = self.status.value
        data["tool_traces"] = [trace.to_dict() for trace in self.tool_traces]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopicBlock":
        """딕셔너리에서 생성."""
        data_copy = data.copy()
        if isinstance(data_copy.get("status"), str):
            data_copy["status"] = TopicStatus(data_copy["status"])
        if "tool_traces" in data_copy:
            data_copy["tool_traces"] = [
                ToolTrace.from_dict(t) if isinstance(t, dict) else t
                for t in data_copy["tool_traces"]
            ]
        return cls(**data_copy)


@dataclass
class TaskQueueItem:
    """작업 큐 아이템 (하위 호환성 유지)."""
    task_id: str
    task: Dict[str, Any]
    priority: int = 0  # 높을수록 우선순위 높음
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_topic_block(self, block_id: Optional[str] = None) -> TopicBlock:
        """
        TopicBlock으로 변환 (9대 혁신: Task Queue 구조화).
        
        Args:
            block_id: Block ID (None이면 task_id 사용)
        
        Returns:
            TopicBlock 객체
        """
        sub_topic = self.task.get("sub_topic") or self.task.get("title") or self.task_id
        overview = self.task.get("overview") or self.task.get("description") or ""
        
        return TopicBlock(
            block_id=block_id or self.task_id,
            sub_topic=sub_topic,
            overview=overview,
            status=TopicStatus.PENDING,
            task=self.task,
            priority=self.priority,
            dependencies=self.dependencies,
            created_at=self.created_at.isoformat() if self.created_at else datetime.now().isoformat()
        )


class TaskQueue:
    """
    작업 큐 및 우선순위 관리 시스템 (9대 혁신: Task Queue 구조화).
    
    기존 기능을 유지하면서 TopicBlock 기반 상태 관리를 추가합니다.
    """
    
    def __init__(self, research_id: Optional[str] = None, max_length: Optional[int] = None, state_file: Optional[str] = None):
        """
        초기화.
        
        Args:
            research_id: 연구 작업 ID (optional)
            max_length: 최대 큐 길이 (None이면 무제한)
            state_file: 자동 저장 파일 경로 (optional)
        """
        self.research_id = research_id or f"research_{int(datetime.now().timestamp())}"
        self.max_length = max_length
        self.state_file = state_file
        
        # 기존 시스템 (하위 호환성)
        self.tasks: Dict[str, TaskQueueItem] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.completed_tasks: Set[str] = set()
        self.parallel_groups: List[List[str]] = []
        
        # 9대 혁신: TopicBlock 기반 큐
        self.blocks: List[TopicBlock] = []
        self.block_counter = 0
        self.created_at = datetime.now().isoformat()
        
        # Performance optimizations: Indexed lookups
        self._task_index: Dict[str, int] = {}  # task_id -> index in blocks
        self._block_index: Dict[str, int] = {}  # block_id -> index in blocks
        self._priority_queues: Dict[int, List[str]] = defaultdict(list)  # priority -> [block_ids]
        self._sorted_priorities: List[int] = []  # Cached sorted priorities for faster lookups
        
    def add_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """작업들을 큐에 추가."""
        logger.info(f"Adding {len(tasks)} tasks to queue")
        
        for task in tasks:
            # task_id 또는 id 필드에서 ID 추출
            task_id = task.get('task_id') or task.get('id', '')
            if not task_id:
                # ID가 없으면 생성
                import uuid
                task_id = str(uuid.uuid4())
                task['task_id'] = task_id
                task['id'] = task_id
                logger.warning(f"Generated task_id for task without ID: {task_id}")
            
            dependencies = task.get('dependencies') or []
            self.dependency_graph[task_id] = dependencies
            
            # TaskQueueItem 생성
            queue_item = TaskQueueItem(
                task_id=task_id,
                task=task,
                priority=task.get('priority', 0),
                dependencies=dependencies
            )
            
            self.tasks[task_id] = queue_item
        
        # 병렬 그룹 식별
        self._identify_parallel_groups()
    
    def _identify_parallel_groups(self) -> None:
        """병렬 실행 가능한 작업 그룹 식별."""
        self.parallel_groups = []
        processed = set()
        
        # 의존성 그래프 기반으로 병렬 그룹 생성
        for task_id, dependencies in self.dependency_graph.items():
            if task_id in processed:
                continue
            
            # 의존성이 없거나 모든 의존성이 완료된 작업들로 그룹 생성
            if not dependencies or all(dep in self.completed_tasks for dep in dependencies):
                group = [task_id]
                
                # 다른 독립적인 작업들 찾기
                for other_task_id, other_deps in self.dependency_graph.items():
                    if (other_task_id != task_id and 
                        other_task_id not in processed and
                        (not other_deps or all(dep in self.completed_tasks for dep in other_deps))):
                        group.append(other_task_id)
                
                if len(group) > 1:
                    self.parallel_groups.append(group)
                    processed.update(group)
        
        logger.info(f"Identified {len(self.parallel_groups)} parallel groups")
    
    def get_next_task_group(self, max_group_size: Optional[int] = None) -> Optional[List[str]]:
        """다음 실행 가능한 작업 그룹 반환."""
        # 병렬 그룹이 있으면 반환
        if self.parallel_groups:
            group = self.parallel_groups[0]
            if max_group_size:
                group = group[:max_group_size]
            self.parallel_groups.pop(0)
            
            # 그룹의 모든 작업이 여전히 실행 가능한지 확인
            ready_tasks = [
                task_id for task_id in group
                if task_id not in self.completed_tasks and
                all(dep in self.completed_tasks for dep in self.dependency_graph.get(task_id, []))
            ]
            
            if ready_tasks:
                return ready_tasks
        
        # 병렬 그룹이 없으면 의존성 없는 단일 작업 반환
        for task_id, dependencies in self.dependency_graph.items():
            if (task_id not in self.completed_tasks and
                all(dep in self.completed_tasks for dep in dependencies)):
                return [task_id]
        
        return None
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 정보 반환 (O(1) lookup with index)."""
        # Use indexed lookup for better performance
        if task_id in self.tasks:
            return self.tasks[task_id].task
        return None
    
    def mark_completed(self, task_id: str) -> None:
        """작업 완료 표시."""
        if task_id in self.tasks:
            self.completed_tasks.add(task_id)
            logger.debug(f"Task {task_id} marked as completed")
            
            # 병렬 그룹 재계산
            self._identify_parallel_groups()
    
    def is_completed(self, task_id: str) -> bool:
        """작업 완료 여부 확인."""
        return task_id in self.completed_tasks
    
    def get_remaining_count(self) -> int:
        """남은 작업 수 반환."""
        return len(self.tasks) - len(self.completed_tasks)
    
    def has_pending_tasks(self) -> bool:
        """대기 중인 작업이 있는지 확인."""
        return self.get_remaining_count() > 0
    
    def get_progress(self) -> Dict[str, Any]:
        """작업 진행 상황 반환."""
        total = len(self.tasks)
        completed = len(self.completed_tasks)
        remaining = total - completed
        progress_percentage = (completed / total * 100) if total > 0 else 0.0
        
        return {
            'total_tasks': total,
            'completed_tasks': completed,
            'remaining_tasks': remaining,
            'progress_percentage': progress_percentage,
            'parallel_groups_count': len(self.parallel_groups)
        }
    
    # ========== 9대 혁신: TopicBlock 기반 큐 관리 ==========
    
    def add_block(self, sub_topic: str, overview: str, task: Optional[Dict[str, Any]] = None) -> TopicBlock:
        """
        새로운 topic block을 큐 끝에 추가.
        
        Args:
            sub_topic: 하위 주제 이름
            overview: 주제 개요
            task: 작업 정보 (optional)
        
        Returns:
            생성된 TopicBlock
        """
        if self.max_length and len(self.blocks) >= self.max_length:
            raise RuntimeError(
                f"Queue has reached maximum capacity ({self.max_length}), cannot add new topic."
            )
        
        self.block_counter += 1
        block_id = f"block_{self.block_counter}"
        block = TopicBlock(
            block_id=block_id,
            sub_topic=sub_topic,
            overview=overview,
            task=task or {},
            status=TopicStatus.PENDING
        )
        
        # Add to blocks and update indices
        self.blocks.append(block)
        self._block_index[block_id] = len(self.blocks) - 1
        self._priority_queues[block.priority].append(block_id)
        
        self._auto_save()
        return block
    
    def has_topic(self, sub_topic: str) -> bool:
        """
        주제가 이미 존재하는지 확인 (대소문자 무시, 앞뒤 공백 무시).
        
        Args:
            sub_topic: 확인할 하위 주제
        
        Returns:
            존재 여부
        """
        target = (sub_topic or "").strip().lower()
        if not target:
            return False
        return any((b.sub_topic or "").strip().lower() == target for b in self.blocks)
    
    def get_pending_block(self) -> Optional[TopicBlock]:
        """
        첫 번째 대기 중인 topic block 반환 (priority-based lookup).
        
        Returns:
            PENDING 상태의 첫 번째 TopicBlock, 없으면 None
        """
        # Check priority queues first for better performance
        # Sort priorities once and cache for repeated calls
        if not hasattr(self, '_sorted_priorities'):
            self._sorted_priorities = sorted(self._priority_queues.keys(), reverse=True)
        
        for priority in self._sorted_priorities:
            queue = self._priority_queues.get(priority, [])
            if queue:
                for block_id in queue:
                    block = self.get_block_by_id(block_id)
                    if block and block.status == TopicStatus.PENDING:
                        return block
        
        # Fallback to linear search if priority queues are empty
        for block in self.blocks:
            if block.status == TopicStatus.PENDING:
                return block
        return None
    
    def get_block_by_id(self, block_id: str) -> Optional[TopicBlock]:
        """
        Block ID로 topic block 조회 (O(1) lookup with index).
        
        Args:
            block_id: Topic block ID
        
        Returns:
            해당 TopicBlock, 없으면 None
        """
        # Use indexed lookup for O(1) performance
        if block_id in self._block_index:
            index = self._block_index[block_id]
            if 0 <= index < len(self.blocks):
                return self.blocks[index]
        
        # Fallback to linear search if index is out of sync
        for i, block in enumerate(self.blocks):
            if block.block_id == block_id:
                self._block_index[block_id] = i  # Rebuild index
                return block
        return None
    
    def mark_researching(self, block_id: str) -> bool:
        """
        Topic block을 researching 상태로 표시.
        
        Args:
            block_id: Topic block ID
        
        Returns:
            성공 여부
        """
        block = self.get_block_by_id(block_id)
        if block:
            block.status = TopicStatus.RESEARCHING
            block.updated_at = datetime.now().isoformat()
            self._auto_save()
            return True
        return False
    
    def mark_completed_block(self, block_id: str) -> bool:
        """
        Topic block을 completed 상태로 표시.
        
        Args:
            block_id: Topic block ID
        
        Returns:
            성공 여부
        """
        block = self.get_block_by_id(block_id)
        if block:
            block.status = TopicStatus.COMPLETED
            block.updated_at = datetime.now().isoformat()
            self._auto_save()
            return True
        return False
    
    def mark_failed_block(self, block_id: str) -> bool:
        """
        Topic block을 failed 상태로 표시.
        
        Args:
            block_id: Topic block ID
        
        Returns:
            성공 여부
        """
        block = self.get_block_by_id(block_id)
        if block:
            block.status = TopicStatus.FAILED
            block.updated_at = datetime.now().isoformat()
            self._auto_save()
            return True
        return False
    
    def get_all_completed_blocks(self) -> List[TopicBlock]:
        """모든 완료된 topic block 반환."""
        return [b for b in self.blocks if b.status == TopicStatus.COMPLETED]
    
    def get_all_pending_blocks(self) -> List[TopicBlock]:
        """모든 대기 중인 topic block 반환."""
        return [b for b in self.blocks if b.status == TopicStatus.PENDING]
    
    def is_all_completed(self) -> bool:
        """모든 topic block이 완료되었는지 확인."""
        if not self.blocks:
            return False
        return all(b.status == TopicStatus.COMPLETED for b in self.blocks)
    
    def get_statistics(self) -> Dict[str, Any]:
        """큐 통계 정보 반환."""
        return {
            "total_blocks": len(self.blocks),
            "pending": len(self.get_all_pending_blocks()),
            "researching": len([b for b in self.blocks if b.status == TopicStatus.RESEARCHING]),
            "completed": len(self.get_all_completed_blocks()),
            "failed": len([b for b in self.blocks if b.status == TopicStatus.FAILED]),
            "total_tool_calls": sum(len(b.tool_traces) for b in self.blocks),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "research_id": self.research_id,
            "created_at": self.created_at,
            "blocks": [b.to_dict() for b in self.blocks],
            "statistics": self.get_statistics(),
        }
    
    def save_to_json(self, filepath: str) -> None:
        """큐를 JSON 파일로 저장."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    def _auto_save(self) -> None:
        """state_file이 설정된 경우 자동 저장."""
        if self.state_file:
            try:
                self.save_to_json(self.state_file)
            except Exception as exc:
                logger.warning(f"Failed to save queue progress: {exc}")
    
    @classmethod
    def load_from_json(cls, filepath: str) -> "TaskQueue":
        """JSON 파일에서 큐 로드."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        
        queue = cls(
            research_id=data.get("research_id"),
            state_file=filepath
        )
        queue.created_at = data.get("created_at", queue.created_at)
        
        for block_data in data.get("blocks", []):
            block = TopicBlock.from_dict(block_data)
            queue.blocks.append(block)
            # 카운터 업데이트
            if block.block_id.startswith("block_"):
                try:
                    block_num = int(block.block_id.split("_")[1])
                    queue.block_counter = max(queue.block_counter, block_num)
                except (ValueError, IndexError):
                    pass
        
        return queue

