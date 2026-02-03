"""
Memory Provenance

메모리 출처 추적 및 계보 관리 시스템 (백서 요구사항)
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ProvenanceEventType(Enum):
    """Provenance 이벤트 타입."""
    EXTRACTION = "extraction"  # 메모리 추출
    CONSOLIDATION = "consolidation"  # 메모리 통합
    UPDATE = "update"  # 메모리 업데이트
    DELETE = "delete"  # 메모리 삭제
    RETRIEVAL = "retrieval"  # 메모리 조회
    MERGE = "merge"  # 메모리 병합


@dataclass
class ProvenanceEvent:
    """Provenance 이벤트."""
    event_id: str
    event_type: ProvenanceEventType
    timestamp: datetime
    memory_id: str
    source_session_id: Optional[str] = None
    source_turn: Optional[int] = None
    source_memory_ids: List[str] = field(default_factory=list)  # 통합/병합 시 원본 메모리 ID들
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "memory_id": self.memory_id,
            "source_session_id": self.source_session_id,
            "source_turn": self.source_turn,
            "source_memory_ids": self.source_memory_ids,
            "agent_id": self.agent_id,
            "metadata": self.metadata
        }


@dataclass
class MemoryProvenance:
    """메모리 Provenance 정보."""
    memory_id: str
    creation_event: ProvenanceEvent
    lineage: List[ProvenanceEvent] = field(default_factory=list)  # 계보 (이벤트 히스토리)
    source_memories: List[str] = field(default_factory=list)  # 이 메모리를 생성한 원본 메모리들
    derived_memories: List[str] = field(default_factory=list)  # 이 메모리에서 파생된 메모리들
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "memory_id": self.memory_id,
            "creation_event": self.creation_event.to_dict(),
            "lineage": [event.to_dict() for event in self.lineage],
            "source_memories": self.source_memories,
            "derived_memories": self.derived_memories
        }


class ProvenanceTracker:
    """
    Provenance 추적 시스템.
    
    모든 메모리 작업의 출처와 계보를 추적합니다.
    """
    
    def __init__(self):
        """초기화."""
        self.provenance_records: Dict[str, MemoryProvenance] = {}  # memory_id -> Provenance
        self.event_log: List[ProvenanceEvent] = []
        
        logger.info("ProvenanceTracker initialized")
    
    def record_extraction(
        self,
        memory_id: str,
        source_session_id: str,
        source_turn: Optional[int] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProvenanceEvent:
        """
        메모리 추출 이벤트 기록.
        
        Args:
            memory_id: 생성된 메모리 ID
            source_session_id: 소스 세션 ID
            source_turn: 소스 턴 번호
            agent_id: 에이전트 ID
            metadata: 추가 메타데이터
            
        Returns:
            ProvenanceEvent
        """
        event = ProvenanceEvent(
            event_id=f"extract_{memory_id}_{int(datetime.now().timestamp() * 1000)}",
            event_type=ProvenanceEventType.EXTRACTION,
            timestamp=datetime.now(),
            memory_id=memory_id,
            source_session_id=source_session_id,
            source_turn=source_turn,
            agent_id=agent_id,
            metadata=metadata or {}
        )
        
        # Provenance 기록 생성
        if memory_id not in self.provenance_records:
            self.provenance_records[memory_id] = MemoryProvenance(
                memory_id=memory_id,
                creation_event=event
            )
        else:
            self.provenance_records[memory_id].lineage.append(event)
        
        self.event_log.append(event)
        logger.debug(f"Recorded extraction event: {memory_id} from session {source_session_id}")
        
        return event
    
    def record_consolidation(
        self,
        memory_id: str,
        source_memory_ids: List[str],
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProvenanceEvent:
        """
        메모리 통합 이벤트 기록.
        
        Args:
            memory_id: 통합된 메모리 ID
            source_memory_ids: 원본 메모리 ID 목록
            agent_id: 에이전트 ID
            metadata: 추가 메타데이터
            
        Returns:
            ProvenanceEvent
        """
        event = ProvenanceEvent(
            event_id=f"consolidate_{memory_id}_{int(datetime.now().timestamp() * 1000)}",
            event_type=ProvenanceEventType.CONSOLIDATION,
            timestamp=datetime.now(),
            memory_id=memory_id,
            source_memory_ids=source_memory_ids,
            agent_id=agent_id,
            metadata=metadata or {}
        )
        
        # Provenance 기록 업데이트
        if memory_id not in self.provenance_records:
            self.provenance_records[memory_id] = MemoryProvenance(
                memory_id=memory_id,
                creation_event=event,
                source_memories=source_memory_ids
            )
        else:
            self.provenance_records[memory_id].lineage.append(event)
            self.provenance_records[memory_id].source_memories.extend(source_memory_ids)
        
        # 원본 메모리들의 derived_memories 업데이트
        for source_id in source_memory_ids:
            if source_id in self.provenance_records:
                if memory_id not in self.provenance_records[source_id].derived_memories:
                    self.provenance_records[source_id].derived_memories.append(memory_id)
        
        self.event_log.append(event)
        logger.debug(f"Recorded consolidation event: {memory_id} from {len(source_memory_ids)} sources")
        
        return event
    
    def record_update(
        self,
        memory_id: str,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProvenanceEvent:
        """메모리 업데이트 이벤트 기록."""
        event = ProvenanceEvent(
            event_id=f"update_{memory_id}_{int(datetime.now().timestamp() * 1000)}",
            event_type=ProvenanceEventType.UPDATE,
            timestamp=datetime.now(),
            memory_id=memory_id,
            agent_id=agent_id,
            metadata=metadata or {}
        )
        
        if memory_id in self.provenance_records:
            self.provenance_records[memory_id].lineage.append(event)
        
        self.event_log.append(event)
        return event
    
    def record_retrieval(
        self,
        memory_id: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> ProvenanceEvent:
        """메모리 조회 이벤트 기록."""
        event = ProvenanceEvent(
            event_id=f"retrieve_{memory_id}_{int(datetime.now().timestamp() * 1000)}",
            event_type=ProvenanceEventType.RETRIEVAL,
            timestamp=datetime.now(),
            memory_id=memory_id,
            source_session_id=session_id,
            agent_id=agent_id
        )
        
        if memory_id in self.provenance_records:
            self.provenance_records[memory_id].lineage.append(event)
        
        self.event_log.append(event)
        return event
    
    def get_provenance(self, memory_id: str) -> Optional[MemoryProvenance]:
        """메모리의 Provenance 정보 조회."""
        return self.provenance_records.get(memory_id)
    
    def get_lineage(self, memory_id: str) -> List[ProvenanceEvent]:
        """메모리의 계보 조회."""
        provenance = self.get_provenance(memory_id)
        if provenance:
            return [provenance.creation_event] + provenance.lineage
        return []
    
    def get_source_memories(self, memory_id: str) -> List[str]:
        """메모리의 원본 메모리 목록 조회."""
        provenance = self.get_provenance(memory_id)
        if provenance:
            return provenance.source_memories
        return []
    
    def get_derived_memories(self, memory_id: str) -> List[str]:
        """메모리에서 파생된 메모리 목록 조회."""
        provenance = self.get_provenance(memory_id)
        if provenance:
            return provenance.derived_memories
        return []


# 전역 인스턴스
_provenance_tracker: Optional[ProvenanceTracker] = None


def get_provenance_tracker() -> ProvenanceTracker:
    """전역 Provenance 추적기 인스턴스 반환."""
    global _provenance_tracker
    if _provenance_tracker is None:
        _provenance_tracker = ProvenanceTracker()
    return _provenance_tracker

