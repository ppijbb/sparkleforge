"""
Memory Types

백서에 따른 메모리 타입 정의: Semantic, Episodic, Procedural
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """메모리 타입 (백서 기준)."""
    SEMANTIC = "semantic"      # 사실적 지식 (factual knowledge)
    EPISODIC = "episodic"      # 특정 이벤트/경험 (specific events/experiences)
    PROCEDURAL = "procedural"  # 절차적 지식 (how-to knowledge, tool usage)


@dataclass
class BaseMemory:
    """기본 메모리 구조."""
    memory_id: str
    memory_type: MemoryType
    content: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance: float = 0.5  # 0.0 ~ 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "metadata": self.metadata
        }


@dataclass
class SemanticMemory(BaseMemory):
    """
    Semantic Memory: 사실적 지식
    
    예: "사용자는 파이썬 개발자입니다", "사용자는 데이터 분석을 선호합니다"
    """
    memory_type: MemoryType = field(default=MemoryType.SEMANTIC, init=False)
    facts: List[str] = field(default_factory=list)  # 추출된 사실들
    domain: Optional[str] = None  # 도메인 (예: "programming", "data_science")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        base_dict = super().to_dict()
        base_dict.update({
            "facts": self.facts,
            "domain": self.domain
        })
        return base_dict


@dataclass
class EpisodicMemory(BaseMemory):
    """
    Episodic Memory: 특정 이벤트/경험
    
    예: "2024년 1월에 사용자가 머신러닝 프로젝트를 완료했습니다"
    """
    memory_type: MemoryType = field(default=MemoryType.EPISODIC, init=False)
    event_timestamp: Optional[datetime] = None  # 이벤트 발생 시점
    session_id: Optional[str] = None  # 관련 세션 ID
    context: Dict[str, Any] = field(default_factory=dict)  # 이벤트 컨텍스트
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        base_dict = super().to_dict()
        base_dict.update({
            "event_timestamp": self.event_timestamp.isoformat() if self.event_timestamp else None,
            "session_id": self.session_id,
            "context": self.context
        })
        return base_dict


@dataclass
class ProceduralMemory(BaseMemory):
    """
    Procedural Memory: 절차적 지식 (도구 사용법 등)
    
    예: "사용자는 pandas를 사용하여 데이터를 분석하는 방법을 알고 있습니다"
    """
    memory_type: MemoryType = field(default=MemoryType.PROCEDURAL, init=False)
    procedure_steps: List[str] = field(default_factory=list)  # 절차 단계
    tool_name: Optional[str] = None  # 관련 도구 이름
    success_rate: float = 0.0  # 성공률
    examples: List[str] = field(default_factory=list)  # 사용 예시
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        base_dict = super().to_dict()
        base_dict.update({
            "procedure_steps": self.procedure_steps,
            "tool_name": self.tool_name,
            "success_rate": self.success_rate,
            "examples": self.examples
        })
        return base_dict


def create_memory_from_dict(data: Dict[str, Any]) -> BaseMemory:
    """딕셔너리에서 메모리 객체 생성."""
    memory_type = MemoryType(data.get("memory_type", "semantic"))
    
    # 공통 필드
    base_fields = {
        "memory_id": data["memory_id"],
        "content": data["content"],
        "user_id": data["user_id"],
        "created_at": datetime.fromisoformat(data["created_at"]),
        "last_accessed": datetime.fromisoformat(data["last_accessed"]),
        "access_count": data.get("access_count", 0),
        "importance": data.get("importance", 0.5),
        "metadata": data.get("metadata", {})
    }
    
    if memory_type == MemoryType.SEMANTIC:
        return SemanticMemory(
            **base_fields,
            facts=data.get("facts", []),
            domain=data.get("domain")
        )
    elif memory_type == MemoryType.EPISODIC:
        return EpisodicMemory(
            **base_fields,
            event_timestamp=datetime.fromisoformat(data["event_timestamp"]) if data.get("event_timestamp") else None,
            session_id=data.get("session_id"),
            context=data.get("context", {})
        )
    elif memory_type == MemoryType.PROCEDURAL:
        return ProceduralMemory(
            **base_fields,
            procedure_steps=data.get("procedure_steps", []),
            tool_name=data.get("tool_name"),
            success_rate=data.get("success_rate", 0.0),
            examples=data.get("examples", [])
        )
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")

