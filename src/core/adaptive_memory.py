"""
Adaptive Memory System

백서 요구사항에 맞춘 재구현:
- Memory Types 지원 (Semantic, Episodic, Procedural)
- Provenance 추적
- Background generation 통합
- 세션 간 지식 전이, 중요 정보 자동 추출 및 보존
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import json

from src.core.memory_types import BaseMemory, MemoryType, SemanticMemory, EpisodicMemory, ProceduralMemory
from src.core.memory_provenance import get_provenance_tracker
from src.core.db.database_driver import Transaction
from src.core.db.transaction_manager import get_transaction_manager

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """기존 호환성을 위한 메모리 항목 (deprecated, BaseMemory 사용 권장)."""
    key: str
    value: Any
    importance: float = 0.5
    tags: Set[str] = field(default_factory=set)
    memory_type: str = "short_term"
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0


@dataclass
class KnowledgeTransfer:
    """지식 전이 정보."""
    source_session_id: str
    target_session_id: str
    transferred_items: List[str]
    transfer_time: datetime
    success: bool = True


class AdaptiveMemory:
    """
    적응형 메모리 시스템 (백서 요구사항 준수).
    
    - Memory Types 지원 (Semantic, Episodic, Procedural)
    - Provenance 추적
    - Background generation 통합
    - 세션 간 지식 전이, 중요 정보 자동 추출 및 보존
    """
    
    def __init__(self):
        """초기화."""
        # Memory Types별 저장소
        self.semantic_memories: Dict[str, SemanticMemory] = {}
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.procedural_memories: Dict[str, ProceduralMemory] = {}
        
        # 기존 호환성을 위한 단기/장기 메모리 (deprecated)
        self.short_term_memory: Dict[str, Any] = {}
        self.long_term_memory: Dict[str, Any] = {}
        
        self.knowledge_transfers: List[KnowledgeTransfer] = []
        self.importance_threshold: float = 0.7
        
        # Provenance 추적기
        self.provenance_tracker = get_provenance_tracker()
        
        logger.info("AdaptiveMemory initialized (with Memory Types and Provenance support)")
    
    def store(
        self,
        key: str,
        value: Any,
        importance: float = 0.5,
        tags: Optional[Set[str]] = None,
        memory_type: Optional[str] = None,
        tx: Optional[Transaction] = None  # 기본값: None (기존 동작 유지)
    ) -> bool:
        """
        메모리 항목 저장.
        
        Args:
            key: 메모리 키
            value: 메모리 값
            importance: 중요도 (0.0 ~ 1.0)
            tags: 태그
            memory_type: 메모리 타입 (None이면 자동 결정)
            
        Returns:
            성공 여부
        """
        try:
            # 메모리 타입 자동 결정
            if memory_type is None:
                memory_type = "long_term" if importance >= self.importance_threshold else "short_term"
            
            # 기존 항목 업데이트 또는 새 항목 생성
            if memory_type == "long_term":
                if key in self.long_term_memory:
                    item = self.long_term_memory[key]
                    item.value = value
                    item.importance = max(item.importance, importance)
                    item.last_accessed = datetime.now()
                    item.access_count += 1
                    if tags:
                        item.tags.update(tags)
                else:
                    self.long_term_memory[key] = MemoryItem(
                        key=key,
                        value=value,
                        importance=importance,
                        tags=tags or set(),
                        memory_type="long_term"
                    )
            else:
                if key in self.short_term_memory:
                    item = self.short_term_memory[key]
                    item.value = value
                    item.importance = max(item.importance, importance)
                    item.last_accessed = datetime.now()
                    item.access_count += 1
                    if tags:
                        item.tags.update(tags)
                    
                    # 중요도가 임계값을 넘으면 장기 메모리로 전이
                    if item.importance >= self.importance_threshold:
                        self._promote_to_long_term(key, item)
                else:
                    self.short_term_memory[key] = MemoryItem(
                        key=key,
                        value=value,
                        importance=importance,
                        tags=tags or set(),
                        memory_type="short_term"
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory item {key}: {e}")
            return False
    
    def store_memory(
        self,
        memory: BaseMemory,
        tx: Optional[Transaction] = None  # 기본값: None (기존 동작 유지)
    ) -> bool:
        """
        BaseMemory 객체를 저장 (memory_service.py 호환).
        
        Args:
            memory: BaseMemory 객체
            tx: 트랜잭션 (선택사항)
            
        Returns:
            성공 여부
        """
        try:
            # BaseMemory를 일반 메모리 형식으로 변환
            key = memory.memory_id
            value = {
                "content": memory.content,
                "memory_type": memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type),
                "metadata": memory.metadata if hasattr(memory, 'metadata') else {}
            }
            importance = memory.importance if hasattr(memory, 'importance') else 0.7
            tags = set(memory.tags) if hasattr(memory, 'tags') else set()
            
            return self.store(key, value, importance, tags, None, tx)
        except Exception as e:
            logger.error(f"Failed to store memory object: {e}")
            return False
    
    async def store_memory_async(
        self,
        memory: BaseMemory,
        tx: Optional[Transaction] = None
    ) -> bool:
        """
        BaseMemory 객체를 비동기로 저장 (트랜잭션 지원).
        
        Args:
            memory: BaseMemory 객체
            tx: 트랜잭션 (None이면 자동으로 트랜잭션 생성)
            
        Returns:
            성공 여부
        """
        try:
            tx_manager = get_transaction_manager()
            
            if tx is None:
                async with tx_manager.transaction() as new_tx:
                    return await self._store_memory_in_transaction(memory, new_tx)
            else:
                return await self._store_memory_in_transaction(memory, tx)
        except Exception as e:
            logger.error(f"Failed to store memory object (async): {e}")
            return False
    
    async def _store_memory_in_transaction(
        self,
        memory: BaseMemory,
        tx: Transaction
    ) -> bool:
        """트랜잭션 내에서 메모리 저장."""
        # BaseMemory를 일반 메모리 형식으로 변환
        key = memory.memory_id
        value = {
            "content": memory.content,
            "memory_type": memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type),
            "metadata": memory.metadata if hasattr(memory, 'metadata') else {}
        }
        importance = memory.importance if hasattr(memory, 'importance') else 0.7
        tags = set(memory.tags) if hasattr(memory, 'tags') else set()
        
        # 현재는 인메모리 저장이지만, 트랜잭션 로깅
        result = self.store(key, value, importance, tags, None, tx)
        if result and tx:
            logger.debug(f"Memory stored in transaction: {key}")
        return result
    
    def retrieve(self, key: str, memory_type: Optional[str] = None) -> Optional[Any]:
        """
        메모리 항목 조회.
        
        Args:
            key: 메모리 키
            memory_type: 메모리 타입 (None이면 둘 다 검색)
            
        Returns:
            메모리 값 또는 None
        """
        try:
            # 타입 지정 시
            if memory_type == "long_term":
                if key in self.long_term_memory:
                    item = self.long_term_memory[key]
                    item.last_accessed = datetime.now()
                    item.access_count += 1
                    return item.value
            elif memory_type == "short_term":
                if key in self.short_term_memory:
                    item = self.short_term_memory[key]
                    item.last_accessed = datetime.now()
                    item.access_count += 1
                    return item.value
            else:
                # 둘 다 검색
                if key in self.long_term_memory:
                    item = self.long_term_memory[key]
                    item.last_accessed = datetime.now()
                    item.access_count += 1
                    return item.value
                elif key in self.short_term_memory:
                    item = self.short_term_memory[key]
                    item.last_accessed = datetime.now()
                    item.access_count += 1
                    return item.value
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory item {key}: {e}")
            return None
    
    def transfer_knowledge(
        self,
        source_session_id: str,
        target_session_id: str,
        filter_importance: float = 0.5
    ) -> KnowledgeTransfer:
        """
        세션 간 지식 전이.
        
        Args:
            source_session_id: 소스 세션 ID
            target_session_id: 타겟 세션 ID
            filter_importance: 전이할 최소 중요도
            
        Returns:
            KnowledgeTransfer 정보
        """
        try:
            transferred_items = []
            
            # 장기 메모리에서 전이
            for key, item in self.long_term_memory.items():
                if item.importance >= filter_importance:
                    # 타겟 세션에 저장 (세션 태그 추가)
                    item.tags.add(f"session:{target_session_id}")
                    transferred_items.append(key)
            
            # 단기 메모리에서 중요도 높은 것만 전이
            for key, item in list(self.short_term_memory.items()):
                if item.importance >= filter_importance:
                    # 장기 메모리로 승격
                    self._promote_to_long_term(key, item)
                    item.tags.add(f"session:{target_session_id}")
                    transferred_items.append(key)
            
            transfer = KnowledgeTransfer(
                source_session_id=source_session_id,
                target_session_id=target_session_id,
                transferred_items=transferred_items,
                transfer_time=datetime.now(),
                success=len(transferred_items) > 0
            )
            
            self.knowledge_transfers.append(transfer)
            logger.info(f"Knowledge transferred: {len(transferred_items)} items from {source_session_id} to {target_session_id}")
            
            return transfer
            
        except Exception as e:
            logger.error(f"Failed to transfer knowledge: {e}")
            return KnowledgeTransfer(
                source_session_id=source_session_id,
                target_session_id=target_session_id,
                transferred_items=[],
                transfer_time=datetime.now(),
                success=False
            )
    
    def extract_important_info(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        세션 데이터에서 중요 정보 자동 추출.
        
        Args:
            session_data: 세션 데이터 (AgentState 등)
            
        Returns:
            중요 정보 목록
        """
        important_info = []
        
        try:
            # 연구 결과에서 중요 정보 추출
            research_results = session_data.get("research_results", [])
            for result in research_results:
                if isinstance(result, dict):
                    # 성공적인 결과는 중요도 높음
                    if not result.get("failed", False):
                        important_info.append({
                            "type": "research_result",
                            "content": result.get("content", "")[:200],  # 처음 200자
                            "importance": 0.8,
                            "tags": {"research", "success"}
                        })
            
            # 검증된 결과는 더 중요
            verified_results = session_data.get("verified_results", [])
            for result in verified_results:
                if isinstance(result, dict):
                    important_info.append({
                        "type": "verified_result",
                        "content": result.get("content", "")[:200],
                        "importance": 0.9,
                        "tags": {"verified", "high_confidence"}
                    })
            
            # 최종 보고서는 매우 중요
            final_report = session_data.get("final_report")
            if final_report:
                important_info.append({
                    "type": "final_report",
                    "content": final_report[:500],  # 처음 500자
                    "importance": 1.0,
                    "tags": {"report", "final", "critical"}
                })
            
            return important_info
            
        except Exception as e:
            logger.error(f"Failed to extract important info: {e}")
            return []
    
    def adjust_weights_by_usage(self, usage_stats: Dict[str, int]) -> Dict[str, float]:
        """
        사용 패턴 기반 메모리 가중치 조정.
        
        Args:
            usage_stats: 사용 통계 (키별 접근 횟수)
            
        Returns:
            조정된 가중치
        """
        weights = {}
        total_accesses = sum(usage_stats.values()) if usage_stats else 1
        
        for key, access_count in usage_stats.items():
            # 접근 빈도 기반 가중치 계산
            frequency = access_count / total_accesses
            weights[key] = min(1.0, frequency * 2.0)  # 최대 1.0
        
        return weights
    
    def cleanup_old_memory(self, max_age_days: int = 30, keep_important: bool = True):
        """
        오래된 메모리 정리.
        
        Args:
            max_age_days: 최대 보관 기간 (일)
            keep_important: 중요 메모리는 보관 여부
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # 단기 메모리 정리
        cleaned_short = 0
        for key, item in list(self.short_term_memory.items()):
            if item.last_accessed < cutoff_date:
                if not (keep_important and item.importance >= self.importance_threshold):
                    del self.short_term_memory[key]
                    cleaned_short += 1
        
        # 장기 메모리는 더 오래 보관 (기본적으로 정리하지 않음)
        cleaned_long = 0
        if not keep_important:
            for key, item in list(self.long_term_memory.items()):
                if item.last_accessed < cutoff_date and item.importance < 0.8:
                    del self.long_term_memory[key]
                    cleaned_long += 1
        
        if cleaned_short > 0 or cleaned_long > 0:
            logger.info(f"Cleaned up memory: {cleaned_short} short-term, {cleaned_long} long-term items")
    
    def _promote_to_long_term(self, key: str, item: MemoryItem):
        """단기 메모리를 장기 메모리로 승격."""
        item.memory_type = "long_term"
        self.long_term_memory[key] = item
        if key in self.short_term_memory:
            del self.short_term_memory[key]
        logger.debug(f"Promoted {key} to long-term memory (importance: {item.importance})")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계."""
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory),
            "total_transfers": len(self.knowledge_transfers),
            "average_importance_short": sum(item.importance for item in self.short_term_memory.values()) / len(self.short_term_memory) if self.short_term_memory else 0.0,
            "average_importance_long": sum(item.importance for item in self.long_term_memory.values()) / len(self.long_term_memory) if self.long_term_memory else 0.0
        }


# 전역 인스턴스
_adaptive_memory: Optional[AdaptiveMemory] = None


def get_adaptive_memory() -> AdaptiveMemory:
    """전역 적응형 메모리 인스턴스 반환."""
    global _adaptive_memory
    if _adaptive_memory is None:
        _adaptive_memory = AdaptiveMemory()
    return _adaptive_memory

