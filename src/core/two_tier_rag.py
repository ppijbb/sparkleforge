"""
Two-Tier RAG Memory System

Hephaestus 영감을 받은 2계층 RAG 메모리 시스템.
Pre-loaded + Dynamic Search 구조로 검색 효율성 극대화.

핵심 특징:
- Tier 1 (Pre-loaded): 세션 시작 시 관련 컨텍스트 미리 로드
- Tier 2 (Dynamic): 실행 중 필요시 추가 검색
- Memory Types: errors, discoveries, decisions, learnings
- Bidirectional RAG: 양방향 컨텍스트 연결
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from enum import Enum
from collections import defaultdict
import hashlib

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """메모리 유형."""
    ERROR = "error"
    DISCOVERY = "discovery"
    DECISION = "decision"
    LEARNING = "learning"
    FACT = "fact"
    CONTEXT = "context"
    REFERENCE = "reference"


class MemoryEntry(BaseModel):
    """메모리 엔트리."""
    entry_id: str = Field(description="엔트리 고유 ID")
    memory_type: MemoryType = Field(description="메모리 유형")
    content: str = Field(description="내용")
    summary: str = Field(default="", description="요약")
    
    # 메타데이터
    source: str = Field(default="", description="출처")
    session_id: Optional[str] = Field(default=None, description="세션 ID")
    agent_id: Optional[str] = Field(default=None, description="에이전트 ID")
    
    # 검색 관련
    keywords: List[str] = Field(default_factory=list, description="키워드")
    embedding: Optional[List[float]] = Field(default=None, description="임베딩 벡터")
    
    # 중요도 및 관련성
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="중요도")
    relevance_score: float = Field(default=0.0, description="관련성 점수")
    access_count: int = Field(default=0, description="접근 횟수")
    
    # 타임스탬프
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = Field(default=None)
    
    # 연결 정보 (Bidirectional)
    related_entries: List[str] = Field(default_factory=list, description="관련 엔트리 ID들")
    parent_entry: Optional[str] = Field(default=None, description="부모 엔트리 ID")
    child_entries: List[str] = Field(default_factory=list, description="자식 엔트리 ID들")
    
    class Config:
        arbitrary_types_allowed = True


class TierOneCache:
    """
    Tier 1: Pre-loaded Cache
    
    세션 시작 시 관련 컨텍스트를 미리 로드하여 빠른 검색 제공.
    """
    
    def __init__(self, max_entries: int = 20):
        self.max_entries = max_entries
        self.entries: Dict[str, MemoryEntry] = {}
        self.entry_order: List[str] = []  # LRU 순서
        self.loaded_at: Optional[datetime] = None
        
    def load(self, entries: List[MemoryEntry]):
        """엔트리 로드."""
        self.entries.clear()
        self.entry_order.clear()
        
        # 중요도 기준 정렬 후 상위 N개 선택
        sorted_entries = sorted(entries, key=lambda e: e.importance, reverse=True)
        
        for entry in sorted_entries[:self.max_entries]:
            self.entries[entry.entry_id] = entry
            self.entry_order.append(entry.entry_id)
        
        self.loaded_at = datetime.now()
        
        logger.info(f"Tier 1 cache loaded with {len(self.entries)} entries")
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """엔트리 조회 (LRU 업데이트)."""
        if entry_id in self.entries:
            # LRU 업데이트
            self.entry_order.remove(entry_id)
            self.entry_order.append(entry_id)
            
            entry = self.entries[entry_id]
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            
            return entry
        return None
    
    def search(
        self,
        query_keywords: List[str],
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """키워드 기반 검색."""
        results = []
        query_set = set(k.lower() for k in query_keywords)
        
        for entry in self.entries.values():
            # 타입 필터
            if memory_type and entry.memory_type != memory_type:
                continue
            
            # 키워드 매칭
            entry_keywords = set(k.lower() for k in entry.keywords)
            common = query_set & entry_keywords
            
            if common:
                score = len(common) / max(len(query_set), 1)
                score *= entry.importance  # 중요도 가중치
                results.append((entry, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def add(self, entry: MemoryEntry):
        """엔트리 추가 (캐시 용량 초과 시 LRU 제거)."""
        if entry.entry_id in self.entries:
            return
        
        if len(self.entries) >= self.max_entries:
            # LRU 제거
            oldest_id = self.entry_order.pop(0)
            del self.entries[oldest_id]
        
        self.entries[entry.entry_id] = entry
        self.entry_order.append(entry.entry_id)
    
    @property
    def size(self) -> int:
        """캐시 크기."""
        return len(self.entries)
    
    @property
    def hit_rate(self) -> float:
        """캐시 히트율 (추정)."""
        if not self.entries:
            return 0.0
        total_access = sum(e.access_count for e in self.entries.values())
        return total_access / max(len(self.entries), 1)


class TierTwoStore:
    """
    Tier 2: Dynamic Store
    
    실행 중 동적 검색을 위한 전체 메모리 저장소.
    """
    
    def __init__(self, enable_persistence: bool = True):
        self.entries: Dict[str, MemoryEntry] = {}
        self.type_index: Dict[MemoryType, Set[str]] = defaultdict(set)
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self.enable_persistence = enable_persistence
        
        # 양방향 연결 그래프
        self.connections: Dict[str, Set[str]] = defaultdict(set)
        
    def add(self, entry: MemoryEntry):
        """엔트리 추가."""
        self.entries[entry.entry_id] = entry
        
        # 타입 인덱스
        self.type_index[entry.memory_type].add(entry.entry_id)
        
        # 키워드 인덱스
        for keyword in entry.keywords:
            self.keyword_index[keyword.lower()].add(entry.entry_id)
        
        # 연결 업데이트
        for related_id in entry.related_entries:
            self._add_connection(entry.entry_id, related_id)
        
        logger.debug(f"Added entry {entry.entry_id} to Tier 2 store")
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """엔트리 조회."""
        entry = self.entries.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
        return entry
    
    def search(
        self,
        query_keywords: List[str],
        top_k: int = 10,
        memory_type: Optional[MemoryType] = None,
        min_relevance: float = 0.0
    ) -> List[Tuple[MemoryEntry, float]]:
        """동적 검색."""
        # 키워드 인덱스 활용
        candidate_ids: Set[str] = set()
        
        for keyword in query_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.keyword_index:
                candidate_ids.update(self.keyword_index[keyword_lower])
        
        # 타입 필터
        if memory_type:
            candidate_ids &= self.type_index[memory_type]
        
        # 관련성 점수 계산
        results = []
        query_set = set(k.lower() for k in query_keywords)
        
        for entry_id in candidate_ids:
            entry = self.entries.get(entry_id)
            if not entry:
                continue
            
            # 키워드 매칭 점수
            entry_keywords = set(k.lower() for k in entry.keywords)
            keyword_score = len(query_set & entry_keywords) / max(len(query_set), 1)
            
            # 중요도 가중치
            importance_weight = entry.importance
            
            # 접근 빈도 가중치
            access_weight = min(1.0, entry.access_count / 10)
            
            # 최종 점수
            score = (
                keyword_score * 0.6 +
                importance_weight * 0.3 +
                access_weight * 0.1
            )
            
            if score >= min_relevance:
                results.append((entry, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_by_type(
        self,
        memory_type: MemoryType,
        top_k: int = 10
    ) -> List[MemoryEntry]:
        """타입별 검색."""
        entry_ids = self.type_index.get(memory_type, set())
        entries = [self.entries[eid] for eid in entry_ids if eid in self.entries]
        
        # 중요도 기준 정렬
        entries.sort(key=lambda e: e.importance, reverse=True)
        return entries[:top_k]
    
    def get_related(
        self,
        entry_id: str,
        depth: int = 1
    ) -> List[MemoryEntry]:
        """관련 엔트리 조회 (Bidirectional)."""
        visited: Set[str] = {entry_id}
        current_level = {entry_id}
        
        for _ in range(depth):
            next_level: Set[str] = set()
            for eid in current_level:
                connected = self.connections.get(eid, set())
                next_level.update(connected - visited)
            
            visited.update(next_level)
            current_level = next_level
        
        # 원본 제외
        visited.discard(entry_id)
        
        return [self.entries[eid] for eid in visited if eid in self.entries]
    
    def _add_connection(self, id1: str, id2: str):
        """양방향 연결 추가."""
        self.connections[id1].add(id2)
        self.connections[id2].add(id1)
        
        # 엔트리의 related_entries도 업데이트
        if id1 in self.entries and id2 not in self.entries[id1].related_entries:
            self.entries[id1].related_entries.append(id2)
        if id2 in self.entries and id1 not in self.entries[id2].related_entries:
            self.entries[id2].related_entries.append(id1)
    
    @property
    def size(self) -> int:
        """저장소 크기."""
        return len(self.entries)
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 반환."""
        type_counts = {
            t.value: len(ids) 
            for t, ids in self.type_index.items()
        }
        
        return {
            "total_entries": len(self.entries),
            "type_distribution": type_counts,
            "keyword_count": len(self.keyword_index),
            "connection_count": sum(len(c) for c in self.connections.values()) // 2
        }


class TwoTierRAGSystem:
    """
    Two-Tier RAG 시스템.
    
    Tier 1 (Pre-loaded) + Tier 2 (Dynamic) 통합 관리.
    """
    
    def __init__(
        self,
        tier1_size: int = 20,
        enable_persistence: bool = True,
        auto_promote: bool = True
    ):
        self.tier1 = TierOneCache(max_entries=tier1_size)
        self.tier2 = TierTwoStore(enable_persistence=enable_persistence)
        self.auto_promote = auto_promote
        
        # 프로모션 임계값
        self.promotion_access_threshold = 5
        self.promotion_importance_threshold = 0.7
        
        logger.info(
            f"TwoTierRAGSystem initialized: tier1_size={tier1_size}, "
            f"persistence={enable_persistence}, auto_promote={auto_promote}"
        )
    
    def add(
        self,
        content: str,
        memory_type: MemoryType,
        keywords: Optional[List[str]] = None,
        importance: float = 0.5,
        source: str = "",
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        related_entries: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """
        새 메모리 엔트리 추가.
        """
        # ID 생성
        entry_id = self._generate_id(content)
        
        # 키워드 추출 (제공되지 않은 경우)
        if keywords is None:
            keywords = self._extract_keywords(content)
        
        # 요약 생성
        summary = content[:200] + "..." if len(content) > 200 else content
        
        entry = MemoryEntry(
            entry_id=entry_id,
            memory_type=memory_type,
            content=content,
            summary=summary,
            source=source,
            session_id=session_id,
            agent_id=agent_id,
            keywords=keywords,
            importance=importance,
            related_entries=related_entries or []
        )
        
        # Tier 2에 추가
        self.tier2.add(entry)
        
        # 높은 중요도면 Tier 1에도 추가
        if importance >= self.promotion_importance_threshold:
            self.tier1.add(entry)
        
        return entry
    
    def query(
        self,
        query: str,
        top_k: int = 10,
        memory_type: Optional[MemoryType] = None,
        include_related: bool = True,
        search_depth: int = 1
    ) -> List[Tuple[MemoryEntry, float, str]]:
        """
        통합 검색.
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 개수
            memory_type: 메모리 타입 필터
            include_related: 관련 엔트리 포함 여부
            search_depth: 관련 엔트리 검색 깊이
            
        Returns:
            (엔트리, 점수, 소스 티어) 튜플 리스트
        """
        keywords = self._extract_keywords(query)
        results: List[Tuple[MemoryEntry, float, str]] = []
        seen_ids: Set[str] = set()
        
        # Tier 1 검색 (빠른 캐시)
        tier1_results = self.tier1.search(keywords, top_k=top_k, memory_type=memory_type)
        for entry, score in tier1_results:
            if entry.entry_id not in seen_ids:
                results.append((entry, score * 1.2, "tier1"))  # Tier 1 보너스
                seen_ids.add(entry.entry_id)
        
        # Tier 2 검색 (남은 슬롯만큼)
        remaining = top_k - len(results)
        if remaining > 0:
            tier2_results = self.tier2.search(
                keywords,
                top_k=remaining * 2,  # 더 많이 검색
                memory_type=memory_type
            )
            
            for entry, score in tier2_results:
                if entry.entry_id not in seen_ids:
                    results.append((entry, score, "tier2"))
                    seen_ids.add(entry.entry_id)
                    
                    # 자동 프로모션 체크
                    if self.auto_promote and self._should_promote(entry):
                        self.tier1.add(entry)
                        logger.debug(f"Promoted entry {entry.entry_id} to Tier 1")
                    
                    if len(results) >= top_k:
                        break
        
        # 관련 엔트리 포함
        if include_related and results:
            top_entry = results[0][0]
            related = self.tier2.get_related(top_entry.entry_id, depth=search_depth)
            
            for entry in related[:3]:  # 상위 3개만
                if entry.entry_id not in seen_ids:
                    results.append((entry, 0.5, "related"))
                    seen_ids.add(entry.entry_id)
        
        # 점수 기준 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def query_by_type(
        self,
        memory_type: MemoryType,
        top_k: int = 10
    ) -> List[MemoryEntry]:
        """타입별 검색."""
        return self.tier2.search_by_type(memory_type, top_k=top_k)
    
    def get_errors(self, top_k: int = 10) -> List[MemoryEntry]:
        """오류 메모리 조회."""
        return self.query_by_type(MemoryType.ERROR, top_k=top_k)
    
    def get_discoveries(self, top_k: int = 10) -> List[MemoryEntry]:
        """발견 메모리 조회."""
        return self.query_by_type(MemoryType.DISCOVERY, top_k=top_k)
    
    def get_decisions(self, top_k: int = 10) -> List[MemoryEntry]:
        """결정 메모리 조회."""
        return self.query_by_type(MemoryType.DECISION, top_k=top_k)
    
    def get_learnings(self, top_k: int = 10) -> List[MemoryEntry]:
        """학습 메모리 조회."""
        return self.query_by_type(MemoryType.LEARNING, top_k=top_k)
    
    def preload_for_session(
        self,
        session_context: str,
        session_id: Optional[str] = None
    ):
        """
        세션 시작 시 관련 컨텍스트 사전 로드.
        
        Args:
            session_context: 세션 컨텍스트 (쿼리, 주제 등)
            session_id: 세션 ID (이전 세션 메모리 로드용)
        """
        keywords = self._extract_keywords(session_context)
        
        # 키워드 기반 관련 엔트리 검색
        results = self.tier2.search(keywords, top_k=50)
        
        # 세션 ID가 있으면 해당 세션의 메모리도 포함
        if session_id:
            session_entries = [
                e for e in self.tier2.entries.values()
                if e.session_id == session_id
            ]
            results.extend([(e, e.importance) for e in session_entries])
        
        # 중복 제거 및 중요도 기준 정렬
        seen = set()
        unique_entries = []
        for entry, _ in sorted(results, key=lambda x: x[1], reverse=True):
            if entry.entry_id not in seen:
                unique_entries.append(entry)
                seen.add(entry.entry_id)
        
        # Tier 1에 로드
        self.tier1.load(unique_entries)
        
        logger.info(
            f"Preloaded {self.tier1.size} entries for session "
            f"(context: {session_context[:50]}...)"
        )
    
    def connect_entries(self, entry_id1: str, entry_id2: str):
        """두 엔트리 간 양방향 연결."""
        self.tier2._add_connection(entry_id1, entry_id2)
    
    def _should_promote(self, entry: MemoryEntry) -> bool:
        """Tier 1 프로모션 여부 결정."""
        return (
            entry.access_count >= self.promotion_access_threshold or
            entry.importance >= self.promotion_importance_threshold
        )
    
    def _generate_id(self, content: str) -> str:
        """엔트리 ID 생성."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{content[:100]}{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "can", "to", "of", "in", "for", "on",
            "with", "at", "by", "from", "as", "and", "or", "but", "if",
            "this", "that", "these", "those", "it", "its"
        }
        
        words = text.lower().split()
        keywords = []
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word and clean_word not in stopwords and len(clean_word) > 2:
                keywords.append(clean_word)
        
        # 빈도 기반 상위 키워드
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:15]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """시스템 통계."""
        return {
            "tier1": {
                "size": self.tier1.size,
                "max_size": self.tier1.max_entries,
                "hit_rate": self.tier1.hit_rate,
                "loaded_at": self.tier1.loaded_at.isoformat() if self.tier1.loaded_at else None
            },
            "tier2": self.tier2.get_statistics(),
            "auto_promote": self.auto_promote,
            "promotion_thresholds": {
                "access": self.promotion_access_threshold,
                "importance": self.promotion_importance_threshold
            }
        }


# Singleton instance
_two_tier_rag: Optional[TwoTierRAGSystem] = None


def get_two_tier_rag(
    tier1_size: int = 20,
    enable_persistence: bool = True,
    auto_promote: bool = True
) -> TwoTierRAGSystem:
    """TwoTierRAGSystem 싱글톤 인스턴스 반환."""
    global _two_tier_rag
    
    if _two_tier_rag is None:
        _two_tier_rag = TwoTierRAGSystem(
            tier1_size=tier1_size,
            enable_persistence=enable_persistence,
            auto_promote=auto_promote
        )
    
    return _two_tier_rag
