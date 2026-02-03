"""
Topic Segmenter Module

LightMem 영감을 받은 토픽 기반 메모리 세분화 시스템.
대화와 연구 컨텍스트를 토픽별로 자동 분할하여 효율적인 검색 지원.

핵심 특징:
- 자동 토픽 감지 및 세분화
- Sensory Buffer for real-time processing
- 토픽 경계 감지 (topic boundary detection)
- 토픽별 메모리 클러스터링
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import deque
from enum import Enum
import hashlib

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TopicType(Enum):
    """토픽 유형."""
    RESEARCH = "research"
    QUESTION = "question"
    ANSWER = "answer"
    DISCUSSION = "discussion"
    FINDING = "finding"
    SYNTHESIS = "synthesis"
    META = "meta"


class TopicSegment(BaseModel):
    """개별 토픽 세그먼트."""
    segment_id: str = Field(description="세그먼트 고유 ID")
    topic_type: TopicType = Field(default=TopicType.RESEARCH, description="토픽 유형")
    topic_label: str = Field(default="", description="토픽 레이블")
    content: str = Field(description="세그먼트 내용")
    keywords: List[str] = Field(default_factory=list, description="핵심 키워드")
    embedding_vector: Optional[List[float]] = Field(default=None, description="임베딩 벡터")
    
    # 메타데이터
    created_at: datetime = Field(default_factory=datetime.now)
    token_count: int = Field(default=0, description="토큰 수")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="중요도")
    
    # 연결 정보
    parent_segment_id: Optional[str] = Field(default=None, description="부모 세그먼트")
    child_segment_ids: List[str] = Field(default_factory=list, description="자식 세그먼트들")
    related_segment_ids: List[str] = Field(default_factory=list, description="관련 세그먼트들")
    
    class Config:
        arbitrary_types_allowed = True


class SensoryBuffer:
    """
    실시간 입력을 위한 Sensory Buffer.
    
    LightMem의 sensory memory 개념 적용.
    새로운 입력을 일시적으로 버퍼링하고 토픽 경계 감지.
    """
    
    def __init__(
        self,
        max_tokens: int = 512,
        boundary_threshold: float = 0.3
    ):
        self.max_tokens = max_tokens
        self.boundary_threshold = boundary_threshold
        self.buffer: deque = deque()
        self.current_tokens = 0
        self.last_topic_keywords: List[str] = []
        
    def add(self, content: str, token_count: int) -> Optional[str]:
        """
        버퍼에 내용 추가.
        
        버퍼가 가득 차면 flush하여 세그먼트 반환.
        """
        self.buffer.append({
            "content": content,
            "tokens": token_count,
            "timestamp": datetime.now()
        })
        self.current_tokens += token_count
        
        # 버퍼 오버플로우 시 flush
        if self.current_tokens >= self.max_tokens:
            return self.flush()
        
        return None
    
    def flush(self) -> str:
        """버퍼 비우고 내용 반환."""
        if not self.buffer:
            return ""
        
        content = " ".join(item["content"] for item in self.buffer)
        self.buffer.clear()
        self.current_tokens = 0
        
        return content
    
    def peek(self) -> str:
        """현재 버퍼 내용 확인 (비우지 않음)."""
        return " ".join(item["content"] for item in self.buffer)
    
    def is_empty(self) -> bool:
        """버퍼가 비어있는지 확인."""
        return len(self.buffer) == 0
    
    @property
    def fill_ratio(self) -> float:
        """버퍼 채움 비율."""
        return self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0


class TopicBoundaryDetector:
    """
    토픽 경계 감지기.
    
    텍스트 흐름에서 토픽 전환점을 감지.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.3,
        min_segment_tokens: int = 100
    ):
        self.similarity_threshold = similarity_threshold
        self.min_segment_tokens = min_segment_tokens
        self.previous_keywords: List[str] = []
    
    def detect_boundary(
        self,
        current_content: str,
        current_keywords: List[str]
    ) -> Tuple[bool, float]:
        """
        토픽 경계 감지.
        
        Args:
            current_content: 현재 컨텐츠
            current_keywords: 현재 키워드들
            
        Returns:
            (경계 여부, 유사도 점수)
        """
        if not self.previous_keywords:
            self.previous_keywords = current_keywords
            return False, 1.0
        
        # 키워드 기반 유사도 계산
        similarity = self._calculate_keyword_similarity(
            self.previous_keywords,
            current_keywords
        )
        
        # 토픽 전환 감지
        is_boundary = similarity < self.similarity_threshold
        
        # 이전 키워드 업데이트
        self.previous_keywords = current_keywords
        
        return is_boundary, similarity
    
    def _calculate_keyword_similarity(
        self,
        keywords1: List[str],
        keywords2: List[str]
    ) -> float:
        """Jaccard 유사도 계산."""
        if not keywords1 or not keywords2:
            return 0.0
        
        set1 = set(k.lower() for k in keywords1)
        set2 = set(k.lower() for k in keywords2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


class TopicSegmenter:
    """
    메인 토픽 세분화 엔진.
    
    대화와 연구 컨텐츠를 토픽별로 자동 세분화.
    """
    
    def __init__(
        self,
        sensory_buffer_size: int = 512,
        boundary_threshold: float = 0.3,
        min_segment_tokens: int = 100,
        max_segment_tokens: int = 2000
    ):
        self.sensory_buffer = SensoryBuffer(
            max_tokens=sensory_buffer_size,
            boundary_threshold=boundary_threshold
        )
        self.boundary_detector = TopicBoundaryDetector(
            similarity_threshold=boundary_threshold,
            min_segment_tokens=min_segment_tokens
        )
        self.max_segment_tokens = max_segment_tokens
        
        # 세그먼트 저장소
        self.segments: Dict[str, TopicSegment] = {}
        self.segment_order: List[str] = []  # 순서 유지
        self.topic_clusters: Dict[str, List[str]] = {}  # 토픽 -> 세그먼트 IDs
        
        # 현재 상태
        self.current_segment_buffer: List[str] = []
        self.current_segment_tokens: int = 0
        self.current_keywords: List[str] = []
        
        logger.info(
            f"TopicSegmenter initialized: buffer_size={sensory_buffer_size}, "
            f"threshold={boundary_threshold}"
        )
    
    async def process(
        self,
        content: str,
        topic_type: TopicType = TopicType.RESEARCH,
        force_new_segment: bool = False
    ) -> Optional[TopicSegment]:
        """
        컨텐츠 처리 및 세분화.
        
        Args:
            content: 입력 컨텐츠
            topic_type: 토픽 유형
            force_new_segment: 강제로 새 세그먼트 생성
            
        Returns:
            완성된 세그먼트 (있을 경우)
        """
        # 토큰 수 추정 (간단한 방식)
        token_count = len(content.split())
        
        # 키워드 추출
        keywords = self._extract_keywords(content)
        
        # 토픽 경계 감지
        is_boundary, similarity = self.boundary_detector.detect_boundary(
            content, keywords
        )
        
        # 새 세그먼트 생성 조건
        should_create_segment = (
            force_new_segment or
            is_boundary or
            self.current_segment_tokens + token_count > self.max_segment_tokens
        )
        
        completed_segment = None
        
        if should_create_segment and self.current_segment_buffer:
            # 현재 버퍼로 세그먼트 생성
            completed_segment = self._create_segment(
                " ".join(self.current_segment_buffer),
                topic_type,
                self.current_keywords
            )
            
            # 버퍼 초기화
            self.current_segment_buffer = []
            self.current_segment_tokens = 0
            self.current_keywords = []
        
        # 새 컨텐츠를 버퍼에 추가
        self.current_segment_buffer.append(content)
        self.current_segment_tokens += token_count
        self.current_keywords.extend(keywords)
        
        return completed_segment
    
    def flush(self, topic_type: TopicType = TopicType.RESEARCH) -> Optional[TopicSegment]:
        """현재 버퍼의 내용으로 세그먼트 생성."""
        if not self.current_segment_buffer:
            return None
        
        segment = self._create_segment(
            " ".join(self.current_segment_buffer),
            topic_type,
            self.current_keywords
        )
        
        self.current_segment_buffer = []
        self.current_segment_tokens = 0
        self.current_keywords = []
        
        return segment
    
    def _create_segment(
        self,
        content: str,
        topic_type: TopicType,
        keywords: List[str]
    ) -> TopicSegment:
        """세그먼트 생성 및 저장."""
        # 고유 ID 생성
        segment_id = self._generate_segment_id(content)
        
        # 토픽 레이블 생성
        topic_label = self._generate_topic_label(keywords)
        
        # 중요도 점수 계산
        importance = self._calculate_importance(content, keywords)
        
        segment = TopicSegment(
            segment_id=segment_id,
            topic_type=topic_type,
            topic_label=topic_label,
            content=content,
            keywords=list(set(keywords))[:20],  # 상위 20개 키워드
            token_count=len(content.split()),
            importance_score=importance
        )
        
        # 저장
        self.segments[segment_id] = segment
        self.segment_order.append(segment_id)
        
        # 토픽 클러스터 업데이트
        if topic_label not in self.topic_clusters:
            self.topic_clusters[topic_label] = []
        self.topic_clusters[topic_label].append(segment_id)
        
        logger.debug(f"Created segment {segment_id}: {topic_label} ({segment.token_count} tokens)")
        
        return segment
    
    def _generate_segment_id(self, content: str) -> str:
        """세그먼트 ID 생성."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{content[:100]}{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _generate_topic_label(self, keywords: List[str]) -> str:
        """키워드 기반 토픽 레이블 생성."""
        if not keywords:
            return "general"
        
        # 상위 3개 키워드로 레이블 생성
        top_keywords = keywords[:3]
        return "_".join(k.lower().replace(" ", "-") for k in top_keywords)
    
    def _extract_keywords(self, content: str) -> List[str]:
        """간단한 키워드 추출."""
        # 불용어
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because",
            "until", "while", "this", "that", "these", "those", "it"
        }
        
        # 단어 추출
        words = content.lower().split()
        
        # 필터링
        keywords = []
        for word in words:
            # 알파벳만 유지
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word and clean_word not in stopwords and len(clean_word) > 2:
                keywords.append(clean_word)
        
        # 빈도 기반 상위 키워드
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:15]]
    
    def _calculate_importance(self, content: str, keywords: List[str]) -> float:
        """중요도 점수 계산."""
        # 기본 점수
        score = 0.5
        
        # 길이 기반 조정
        word_count = len(content.split())
        if word_count > 200:
            score += 0.1
        if word_count > 500:
            score += 0.1
        
        # 키워드 다양성
        unique_keywords = len(set(keywords))
        if unique_keywords > 10:
            score += 0.1
        
        # 특정 패턴 감지 (finding, conclusion 등)
        important_patterns = ["finding", "conclusion", "result", "important", "key", "significant"]
        content_lower = content.lower()
        for pattern in important_patterns:
            if pattern in content_lower:
                score += 0.05
        
        return min(1.0, score)
    
    def get_segment(self, segment_id: str) -> Optional[TopicSegment]:
        """세그먼트 조회."""
        return self.segments.get(segment_id)
    
    def get_segments_by_topic(self, topic_label: str) -> List[TopicSegment]:
        """토픽별 세그먼트 조회."""
        segment_ids = self.topic_clusters.get(topic_label, [])
        return [self.segments[sid] for sid in segment_ids if sid in self.segments]
    
    def get_recent_segments(self, n: int = 10) -> List[TopicSegment]:
        """최근 n개 세그먼트 조회."""
        recent_ids = self.segment_order[-n:] if self.segment_order else []
        return [self.segments[sid] for sid in reversed(recent_ids) if sid in self.segments]
    
    def search_segments(
        self,
        query: str,
        top_k: int = 5,
        topic_filter: Optional[str] = None
    ) -> List[Tuple[TopicSegment, float]]:
        """
        키워드 기반 세그먼트 검색.
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 개수
            topic_filter: 토픽 필터 (선택)
            
        Returns:
            (세그먼트, 관련도 점수) 튜플 리스트
        """
        query_keywords = self._extract_keywords(query)
        
        results = []
        segments_to_search = list(self.segments.values())
        
        if topic_filter:
            segment_ids = self.topic_clusters.get(topic_filter, [])
            segments_to_search = [self.segments[sid] for sid in segment_ids if sid in self.segments]
        
        for segment in segments_to_search:
            # 키워드 매칭 점수
            common = set(query_keywords) & set(k.lower() for k in segment.keywords)
            if not common:
                continue
            
            score = len(common) / max(len(query_keywords), 1)
            score *= segment.importance_score  # 중요도 가중치
            
            results.append((segment, score))
        
        # 점수 기준 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """세분화 통계 반환."""
        total_segments = len(self.segments)
        total_tokens = sum(s.token_count for s in self.segments.values())
        
        return {
            "total_segments": total_segments,
            "total_tokens": total_tokens,
            "topic_clusters": len(self.topic_clusters),
            "topics": list(self.topic_clusters.keys()),
            "avg_tokens_per_segment": total_tokens / total_segments if total_segments > 0 else 0,
            "buffer_fill_ratio": self.sensory_buffer.fill_ratio
        }


# Singleton instance
_topic_segmenter: Optional[TopicSegmenter] = None


def get_topic_segmenter(
    sensory_buffer_size: int = 512,
    boundary_threshold: float = 0.3
) -> TopicSegmenter:
    """TopicSegmenter 싱글톤 인스턴스 반환."""
    global _topic_segmenter
    
    if _topic_segmenter is None:
        _topic_segmenter = TopicSegmenter(
            sensory_buffer_size=sensory_buffer_size,
            boundary_threshold=boundary_threshold
        )
    
    return _topic_segmenter
