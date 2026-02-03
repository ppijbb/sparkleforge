"""
Pre-Compressor Module

LightMem 영감을 받은 사전 압축 시스템.
LLMLingua-2 스타일의 토큰 압축으로 메모리 효율성 극대화.

핵심 특징:
- Entropy 기반 압축
- 중요도 기반 토큰 선택
- 의미 보존 압축
- 압축률 조절 가능
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """압축 수준."""
    NONE = "none"
    LIGHT = "light"  # ~80% 유지
    MODERATE = "moderate"  # ~60% 유지
    AGGRESSIVE = "aggressive"  # ~40% 유지
    EXTREME = "extreme"  # ~20% 유지


class CompressionResult(BaseModel):
    """압축 결과."""
    original_text: str = Field(description="원본 텍스트")
    compressed_text: str = Field(description="압축된 텍스트")
    original_tokens: int = Field(description="원본 토큰 수")
    compressed_tokens: int = Field(description="압축 후 토큰 수")
    compression_ratio: float = Field(description="압축률")
    preserved_keywords: List[str] = Field(default_factory=list, description="보존된 키워드")
    removed_phrases: List[str] = Field(default_factory=list, description="제거된 구문")


@dataclass
class TokenScore:
    """토큰 중요도 점수."""
    token: str
    position: int
    importance: float
    is_keyword: bool
    is_entity: bool


class PreCompressor:
    """
    사전 압축 엔진.
    
    LLMLingua-2 영감을 받은 토큰 레벨 압축.
    """
    
    def __init__(
        self,
        default_level: CompressionLevel = CompressionLevel.MODERATE,
        preserve_keywords: bool = True,
        preserve_entities: bool = True,
        min_sentence_length: int = 5
    ):
        self.default_level = default_level
        self.preserve_keywords = preserve_keywords
        self.preserve_entities = preserve_entities
        self.min_sentence_length = min_sentence_length
        
        # 압축 수준별 유지 비율
        self.retention_ratios = {
            CompressionLevel.NONE: 1.0,
            CompressionLevel.LIGHT: 0.8,
            CompressionLevel.MODERATE: 0.6,
            CompressionLevel.AGGRESSIVE: 0.4,
            CompressionLevel.EXTREME: 0.2
        }
        
        # 중요 패턴 (항상 보존)
        self.important_patterns = [
            r'\b\d+\.?\d*%\b',  # 백분율
            r'\$\d+\.?\d*[KMB]?\b',  # 금액
            r'\b\d{4}\b',  # 연도
            r'\b[A-Z]{2,}\b',  # 약어
        ]
        
        # 불필요한 패턴 (우선 제거)
        self.removable_patterns = [
            r'\b(however|therefore|moreover|furthermore|additionally)\b',
            r'\b(in order to|as a result of|due to the fact that)\b',
            r'\b(it is|there is|there are)\b',
            r'\b(very|really|quite|rather|somewhat)\b',
        ]
        
        logger.info(f"PreCompressor initialized: default_level={default_level.value}")
    
    def compress(
        self,
        text: str,
        level: Optional[CompressionLevel] = None,
        target_tokens: Optional[int] = None,
        preserve_keywords: Optional[List[str]] = None
    ) -> CompressionResult:
        """
        텍스트 압축.
        
        Args:
            text: 압축할 텍스트
            level: 압축 수준 (None이면 default 사용)
            target_tokens: 목표 토큰 수 (level 대신 사용)
            preserve_keywords: 반드시 보존할 키워드들
            
        Returns:
            압축 결과
        """
        level = level or self.default_level
        original_tokens = self._count_tokens(text)
        
        if level == CompressionLevel.NONE:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                preserved_keywords=preserve_keywords or []
            )
        
        # 목표 토큰 수 계산
        if target_tokens:
            target_ratio = target_tokens / original_tokens
        else:
            target_ratio = self.retention_ratios[level]
        
        # 문장 분리
        sentences = self._split_sentences(text)
        
        # 문장별 중요도 계산
        scored_sentences = [
            (s, self._score_sentence(s, preserve_keywords))
            for s in sentences
        ]
        
        # 중요도 기반 압축
        compressed_text, preserved, removed = self._compress_by_importance(
            scored_sentences,
            target_ratio,
            preserve_keywords or []
        )
        
        compressed_tokens = self._count_tokens(compressed_text)
        
        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            preserved_keywords=preserved,
            removed_phrases=removed
        )
    
    def compress_for_context(
        self,
        text: str,
        max_tokens: int,
        query: Optional[str] = None
    ) -> str:
        """
        컨텍스트 윈도우에 맞춰 압축.
        
        Args:
            text: 압축할 텍스트
            max_tokens: 최대 허용 토큰
            query: 관련 쿼리 (중요도 계산에 사용)
            
        Returns:
            압축된 텍스트
        """
        current_tokens = self._count_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        # 쿼리에서 키워드 추출
        query_keywords = self._extract_keywords(query) if query else []
        
        # 동적 압축 수준 결정
        ratio = max_tokens / current_tokens
        if ratio >= 0.8:
            level = CompressionLevel.LIGHT
        elif ratio >= 0.6:
            level = CompressionLevel.MODERATE
        elif ratio >= 0.4:
            level = CompressionLevel.AGGRESSIVE
        else:
            level = CompressionLevel.EXTREME
        
        result = self.compress(
            text,
            level=level,
            target_tokens=max_tokens,
            preserve_keywords=query_keywords
        )
        
        return result.compressed_text
    
    def _count_tokens(self, text: str) -> int:
        """간단한 토큰 수 추정."""
        # 공백 기준 + 특수문자 고려
        words = text.split()
        return int(len(words) * 1.3)  # 평균적으로 1 word = ~1.3 tokens
    
    def _split_sentences(self, text: str) -> List[str]:
        """문장 분리."""
        # 간단한 문장 분리
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _score_sentence(
        self,
        sentence: str,
        preserve_keywords: Optional[List[str]] = None
    ) -> float:
        """문장 중요도 점수 계산."""
        score = 0.5  # 기본 점수
        
        words = sentence.lower().split()
        
        # 길이 패널티 (너무 짧은 문장은 덜 중요)
        if len(words) < self.min_sentence_length:
            score -= 0.2
        
        # 중요 패턴 보너스
        for pattern in self.important_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                score += 0.15
        
        # 제거 가능 패턴 페널티
        for pattern in self.removable_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                score -= 0.1
        
        # 보존 키워드 보너스
        if preserve_keywords:
            for keyword in preserve_keywords:
                if keyword.lower() in sentence.lower():
                    score += 0.2
        
        # 대문자 단어 (고유명사) 보너스
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', sentence)
        score += len(proper_nouns) * 0.05
        
        # 숫자 포함 보너스
        if re.search(r'\d+', sentence):
            score += 0.1
        
        # 인용구 보너스
        if '"' in sentence or "'" in sentence:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _compress_by_importance(
        self,
        scored_sentences: List[Tuple[str, float]],
        target_ratio: float,
        preserve_keywords: List[str]
    ) -> Tuple[str, List[str], List[str]]:
        """
        중요도 기반 압축.
        
        Returns:
            (압축된 텍스트, 보존된 키워드, 제거된 구문)
        """
        if not scored_sentences:
            return "", [], []
        
        # 총 토큰 수 계산
        total_tokens = sum(self._count_tokens(s) for s, _ in scored_sentences)
        target_tokens = int(total_tokens * target_ratio)
        
        # 점수 기준 정렬
        sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
        
        # 상위 문장 선택
        selected = []
        current_tokens = 0
        removed = []
        preserved = list(preserve_keywords)
        
        for sentence, score in sorted_sentences:
            sent_tokens = self._count_tokens(sentence)
            
            if current_tokens + sent_tokens <= target_tokens:
                selected.append((sentence, score))
                current_tokens += sent_tokens
                
                # 키워드 추출 및 보존 목록 업데이트
                for keyword in preserve_keywords:
                    if keyword.lower() in sentence.lower() and keyword not in preserved:
                        preserved.append(keyword)
            else:
                # 제거된 문장 기록 (중요도가 낮은 문장)
                if score < 0.4:
                    removed.append(sentence[:50] + "..." if len(sentence) > 50 else sentence)
        
        # 원래 순서로 재정렬
        original_order = {s: i for i, (s, _) in enumerate(scored_sentences)}
        selected.sort(key=lambda x: original_order.get(x[0], 0))
        
        # 텍스트 조합
        compressed = " ".join(s for s, _ in selected)
        
        return compressed, preserved, removed[:5]  # 제거된 구문은 상위 5개만
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출."""
        if not text:
            return []
        
        # 불용어 제거
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "and", "or", "but"
        }
        
        words = text.lower().split()
        keywords = [
            word.strip('.,!?:;()[]{}')
            for word in words
            if word.lower() not in stopwords and len(word) > 2
        ]
        
        # 빈도 기반 상위 키워드
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:10]]


class HybridRetriever:
    """
    하이브리드 검색기.
    
    Context-based + Embedding-based 검색 융합.
    """
    
    def __init__(
        self,
        context_weight: float = 0.4,
        embedding_weight: float = 0.6
    ):
        self.context_weight = context_weight
        self.embedding_weight = embedding_weight
        
        logger.info(
            f"HybridRetriever initialized: context={context_weight}, embedding={embedding_weight}"
        )
    
    def retrieve(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        context: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        하이브리드 검색 수행.
        
        Args:
            query: 검색 쿼리
            documents: 검색 대상 문서들
            top_k: 반환할 최대 개수
            context: 추가 컨텍스트
            
        Returns:
            (문서, 점수) 튜플 리스트
        """
        if not documents:
            return []
        
        results = []
        query_keywords = set(self._extract_keywords(query))
        
        for doc in documents:
            content = doc.get("content", "")
            
            # 1. 키워드 기반 점수 (context-based)
            keyword_score = self._calculate_keyword_score(query_keywords, content)
            
            # 2. 임베딩 기반 점수 (placeholder - 실제 구현에서는 벡터 유사도 사용)
            embedding_score = self._estimate_semantic_similarity(query, content)
            
            # 3. 컨텍스트 보너스
            context_bonus = 0.0
            if context:
                context_keywords = set(self._extract_keywords(context))
                common = context_keywords & set(self._extract_keywords(content))
                context_bonus = len(common) / max(len(context_keywords), 1) * 0.2
            
            # 최종 점수 계산
            final_score = (
                self.context_weight * keyword_score +
                self.embedding_weight * embedding_score +
                context_bonus
            )
            
            results.append((doc, final_score))
        
        # 점수 기준 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _calculate_keyword_score(self, query_keywords: set, content: str) -> float:
        """키워드 매칭 점수."""
        content_lower = content.lower()
        matches = sum(1 for kw in query_keywords if kw in content_lower)
        return matches / max(len(query_keywords), 1)
    
    def _estimate_semantic_similarity(self, query: str, content: str) -> float:
        """
        의미적 유사도 추정 (간단한 버전).
        
        실제 구현에서는 임베딩 벡터 코사인 유사도 사용.
        """
        # 간단한 n-gram 기반 유사도
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        intersection = len(query_words & content_words)
        union = len(query_words | content_words)
        
        return intersection / max(union, 1)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "can", "to", "of", "in", "for", "on",
            "with", "at", "by", "from", "as", "and", "or", "but", "if"
        }
        
        words = text.lower().split()
        return [
            word.strip('.,!?:;()[]{}')
            for word in words
            if word.lower() not in stopwords and len(word) > 2
        ]


# Singleton instances
_pre_compressor: Optional[PreCompressor] = None
_hybrid_retriever: Optional[HybridRetriever] = None


def get_pre_compressor(
    default_level: CompressionLevel = CompressionLevel.MODERATE
) -> PreCompressor:
    """PreCompressor 싱글톤 인스턴스 반환."""
    global _pre_compressor
    
    if _pre_compressor is None:
        _pre_compressor = PreCompressor(default_level=default_level)
    
    return _pre_compressor


def get_hybrid_retriever(
    context_weight: float = 0.4,
    embedding_weight: float = 0.6
) -> HybridRetriever:
    """HybridRetriever 싱글톤 인스턴스 반환."""
    global _hybrid_retriever
    
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(
            context_weight=context_weight,
            embedding_weight=embedding_weight
        )
    
    return _hybrid_retriever
