"""
Hierarchical Compression (혁신 2)

3단계 압축, 중요도 기반 보존, 압축 검증, 압축 히스토리를 통한
정보 손실 최소화된 계층적 압축 시스템.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from datetime import datetime

from src.core.researcher_config import get_compression_config, get_llm_config
from src.core.reliability import execute_with_reliability

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """압축 레벨."""
    RAW = "raw"
    INTERMEDIATE = "intermediate"
    COMPRESSED = "compressed"
    FINAL = "final"


@dataclass
class CompressionResult:
    """압축 결과."""
    level: CompressionLevel
    data: Any
    original_size: int
    compressed_size: int
    compression_ratio: float
    important_info_preserved: List[str]
    validation_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class ImportantInfo:
    """중요 정보."""
    content: str
    importance_score: float
    category: str
    position: int


class ImportanceAnalyzer:
    """중요도 분석기."""
    
    def __init__(self):
        self.llm_config = get_llm_config()
        self.importance_keywords = [
            "결론", "요약", "핵심", "중요", "주요", "결과", "발견",
            "conclusion", "summary", "key", "important", "main", "result", "finding"
        ]
        self.category_weights = {
            "conclusion": 1.0,
            "summary": 0.9,
            "key_findings": 0.95,
            "data": 0.8,
            "methodology": 0.7,
            "background": 0.5,
            "references": 0.3
        }
    
    async def analyze_importance(self, text: str) -> List[ImportantInfo]:
        """텍스트 중요도 분석."""
        important_info = []
        sentences = text.split('. ')
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # 키워드 기반 중요도 점수
            keyword_score = self._calculate_keyword_score(sentence)
            
            # 카테고리 기반 중요도 점수
            category_score = self._calculate_category_score(sentence)
            
            # 위치 기반 중요도 점수 (시작과 끝이 더 중요)
            position_score = self._calculate_position_score(i, len(sentences))
            
            # 종합 중요도 점수
            total_score = (keyword_score * 0.4 + category_score * 0.4 + position_score * 0.2)
            
            if total_score > 0.3:  # 임계값 이상만 중요 정보로 간주
                important_info.append(ImportantInfo(
                    content=sentence.strip(),
                    importance_score=total_score,
                    category=self._categorize_sentence(sentence),
                    position=i
                ))
        
        # 중요도 순으로 정렬
        important_info.sort(key=lambda x: x.importance_score, reverse=True)
        return important_info
    
    def _calculate_keyword_score(self, sentence: str) -> float:
        """키워드 기반 점수 계산."""
        sentence_lower = sentence.lower()
        score = 0.0
        
        for keyword in self.importance_keywords:
            if keyword in sentence_lower:
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_category_score(self, sentence: str) -> float:
        """카테고리 기반 점수 계산."""
        sentence_lower = sentence.lower()
        
        for category, weight in self.category_weights.items():
            if category in sentence_lower:
                return weight
        
        return 0.5  # 기본 점수
    
    def _calculate_position_score(self, position: int, total: int) -> float:
        """위치 기반 점수 계산."""
        if total <= 1:
            return 1.0
        
        # 시작과 끝 부분이 더 중요
        relative_position = position / (total - 1)
        if relative_position < 0.1 or relative_position > 0.9:
            return 1.0
        elif relative_position < 0.2 or relative_position > 0.8:
            return 0.8
        else:
            return 0.6
    
    def _categorize_sentence(self, sentence: str) -> str:
        """문장 카테고리 분류."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ["결론", "conclusion", "요약", "summary"]):
            return "conclusion"
        elif any(word in sentence_lower for word in ["핵심", "key", "주요", "main"]):
            return "key_findings"
        elif any(word in sentence_lower for word in ["데이터", "data", "수치", "number"]):
            return "data"
        elif any(word in sentence_lower for word in ["방법", "method", "방법론", "methodology"]):
            return "methodology"
        elif any(word in sentence_lower for word in ["참고문헌", "reference", "인용", "cite"]):
            return "references"
        else:
            return "background"


class CompressionValidator:
    """압축 검증기."""
    
    def __init__(self):
        self.config = get_compression_config()
    
    async def validate_compression(
        self,
        original: str,
        compressed: str,
        important_info: List[ImportantInfo]
    ) -> Tuple[bool, float, List[str]]:
        """압축 검증."""
        validation_issues = []
        
        # 1. 압축률 검증
        compression_ratio = len(compressed) / len(original) if len(original) > 0 else 0
        if compression_ratio < self.config.min_compression_ratio:
            validation_issues.append(f"Compression too aggressive: {compression_ratio:.2%}")
        
        # 2. 중요 정보 보존 검증
        preserved_info = []
        for info in important_info:
            if info.content in compressed or self._is_similar_content(info.content, compressed):
                preserved_info.append(info.content)
            else:
                validation_issues.append(f"Important info lost: {info.content[:50]}...")
        
        # 3. 의미 보존 검증
        semantic_score = await self._calculate_semantic_similarity(original, compressed)
        if semantic_score < 0.7:
            validation_issues.append(f"Low semantic similarity: {semantic_score:.2f}")
        
        # 4. 구조 보존 검증
        structure_score = self._calculate_structure_similarity(original, compressed)
        if structure_score < 0.8:
            validation_issues.append(f"Low structure similarity: {structure_score:.2f}")
        
        # 종합 검증 점수
        validation_score = (
            min(compression_ratio / self.config.min_compression_ratio, 1.0) * 0.3 +
            len(preserved_info) / len(important_info) * 0.4 +
            semantic_score * 0.2 +
            structure_score * 0.1
        )
        
        is_valid = len(validation_issues) == 0 and validation_score > 0.6
        
        return is_valid, validation_score, validation_issues
    
    def _is_similar_content(self, original: str, compressed: str) -> bool:
        """유사한 내용인지 확인."""
        # 간단한 유사도 검사 (실제로는 더 정교한 방법 사용)
        original_words = set(original.lower().split())
        compressed_words = set(compressed.lower().split())
        
        if not original_words:
            return True
        
        overlap = len(original_words.intersection(compressed_words))
        similarity = overlap / len(original_words)
        
        return similarity > 0.5
    
    async def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """의미적 유사도 계산."""
        # 실제 구현에서는 임베딩 모델 사용
        # 여기서는 간단한 키워드 기반 유사도 계산
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_structure_similarity(self, original: str, compressed: str) -> float:
        """구조적 유사도 계산."""
        # 문단 수, 문장 수, 키워드 분포 등으로 구조 유사도 계산
        original_paragraphs = original.split('\n\n')
        compressed_paragraphs = compressed.split('\n\n')
        
        if not original_paragraphs or not compressed_paragraphs:
            return 0.0
        
        # 문단 수 비율
        paragraph_ratio = min(len(compressed_paragraphs) / len(original_paragraphs), 1.0)
        
        # 문장 수 비율
        original_sentences = original.count('.')
        compressed_sentences = compressed.count('.')
        sentence_ratio = min(compressed_sentences / max(original_sentences, 1), 1.0)
        
        return (paragraph_ratio + sentence_ratio) / 2


class CompressionHistory:
    """압축 히스토리 관리."""
    
    def __init__(self):
        self.history: List[CompressionResult] = []
        self.max_history = 100
    
    def add_compression(self, result: CompressionResult):
        """압축 결과 추가."""
        self.history.append(result)
        
        # 최대 히스토리 수 제한
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_compression_history(self, limit: int = 10) -> List[CompressionResult]:
        """압축 히스토리 반환."""
        return self.history[-limit:]
    
    def get_best_compression(self, target_ratio: float = 0.5) -> Optional[CompressionResult]:
        """목표 압축률에 가장 가까운 결과 반환."""
        if not self.history:
            return None
        
        best_result = None
        best_diff = float('inf')
        
        for result in self.history:
            diff = abs(result.compression_ratio - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_result = result
        
        return best_result


class HierarchicalCompressor:
    """계층적 압축기 (혁신 2)."""
    
    def __init__(self):
        self.config = get_compression_config()
        self.llm_config = get_llm_config()
        
        self.importance_analyzer = ImportanceAnalyzer()
        self.validator = CompressionValidator()
        self.history = CompressionHistory()
    
    async def compress_with_validation(self, data: Union[str, Dict[str, Any]]) -> CompressionResult:
        """검증과 함께 압축."""
        if isinstance(data, dict):
            text = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            text = str(data)
        
        original_size = len(text)
        
        # 중요도 분석
        important_info = await self.importance_analyzer.analyze_importance(text)
        
        # 3단계 압축
        compression_results = []
        
        # Level 1: 불필요한 정보 제거 (50% 압축)
        intermediate = await self._remove_redundant(text, important_info)
        intermediate_size = len(intermediate)
        compression_results.append(CompressionResult(
            level=CompressionLevel.INTERMEDIATE,
            data=intermediate,
            original_size=original_size,
            compressed_size=intermediate_size,
            compression_ratio=intermediate_size / original_size,
            important_info_preserved=[info.content for info in important_info if info.content in intermediate],
            validation_score=1.0,
            timestamp=datetime.now()
        ))
        
        # Level 2: 핵심 정보 추출 (70% 압축)
        compressed = await self._extract_key_info(intermediate, important_info)
        compressed_size = len(compressed)
        compression_results.append(CompressionResult(
            level=CompressionLevel.COMPRESSED,
            data=compressed,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / original_size,
            important_info_preserved=[info.content for info in important_info if info.content in compressed],
            validation_score=1.0,
            timestamp=datetime.now()
        ))
        
        # Level 3: 최종 압축 및 검증
        final = await self._final_compression(compressed, important_info)
        final_size = len(final)
        
        # 검증
        is_valid, validation_score, issues = await self.validator.validate_compression(
            text, final, important_info
        )
        
        final_result = CompressionResult(
            level=CompressionLevel.FINAL,
            data=final,
            original_size=original_size,
            compressed_size=final_size,
            compression_ratio=final_size / original_size,
            important_info_preserved=[info.content for info in important_info if info.content in final],
            validation_score=validation_score,
            timestamp=datetime.now(),
            metadata={
                "validation_issues": issues,
                "is_valid": is_valid,
                "compression_stages": len(compression_results)
            }
        )
        
        # 압축률이 너무 높으면 중간 단계로 되돌리기
        if final_result.compression_ratio < self.config.min_compression_ratio:
            logger.warning(f"Compression too aggressive: {final_result.compression_ratio:.2%}, using intermediate")
            final_result = compression_results[0]  # 중간 단계 사용
        
        # 히스토리에 추가
        self.history.add_compression(final_result)
        
        return final_result
    
    async def _remove_redundant(self, text: str, important_info: List[ImportantInfo]) -> str:
        """불필요한 정보 제거."""
        # 중요하지 않은 문장들 제거
        sentences = text.split('. ')
        important_sentences = [info.content for info in important_info]
        
        filtered_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # 중요 정보이거나 짧은 문장은 보존
            if (sentence.strip() in important_sentences or 
                len(sentence.strip()) < 50 or
                any(keyword in sentence.lower() for keyword in ["결론", "요약", "핵심"])):
                filtered_sentences.append(sentence.strip())
        
        return '. '.join(filtered_sentences)
    
    async def _extract_key_info(self, text: str, important_info: List[ImportantInfo]) -> str:
        """핵심 정보 추출."""
        # 중요도 순으로 정렬된 정보만 추출
        important_sentences = [info.content for info in important_info[:10]]  # 상위 10개만
        
        # 원본 텍스트에서 해당 문장들 찾기
        result_sentences = []
        for sentence in important_sentences:
            if sentence in text:
                result_sentences.append(sentence)
        
        return '. '.join(result_sentences)
    
    async def _final_compression(self, text: str, important_info: List[ImportantInfo]) -> str:
        """최종 압축."""
        # 문장 단위로 압축
        sentences = text.split('. ')
        compressed_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # 문장 압축 (실제로는 LLM 사용)
            compressed_sentence = await self._compress_sentence(sentence)
            compressed_sentences.append(compressed_sentence)
        
        return '. '.join(compressed_sentences)
    
    async def _compress_sentence(self, sentence: str) -> str:
        """문장 압축."""
        # 실제 구현에서는 LLM을 사용하여 문장 압축
        # 여기서는 간단한 키워드 추출
        words = sentence.split()
        if len(words) <= 5:
            return sentence
        
        # 핵심 단어들만 추출
        important_words = []
        for word in words:
            if len(word) > 3 and not word.lower() in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']:
                important_words.append(word)
        
        if important_words:
            return ' '.join(important_words[:5])  # 최대 5개 단어
        else:
            return sentence[:50] + "..." if len(sentence) > 50 else sentence
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """압축 통계 반환."""
        if not self.history.history:
            return {"total_compressions": 0}
        
        total_compressions = len(self.history.history)
        avg_compression_ratio = sum(r.compression_ratio for r in self.history.history) / total_compressions
        avg_validation_score = sum(r.validation_score for r in self.history.history) / total_compressions
        
        return {
            "total_compressions": total_compressions,
            "avg_compression_ratio": avg_compression_ratio,
            "avg_validation_score": avg_validation_score,
            "recent_compressions": [
                {
                    "level": r.level.value,
                    "compression_ratio": r.compression_ratio,
                    "validation_score": r.validation_score,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.history.get_compression_history(5)
            ]
        }


# Global compressor instance (lazy initialization)
_compressor = None

def get_compressor() -> 'HierarchicalCompressor':
    """Get or initialize global compressor."""
    global _compressor
    if _compressor is None:
        _compressor = HierarchicalCompressor()
    return _compressor


async def compress_data(data: Union[str, Dict[str, Any]]) -> CompressionResult:
    """데이터 압축."""
    compressor = get_compressor()
    return await execute_with_reliability(
        compressor.compress_with_validation,
        data,
        component_name="hierarchical_compression",
        save_state=True
    )


def get_compression_stats() -> Dict[str, Any]:
    """압축 통계 반환."""
    compressor = get_compressor()
    return compressor.get_compression_stats()


async def restore_from_history(target_ratio: float = 0.5) -> Optional[CompressionResult]:
    """히스토리에서 복원."""
    compressor = get_compressor()
    return compressor.history.get_best_compression(target_ratio)
