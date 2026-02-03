"""
Context Cache

자주 사용되는 컨텍스트 조합 캐싱, 유사 쿼리 감지 및 캐시 활용,
캐시 무효화 전략, 메모리 효율적 캐시 관리.
"""

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class CachedContext:
    """캐시된 컨텍스트."""
    cache_key: str
    context_chunks: List[Dict[str, Any]]
    token_allocation: Dict[str, int]
    query_signature: str
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    hit_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextCache:
    """
    컨텍스트 캐시 시스템.
    
    자주 사용되는 컨텍스트 조합을 캐싱하여
    유사한 쿼리에서 재사용합니다.
    """
    
    def __init__(self, max_size: int = 100, ttl_hours: int = 24):
        """
        초기화.
        
        Args:
            max_size: 최대 캐시 크기
            ttl_hours: 캐시 TTL (시간)
        """
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.cache: OrderedDict[str, CachedContext] = OrderedDict()
        self.query_signatures: Dict[str, str] = {}  # query -> signature
        
        logger.info(f"ContextCache initialized: max_size={max_size}, ttl={ttl_hours}h")
    
    def generate_cache_key(
        self,
        context_types: List[str],
        query: str,
        token_allocation: Dict[str, int]
    ) -> str:
        """
        캐시 키 생성.
        
        Args:
            context_types: 컨텍스트 타입 목록
            query: 사용자 쿼리
            token_allocation: 토큰 할당
            
        Returns:
            캐시 키
        """
        # 쿼리 시그니처 생성 (유사 쿼리 감지용)
        query_sig = self._generate_query_signature(query)
        self.query_signatures[query] = query_sig
        
        # 캐시 키 생성
        key_data = {
            "types": sorted(context_types),
            "query_sig": query_sig,
            "allocation": sorted(token_allocation.items())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.md5(key_str.encode()).hexdigest()
        
        return cache_key
    
    def get(
        self,
        cache_key: str,
        query: Optional[str] = None
    ) -> Optional[CachedContext]:
        """
        캐시에서 조회.
        
        Args:
            cache_key: 캐시 키
            query: 쿼리 (유사도 검사용)
            
        Returns:
            캐시된 컨텍스트 또는 None
        """
        try:
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                
                # TTL 확인
                if datetime.now() - cached.created_at > timedelta(hours=self.ttl_hours):
                    logger.debug(f"Cache expired: {cache_key}")
                    del self.cache[cache_key]
                    return None
                
                # 접근 시간 및 횟수 업데이트
                cached.last_accessed = datetime.now()
                cached.access_count += 1
                
                # LRU: 가장 최근 사용으로 이동
                self.cache.move_to_end(cache_key)
                
                logger.debug(f"Cache hit: {cache_key} (access_count={cached.access_count})")
                return cached
            
            # 유사 쿼리 검색 (쿼리가 제공된 경우)
            if query:
                similar = self._find_similar_context(query, cache_key)
                if similar:
                    logger.debug(f"Found similar cached context for query")
                    return similar
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get from cache: {e}")
            return None
    
    def put(
        self,
        cache_key: str,
        context_chunks: List[Dict[str, Any]],
        token_allocation: Dict[str, int],
        query: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        캐시에 저장.
        
        Args:
            cache_key: 캐시 키
            context_chunks: 컨텍스트 청크 목록
            token_allocation: 토큰 할당
            query: 쿼리
            metadata: 메타데이터
            
        Returns:
            성공 여부
        """
        try:
            # 캐시 크기 제한
            if len(self.cache) >= self.max_size:
                # LRU: 가장 오래된 항목 제거
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                logger.debug(f"Cache evicted: {oldest_key} (max_size reached)")
            
            # 새 캐시 항목 생성
            query_sig = self._generate_query_signature(query)
            cached = CachedContext(
                cache_key=cache_key,
                context_chunks=context_chunks,
                token_allocation=token_allocation,
                query_signature=query_sig,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                metadata=metadata or {}
            )
            
            self.cache[cache_key] = cached
            self.cache.move_to_end(cache_key)  # LRU
            
            logger.debug(f"Context cached: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to put into cache: {e}")
            return False
    
    def invalidate(self, cache_key: Optional[str] = None, pattern: Optional[str] = None):
        """
        캐시 무효화.
        
        Args:
            cache_key: 특정 키 무효화 (None이면 패턴 사용)
            pattern: 무효화할 키 패턴 (정규식)
        """
        try:
            if cache_key:
                if cache_key in self.cache:
                    del self.cache[cache_key]
                    logger.info(f"Cache invalidated: {cache_key}")
            elif pattern:
                import re
                pattern_re = re.compile(pattern)
                keys_to_remove = [
                    key for key in self.cache.keys()
                    if pattern_re.search(key)
                ]
                for key in keys_to_remove:
                    del self.cache[key]
                logger.info(f"Cache invalidated: {len(keys_to_remove)} entries matching pattern {pattern}")
            else:
                # 전체 캐시 클리어
                self.cache.clear()
                logger.info("Cache cleared")
                
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
    
    def cleanup_expired(self) -> int:
        """
        만료된 캐시 정리.
        
        Returns:
            정리된 항목 수
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.ttl_hours)
            expired_keys = [
                key for key, cached in self.cache.items()
                if cached.created_at < cutoff_time
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계."""
        total_accesses = sum(cached.access_count for cached in self.cache.values())
        avg_access = total_accesses / len(self.cache) if self.cache else 0.0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "total_accesses": total_accesses,
            "average_accesses": avg_access,
            "oldest_entry": min(
                (cached.created_at for cached in self.cache.values()),
                default=None
            ).isoformat() if self.cache else None,
            "newest_entry": max(
                (cached.created_at for cached in self.cache.values()),
                default=None
            ).isoformat() if self.cache else None
        }
    
    def _generate_query_signature(self, query: str) -> str:
        """쿼리 시그니처 생성 (유사 쿼리 감지용)."""
        # 간단한 해싱 (실제로는 더 정교한 유사도 계산 가능)
        normalized = query.lower().strip()
        # 주요 키워드 추출 (간단한 버전)
        words = normalized.split()[:10]  # 상위 10개 단어
        sig_str = " ".join(sorted(set(words)))
        return hashlib.md5(sig_str.encode()).hexdigest()[:16]
    
    def _find_similar_context(self, query: str, exclude_key: str) -> Optional[CachedContext]:
        """유사한 컨텍스트 찾기."""
        try:
            query_sig = self._generate_query_signature(query)
            
            # 시그니처가 같은 캐시 항목 찾기
            for key, cached in self.cache.items():
                if key != exclude_key and cached.query_signature == query_sig:
                    # TTL 확인
                    if datetime.now() - cached.created_at <= timedelta(hours=self.ttl_hours):
                        cached.last_accessed = datetime.now()
                        cached.access_count += 1
                        self.cache.move_to_end(key)
                        return cached
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find similar context: {e}")
            return None


# 전역 인스턴스
_context_cache: Optional[ContextCache] = None


def get_context_cache() -> ContextCache:
    """전역 컨텍스트 캐시 인스턴스 반환."""
    global _context_cache
    if _context_cache is None:
        import os
        max_size = int(os.getenv("CONTEXT_CACHE_MAX_SIZE", "100"))
        ttl_hours = int(os.getenv("CONTEXT_CACHE_TTL_HOURS", "24"))
        _context_cache = ContextCache(max_size=max_size, ttl_hours=ttl_hours)
    return _context_cache

