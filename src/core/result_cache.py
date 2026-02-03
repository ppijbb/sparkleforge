#!/usr/bin/env python3
"""
Result Cache System for Performance Improvement

TTL-based caching for search results, API responses, and task results.
Includes similar task detection and cache invalidation strategies.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import OrderedDict
import difflib

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class ResultCache:
    """Result cache with TTL, similarity detection, and statistics."""
    
    def __init__(
        self,
        default_ttl: int = 3600,  # 1 hour default TTL
        max_size: int = 1000,  # Maximum cache entries
        similarity_threshold: float = 0.8  # Similarity threshold for fuzzy matching
    ):
        """
        Initialize result cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
            max_size: Maximum number of cache entries
            similarity_threshold: Similarity threshold (0.0-1.0) for fuzzy matching
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        
        # LRU cache using OrderedDict
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'similarity_matches': 0,
            'total_requests': 0
        }
        
        logger.info(f"ResultCache initialized: default_ttl={default_ttl}s, max_size={max_size}")
    
    def _generate_cache_key(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        task_id: Optional[str] = None
    ) -> str:
        """
        Generate cache key from tool name and parameters.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            task_id: Optional task ID for additional uniqueness
            
        Returns:
            Cache key string
        """
        # Normalize parameters (sort keys, remove None values)
        normalized_params = {
            k: v for k, v in sorted(parameters.items())
            if v is not None
        }
        
        # Create key components
        key_parts = [tool_name]
        
        if task_id:
            key_parts.append(f"task:{task_id}")
        
        # Add normalized parameters
        key_parts.append(json.dumps(normalized_params, sort_keys=True))
        
        # Generate hash
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"{tool_name}:{key_hash}"
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate similarity between two queries.
        
        Args:
            query1: First query string
            query2: Second query string
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if not query1 or not query2:
            return 0.0
        
        # Normalize queries
        q1_lower = query1.lower().strip()
        q2_lower = query2.lower().strip()
        
        if q1_lower == q2_lower:
            return 1.0
        
        # Use SequenceMatcher for similarity
        similarity = difflib.SequenceMatcher(None, q1_lower, q2_lower).ratio()
        return similarity
    
    async def _find_similar_entry(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        current_time: datetime
    ) -> Optional[Tuple[str, CacheEntry]]:
        """
        Find similar cache entry using fuzzy matching.
        
        Args:
            tool_name: Tool name
            parameters: Tool parameters
            current_time: Current time for expiration check
            
        Returns:
            Tuple of (cache_key, cache_entry) if found, None otherwise
        """
        query = parameters.get('query', '')
        if not query:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        async with self.lock:
            for key, entry in self.cache.items():
                # Check if same tool
                if not key.startswith(f"{tool_name}:"):
                    continue
                
                # Check if expired
                if entry.expires_at < current_time:
                    continue
                
                # Get query from entry metadata
                entry_query = entry.metadata.get('query', '')
                if not entry_query:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(query, entry_query)
                
                if similarity >= self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (key, entry)
        
        if best_match:
            self.stats['similarity_matches'] += 1
            logger.debug(f"Found similar cache entry: similarity={best_similarity:.2f}")
        
        return best_match
    
    async def get(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        task_id: Optional[str] = None,
        check_similarity: bool = True
    ) -> Optional[Any]:
        """
        Get cached result.
        
        Args:
            tool_name: Tool name
            parameters: Tool parameters
            task_id: Optional task ID
            check_similarity: Whether to check for similar entries
            
        Returns:
            Cached value if found, None otherwise
        """
        async with self.lock:
            self.stats['total_requests'] += 1
        
        current_time = datetime.now()
        
        # Generate cache key
        cache_key = self._generate_cache_key(tool_name, parameters, task_id)
        
        # Check exact match
        async with self.lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check expiration
                if entry.expires_at >= current_time:
                    # Update access info
                    entry.access_count += 1
                    entry.last_accessed = current_time
                    # Move to end (LRU)
                    self.cache.move_to_end(cache_key)
                    
                    self.stats['hits'] += 1
                    logger.debug(f"Cache hit: {cache_key}")
                    return entry.value
                else:
                    # Expired, remove
                    del self.cache[cache_key]
                    logger.debug(f"Cache entry expired: {cache_key}")
        
        # Check for similar entries if enabled
        if check_similarity:
            similar_match = await self._find_similar_entry(tool_name, parameters, current_time)
            if similar_match:
                key, entry = similar_match
                async with self.lock:
                    # Update access info
                    entry.access_count += 1
                    entry.last_accessed = current_time
                    self.cache.move_to_end(key)
                    
                    self.stats['hits'] += 1
                    logger.debug(f"Cache similarity hit: {key}, similarity={self._calculate_similarity(parameters.get('query', ''), entry.metadata.get('query', '')):.2f}")
                    return entry.value
        
        self.stats['misses'] += 1
        logger.debug(f"Cache miss: {cache_key}")
        return None
    
    async def set(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        value: Any,
        ttl: Optional[int] = None,
        task_id: Optional[str] = None
    ) -> str:
        """
        Store result in cache.
        
        Args:
            tool_name: Tool name
            parameters: Tool parameters
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            task_id: Optional task ID
            
        Returns:
            Cache key
        """
        current_time = datetime.now()
        ttl = ttl or self.default_ttl
        expires_at = current_time + timedelta(seconds=ttl)
        
        cache_key = self._generate_cache_key(tool_name, parameters, task_id)
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            value=value,
            created_at=current_time,
            expires_at=expires_at,
            access_count=1,
            last_accessed=current_time,
            metadata={
                'tool_name': tool_name,
                'query': parameters.get('query', ''),
                'parameters': parameters
            }
        )
        
        async with self.lock:
            # Remove if exists (update)
            if cache_key in self.cache:
                del self.cache[cache_key]
            
            # Evict if cache is full (LRU)
            while len(self.cache) >= self.max_size:
                # Remove oldest (least recently used)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
                logger.debug(f"Cache eviction: {oldest_key}")
            
            # Add new entry
            self.cache[cache_key] = entry
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
        
        logger.debug(f"Cache set: {cache_key}, ttl={ttl}s")
        return cache_key
    
    async def invalidate(
        self,
        tool_name: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            tool_name: Invalidate entries for specific tool (None = all)
            pattern: Pattern to match (None = all)
            
        Returns:
            Number of entries invalidated
        """
        invalidated = 0
        
        async with self.lock:
            keys_to_remove = []
            
            for key, entry in self.cache.items():
                should_remove = False
                
                if tool_name and not key.startswith(f"{tool_name}:"):
                    continue
                
                if pattern:
                    if pattern.lower() in key.lower():
                        should_remove = True
                else:
                    should_remove = True
                
                if should_remove:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
                invalidated += 1
        
        logger.info(f"Cache invalidated: {invalidated} entries")
        return invalidated
    
    async def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        async with self.lock:
            count = len(self.cache)
            self.cache.clear()
        
        logger.info(f"Cache cleared: {count} entries")
        return count
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        current_time = datetime.now()
        removed = 0
        
        async with self.lock:
            keys_to_remove = []
            
            for key, entry in self.cache.items():
                if entry.expires_at < current_time:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
                removed += 1
        
        if removed > 0:
            logger.debug(f"Cache cleanup: removed {removed} expired entries")
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        async def _get_stats():
            async with self.lock:
                hit_rate = (
                    self.stats['hits'] / self.stats['total_requests']
                    if self.stats['total_requests'] > 0 else 0.0
                )
                
                return {
                    'hits': self.stats['hits'],
                    'misses': self.stats['misses'],
                    'total_requests': self.stats['total_requests'],
                    'hit_rate': hit_rate,
                    'evictions': self.stats['evictions'],
                    'similarity_matches': self.stats['similarity_matches'],
                    'current_size': len(self.cache),
                    'max_size': self.max_size,
                    'default_ttl': self.default_ttl,
                    'similarity_threshold': self.similarity_threshold
                }
        
        # Run in event loop if available
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _get_stats())
                    # timeout 설정으로 무한 대기 방지
                    return future.result(timeout=30)  # 최대 30초
            else:
                return loop.run_until_complete(_get_stats())
        except RuntimeError:
            return asyncio.run(_get_stats())


# Global cache instance
_cache_instance: Optional[ResultCache] = None


def get_result_cache() -> ResultCache:
    """Get or create global result cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResultCache()
    return _cache_instance

