"""
Shared Memory System for Multi-Agent Orchestration

Existing local_researcher uses ChromaDB and LangGraph,
so this implementation integrates both without additional dependencies.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from collections import OrderedDict
import threading
import weakref

logger = logging.getLogger(__name__)


class MemoryScope:
    """Memory scope enumeration."""
    GLOBAL = "global"
    SESSION = "session"
    AGENT = "agent"


class LRUCache:
    """Thread-safe LRU Cache with size limits and TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = None):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items in cache
            ttl_seconds: Time-to-live for items (None for no expiration)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, moving it to end (most recently used)."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check TTL
            if self._is_expired(key):
                del self.cache[key]
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value["value"]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache, evicting oldest if necessary."""
        with self.lock:
            current_time = datetime.now().timestamp()
            
            # If key exists, update and move to end
            if key in self.cache:
                self.cache.pop(key)
            
            # Add new item
            self.cache[key] = {
                "value": value,
                "timestamp": current_time
            }
            
            # Evict if over capacity
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }
    
    def _is_expired(self, key: str) -> bool:
        """Check if item has expired."""
        if self.ttl_seconds is None:
            return False
        
        item = self.cache[key]
        current_time = datetime.now().timestamp()
        return (current_time - item["timestamp"]) > self.ttl_seconds
    
    def cleanup_expired(self) -> int:
        """Remove expired items and return count removed."""
        if self.ttl_seconds is None:
            return 0
        
        with self.lock:
            current_time = datetime.now().timestamp()
            expired_keys = []
            
            for key, item in self.cache.items():
                if (current_time - item["timestamp"]) > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)


class SharedMemory:
    """
    Multi-Agent Shared Memory System with LRU eviction
    
    ChromaDB를 벡터 저장소로 사용하되, 
    간단한 파일 기반 메모리 시스템을 제공하여 추가 의존성 없이 작동
    """
    
    def __init__(self, storage_path: str = "./storage/memori", enable_chromadb: bool = True,
                 max_memory_size: int = 1000, memory_ttl: Optional[int] = None):
        """
        Initialize shared memory system.
        
        Args:
            storage_path: Path for persistent storage
            enable_chromadb: Enable ChromaDB vector search
            max_memory_size: Maximum items in memory cache
            memory_ttl: Time-to-live for memory items (seconds)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.enable_chromadb = enable_chromadb
        
        # LRU caches for performance and memory management
        self.global_cache = LRUCache(max_size=max_memory_size, ttl_seconds=memory_ttl)
        self.session_caches: Dict[str, LRUCache] = {}
        self.agent_caches: Dict[str, LRUCache] = {}
        
        # In-memory state (legacy compatibility)
        self.memories: Dict[str, Any] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Periodic cleanup
        self.last_cleanup = datetime.now().timestamp()
        self.cleanup_interval = 300  # 5 minutes
        
        # Try to initialize ChromaDB if enabled
        self.chroma_client = None
        if self.enable_chromadb:
            try:
                import chromadb
                self.chroma_client = chromadb.Client()
                logger.info("✅ ChromaDB initialized for vector search")
            except ImportError:
                logger.warning("⚠️ ChromaDB not available - using file-based storage only")
                logger.info("   Install with: pip install chromadb")
                self.enable_chromadb = False
        
        logger.info(f"SharedMemory initialized at {self.storage_path} (LRU cache: {max_memory_size} items)")
    
    def write(self, 
              key: str, 
              value: Any, 
              scope: str = MemoryScope.GLOBAL,
              session_id: Optional[str] = None,
              agent_id: Optional[str] = None) -> bool:
        """
        Write to shared memory with LRU cache optimization.
        
        Args:
            key: Memory key
            value: Memory value
            scope: Memory scope (global, session, agent)
            session_id: Session ID
            agent_id: Agent ID
            
        Returns:
            Success status
        """
        try:
            # Periodic cleanup
            self._periodic_cleanup()
            
            # Create memory entry
            memory_entry = {
                "key": key,
                "value": value,
                "scope": scope,
                "session_id": session_id,
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Use LRU cache based on scope
            if scope == MemoryScope.GLOBAL:
                self.global_cache.put(key, memory_entry)
                # Legacy compatibility
                self.memories[key] = memory_entry
                
            elif scope == MemoryScope.SESSION:
                if session_id:
                    if session_id not in self.session_caches:
                        self.session_caches[session_id] = LRUCache(max_size=500, ttl_seconds=3600)
                    self.session_caches[session_id].put(key, memory_entry)
                    
                    # Legacy compatibility
                    if session_id not in self.sessions:
                        self.sessions[session_id] = {}
                    self.sessions[session_id][key] = memory_entry
                    
            elif scope == MemoryScope.AGENT:
                if agent_id and session_id:
                    cache_key = f"{session_id}:{agent_id}"
                    if cache_key not in self.agent_caches:
                        self.agent_caches[cache_key] = LRUCache(max_size=200, ttl_seconds=1800)
                    self.agent_caches[cache_key].put(key, memory_entry)
                    
                    # Legacy compatibility
                    if session_id not in self.sessions:
                        self.sessions[session_id] = {}
                    if "agents" not in self.sessions[session_id]:
                        self.sessions[session_id]["agents"] = {}
                    if agent_id not in self.sessions[session_id]["agents"]:
                        self.sessions[session_id]["agents"][agent_id] = {}
                    self.sessions[session_id]["agents"][agent_id][key] = memory_entry
            
            # Persist to file
            self._persist_memory(memory_entry)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write memory: {e}")
            return False
    
    def _periodic_cleanup(self) -> None:
        """Perform periodic cleanup of expired items."""
        current_time = datetime.now().timestamp()
        if current_time - self.last_cleanup > self.cleanup_interval:
            total_cleaned = 0
            total_cleaned += self.global_cache.cleanup_expired()
            
            for cache in self.session_caches.values():
                total_cleaned += cache.cleanup_expired()
                
            for cache in self.agent_caches.values():
                total_cleaned += cache.cleanup_expired()
            
            if total_cleaned > 0:
                logger.debug(f"Cleaned up {total_cleaned} expired memory items")
            
            self.last_cleanup = current_time
    
    def read(self, 
             key: str, 
             scope: str = MemoryScope.GLOBAL,
             session_id: Optional[str] = None,
             agent_id: Optional[str] = None) -> Optional[Any]:
        """
        Read from shared memory using LRU cache optimization.
        
        Args:
            key: Memory key
            scope: Memory scope
            session_id: Session ID
            agent_id: Agent ID
            
        Returns:
            Memory value or None
        """
        try:
            # Periodic cleanup
            self._periodic_cleanup()
            
            # Use LRU cache based on scope
            if scope == MemoryScope.GLOBAL:
                cached_entry = self.global_cache.get(key)
                if cached_entry:
                    return cached_entry["value"]
                # Fallback to legacy
                if key in self.memories:
                    return self.memories[key]["value"]
                    
            elif scope == MemoryScope.SESSION:
                if session_id and session_id in self.session_caches:
                    cached_entry = self.session_caches[session_id].get(key)
                    if cached_entry:
                        return cached_entry["value"]
                # Fallback to legacy
                if session_id and session_id in self.sessions:
                    if key in self.sessions[session_id]:
                        return self.sessions[session_id][key]["value"]
                        
            elif scope == MemoryScope.AGENT:
                if agent_id and session_id:
                    cache_key = f"{session_id}:{agent_id}"
                    if cache_key in self.agent_caches:
                        cached_entry = self.agent_caches[cache_key].get(key)
                        if cached_entry:
                            return cached_entry["value"]
                    # Fallback to legacy
                    if session_id in self.sessions:
                        if "agents" in self.sessions[session_id]:
                            if agent_id in self.sessions[session_id]["agents"]:
                                if key in self.sessions[session_id]["agents"][agent_id]:
                                    return self.sessions[session_id]["agents"][agent_id][key]["value"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to read memory: {e}")
            return None
    
    def search(self, 
               query: str, 
               limit: int = 10,
               scope: str = MemoryScope.GLOBAL,
               session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search memories by query.
        
        Args:
            query: Search query
            limit: Result limit
            scope: Memory scope
            session_id: Optional session ID to filter results (prevents cross-session contamination)
            
        Returns:
            List of matching memories
        """
        results = []
        
        try:
            # Simple keyword matching for now
            # If ChromaDB is available, use vector search
            if self.enable_chromadb and self.chroma_client:
                # ChromaDB vector search would be implemented here
                # For now, fall through to simple search
                pass
            
            # Simple keyword search
            query_lower = query.lower()
            search_spaces = []
            
            if scope == MemoryScope.GLOBAL:
                # Global scope: search all memories (but filter by session_id if provided)
                if session_id:
                    # If session_id provided, only search within that session even for GLOBAL scope
                    if session_id in self.sessions:
                        search_spaces.append(self.sessions[session_id])
                else:
                    search_spaces.append(self.memories)
            elif scope == MemoryScope.SESSION:
                # Session scope: only search within specified session or all sessions
                if session_id:
                    # Only search within the specified session
                    if session_id in self.sessions:
                        search_spaces.append(self.sessions[session_id])
                else:
                    # Search all sessions (legacy behavior, but should be avoided)
                    logger.warning("Searching across all sessions without session_id filter - this may cause cross-session contamination")
                    for session_data in self.sessions.values():
                        if isinstance(session_data, dict):
                            search_spaces.append(session_data)
            
            for space in search_spaces:
                for key, memory_entry in space.items():
                    if isinstance(memory_entry, dict):
                        # Skip agent-specific entries in session search
                        if key == "agents":
                            continue
                        
                        # Verify session_id matches if provided
                        if session_id and memory_entry.get("session_id") != session_id:
                            continue
                        
                        key_str = key.lower()
                        value_str = str(memory_entry.get("value", "")).lower()
                        
                        if query_lower in key_str or query_lower in value_str:
                            results.append({
                                "key": key,
                                "value": memory_entry.get("value"),
                                "timestamp": memory_entry.get("timestamp"),
                                "session_id": memory_entry.get("session_id")
                            })
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            return []
    
    def list_session_memories(self, session_id: str) -> Dict[str, Any]:
        """List all memories for a session."""
        return self.sessions.get(session_id, {})
    
    def clear_session(self, session_id: str) -> bool:
        """Clear all memories for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "global_cache": self.global_cache.get_stats(),
            "session_caches": {
                session_id: cache.get_stats()
                for session_id, cache in self.session_caches.items()
            },
            "agent_caches": {
                cache_key: cache.get_stats()
                for cache_key, cache in self.agent_caches.items()
            },
            "total_session_caches": len(self.session_caches),
            "total_agent_caches": len(self.agent_caches)
        }
        
        # Calculate totals
        total_memory_usage = stats["global_cache"]["size"]
        total_memory_usage += sum(cache_stats["size"] for cache_stats in stats["session_caches"].values())
        total_memory_usage += sum(cache_stats["size"] for cache_stats in stats["agent_caches"].values())
        
        stats["total_cached_items"] = total_memory_usage
        return stats
    
    def clear_expired_caches(self) -> Dict[str, int]:
        """Manually clear all expired items and return counts."""
        result = {
            "global": self.global_cache.cleanup_expired(),
            "sessions": {},
            "agents": {}
        }
        
        total_cleaned = result["global"]
        
        for session_id, cache in self.session_caches.items():
            cleaned = cache.cleanup_expired()
            result["sessions"][session_id] = cleaned
            total_cleaned += cleaned
            
        for cache_key, cache in self.agent_caches.items():
            cleaned = cache.cleanup_expired()
            result["agents"][cache_key] = cleaned
            total_cleaned += cleaned
        
        result["total_cleaned"] = total_cleaned
        return result
    
    def _persist_memory(self, memory_entry: Dict[str, Any]) -> None:
        """Persist memory to disk."""
        try:
            memory_file = self.storage_path / "memories.jsonl"
            
            with open(memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(memory_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.warning(f"Failed to persist memory: {e}")


# Global shared memory instance
_shared_memory: Optional[SharedMemory] = None


def get_shared_memory() -> SharedMemory:
    """Get global shared memory instance."""
    global _shared_memory
    
    if _shared_memory is None:
        storage_path = os.getenv("MEMORI_STORAGE_PATH", "./storage/memori")
        enable_chromadb = os.getenv("ENABLE_CHROMADB", "true").lower() == "true"
        _shared_memory = SharedMemory(storage_path=storage_path, enable_chromadb=enable_chromadb)
    
    return _shared_memory


def init_shared_memory(storage_path: Optional[str] = None, enable_chromadb: Optional[bool] = None) -> SharedMemory:
    """Initialize shared memory system."""
    global _shared_memory
    
    if storage_path is None:
        storage_path = os.getenv("MEMORI_STORAGE_PATH", "./storage/memori")
    if enable_chromadb is None:
        enable_chromadb = os.getenv("ENABLE_CHROMADB", "true").lower() == "true"
    
    _shared_memory = SharedMemory(storage_path=storage_path, enable_chromadb=enable_chromadb)
    
    return _shared_memory

