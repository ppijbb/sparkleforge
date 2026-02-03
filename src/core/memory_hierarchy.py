"""
Memory Hierarchy

L1-L4 계층 구조를 통한 빠른 접근 경로 제공.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryLevel:
    """메모리 레벨."""
    level: int
    name: str
    access_time_ms: float
    capacity: int
    current_size: int = 0
    hit_count: int = 0
    miss_count: int = 0


class MemoryHierarchy:
    """
    메모리 계층 구조.
    
    L1: 현재 세션 메모리 (가장 빠름)
    L2: 사용자 프로필 메모리
    L3: 전역 공유 메모리
    L4: 장기 저장소 (디스크)
    """
    
    def __init__(self):
        """초기화."""
        self.levels = {
            1: MemoryLevel(level=1, name="L1_Session", access_time_ms=0.1, capacity=1000),
            2: MemoryLevel(level=2, name="L2_UserProfile", access_time_ms=1.0, capacity=5000),
            3: MemoryLevel(level=3, name="L3_Global", access_time_ms=5.0, capacity=50000),
            4: MemoryLevel(level=4, name="L4_Disk", access_time_ms=50.0, capacity=1000000)
        }
        
        # L1: 현재 세션 메모리 (in-memory)
        self.l1_cache: Dict[str, Any] = {}
        
        # L2: 사용자 프로필 메모리 (in-memory, 사용자별)
        self.l2_cache: Dict[str, Dict[str, Any]] = {}
        
        # L3: 전역 공유 메모리 (SharedMemory 참조)
        self.l3_memory = None
        
        # L4: 장기 저장소 (SessionStorage 참조)
        self.l4_storage = None
        
        logger.info("MemoryHierarchy initialized")
    
    def get(
        self,
        key: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[Any]:
        """
        메모리 계층에서 조회 (L1 -> L2 -> L3 -> L4 순서).
        
        Args:
            key: 메모리 키
            user_id: 사용자 ID
            session_id: 세션 ID
            
        Returns:
            메모리 값 또는 None
        """
        import time
        start_time = time.time()
        
        # L1: 현재 세션 메모리
        if session_id and session_id in self.l1_cache:
            if key in self.l1_cache[session_id]:
                self.levels[1].hit_count += 1
                elapsed = (time.time() - start_time) * 1000
                logger.debug(f"L1 hit: {key} ({elapsed:.2f}ms)")
                return self.l1_cache[session_id][key]
            self.levels[1].miss_count += 1
        
        # L2: 사용자 프로필 메모리
        if user_id and user_id in self.l2_cache:
            if key in self.l2_cache[user_id]:
                self.levels[2].hit_count += 1
                elapsed = (time.time() - start_time) * 1000
                logger.debug(f"L2 hit: {key} ({elapsed:.2f}ms)")
                
                # L1로 승격 (다음 접근을 위해)
                if session_id:
                    if session_id not in self.l1_cache:
                        self.l1_cache[session_id] = {}
                    self.l1_cache[session_id][key] = self.l2_cache[user_id][key]
                
                return self.l2_cache[user_id][key]
            self.levels[2].miss_count += 1
        
        # L3: 전역 공유 메모리
        if self.l3_memory:
            try:
                value = self.l3_memory.read(key, scope="global", session_id=session_id)
                if value is not None:
                    self.levels[3].hit_count += 1
                    elapsed = (time.time() - start_time) * 1000
                    logger.debug(f"L3 hit: {key} ({elapsed:.2f}ms)")
                    
                    # L2로 승격
                    if user_id:
                        if user_id not in self.l2_cache:
                            self.l2_cache[user_id] = {}
                        self.l2_cache[user_id][key] = value
                    
                    # L1로 승격
                    if session_id:
                        if session_id not in self.l1_cache:
                            self.l1_cache[session_id] = {}
                        self.l1_cache[session_id][key] = value
                    
                    return value
                self.levels[3].miss_count += 1
            except Exception as e:
                logger.debug(f"L3 access failed: {e}")
        
        # L4: 장기 저장소
        if self.l4_storage and session_id:
            try:
                session_data = self.l4_storage.load_session(session_id)
                if session_data:
                    memory_data = session_data.get("memory", {})
                    entries = memory_data.get("entries", [])
                    for entry in entries:
                        if entry.get("key") == key:
                            self.levels[4].hit_count += 1
                            elapsed = (time.time() - start_time) * 1000
                            logger.debug(f"L4 hit: {key} ({elapsed:.2f}ms)")
                            
                            value = entry.get("value")
                            
                            # 상위 레벨로 승격
                            if user_id:
                                if user_id not in self.l2_cache:
                                    self.l2_cache[user_id] = {}
                                self.l2_cache[user_id][key] = value
                            
                            if session_id:
                                if session_id not in self.l1_cache:
                                    self.l1_cache[session_id] = {}
                                self.l1_cache[session_id][key] = value
                            
                            return value
                    self.levels[4].miss_count += 1
            except Exception as e:
                logger.debug(f"L4 access failed: {e}")
        
        # 모든 레벨에서 미스
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Memory miss: {key} (searched all levels, {elapsed:.2f}ms)")
        return None
    
    def put(
        self,
        key: str,
        value: Any,
        level: int = 1,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """
        메모리 계층에 저장.
        
        Args:
            key: 메모리 키
            value: 메모리 값
            level: 저장할 레벨 (1-4)
            user_id: 사용자 ID
            session_id: 세션 ID
            
        Returns:
            성공 여부
        """
        try:
            # L1 저장
            if level <= 1 and session_id:
                if session_id not in self.l1_cache:
                    self.l1_cache[session_id] = {}
                self.l1_cache[session_id][key] = value
                self.levels[1].current_size += 1
            
            # L2 저장
            if level <= 2 and user_id:
                if user_id not in self.l2_cache:
                    self.l2_cache[user_id] = {}
                self.l2_cache[user_id][key] = value
                self.levels[2].current_size += 1
            
            # L3 저장
            if level <= 3 and self.l3_memory:
                self.l3_memory.write(key, value, scope="global", session_id=session_id)
                self.levels[3].current_size += 1
            
            # L4 저장 (세션 저장소)
            if level <= 4 and self.l4_storage and session_id:
                # 세션 저장 시 자동으로 L4에 저장됨
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to put into memory hierarchy: {e}")
            return False
    
    def set_l3_memory(self, shared_memory):
        """L3 메모리 설정."""
        self.l3_memory = shared_memory
    
    def set_l4_storage(self, session_storage):
        """L4 저장소 설정."""
        self.l4_storage = session_storage
    
    def get_hierarchy_stats(self) -> Dict[str, Any]:
        """계층 구조 통계."""
        stats = {}
        
        for level_num, level in self.levels.items():
            total_accesses = level.hit_count + level.miss_count
            hit_rate = level.hit_count / total_accesses if total_accesses > 0 else 0.0
            
            stats[level.name] = {
                "level": level.level,
                "capacity": level.capacity,
                "current_size": level.current_size,
                "hit_count": level.hit_count,
                "miss_count": level.miss_count,
                "hit_rate": hit_rate,
                "access_time_ms": level.access_time_ms
            }
        
        return stats
    
    def cleanup_l1(self, session_id: Optional[str] = None):
        """L1 캐시 정리."""
        if session_id:
            if session_id in self.l1_cache:
                del self.l1_cache[session_id]
                self.levels[1].current_size = sum(len(cache) for cache in self.l1_cache.values())
        else:
            self.l1_cache.clear()
            self.levels[1].current_size = 0
        
        logger.debug(f"L1 cache cleaned for session: {session_id or 'all'}")


# 전역 인스턴스
_memory_hierarchy: Optional[MemoryHierarchy] = None


def get_memory_hierarchy() -> MemoryHierarchy:
    """전역 메모리 계층 인스턴스 반환."""
    global _memory_hierarchy
    if _memory_hierarchy is None:
        _memory_hierarchy = MemoryHierarchy()
        # L3, L4 자동 연결
        try:
            from src.core.shared_memory import get_shared_memory
            from src.core.session_storage import get_session_storage
            _memory_hierarchy.set_l3_memory(get_shared_memory())
            _memory_hierarchy.set_l4_storage(get_session_storage())
        except Exception as e:
            logger.debug(f"Failed to auto-connect L3/L4: {e}")
    return _memory_hierarchy

