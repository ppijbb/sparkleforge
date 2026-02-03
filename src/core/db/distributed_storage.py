"""
Distributed Storage

분산 저장소 지원: Redis 캐시 레이어 및 분산 락.
"""

import logging
from typing import Any, Dict, List, Optional
import json

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None

from src.core.db.database_driver import DatabaseDriver, Transaction

logger = logging.getLogger(__name__)


class DistributedStorage:
    """
    분산 저장소.
    
    PostgreSQL + Redis 캐시 레이어를 제공합니다.
    """
    
    def __init__(
        self,
        postgres_driver: DatabaseDriver,
        redis_url: str = "redis://localhost:6379",
        cache_ttl: int = 3600,
        enable_cache: bool = True
    ):
        """
        초기화.
        
        Args:
            postgres_driver: PostgreSQL 드라이버
            redis_url: Redis 연결 URL
            cache_ttl: 캐시 TTL (초)
            enable_cache: 캐시 활성화 여부
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Install with: pip install redis")
            enable_cache = False
        
        self.postgres = postgres_driver
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        self.enable_cache = enable_cache and REDIS_AVAILABLE
        
        self._redis: Optional[aioredis.Redis] = None
        
        if self.enable_cache:
            logger.info(f"DistributedStorage initialized with Redis cache: {redis_url}")
        else:
            logger.info("DistributedStorage initialized without Redis cache")
    
    async def connect(self) -> None:
        """Redis 연결."""
        if self.enable_cache and self._redis is None:
            try:
                self._redis = await aioredis.from_url(
                    self.redis_url,
                    decode_responses=True
                )
                await self._redis.ping()
                logger.info("Redis connected")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.enable_cache = False
    
    async def disconnect(self) -> None:
        """Redis 연결 종료."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis disconnected")
    
    async def get_session(
        self,
        session_id: str,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        세션 조회 (캐시 우선).
        
        Args:
            session_id: 세션 ID
            use_cache: 캐시 사용 여부
            
        Returns:
            세션 데이터 또는 None
        """
        # 1. Redis 캐시 확인
        if self.enable_cache and use_cache:
            cached = await self._get_from_cache(f"session:{session_id}")
            if cached:
                logger.debug(f"Session cache hit: {session_id}")
                return cached
        
        # 2. PostgreSQL에서 로드
        query = "SELECT * FROM sessions WHERE session_id = :session_id"
        session = await self.postgres.fetch_one(query, {"session_id": session_id})
        
        if session:
            # 3. Redis에 캐시
            if self.enable_cache and use_cache:
                await self._set_cache(f"session:{session_id}", session)
        
        return session
    
    async def save_session(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        tx: Optional[Transaction] = None
    ) -> bool:
        """
        세션 저장 (캐시 무효화).
        
        Args:
            session_id: 세션 ID
            session_data: 세션 데이터
            tx: 트랜잭션
            
        Returns:
            성공 여부
        """
        # PostgreSQL에 저장
        if tx:
            query = """
                INSERT INTO sessions 
                (session_id, user_id, created_at, last_accessed, state_version,
                 context_size, memory_size, tags, description)
                VALUES (:session_id, :user_id, :created_at, :last_accessed, :state_version,
                 :context_size, :memory_size, :tags, :description)
                ON CONFLICT (session_id) DO UPDATE SET
                    last_accessed = :last_accessed,
                    state_version = :state_version,
                    context_size = :context_size,
                    memory_size = :memory_size,
                    tags = :tags,
                    description = :description
            """
            await tx.execute(query, session_data)
        else:
            query = """
                INSERT INTO sessions 
                (session_id, user_id, created_at, last_accessed, state_version,
                 context_size, memory_size, tags, description)
                VALUES (:session_id, :user_id, :created_at, :last_accessed, :state_version,
                 :context_size, :memory_size, :tags, :description)
                ON CONFLICT (session_id) DO UPDATE SET
                    last_accessed = :last_accessed,
                    state_version = :state_version,
                    context_size = :context_size,
                    memory_size = :memory_size,
                    tags = :tags,
                    description = :description
            """
            await self.postgres.execute(query, session_data)
        
        # 캐시 무효화
        if self.enable_cache:
            await self._delete_cache(f"session:{session_id}")
        
        return True
    
    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """캐시에서 조회."""
        if not self._redis:
            return None
        
        try:
            value = await self._redis.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
        
        return None
    
    async def _set_cache(self, key: str, value: Dict[str, Any]) -> None:
        """캐시에 저장."""
        if not self._redis:
            return
        
        try:
            await self._redis.setex(
                key,
                self.cache_ttl,
                json.dumps(value, ensure_ascii=False)
            )
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
    
    async def _delete_cache(self, key: str) -> None:
        """캐시 삭제."""
        if not self._redis:
            return
        
        try:
            await self._redis.delete(key)
        except Exception as e:
            logger.debug(f"Cache delete failed: {e}")
    
    async def acquire_lock(
        self,
        lock_key: str,
        timeout: int = 10,
        expire: int = 30
    ) -> bool:
        """
        분산 락 획득.
        
        Args:
            lock_key: 락 키
            timeout: 대기 시간 (초)
            expire: 락 만료 시간 (초)
            
        Returns:
            락 획득 성공 여부
        """
        if not self.enable_cache:
            return True  # 캐시 없으면 항상 성공
        
        try:
            # Redis SET NX EX로 분산 락 구현
            result = await self._redis.set(
                f"lock:{lock_key}",
                "1",
                nx=True,
                ex=expire
            )
            return result is True
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return False
    
    async def release_lock(self, lock_key: str) -> None:
        """분산 락 해제."""
        if not self.enable_cache:
            return
        
        try:
            await self._redis.delete(f"lock:{lock_key}")
        except Exception as e:
            logger.debug(f"Failed to release lock: {e}")

