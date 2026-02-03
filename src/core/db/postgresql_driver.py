"""
PostgreSQL Driver Implementation

PostgreSQL 데이터베이스 드라이버 구현.
asyncpg를 사용한 비동기 연결 풀링 및 트랜잭션 지원.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from contextlib import asynccontextmanager

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None

from src.core.db.database_driver import (
    DatabaseDriver,
    Transaction,
    TransactionIsolation,
)

if TYPE_CHECKING:
    # 타입 체크 시에만 asyncpg.Connection 사용
    if asyncpg is not None:
        ConnectionType = asyncpg.Connection
    else:
        ConnectionType = Any
else:
    # 런타임에는 Any로 처리
    ConnectionType = Any

logger = logging.getLogger(__name__)


class PostgreSQLTransaction(Transaction):
    """PostgreSQL 트랜잭션."""
    
    def __init__(
        self,
        driver: "PostgreSQLDriver",
        connection: ConnectionType,
        isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED,
        parent: Optional[Transaction] = None
    ):
        super().__init__(driver, isolation_level, parent)
        self.connection = connection
        self.savepoint_name: Optional[str] = None
    
    async def commit(self) -> None:
        """트랜잭션 커밋."""
        if self.rolled_back:
            raise RuntimeError("Cannot commit a rolled back transaction")
        if self.committed:
            return
        
        if self.parent:
            # 중첩 트랜잭션: Savepoint 해제
            if self.savepoint_name:
                await self.connection.execute(f"RELEASE SAVEPOINT {self.savepoint_name}")
                logger.debug(f"Savepoint released: {self.savepoint_name}")
        else:
            # 루트 트랜잭션: 커밋
            await self.connection.commit()
        
        self.committed = True
        logger.debug(f"PostgreSQL transaction committed (level: {self._nested_level})")
    
    async def rollback(self) -> None:
        """트랜잭션 롤백."""
        if self.committed:
            raise RuntimeError("Cannot rollback a committed transaction")
        if self.rolled_back:
            return
        
        if self.parent:
            # 중첩 트랜잭션: Savepoint로 롤백
            if self.savepoint_name:
                await self.connection.execute(f"ROLLBACK TO SAVEPOINT {self.savepoint_name}")
                logger.debug(f"Rolled back to savepoint: {self.savepoint_name}")
        else:
            # 루트 트랜잭션: 롤백
            await self.connection.rollback()
        
        self.rolled_back = True
        logger.debug(f"PostgreSQL transaction rolled back (level: {self._nested_level})")


class PostgreSQLDriver(DatabaseDriver):
    """
    PostgreSQL 데이터베이스 드라이버.
    
    asyncpg를 사용한 비동기 연결 풀링 및 트랜잭션 지원.
    """
    
    def __init__(
        self,
        connection_string: str,
        pool_size: int = 20,
        max_overflow: int = 10,
        isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED,
        min_size: int = 10,
        max_size: int = 30
    ):
        """
        초기화.
        
        Args:
            connection_string: PostgreSQL 연결 문자열 (postgresql://user:pass@host:port/dbname)
            pool_size: 연결 풀 크기
            max_overflow: 최대 오버플로우 연결 수
            isolation_level: 기본 격리 수준
            min_size: 최소 연결 풀 크기
            max_size: 최대 연결 풀 크기
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is required for PostgreSQL driver. "
                "Install it with: pip install asyncpg"
            )
        
        super().__init__(connection_string, pool_size, max_overflow, isolation_level)
        self.connection_string = connection_string
        self.min_size = min_size
        self.max_size = max_size
        self._pool: Optional[Any] = None  # asyncpg.Pool이지만 타입 체크 시에만
        self._savepoint_counter = 0
    
    async def connect(self) -> None:
        """데이터베이스 연결 풀 생성."""
        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    command_timeout=60.0
                )
                logger.info(f"PostgreSQL connection pool created: {self.min_size}-{self.max_size} connections")
            except Exception as e:
                logger.error(f"Failed to create PostgreSQL connection pool: {e}")
                raise
    
    async def disconnect(self) -> None:
        """데이터베이스 연결 풀 종료."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")
    
    async def begin(
        self,
        isolation_level: Optional[TransactionIsolation] = None
    ) -> PostgreSQLTransaction:
        """
        트랜잭션 시작.
        
        PostgreSQL은 중첩 트랜잭션을 Savepoint로 구현합니다.
        """
        await self.connect()
        
        if self._pool is None:
            raise RuntimeError("Database connection pool not established")
        
        # 연결 풀에서 연결 가져오기
        connection = await self._pool.acquire()
        
        # 현재 트랜잭션 확인 (Context에서)
        from src.core.db.database_driver import TransactionContext
        current_tx = TransactionContext.get_current()
        
        isolation = isolation_level or self.isolation_level
        
        if current_tx and isinstance(current_tx, PostgreSQLTransaction):
            # 중첩 트랜잭션: Savepoint 생성
            self._savepoint_counter += 1
            savepoint_name = f"sp_{self._savepoint_counter}"
            await connection.execute(f"SAVEPOINT {savepoint_name}")
            
            tx = PostgreSQLTransaction(
                self,
                connection,
                isolation,
                current_tx
            )
            tx.savepoint_name = savepoint_name
            logger.debug(f"PostgreSQL savepoint created: {savepoint_name}")
        else:
            # 루트 트랜잭션: BEGIN
            isolation_sql = isolation.value if isolation else "READ COMMITTED"
            await connection.execute(f"BEGIN ISOLATION LEVEL {isolation_sql}")
            
            tx = PostgreSQLTransaction(
                self,
                connection,
                isolation
            )
            logger.debug(f"PostgreSQL root transaction started (isolation: {isolation_sql})")
        
        return tx
    
    async def _commit_transaction(self, tx: PostgreSQLTransaction) -> None:
        """트랜잭션 커밋 (내부 메서드)."""
        # PostgreSQLTransaction.commit()에서 처리
        # 연결 반환
        if not tx.parent and self._pool:
            await self._pool.release(tx.connection)
    
    async def _rollback_transaction(self, tx: PostgreSQLTransaction) -> None:
        """트랜잭션 롤백 (내부 메서드)."""
        # PostgreSQLTransaction.rollback()에서 처리
        # 연결 반환
        if not tx.parent and self._pool:
            await self._pool.release(tx.connection)
    
    async def _execute_in_transaction(
        self,
        tx: PostgreSQLTransaction,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """트랜잭션 내에서 쿼리 실행."""
        if params:
            # Dict를 위치 기반 파라미터로 변환
            # PostgreSQL은 $1, $2 형식 사용
            param_values = list(params.values())
            return await tx.connection.execute(query, *param_values)
        else:
            return await tx.connection.execute(query)
    
    async def _execute_many_in_transaction(
        self,
        tx: PostgreSQLTransaction,
        query: str,
        params_list: List[Dict[str, Any]]
    ) -> Any:
        """트랜잭션 내에서 여러 쿼리 일괄 실행."""
        # Dict 리스트를 값 리스트로 변환
        param_values_list = [list(params.values()) for params in params_list]
        return await tx.connection.executemany(query, param_values_list)
    
    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """트랜잭션 없이 쿼리 실행."""
        await self.connect()
        if self._pool is None:
            raise RuntimeError("Database connection pool not established")
        
        async with self._pool.acquire() as connection:
            if params:
                param_values = list(params.values())
                return await connection.execute(query, *param_values)
            else:
                return await connection.execute(query)
    
    async def execute_many(
        self,
        query: str,
        params_list: List[Dict[str, Any]]
    ) -> Any:
        """트랜잭션 없이 여러 쿼리 일괄 실행."""
        await self.connect()
        if self._pool is None:
            raise RuntimeError("Database connection pool not established")
        
        async with self._pool.acquire() as connection:
            param_values_list = [list(params.values()) for params in params_list]
            return await connection.executemany(query, param_values_list)
    
    async def fetch_one(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """단일 행 조회."""
        await self.connect()
        if self._pool is None:
            raise RuntimeError("Database connection pool not established")
        
        async with self._pool.acquire() as connection:
            if params:
                param_values = list(params.values())
                row = await connection.fetchrow(query, *param_values)
            else:
                row = await connection.fetchrow(query)
            
            if row is None:
                return None
            
            # Record를 Dict로 변환
            return dict(row)
    
    async def fetch_all(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """모든 행 조회."""
        await self.connect()
        if self._pool is None:
            raise RuntimeError("Database connection pool not established")
        
        async with self._pool.acquire() as connection:
            if params:
                param_values = list(params.values())
                rows = await connection.fetch(query, *param_values)
            else:
                rows = await connection.fetch(query)
            
            # Records를 Dict 리스트로 변환
            return [dict(row) for row in rows]
    
    async def health_check(self) -> bool:
        """데이터베이스 연결 상태 확인."""
        try:
            await self.connect()
            if self._pool is None:
                return False
            
            async with self._pool.acquire() as connection:
                result = await connection.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False

