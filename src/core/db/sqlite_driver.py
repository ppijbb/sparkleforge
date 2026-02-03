"""
SQLite Driver Implementation

SQLite 데이터베이스 드라이버 구현.
기존 SessionStorage와의 호환성을 유지하면서 트랜잭션 지원을 추가합니다.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from contextlib import asynccontextmanager

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None

from src.core.db.database_driver import (
    DatabaseDriver,
    Transaction,
    TransactionIsolation,
)

if TYPE_CHECKING:
    # 타입 체크 시에만 aiosqlite.Connection 사용
    if aiosqlite is not None:
        ConnectionType = aiosqlite.Connection
    else:
        ConnectionType = Any
else:
    # 런타임에는 Any로 처리
    ConnectionType = Any

logger = logging.getLogger(__name__)


class SQLiteTransaction(Transaction):
    """SQLite 트랜잭션."""
    
    def __init__(
        self,
        driver: "SQLiteDriver",
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
        logger.debug(f"SQLite transaction committed (level: {self._nested_level})")
    
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
        logger.debug(f"SQLite transaction rolled back (level: {self._nested_level})")


class SQLiteDriver(DatabaseDriver):
    """
    SQLite 데이터베이스 드라이버.
    
    기존 SessionStorage와의 호환성을 유지하면서 트랜잭션 지원을 추가합니다.
    """
    
    def __init__(
        self,
        db_path: str,
        pool_size: int = 10,
        max_overflow: int = 5,
        isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED,
        timeout: float = 5.0
    ):
        """
        초기화.
        
        Args:
            db_path: SQLite 데이터베이스 파일 경로
            pool_size: 연결 풀 크기 (SQLite는 단일 연결이지만 호환성을 위해 유지)
            max_overflow: 최대 오버플로우 연결 수
            isolation_level: 기본 격리 수준
            timeout: 연결 타임아웃 (초)
        """
        if not AIOSQLITE_AVAILABLE:
            raise ImportError(
                "aiosqlite is required for SQLite driver. "
                "Install it with: pip install aiosqlite"
            )
        
        # SQLite는 파일 경로를 connection_string으로 사용
        super().__init__(db_path, pool_size, max_overflow, isolation_level)
        self.db_path = Path(db_path)
        self.timeout = timeout
        self._connection: Optional[ConnectionType] = None
        self._savepoint_counter = 0
        
        # 디렉토리 생성
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def connect(self) -> None:
        """데이터베이스 연결."""
        if self._connection is None:
            self._connection = await aiosqlite.connect(
                str(self.db_path),
                timeout=self.timeout
            )
            # 외래 키 제약 조건 활성화
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._connection.commit()
            logger.info(f"SQLite connected: {self.db_path}")
    
    async def disconnect(self) -> None:
        """데이터베이스 연결 종료."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("SQLite disconnected")
    
    async def begin(
        self,
        isolation_level: Optional[TransactionIsolation] = None
    ) -> SQLiteTransaction:
        """
        트랜잭션 시작.
        
        SQLite는 중첩 트랜잭션을 Savepoint로 구현합니다.
        """
        await self.connect()
        
        if self._connection is None:
            raise RuntimeError("Database connection not established")
        
        # 현재 트랜잭션 확인 (Context에서)
        from src.core.db.database_driver import TransactionContext
        current_tx = TransactionContext.get_current()
        
        if current_tx and isinstance(current_tx, SQLiteTransaction):
            # 중첩 트랜잭션: Savepoint 생성
            self._savepoint_counter += 1
            savepoint_name = f"sp_{self._savepoint_counter}"
            await self._connection.execute(f"SAVEPOINT {savepoint_name}")
            
            tx = SQLiteTransaction(
                self,
                self._connection,
                isolation_level or self.isolation_level,
                current_tx
            )
            tx.savepoint_name = savepoint_name
            logger.debug(f"SQLite savepoint created: {savepoint_name}")
        else:
            # 루트 트랜잭션
            tx = SQLiteTransaction(
                self,
                self._connection,
                isolation_level or self.isolation_level
            )
            logger.debug("SQLite root transaction started")
        
        return tx
    
    async def _commit_transaction(self, tx: SQLiteTransaction) -> None:
        """트랜잭션 커밋 (내부 메서드)."""
        # SQLiteTransaction.commit()에서 처리
        pass
    
    async def _rollback_transaction(self, tx: SQLiteTransaction) -> None:
        """트랜잭션 롤백 (내부 메서드)."""
        # SQLiteTransaction.rollback()에서 처리
        pass
    
    async def _execute_in_transaction(
        self,
        tx: SQLiteTransaction,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """트랜잭션 내에서 쿼리 실행."""
        if params:
            # Dict를 tuple로 변환 (SQLite는 위치 기반 파라미터 사용)
            # Named parameters도 지원하지만, 간단하게 처리
            cursor = await tx.connection.execute(query, params)
        else:
            cursor = await tx.connection.execute(query)
        
        return cursor
    
    async def _execute_many_in_transaction(
        self,
        tx: SQLiteTransaction,
        query: str,
        params_list: List[Dict[str, Any]]
    ) -> Any:
        """트랜잭션 내에서 여러 쿼리 일괄 실행."""
        cursor = await tx.connection.executemany(query, params_list)
        return cursor
    
    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """트랜잭션 없이 쿼리 실행."""
        await self.connect()
        if self._connection is None:
            raise RuntimeError("Database connection not established")
        
        if params:
            cursor = await self._connection.execute(query, params)
        else:
            cursor = await self._connection.execute(query)
        
        await self._connection.commit()
        return cursor
    
    async def execute_many(
        self,
        query: str,
        params_list: List[Dict[str, Any]]
    ) -> Any:
        """트랜잭션 없이 여러 쿼리 일괄 실행."""
        await self.connect()
        if self._connection is None:
            raise RuntimeError("Database connection not established")
        
        cursor = await self._connection.executemany(query, params_list)
        await self._connection.commit()
        return cursor
    
    async def fetch_one(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """단일 행 조회."""
        await self.connect()
        if self._connection is None:
            raise RuntimeError("Database connection not established")
        
        if params:
            cursor = await self._connection.execute(query, params)
        else:
            cursor = await self._connection.execute(query)
        
        row = await cursor.fetchone()
        if row is None:
            return None
        
        # Row를 Dict로 변환
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    
    async def fetch_all(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """모든 행 조회."""
        await self.connect()
        if self._connection is None:
            raise RuntimeError("Database connection not established")
        
        if params:
            cursor = await self._connection.execute(query, params)
        else:
            cursor = await self._connection.execute(query)
        
        rows = await cursor.fetchall()
        
        # Rows를 Dict 리스트로 변환
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    async def health_check(self) -> bool:
        """데이터베이스 연결 상태 확인."""
        try:
            await self.connect()
            if self._connection is None:
                return False
            
            cursor = await self._connection.execute("SELECT 1")
            await cursor.fetchone()
            return True
        except Exception as e:
            logger.error(f"SQLite health check failed: {e}")
            return False

