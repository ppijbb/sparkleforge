"""
Database Driver Interface

AgentPG 패턴을 참고한 데이터베이스 드라이버 추상화 계층.
SQLite와 PostgreSQL을 지원하는 통합 인터페이스.
"""

import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, TypeVar, Generic
from contextvars import ContextVar
from enum import Enum

logger = logging.getLogger(__name__)

# Context variable for transaction propagation
_current_transaction: ContextVar[Optional["Transaction"]] = ContextVar(
    "current_transaction", default=None
)


class TransactionIsolation(Enum):
    """트랜잭션 격리 수준."""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


class Transaction:
    """
    트랜잭션 추상화 클래스.
    
    AgentPG 패턴을 참고하여 Context 기반으로 트랜잭션을 전파합니다.
    """
    
    def __init__(
        self,
        driver: "DatabaseDriver",
        isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED,
        parent: Optional["Transaction"] = None
    ):
        self.driver = driver
        self.isolation_level = isolation_level
        self.parent = parent
        self.committed = False
        self.rolled_back = False
        self._nested_level = 0 if parent is None else parent._nested_level + 1
    
    async def commit(self) -> None:
        """트랜잭션 커밋."""
        if self.rolled_back:
            raise RuntimeError("Cannot commit a rolled back transaction")
        if self.committed:
            return
        
        await self.driver._commit_transaction(self)
        self.committed = True
        logger.debug(f"Transaction committed (level: {self._nested_level})")
    
    async def rollback(self) -> None:
        """트랜잭션 롤백."""
        if self.committed:
            raise RuntimeError("Cannot rollback a committed transaction")
        if self.rolled_back:
            return
        
        await self.driver._rollback_transaction(self)
        self.rolled_back = True
        logger.debug(f"Transaction rolled back (level: {self._nested_level})")
    
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """쿼리 실행."""
        return await self.driver._execute_in_transaction(self, query, params)
    
    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> Any:
        """여러 쿼리 일괄 실행."""
        return await self.driver._execute_many_in_transaction(self, query, params_list)
    
    def __enter__(self):
        """Context manager 진입."""
        _current_transaction.set(self)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료."""
        _current_transaction.set(self.parent)
        if exc_type is not None:
            # 예외 발생 시 자동 롤백
            import asyncio
            if not self.rolled_back and not self.committed:
                asyncio.create_task(self.rollback())
    
    async def __aenter__(self):
        """Async context manager 진입."""
        _current_transaction.set(self)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager 종료."""
        _current_transaction.set(self.parent)
        if exc_type is not None:
            # 예외 발생 시 자동 롤백
            if not self.rolled_back and not self.committed:
                await self.rollback()


class TransactionContext:
    """트랜잭션 컨텍스트 유틸리티."""
    
    @staticmethod
    def get_current() -> Optional[Transaction]:
        """현재 컨텍스트의 트랜잭션 반환."""
        return _current_transaction.get()
    
    @staticmethod
    def set_current(tx: Optional[Transaction]) -> None:
        """현재 컨텍스트에 트랜잭션 설정."""
        _current_transaction.set(tx)
    
    @staticmethod
    def has_transaction() -> bool:
        """현재 컨텍스트에 트랜잭션이 있는지 확인."""
        return _current_transaction.get() is not None


class DatabaseDriver(ABC):
    """
    데이터베이스 드라이버 추상화 인터페이스.
    
    AgentPG의 driver 패턴을 참고하여 구현했습니다.
    """
    
    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        max_overflow: int = 5,
        isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED
    ):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.isolation_level = isolation_level
        self._pool = None
        logger.info(f"DatabaseDriver initialized: {self.__class__.__name__}")
    
    @abstractmethod
    async def connect(self) -> None:
        """데이터베이스 연결."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """데이터베이스 연결 종료."""
        pass
    
    @abstractmethod
    async def begin(
        self,
        isolation_level: Optional[TransactionIsolation] = None
    ) -> Transaction:
        """
        트랜잭션 시작.
        
        Args:
            isolation_level: 격리 수준 (None이면 기본값 사용)
            
        Returns:
            Transaction 객체
        """
        pass
    
    @abstractmethod
    async def _commit_transaction(self, tx: Transaction) -> None:
        """트랜잭션 커밋 (내부 메서드)."""
        pass
    
    @abstractmethod
    async def _rollback_transaction(self, tx: Transaction) -> None:
        """트랜잭션 롤백 (내부 메서드)."""
        pass
    
    @abstractmethod
    async def _execute_in_transaction(
        self,
        tx: Transaction,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """트랜잭션 내에서 쿼리 실행."""
        pass
    
    @abstractmethod
    async def _execute_many_in_transaction(
        self,
        tx: Transaction,
        query: str,
        params_list: List[Dict[str, Any]]
    ) -> Any:
        """트랜잭션 내에서 여러 쿼리 일괄 실행."""
        pass
    
    @abstractmethod
    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """트랜잭션 없이 쿼리 실행."""
        pass
    
    @abstractmethod
    async def execute_many(
        self,
        query: str,
        params_list: List[Dict[str, Any]]
    ) -> Any:
        """트랜잭션 없이 여러 쿼리 일괄 실행."""
        pass
    
    @abstractmethod
    async def fetch_one(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """단일 행 조회."""
        pass
    
    @abstractmethod
    async def fetch_all(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """모든 행 조회."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """데이터베이스 연결 상태 확인."""
        pass
    
    @asynccontextmanager
    async def transaction(
        self,
        isolation_level: Optional[TransactionIsolation] = None
    ):
        """
        트랜잭션 컨텍스트 매니저.
        
        사용 예:
            async with driver.transaction() as tx:
                await tx.execute("INSERT INTO ...")
                # 자동 커밋 (예외 발생 시 롤백)
        """
        tx = await self.begin(isolation_level)
        try:
            yield tx
            if not tx.committed and not tx.rolled_back:
                await tx.commit()
        except Exception:
            if not tx.rolled_back:
                await tx.rollback()
            raise


# 전역 드라이버 인스턴스
_database_driver: Optional[DatabaseDriver] = None


def get_database_driver() -> Optional[DatabaseDriver]:
    """전역 데이터베이스 드라이버 인스턴스 반환."""
    return _database_driver


def set_database_driver(driver: DatabaseDriver) -> None:
    """전역 데이터베이스 드라이버 설정."""
    global _database_driver
    _database_driver = driver
    logger.info(f"Database driver set: {driver.__class__.__name__}")

