"""
Transaction Manager

트랜잭션 생명주기 관리 및 원자적 작업 실행을 담당합니다.
AgentPG 패턴을 참고하여 Context 기반 트랜잭션 전파를 지원합니다.
"""

import logging
from typing import Any, Callable, List, Optional, TypeVar, Coroutine
from contextlib import asynccontextmanager
from contextvars import copy_context

from src.core.db.database_driver import (
    DatabaseDriver,
    Transaction,
    TransactionContext,
    TransactionIsolation,
    get_database_driver,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TransactionManager:
    """
    트랜잭션 관리자.
    
    여러 작업을 원자적으로 실행하고, Context 기반으로 트랜잭션을 전파합니다.
    """
    
    def __init__(
        self,
        driver: Optional[DatabaseDriver] = None,
        default_isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED
    ):
        """
        초기화.
        
        Args:
            driver: 데이터베이스 드라이버 (None이면 전역 드라이버 사용)
            default_isolation_level: 기본 격리 수준
        """
        self.driver = driver or get_database_driver()
        if self.driver is None:
            raise ValueError("Database driver is required. Set it via set_database_driver() or pass to constructor.")
        
        self.default_isolation_level = default_isolation_level
        logger.info("TransactionManager initialized")
    
    async def begin(
        self,
        isolation_level: Optional[TransactionIsolation] = None
    ) -> Transaction:
        """
        새 트랜잭션 시작.
        
        Args:
            isolation_level: 격리 수준 (None이면 기본값 사용)
            
        Returns:
            Transaction 객체
        """
        level = isolation_level or self.default_isolation_level
        
        # 현재 트랜잭션이 있으면 중첩 트랜잭션으로 처리
        current_tx = TransactionContext.get_current()
        if current_tx:
            # 중첩 트랜잭션: Savepoint 사용
            tx = await self.driver.begin(level)
            tx.parent = current_tx
            logger.debug(f"Nested transaction started (level: {tx._nested_level})")
        else:
            # 루트 트랜잭션
            tx = await self.driver.begin(level)
            logger.debug("Root transaction started")
        
        return tx
    
    @asynccontextmanager
    async def transaction(
        self,
        isolation_level: Optional[TransactionIsolation] = None
    ):
        """
        트랜잭션 컨텍스트 매니저.
        
        사용 예:
            async with transaction_manager.transaction() as tx:
                await operation1(tx)
                await operation2(tx)
                # 자동 커밋 (예외 발생 시 롤백)
        """
        tx = await self.begin(isolation_level)
        try:
            # Context에 트랜잭션 설정
            TransactionContext.set_current(tx)
            yield tx
            # 성공 시 커밋
            if not tx.committed and not tx.rolled_back:
                await tx.commit()
        except Exception as e:
            # 예외 발생 시 롤백
            if not tx.rolled_back:
                await tx.rollback()
            raise
        finally:
            # Context에서 트랜잭션 제거 (부모 트랜잭션으로 복원)
            TransactionContext.set_current(tx.parent)
    
    async def execute_atomic(
        self,
        operations: List[Callable[[Transaction], Coroutine[Any, Any, Any]]],
        isolation_level: Optional[TransactionIsolation] = None
    ) -> List[Any]:
        """
        여러 작업을 원자적으로 실행.
        
        모든 작업이 성공하면 커밋, 하나라도 실패하면 롤백.
        
        Args:
            operations: 트랜잭션을 받아서 실행할 작업 리스트
            isolation_level: 격리 수준
            
        Returns:
            각 작업의 결과 리스트
            
        Raises:
            Exception: 작업 중 예외 발생 시 롤백 후 예외 재발생
        """
        async with self.transaction(isolation_level) as tx:
            results = []
            for i, operation in enumerate(operations):
                try:
                    result = await operation(tx)
                    results.append(result)
                    logger.debug(f"Operation {i+1}/{len(operations)} completed")
                except Exception as e:
                    logger.error(f"Operation {i+1}/{len(operations)} failed: {e}")
                    raise  # 예외 발생 시 트랜잭션 롤백 (컨텍스트 매니저가 처리)
            
            return results
    
    async def execute_in_transaction(
        self,
        operation: Callable[[Transaction], Coroutine[Any, Any, T]],
        isolation_level: Optional[TransactionIsolation] = None
    ) -> T:
        """
        단일 작업을 트랜잭션 내에서 실행.
        
        Args:
            operation: 트랜잭션을 받아서 실행할 작업
            isolation_level: 격리 수준
            
        Returns:
            작업 결과
        """
        async with self.transaction(isolation_level) as tx:
            return await operation(tx)
    
    def get_current_transaction(self) -> Optional[Transaction]:
        """현재 컨텍스트의 트랜잭션 반환."""
        return TransactionContext.get_current()
    
    def has_transaction(self) -> bool:
        """현재 컨텍스트에 트랜잭션이 있는지 확인."""
        return TransactionContext.has_transaction()


# 전역 트랜잭션 관리자 인스턴스
_transaction_manager: Optional[TransactionManager] = None


def get_transaction_manager() -> TransactionManager:
    """전역 트랜잭션 관리자 인스턴스 반환."""
    global _transaction_manager
    if _transaction_manager is None:
        _transaction_manager = TransactionManager()
    return _transaction_manager


def set_transaction_manager(manager: TransactionManager) -> None:
    """전역 트랜잭션 관리자 설정."""
    global _transaction_manager
    _transaction_manager = manager
    logger.info("Transaction manager set")

