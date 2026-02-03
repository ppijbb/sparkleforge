"""
Database Module

데이터베이스 드라이버 및 트랜잭션 관리 모듈.
"""

from src.core.db.database_driver import (
    DatabaseDriver,
    Transaction,
    TransactionContext,
    TransactionIsolation,
    get_database_driver,
    set_database_driver,
)
from src.core.db.transaction_manager import (
    TransactionManager,
    get_transaction_manager,
    set_transaction_manager,
)
# SQLite 드라이버는 선택적 의존성
try:
    from src.core.db.sqlite_driver import (
        SQLiteDriver,
        SQLiteTransaction,
        AIOSQLITE_AVAILABLE,
    )
except ImportError:
    # aiosqlite가 없으면 SQLite 기능 비활성화
    AIOSQLITE_AVAILABLE = False
    SQLiteDriver = None  # type: ignore
    SQLiteTransaction = None  # type: ignore

# PostgreSQL 드라이버는 선택적 의존성
try:
    from src.core.db.postgresql_driver import (
        PostgreSQLDriver,
        PostgreSQLTransaction,
        ASYNCPG_AVAILABLE,
    )
except ImportError:
    # asyncpg가 없으면 PostgreSQL 기능 비활성화
    ASYNCPG_AVAILABLE = False
    PostgreSQLDriver = None  # type: ignore
    PostgreSQLTransaction = None  # type: ignore

__all__ = [
    "DatabaseDriver",
    "Transaction",
    "TransactionContext",
    "TransactionIsolation",
    "get_database_driver",
    "set_database_driver",
    "TransactionManager",
    "get_transaction_manager",
    "set_transaction_manager",
]

# SQLite 관련은 선택적으로 추가
if AIOSQLITE_AVAILABLE:
    __all__.extend([
        "SQLiteDriver",
        "SQLiteTransaction",
        "AIOSQLITE_AVAILABLE",
    ])

# PostgreSQL 관련은 선택적으로 추가
if ASYNCPG_AVAILABLE:
    __all__.extend([
        "PostgreSQLDriver",
        "PostgreSQLTransaction",
        "ASYNCPG_AVAILABLE",
    ])

