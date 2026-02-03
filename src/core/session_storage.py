"""
Session Storage Backend

세션 데이터의 영구 저장을 위한 백엔드 구현.
파일 기반 저장소와 선택적 데이터베이스 지원.
트랜잭션 지원 추가 (Phase 1).
"""

import json
import gzip
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

from src.core.db.database_driver import Transaction
from src.core.db.transaction_manager import get_transaction_manager

logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """세션 메타데이터."""
    session_id: str
    user_id: Optional[str]
    created_at: str
    last_accessed: str
    state_version: int
    context_size: int
    memory_size: int
    tags: List[str]
    description: Optional[str] = None


class SessionStorage:
    """
    세션 저장소 백엔드.
    
    파일 기반 저장소를 기본으로 하며,
    선택적으로 SQLite 데이터베이스를 사용할 수 있습니다.
    """
    
    def __init__(
        self,
        storage_path: str = "./storage/sessions",
        use_database: bool = True,
        compress: bool = True,
        enable_transactions: bool = False  # 기본값: False (기존 동작 유지)
    ):
        """
        초기화.
        
        Args:
            storage_path: 저장소 경로
            use_database: SQLite 데이터베이스 사용 여부
            compress: 압축 저장 여부
            enable_transactions: 트랜잭션 활성화 여부 (기본값: False, 기존 동작 유지)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.use_database = use_database
        self.compress = compress
        self.enable_transactions = enable_transactions  # 기본값: False
        
        # 파일 저장 경로
        self.sessions_dir = self.storage_path / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 초기화
        self.db_path = self.storage_path / "sessions.db"
        if self.use_database:
            self._init_database()
        
        # 트랜잭션 관리자 (선택적)
        self.tx_manager = None
        if self.enable_transactions:
            try:
                from src.core.db.transaction_manager import get_transaction_manager
                self.tx_manager = get_transaction_manager()
            except Exception as e:
                logger.warning(f"Transaction manager not available: {e}. Continuing without transactions.")
        
        logger.info(
            f"SessionStorage initialized at {self.storage_path} "
            f"(transactions: {'enabled' if self.enable_transactions else 'disabled'})"
        )
    
    def _init_database(self):
        """SQLite 데이터베이스 초기화."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 세션 메타데이터 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_metadata (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    state_version INTEGER DEFAULT 1,
                    context_size INTEGER DEFAULT 0,
                    memory_size INTEGER DEFAULT 0,
                    tags TEXT,
                    description TEXT,
                    file_path TEXT NOT NULL
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id 
                ON session_metadata(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON session_metadata(last_accessed)
            """)
            
            conn.commit()
            conn.close()
            logger.info("Session database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.use_database = False
    
    def save_session(
        self,
        session_id: str,
        state: Dict[str, Any],
        context: Dict[str, Any],
        memory: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tx: Optional[Transaction] = None
    ) -> bool:
        """
        세션 저장.
        
        Args:
            session_id: 세션 ID
            state: AgentState 데이터
            context: ContextEngineer 상태
            memory: SharedMemory 상태
            metadata: 추가 메타데이터
            tx: 트랜잭션 (선택사항)
            
        Returns:
            성공 여부
        """
        try:
            # 세션 데이터 구성
            session_data = {
                "session_id": session_id,
                "state": state,
                "context": context,
                "memory": memory,
                "saved_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # 파일 경로 결정
            file_path = self.sessions_dir / f"{session_id}.json"
            if self.compress:
                file_path = self.sessions_dir / f"{session_id}.json.gz"
            
            # 저장
            json_data = json.dumps(session_data, ensure_ascii=False, indent=2)
            
            if self.compress:
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    f.write(json_data)
            else:
                file_path.write_text(json_data, encoding='utf-8')
            
            # 메타데이터 업데이트
            session_metadata = SessionMetadata(
                session_id=session_id,
                user_id=metadata.get("user_id") if metadata else None,
                created_at=metadata.get("created_at", datetime.now().isoformat()) if metadata else datetime.now().isoformat(),
                last_accessed=datetime.now().isoformat(),
                state_version=metadata.get("state_version", 1) if metadata else 1,
                context_size=len(json.dumps(context, ensure_ascii=False)),
                memory_size=len(json.dumps(memory, ensure_ascii=False)),
                tags=metadata.get("tags", []) if metadata else [],
                description=metadata.get("description") if metadata else None
            )
            
            # 트랜잭션이 있으면 트랜잭션 내에서 메타데이터 저장
            if tx:
                self._save_metadata_in_transaction(session_metadata, str(file_path), tx)
            else:
                self._save_metadata(session_metadata, str(file_path))
            
            logger.info(f"Session saved: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False
    
    async def save_session_async(
        self,
        session_id: str,
        state: Dict[str, Any],
        context: Dict[str, Any],
        memory: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tx: Optional[Transaction] = None,
        use_transaction: Optional[bool] = None  # None이면 enable_transactions 설정 사용
    ) -> bool:
        """
        세션 저장 (비동기, 트랜잭션 지원).
        
        Args:
            session_id: 세션 ID
            state: AgentState 데이터
            context: ContextEngineer 상태
            memory: SharedMemory 상태
            metadata: 추가 메타데이터
            tx: 트랜잭션 (None이면 자동으로 트랜잭션 생성)
            use_transaction: 트랜잭션 사용 여부 (None이면 enable_transactions 설정 사용)
            
        Returns:
            성공 여부
        """
        try:
            # 트랜잭션 사용 여부 결정
            should_use_tx = use_transaction if use_transaction is not None else self.enable_transactions
            
            if should_use_tx and self.tx_manager:
                # 트랜잭션 관리자 가져오기
                if tx is None:
                    async with self.tx_manager.transaction() as new_tx:
                        return await self._save_session_in_transaction(
                            session_id, state, context, memory, metadata, new_tx
                        )
                else:
                    return await self._save_session_in_transaction(
                        session_id, state, context, memory, metadata, tx
                    )
            else:
                # 트랜잭션 없이 저장 (기존 동작)
                return self.save_session(session_id, state, context, memory, metadata, tx=None)
        except Exception as e:
            logger.error(f"Failed to save session {session_id} (async): {e}")
            return False
    
    async def _save_session_in_transaction(
        self,
        session_id: str,
        state: Dict[str, Any],
        context: Dict[str, Any],
        memory: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
        tx: Transaction
    ) -> bool:
        """트랜잭션 내에서 세션 저장."""
        # 세션 데이터 구성
        session_data = {
            "session_id": session_id,
            "state": state,
            "context": context,
            "memory": memory,
            "saved_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # 파일 경로 결정
        file_path = self.sessions_dir / f"{session_id}.json"
        if self.compress:
            file_path = self.sessions_dir / f"{session_id}.json.gz"
        
        # 저장 (파일 저장은 트랜잭션 밖에서, 하지만 원자성을 위해 시도)
        json_data = json.dumps(session_data, ensure_ascii=False, indent=2)
        
        if self.compress:
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                f.write(json_data)
        else:
            file_path.write_text(json_data, encoding='utf-8')
        
        # 메타데이터 업데이트 (트랜잭션 내에서)
        session_metadata = SessionMetadata(
            session_id=session_id,
            user_id=metadata.get("user_id") if metadata else None,
            created_at=metadata.get("created_at", datetime.now().isoformat()) if metadata else datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            state_version=metadata.get("state_version", 1) if metadata else 1,
            context_size=len(json.dumps(context, ensure_ascii=False)),
            memory_size=len(json.dumps(memory, ensure_ascii=False)),
            tags=metadata.get("tags", []) if metadata else [],
            description=metadata.get("description") if metadata else None
        )
        
        await self._save_metadata_in_transaction_async(session_metadata, str(file_path), tx)
        
        logger.info(f"Session saved (transaction): {session_id}")
        return True
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        세션 로드.
        
        Args:
            session_id: 세션 ID
            
        Returns:
            세션 데이터 또는 None
        """
        try:
            # 압축 파일 우선 시도
            file_path = self.sessions_dir / f"{session_id}.json.gz"
            if not file_path.exists():
                file_path = self.sessions_dir / f"{session_id}.json"
            
            if not file_path.exists():
                logger.warning(f"Session file not found: {session_id}")
                return None
            
            # 로드
            if file_path.suffix == ".gz":
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    session_data = json.loads(f.read())
            else:
                session_data = json.loads(file_path.read_text(encoding='utf-8'))
            
            # 메타데이터 업데이트 (last_accessed)
            self._update_last_accessed(session_id)
            
            logger.info(f"Session loaded: {session_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        세션 삭제.
        
        Args:
            session_id: 세션 ID
            
        Returns:
            성공 여부
        """
        try:
            # 파일 삭제
            file_path = self.sessions_dir / f"{session_id}.json.gz"
            if not file_path.exists():
                file_path = self.sessions_dir / f"{session_id}.json"
            
            if file_path.exists():
                file_path.unlink()
            
            # 데이터베이스에서 메타데이터 삭제
            if self.use_database:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("DELETE FROM session_metadata WHERE session_id = ?", (session_id,))
                conn.commit()
                conn.close()
            
            logger.info(f"Session deleted: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def list_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "last_accessed",
        order_desc: bool = True
    ) -> List[SessionMetadata]:
        """
        세션 목록 조회.
        
        Args:
            user_id: 사용자 ID 필터
            limit: 최대 결과 수
            offset: 오프셋
            order_by: 정렬 기준
            order_desc: 내림차순 여부
            
        Returns:
            세션 메타데이터 목록
        """
        try:
            if self.use_database:
                return self._list_sessions_from_db(user_id, limit, offset, order_by, order_desc)
            else:
                return self._list_sessions_from_files(user_id, limit, offset, order_by, order_desc)
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    def _list_sessions_from_db(
        self,
        user_id: Optional[str],
        limit: int,
        offset: int,
        order_by: str,
        order_desc: bool
    ) -> List[SessionMetadata]:
        """데이터베이스에서 세션 목록 조회."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        order_dir = "DESC" if order_desc else "ASC"
        query = f"SELECT * FROM session_metadata"
        params = []
        
        if user_id:
            query += " WHERE user_id = ?"
            params.append(user_id)
        
        query += f" ORDER BY {order_by} {order_dir} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        sessions = []
        for row in rows:
            sessions.append(SessionMetadata(
                session_id=row["session_id"],
                user_id=row["user_id"],
                created_at=row["created_at"],
                last_accessed=row["last_accessed"],
                state_version=row["state_version"],
                context_size=row["context_size"],
                memory_size=row["memory_size"],
                tags=json.loads(row["tags"]) if row["tags"] else [],
                description=row["description"]
            ))
        
        return sessions
    
    def _list_sessions_from_files(
        self,
        user_id: Optional[str],
        limit: int,
        offset: int,
        order_by: str,
        order_desc: bool
    ) -> List[SessionMetadata]:
        """파일에서 세션 목록 조회."""
        sessions = []
        
        for file_path in self.sessions_dir.glob("*.json*"):
            try:
                session_id = file_path.stem.replace(".json", "")
                session_data = self.load_session(session_id)
                
                if not session_data:
                    continue
                
                metadata = session_data.get("metadata", {})
                if user_id and metadata.get("user_id") != user_id:
                    continue
                
                sessions.append(SessionMetadata(
                    session_id=session_id,
                    user_id=metadata.get("user_id"),
                    created_at=metadata.get("created_at", session_data.get("saved_at", "")),
                    last_accessed=metadata.get("last_accessed", session_data.get("saved_at", "")),
                    state_version=metadata.get("state_version", 1),
                    context_size=len(json.dumps(session_data.get("context", {}), ensure_ascii=False)),
                    memory_size=len(json.dumps(session_data.get("memory", {}), ensure_ascii=False)),
                    tags=metadata.get("tags", []),
                    description=metadata.get("description")
                ))
            except Exception as e:
                logger.warning(f"Failed to read session from {file_path}: {e}")
                continue
        
        # 정렬
        reverse = order_desc
        if order_by == "last_accessed":
            sessions.sort(key=lambda x: x.last_accessed, reverse=reverse)
        elif order_by == "created_at":
            sessions.sort(key=lambda x: x.created_at, reverse=reverse)
        
        return sessions[offset:offset + limit]
    
    def _save_metadata(self, metadata: SessionMetadata, file_path: str):
        """메타데이터 저장 (동기, 기존 호환성)."""
        if not self.use_database:
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO session_metadata
                (session_id, user_id, created_at, last_accessed, state_version,
                 context_size, memory_size, tags, description, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.session_id,
                metadata.user_id,
                metadata.created_at,
                metadata.last_accessed,
                metadata.state_version,
                metadata.context_size,
                metadata.memory_size,
                json.dumps(metadata.tags),
                metadata.description,
                file_path
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _save_metadata_in_transaction(
        self,
        metadata: SessionMetadata,
        file_path: str,
        tx: Transaction
    ):
        """메타데이터 저장 (트랜잭션 내, 동기)."""
        if not self.use_database:
            return
        
        # 트랜잭션을 통해 실행
        try:
            query = """
                INSERT OR REPLACE INTO session_metadata
                (session_id, user_id, created_at, last_accessed, state_version,
                 context_size, memory_size, tags, description, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = {
                "session_id": metadata.session_id,
                "user_id": metadata.user_id,
                "created_at": metadata.created_at,
                "last_accessed": metadata.last_accessed,
                "state_version": metadata.state_version,
                "context_size": metadata.context_size,
                "memory_size": metadata.memory_size,
                "tags": json.dumps(metadata.tags),
                "description": metadata.description,
                "file_path": file_path
            }
            # 동기 트랜잭션은 나중에 구현
            # 현재는 기존 방식 사용
            self._save_metadata(metadata, file_path)
        except Exception as e:
            logger.error(f"Failed to save metadata in transaction: {e}")
    
    async def _save_metadata_in_transaction_async(
        self,
        metadata: SessionMetadata,
        file_path: str,
        tx: Transaction
    ):
        """메타데이터 저장 (트랜잭션 내, 비동기)."""
        if not self.use_database:
            return
        
        try:
            query = """
                INSERT OR REPLACE INTO session_metadata
                (session_id, user_id, created_at, last_accessed, state_version,
                 context_size, memory_size, tags, description, file_path)
                VALUES (:session_id, :user_id, :created_at, :last_accessed, :state_version,
                 :context_size, :memory_size, :tags, :description, :file_path)
            """
            params = {
                "session_id": metadata.session_id,
                "user_id": metadata.user_id,
                "created_at": metadata.created_at,
                "last_accessed": metadata.last_accessed,
                "state_version": metadata.state_version,
                "context_size": metadata.context_size,
                "memory_size": metadata.memory_size,
                "tags": json.dumps(metadata.tags),
                "description": metadata.description,
                "file_path": file_path
            }
            await tx.execute(query, params)
            logger.debug(f"Metadata saved in transaction: {metadata.session_id}")
        except Exception as e:
            logger.error(f"Failed to save metadata in transaction (async): {e}")
            raise
    
    def _update_last_accessed(self, session_id: str):
        """마지막 접근 시간 업데이트."""
        if not self.use_database:
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE session_metadata SET last_accessed = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update last_accessed: {e}")
    
    def create_snapshot(self, session_id: str) -> Optional[str]:
        """
        세션 스냅샷 생성.
        
        Args:
            session_id: 세션 ID
            
        Returns:
            스냅샷 ID 또는 None
        """
        try:
            session_data = self.load_session(session_id)
            if not session_data:
                return None
            
            snapshot_id = f"{session_id}_snapshot_{int(datetime.now().timestamp())}"
            snapshot_data = session_data.copy()
            snapshot_data["snapshot_id"] = snapshot_id
            snapshot_data["original_session_id"] = session_id
            snapshot_data["snapshot_at"] = datetime.now().isoformat()
            
            # 스냅샷 저장
            snapshot_dir = self.storage_path / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_path = snapshot_dir / f"{snapshot_id}.json.gz"
            json_data = json.dumps(snapshot_data, ensure_ascii=False, indent=2)
            
            with gzip.open(snapshot_path, 'wt', encoding='utf-8') as f:
                f.write(json_data)
            
            logger.info(f"Snapshot created: {snapshot_id} for session {session_id}")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Failed to create snapshot for {session_id}: {e}")
            return None
    
    def restore_from_snapshot(self, session_id: str, snapshot_id: str) -> bool:
        """
        스냅샷에서 세션 복원.
        
        Args:
            session_id: 복원할 세션 ID
            snapshot_id: 스냅샷 ID
            
        Returns:
            성공 여부
        """
        try:
            snapshot_dir = self.storage_path / "snapshots"
            snapshot_path = snapshot_dir / f"{snapshot_id}.json.gz"
            
            if not snapshot_path.exists():
                logger.error(f"Snapshot not found: {snapshot_id}")
                return False
            
            with gzip.open(snapshot_path, 'rt', encoding='utf-8') as f:
                snapshot_data = json.loads(f.read())
            
            # 세션 데이터 복원
            session_data = snapshot_data.copy()
            session_data["session_id"] = session_id
            session_data.pop("snapshot_id", None)
            session_data.pop("original_session_id", None)
            session_data["restored_from_snapshot"] = snapshot_id
            session_data["restored_at"] = datetime.now().isoformat()
            
            # 세션 저장
            return self.save_session(
                session_id,
                session_data.get("state", {}),
                session_data.get("context", {}),
                session_data.get("memory", {}),
                session_data.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to restore from snapshot {snapshot_id}: {e}")
            return False


# 전역 인스턴스
_session_storage: Optional[SessionStorage] = None


def get_session_storage() -> SessionStorage:
    """전역 세션 저장소 인스턴스 반환."""
    global _session_storage
    if _session_storage is None:
        import os
        storage_path = os.getenv("SESSION_STORAGE_PATH", "./storage/sessions")
        use_database = os.getenv("SESSION_USE_DATABASE", "true").lower() == "true"
        compress = os.getenv("SESSION_COMPRESS", "true").lower() == "true"
        _session_storage = SessionStorage(
            storage_path=storage_path,
            use_database=use_database,
            compress=compress
        )
    return _session_storage

