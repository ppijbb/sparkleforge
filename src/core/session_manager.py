"""
Session Manager

세션 상태 전체 저장/복원 및 메타데이터 관리.
AgentState, ContextEngineer, SharedMemory 통합 관리.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from src.core.session_storage import SessionStorage, SessionMetadata, get_session_storage
from src.core.context_engineer import ContextEngineer, get_context_engineer
from src.core.shared_memory import SharedMemory, get_shared_memory
from src.core.pii_redaction import get_pii_redactor

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """완전한 세션 상태."""
    session_id: str
    agent_state: Dict[str, Any]
    context_chunks: List[Dict[str, Any]]
    memory_data: Dict[str, Any]
    metadata: Dict[str, Any]


class SessionManager:
    """
    세션 관리자.
    
    세션 상태 전체 저장/복원 및 메타데이터 관리를 담당합니다.
    """
    
    def __init__(self, storage: Optional[SessionStorage] = None):
        """
        초기화.
        
        Args:
            storage: 세션 저장소 (None이면 전역 인스턴스 사용)
        """
        self.storage = storage or get_session_storage()
        self.context_engineer: Optional[ContextEngineer] = None
        self.shared_memory: Optional[SharedMemory] = None
    
    def set_context_engineer(self, context_engineer: ContextEngineer):
        """ContextEngineer 설정."""
        self.context_engineer = context_engineer
    
    def set_shared_memory(self, shared_memory: SharedMemory):
        """SharedMemory 설정."""
        self.shared_memory = shared_memory
    
    def save_session(
        self,
        session_id: str,
        agent_state: Dict[str, Any],
        context_engineer: Optional[ContextEngineer] = None,
        shared_memory: Optional[SharedMemory] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        세션 저장.
        
        Args:
            session_id: 세션 ID
            agent_state: AgentState 데이터
            context_engineer: ContextEngineer 인스턴스 (None이면 전역 사용)
            shared_memory: SharedMemory 인스턴스 (None이면 전역 사용)
            metadata: 추가 메타데이터
            
        Returns:
            성공 여부
        """
        try:
            # ContextEngineer 상태 추출
            ce = context_engineer or self.context_engineer or get_context_engineer()
            context_data = self._extract_context_state(ce)
            
            # SharedMemory 상태 추출
            sm = shared_memory or self.shared_memory or get_shared_memory()
            memory_data = self._extract_memory_state(sm, session_id)
            
            # PII Redaction (백서 요구사항: 세션 데이터 저장 전 PII 제거)
            pii_redactor = get_pii_redactor()
            redacted_state, pii_matches_state = pii_redactor.redact_session_data(agent_state)
            redacted_context, pii_matches_context = pii_redactor.redact_session_data(context_data)
            redacted_memory, pii_matches_memory = pii_redactor.redact_session_data(memory_data)
            
            # Calculate total PII matches
            total_pii = len(pii_matches_state) + len(pii_matches_context) + len(pii_matches_memory)
            
            if total_pii > 0:
                logger.warning(f"PII redaction: {total_pii} PII matches removed from session {session_id}")
            
            # 메타데이터 구성
            full_metadata = {
                "user_id": metadata.get("user_id") if metadata else None,
                "created_at": metadata.get("created_at", datetime.now().isoformat()) if metadata else datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "state_version": metadata.get("state_version", 1) if metadata else 1,
                "tags": metadata.get("tags", []) if metadata else [],
                "description": metadata.get("description") if metadata else None,
                "agent_state_keys": list(redacted_state.keys()),
                "context_chunk_count": len(redacted_context.get("chunks", [])),
                "memory_entry_count": len(redacted_memory.get("entries", [])),
                "pii_redacted": total_pii > 0,
                "pii_match_count": total_pii
            }
            
            # 저장 (PII 제거된 데이터)
            success = self.storage.save_session(
                session_id=session_id,
                state=redacted_state,
                context=redacted_context,
                memory=redacted_memory,
                metadata=full_metadata
            )
            
            if success:
                logger.info(f"Session saved successfully: {session_id}")
            else:
                logger.error(f"Failed to save session: {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}", exc_info=True)
            return False
    
    def load_session(self, session_id: str) -> Optional[SessionState]:
        """
        세션 로드.
        
        Args:
            session_id: 세션 ID
            
        Returns:
            SessionState 또는 None
        """
        try:
            session_data = self.storage.load_session(session_id)
            if not session_data:
                logger.warning(f"Session not found: {session_id}")
                return None
            
            return SessionState(
                session_id=session_id,
                agent_state=session_data.get("state", {}),
                context_chunks=session_data.get("context", {}).get("chunks", []),
                memory_data=session_data.get("memory", {}),
                metadata=session_data.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}", exc_info=True)
            return None
    
    def restore_session(
        self,
        session_id: str,
        context_engineer: Optional[ContextEngineer] = None,
        shared_memory: Optional[SharedMemory] = None
    ) -> Optional[Dict[str, Any]]:
        """
        세션 복원 (AgentState만 반환, ContextEngineer와 SharedMemory는 자동 복원).
        
        Args:
            session_id: 세션 ID
            context_engineer: ContextEngineer 인스턴스 (None이면 전역 사용)
            shared_memory: SharedMemory 인스턴스 (None이면 전역 사용)
            
        Returns:
            AgentState 또는 None
        """
        try:
            session_state = self.load_session(session_id)
            if not session_state:
                return None
            
            # ContextEngineer 복원
            ce = context_engineer or self.context_engineer or get_context_engineer()
            self._restore_context_state(ce, session_state.context_chunks)
            
            # SharedMemory 복원
            sm = shared_memory or self.shared_memory or get_shared_memory()
            self._restore_memory_state(sm, session_state.memory_data, session_id)
            
            logger.info(f"Session restored: {session_id}")
            return session_state.agent_state
            
        except Exception as e:
            logger.error(f"Error restoring session {session_id}: {e}", exc_info=True)
            return None
    
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
        return self.storage.list_sessions(
            user_id=user_id,
            limit=limit,
            offset=offset,
            order_by=order_by,
            order_desc=order_desc
        )
    
    def delete_session(self, session_id: str) -> bool:
        """
        세션 삭제.
        
        Args:
            session_id: 세션 ID
            
        Returns:
            성공 여부
        """
        return self.storage.delete_session(session_id)
    
    def create_snapshot(self, session_id: str) -> Optional[str]:
        """
        세션 스냅샷 생성.
        
        Args:
            session_id: 세션 ID
            
        Returns:
            스냅샷 ID 또는 None
        """
        return self.storage.create_snapshot(session_id)
    
    def restore_from_snapshot(self, session_id: str, snapshot_id: str) -> bool:
        """
        스냅샷에서 세션 복원.
        
        Args:
            session_id: 복원할 세션 ID
            snapshot_id: 스냅샷 ID
            
        Returns:
            성공 여부
        """
        return self.storage.restore_from_snapshot(session_id, snapshot_id)
    
    def _extract_context_state(self, context_engineer: ContextEngineer) -> Dict[str, Any]:
        """ContextEngineer 상태 추출."""
        try:
            stats = context_engineer.get_context_stats()
            
            # 청크 데이터 추출
            chunks_data = []
            for chunk in context_engineer.context_chunks:
                chunks_data.append({
                    "content": chunk.content,
                    "content_type": chunk.content_type.value,
                    "priority": chunk.priority.value,
                    "timestamp": chunk.timestamp,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                    "dependencies": list(chunk.dependencies),
                    "compressed": chunk.compressed,
                    "original_size": chunk.original_size
                })
            
            return {
                "chunks": chunks_data,
                "stats": stats,
                "config": {
                    "max_tokens": context_engineer.config.max_tokens,
                    "enable_auto_compression": context_engineer.config.enable_auto_compression,
                    "compression_threshold": context_engineer.config.compression_threshold
                }
            }
        except Exception as e:
            logger.error(f"Error extracting context state: {e}")
            return {"chunks": [], "stats": {}, "config": {}}
    
    def _restore_context_state(self, context_engineer: ContextEngineer, chunks_data: List[Dict[str, Any]]):
        """ContextEngineer 상태 복원."""
        try:
            from src.core.context_engineer import ContextChunk, ContextType, ContextPriority
            
            # 기존 청크 클리어
            context_engineer.context_chunks.clear()
            
            # 청크 복원
            for chunk_data in chunks_data:
                chunk = ContextChunk(
                    content=chunk_data["content"],
                    content_type=ContextType(chunk_data["content_type"]),
                    priority=ContextPriority(chunk_data["priority"]),
                    timestamp=chunk_data["timestamp"],
                    metadata=chunk_data.get("metadata", {}),
                    dependencies=set(chunk_data.get("dependencies", [])),
                    compressed=chunk_data.get("compressed", False),
                    original_size=chunk_data.get("original_size", 0)
                )
                chunk.token_count = chunk_data.get("token_count", chunk._estimate_tokens())
                context_engineer.context_chunks.append(chunk)
            
            logger.info(f"Context state restored: {len(chunks_data)} chunks")
            
        except Exception as e:
            logger.error(f"Error restoring context state: {e}", exc_info=True)
    
    def _extract_memory_state(self, shared_memory: SharedMemory, session_id: str) -> Dict[str, Any]:
        """SharedMemory 상태 추출."""
        try:
            # 세션별 메모리 추출
            session_memories = shared_memory.list_session_memories(session_id)
            
            entries = []
            for key, memory_entry in session_memories.items():
                if key == "agents":
                    continue
                if isinstance(memory_entry, dict):
                    entries.append({
                        "key": key,
                        "value": memory_entry.get("value"),
                        "scope": memory_entry.get("scope"),
                        "timestamp": memory_entry.get("timestamp")
                    })
            
            return {
                "session_id": session_id,
                "entries": entries,
                "entry_count": len(entries)
            }
        except Exception as e:
            logger.error(f"Error extracting memory state: {e}")
            return {"session_id": session_id, "entries": [], "entry_count": 0}
    
    def _restore_memory_state(self, shared_memory: SharedMemory, memory_data: Dict[str, Any], session_id: str):
        """SharedMemory 상태 복원."""
        try:
            # 기존 세션 메모리 클리어
            shared_memory.clear_session(session_id)
            
            # 메모리 복원
            entries = memory_data.get("entries", [])
            for entry in entries:
                shared_memory.write(
                    key=entry["key"],
                    value=entry["value"],
                    scope=entry.get("scope", "session"),
                    session_id=session_id
                )
            
            logger.info(f"Memory state restored: {len(entries)} entries")
            
        except Exception as e:
            logger.error(f"Error restoring memory state: {e}", exc_info=True)


# 전역 인스턴스
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """전역 세션 관리자 인스턴스 반환."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager

