"""
Session-Context Bridge

세션 저장 시 ContextEngineer 상태 포함,
세션 복원 시 컨텍스트 자동 재구성,
세션 간 컨텍스트 공유 전략.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.core.session_manager import SessionManager, get_session_manager
from src.core.context_engineer import ContextEngineer, get_context_engineer
from src.core.shared_memory import SharedMemory, get_shared_memory
from src.core.adaptive_memory import AdaptiveMemory, get_adaptive_memory

logger = logging.getLogger(__name__)


class SessionContextBridge:
    """
    세션-컨텍스트 통합 브리지.
    
    세션과 컨텍스트 간의 자동 동기화 및 공유를 담당합니다.
    """
    
    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        context_engineer: Optional[ContextEngineer] = None,
        shared_memory: Optional[SharedMemory] = None,
        adaptive_memory: Optional[AdaptiveMemory] = None
    ):
        """
        초기화.
        
        Args:
            session_manager: 세션 관리자
            context_engineer: 컨텍스트 엔지니어
            shared_memory: 공유 메모리
            adaptive_memory: 적응형 메모리
        """
        self.session_manager = session_manager or get_session_manager()
        self.context_engineer = context_engineer or get_context_engineer()
        self.shared_memory = shared_memory or get_shared_memory()
        self.adaptive_memory = adaptive_memory or get_adaptive_memory()
        
        logger.info("SessionContextBridge initialized")
    
    async def save_session_with_context(
        self,
        session_id: str,
        agent_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        세션과 컨텍스트를 함께 저장.
        
        Args:
            session_id: 세션 ID
            agent_state: AgentState 데이터
            metadata: 추가 메타데이터
            
        Returns:
            성공 여부
        """
        try:
            # 중요 정보 추출 (적응형 메모리)
            important_info = self.adaptive_memory.extract_important_info(agent_state)
            
            # 중요 정보를 적응형 메모리에 저장
            for info in important_info:
                self.adaptive_memory.store(
                    key=f"{session_id}_{info['type']}",
                    value=info['content'],
                    importance=info['importance'],
                    tags=info.get('tags', set())
                )
            
            # 세션 저장 (컨텍스트 포함)
            success = self.session_manager.save_session(
                session_id=session_id,
                agent_state=agent_state,
                context_engineer=self.context_engineer,
                shared_memory=self.shared_memory,
                metadata=metadata
            )
            
            if success:
                logger.info(f"Session and context saved: {session_id} ({len(important_info)} important items extracted)")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save session with context: {e}", exc_info=True)
            return False
    
    async def restore_session_with_context(
        self,
        session_id: str,
        target_session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        세션과 컨텍스트를 함께 복원.
        
        Args:
            session_id: 복원할 세션 ID
            target_session_id: 타겟 세션 ID (지식 전이용, None이면 동일)
            
        Returns:
            복원된 AgentState 또는 None
        """
        try:
            # 세션 복원
            restored_state = self.session_manager.restore_session(
                session_id=session_id,
                context_engineer=self.context_engineer,
                shared_memory=self.shared_memory
            )
            
            if not restored_state:
                return None
            
            # 지식 전이 (다른 세션으로 복원하는 경우)
            if target_session_id and target_session_id != session_id:
                transfer = self.adaptive_memory.transfer_knowledge(
                    source_session_id=session_id,
                    target_session_id=target_session_id
                )
                logger.info(f"Knowledge transferred: {len(transfer.transferred_items)} items")
            
            # 컨텍스트 자동 재구성
            await self._reconstruct_context(session_id)
            
            logger.info(f"Session and context restored: {session_id}")
            return restored_state
            
        except Exception as e:
            logger.error(f"Failed to restore session with context: {e}", exc_info=True)
            return None
    
    async def share_context_between_sessions(
        self,
        source_session_id: str,
        target_session_id: str,
        context_types: Optional[List[str]] = None
    ) -> bool:
        """
        세션 간 컨텍스트 공유.
        
        Args:
            source_session_id: 소스 세션 ID
            target_session_id: 타겟 세션 ID
            context_types: 공유할 컨텍스트 타입 (None이면 모두)
            
        Returns:
            성공 여부
        """
        try:
            # 소스 세션 로드
            source_state = self.session_manager.load_session(source_session_id)
            if not source_state:
                logger.warning(f"Source session not found: {source_session_id}")
                return False
            
            # 컨텍스트 필터링
            source_chunks = source_state.context_chunks
            if context_types:
                source_chunks = [
                    chunk for chunk in source_chunks
                    if chunk.get('content_type') in context_types
                ]
            
            # 타겟 세션의 컨텍스트에 추가
            for chunk_data in source_chunks:
                from src.core.context_engineer import ContextChunk, ContextType, ContextPriority
                
                chunk = ContextChunk(
                    content=chunk_data['content'],
                    content_type=ContextType(chunk_data['content_type']),
                    priority=ContextPriority(chunk_data['priority']),
                    timestamp=chunk_data.get('timestamp', time.time()),
                    metadata=chunk_data.get('metadata', {}),
                    dependencies=set(chunk_data.get('dependencies', []))
                )
                chunk.token_count = chunk_data.get('token_count', chunk._estimate_tokens())
                
                self.context_engineer.context_chunks.append(chunk)
            
            # 지식 전이
            self.adaptive_memory.transfer_knowledge(
                source_session_id=source_session_id,
                target_session_id=target_session_id
            )
            
            logger.info(f"Context shared: {len(source_chunks)} chunks from {source_session_id} to {target_session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to share context between sessions: {e}", exc_info=True)
            return False
    
    async def _reconstruct_context(self, session_id: str):
        """컨텍스트 자동 재구성."""
        try:
            import time
            from src.core.context_engineer import ContextChunk, ContextType, ContextPriority
            
            # 세션 메모리에서 컨텍스트 관련 정보 로드
            session_memories = self.shared_memory.list_session_memories(session_id)
            
            # 중요 정보를 컨텍스트 청크로 변환
            for key, memory_entry in session_memories.items():
                if key == "agents":
                    continue
                
                if isinstance(memory_entry, dict):
                    value = memory_entry.get("value")
                    if isinstance(value, str) and len(value) > 50:  # 충분한 내용이 있는 경우만
                        chunk = ContextChunk(
                            content=value[:500],  # 처음 500자
                            content_type=ContextType.AGENT_MEMORY,
                            priority=ContextPriority.MEDIUM,
                            timestamp=time.time(),
                            metadata={'source': 'session_memory', 'key': key}
                        )
                        self.context_engineer.context_chunks.append(chunk)
            
            logger.debug(f"Context reconstructed for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to reconstruct context: {e}")


# 전역 인스턴스
_session_context_bridge: Optional[SessionContextBridge] = None


def get_session_context_bridge() -> SessionContextBridge:
    """전역 세션-컨텍스트 브리지 인스턴스 반환."""
    global _session_context_bridge
    if _session_context_bridge is None:
        _session_context_bridge = SessionContextBridge()
    return _session_context_bridge

