"""Forge Master Session Manager - Multi-Session & Sub-Agent Lifecycle Engine

24/7 연속 구동을 위한 지속적 인터랙티브 멀티 세션(Multi-Session)과
단발성 서브 에이전트(Sub-Agent) 실행 생명주기 및 문맥 보존 관리자
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.cli_agents.cli_agent_manager import get_cli_agent_manager

logger = logging.getLogger(__name__)


@dataclass
class AgentSession:
    """영속적 또는 단발성 CLI 에이전트 세션 정보"""

    session_id: str
    agent_name: str
    is_persistent: bool
    created_at: float
    last_active_at: float
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ForgeMasterSessionManager:
    """Forge Master 멀티 세션 및 서브 에이전트 생성/관리자"""

    def __init__(self, ttl_seconds: int = 86400):  # 24시간 기본 TTL
        self.sessions: Dict[str, AgentSession] = {}
        self.cli_manager = get_cli_agent_manager()
        self.ttl_seconds = ttl_seconds

    def create_session(
        self, agent_name: str, is_persistent: bool = False, metadata: Optional[Dict[str, Any]] = None
    ) -> AgentSession:
        """신규 에이전트 세션 생성

        Args:
            agent_name: CLI 에이전트 이름 (codex, claude_code 등)
            is_persistent: 24/7 멀티 세션 지속 여부
            metadata: 세션 메타데이터

        Returns:
            생성된 AgentSession
        """
        session_id = f"fmsess_{uuid.uuid4().hex[:12]}"
        now = time.time()

        session = AgentSession(
            session_id=session_id,
            agent_name=agent_name,
            is_persistent=is_persistent,
            created_at=now,
            last_active_at=now,
            metadata=metadata or {},
        )
        self.sessions[session_id] = session
        logger.info(
            f"Created ForgeMaster Session {session_id} for agent '{agent_name}' (persistent={is_persistent})"
        )
        return session

    async def execute_in_session(
        self,
        session_id: str,
        query: str,
        compact_context: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """세션 맥락 내에서 쿼리 실행 및 히스토리 기록

        Args:
            session_id: 세션 ID
            query: 실행할 쿼리
            compact_context: 압축된 전달 문맥
            **kwargs: 추가 옵션

        Returns:
            실행 결과 딕셔너리
        """
        session = self.sessions.get(session_id)
        if not session:
            return {
                "success": False,
                "error": f"Session {session_id} not found",
                "response": "",
            }

        session.last_active_at = time.time()

        # 지속적 멀티 세션인 경우 누적 대화 문맥 추가
        combined_context = compact_context or ""
        if session.is_persistent and session.history:
            prev_summary = f"[Persistent Session Context: {len(session.history)} previous turns]"
            combined_context = f"{prev_summary}\n{combined_context}".strip()

        # CLI 에이전트 실행
        result = await self.cli_manager.execute_with_agent(
            agent_name=session.agent_name,
            query=query,
            context=combined_context,
            **kwargs,
        )

        # 세션 대화 히스토리 업데이트
        session.history.append(
            {
                "timestamp": time.time(),
                "query": query,
                "success": result.get("success", False),
                "response_summary": result.get("response", "")[:200],
            }
        )

        return result

    async def execute_subagent_ephemeral(
        self, agent_name: str, query: str, compact_context: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """단발성 일회성 서브 에이전트(Ephemeral Sub-Agent) 구동 및 자동 해제

        Args:
            agent_name: 에이전트 이름
            query: 쿼리
            compact_context: 문맥
            **kwargs: 기타 옵션

        Returns:
            실행 결과
        """
        session = self.create_session(agent_name=agent_name, is_persistent=False)
        try:
            return await self.execute_in_session(
                session_id=session.session_id,
                query=query,
                compact_context=compact_context,
                **kwargs,
            )
        finally:
            self.close_session(session.session_id)

    def close_session(self, session_id: str) -> bool:
        """세션 종료 및 자원 정리"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Closed ForgeMaster session: {session_id}")
            return True
        return False

    def cleanup_expired_sessions(self) -> int:
        """만료된 세션 정리 (24/7 가비지 컬렉션)"""
        now = time.time()
        expired = [
            sid
            for sid, sess in self.sessions.items()
            if not sess.is_persistent and (now - sess.last_active_at > 3600)
        ]
        for sid in expired:
            self.close_session(sid)
        return len(expired)
