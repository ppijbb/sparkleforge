"""
SparkleForge Agent용 A2A Adapter

sparkleforge/ 폴더의 multi-agent system을 위한 A2A wrapper
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

# 상위 디렉토리의 공통 모듈 import
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.a2a_integration import (
    A2AAdapter,
    A2AMessage,
    MessagePriority,
    get_global_broker,
    get_global_registry,
)
from srcs.common.agent_interface import AgentType

logger = logging.getLogger(__name__)


class SparkleForgeA2AWrapper(A2AAdapter):
    """SparkleForge Multi-Agent System용 A2A Wrapper"""
    
    def __init__(
        self,
        agent_id: str,
        agent_metadata: Dict[str, Any],
        orchestrator: Any = None,
        autonomous_orchestrator: Optional[Any] = None
    ):
        """
        초기화
        
        Args:
            agent_id: Agent ID
            agent_metadata: Agent 메타데이터
            orchestrator: AgentOrchestrator 인스턴스
            autonomous_orchestrator: AutonomousOrchestrator 인스턴스 (선택)
        """
        super().__init__(agent_id, agent_metadata)
        self.orchestrator = orchestrator
        self.autonomous_orchestrator = autonomous_orchestrator
        self._message_processor_task: Optional[asyncio.Task] = None
        self._sub_agents: Dict[str, Any] = {}
    
    async def send_message(
        self,
        target_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: int = MessagePriority.MEDIUM.value,
        correlation_id: Optional[str] = None
    ) -> bool:
        """메시지 전송"""
        message = A2AMessage(
            source_agent=self.agent_id,
            target_agent=target_agent,
            message_type=message_type,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
        )
        
        broker = get_global_broker()
        return await broker.route_message(message)
    
    async def start_listener(self) -> None:
        """메시지 리스너 시작"""
        if self.is_listening:
            logger.warning(f"Listener already started for agent {self.agent_id}")
            return
        
        self.is_listening = True
        self._message_processor_task = asyncio.create_task(self._process_messages())
        logger.info(f"Message listener started for SparkleForge agent {self.agent_id}")
    
    async def stop_listener(self) -> None:
        """메시지 리스너 중지"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        if self._message_processor_task:
            self._message_processor_task.cancel()
            try:
                await self._message_processor_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Message listener stopped for SparkleForge agent {self.agent_id}")
    
    async def _process_messages(self) -> None:
        """메시지 처리 루프"""
        while self.is_listening:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message in SparkleForge agent: {e}")
    
    async def register_capabilities(self, capabilities: List[str]) -> None:
        """Agent 능력 등록"""
        self.agent_metadata["capabilities"] = capabilities
        
        # SparkleForge의 sub-agents도 등록
        if self.orchestrator:
            sub_agent_capabilities = self._extract_sub_agent_capabilities()
            self.agent_metadata["sub_agents"] = sub_agent_capabilities
        
        registry = get_global_registry()
        await registry.register_agent(
            agent_id=self.agent_id,
            agent_type=AgentType.SPARKLEFORGE_AGENT.value,
            metadata=self.agent_metadata,
            a2a_adapter=self,
        )
        logger.info(f"Capabilities registered for SparkleForge agent {self.agent_id}: {capabilities}")
    
    def _extract_sub_agent_capabilities(self) -> Dict[str, List[str]]:
        """SparkleForge의 sub-agents 능력 추출"""
        sub_agents = {}
        
        if self.orchestrator:
            # AgentOrchestrator에서 사용 가능한 agents 확인
            if hasattr(self.orchestrator, "agent_config"):
                agent_config = self.orchestrator.agent_config
                if hasattr(agent_config, "agents"):
                    for agent_name, agent_info in agent_config.agents.items():
                        if isinstance(agent_info, dict):
                            sub_agents[agent_name] = agent_info.get("capabilities", [])
        
        return sub_agents
    
    async def execute_research(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        SparkleForge 연구 실행
        
        Args:
            query: 연구 쿼리
            context: 추가 컨텍스트
            
        Returns:
            실행 결과
        """
        if not self.orchestrator:
            raise ValueError(f"Orchestrator not initialized for agent {self.agent_id}")
        
        context = context or {}
        result = await self.orchestrator.execute(query, context)
        
        return result
    
    async def execute_autonomous(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        AutonomousOrchestrator 실행
        
        Args:
            query: 연구 쿼리
            context: 추가 컨텍스트
            
        Returns:
            실행 결과
        """
        if not self.autonomous_orchestrator:
            raise ValueError(f"AutonomousOrchestrator not initialized for agent {self.agent_id}")
        
        if hasattr(self.autonomous_orchestrator, "astream"):
            # 스트리밍 모드
            context = context or {}
            input_data = {"messages": [{"role": "user", "content": query}], **context}
            return self.autonomous_orchestrator.astream(input_data)
        elif hasattr(self.autonomous_orchestrator, "ainvoke"):
            # 일반 모드
            context = context or {}
            input_data = {"messages": [{"role": "user", "content": query}], **context}
            return await self.autonomous_orchestrator.ainvoke(input_data)
        else:
            raise ValueError(f"AutonomousOrchestrator has no execution method")
    
    def serialize_state(self) -> Dict[str, Any]:
        """상태 직렬화"""
        state = {
            "agent_id": self.agent_id,
            "agent_metadata": self.agent_metadata,
            "is_listening": self.is_listening,
            "message_queue_size": self._message_queue.qsize(),
        }
        
        if self.orchestrator:
            state["orchestrator_available"] = True
            if hasattr(self.orchestrator, "shared_memory"):
                state["shared_memory_size"] = len(getattr(self.orchestrator.shared_memory, "data", {}))
        
        if self.autonomous_orchestrator:
            state["autonomous_orchestrator_available"] = True
        
        if self._sub_agents:
            state["sub_agents"] = list(self._sub_agents.keys())
        
        return state

