"""
서브 에이전트 설계 및 교환 시스템

각기 다른 컨텍스트와 도구를 부여하고, 서로 요약·교환하도록 설계
트리형 에이전트 네트워크 구조, 서브 에이전트 간 데이터 교환 형식 최적화
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Set, Callable, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class SubAgentRole(Enum):
    """서브 에이전트 역할."""
    COORDINATOR = "coordinator"      # 조정자
    RESEARCHER = "researcher"       # 연구자
    ANALYZER = "analyzer"           # 분석자
    SYNTHESIZER = "synthesizer"     # 종합자
    VALIDATOR = "validator"         # 검증자
    SPECIALIST = "specialist"       # 전문가


class SubAgentStatus(Enum):
    """서브 에이전트 상태."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    WORKING = "working"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class ExchangeFormat(Enum):
    """교환 형식."""
    JSON = "json"
    MARKDOWN = "markdown"
    STRUCTURED_TEXT = "structured_text"
    BINARY = "binary"


@dataclass
class SubAgentConfig:
    """서브 에이전트 설정."""
    role: SubAgentRole
    name: str
    capabilities: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    context_window_size: int = 4000
    specialization_area: Optional[str] = None
    collaboration_rules: Dict[str, Any] = field(default_factory=dict)
    exchange_preferences: List[ExchangeFormat] = field(default_factory=lambda: [ExchangeFormat.JSON])


@dataclass
class SubAgentContext:
    """서브 에이전트 컨텍스트."""
    agent_id: str
    config: SubAgentConfig
    status: SubAgentStatus = SubAgentStatus.INITIALIZING
    current_tasks: List[Dict[str, Any]] = field(default_factory=list)
    completed_tasks: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    collaboration_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    @property
    def active_task_count(self) -> int:
        """활성 태스크 수."""
        return len([t for t in self.current_tasks if not t.get('completed', False)])

    @property
    def success_rate(self) -> float:
        """성공률 계산."""
        if not self.completed_tasks:
            return 0.0

        successful = sum(1 for t in self.completed_tasks if t.get('success', False))
        return successful / len(self.completed_tasks)

    def add_task(self, task: Dict[str, Any]):
        """태스크 추가."""
        task_id = str(uuid.uuid4())
        task.update({
            'task_id': task_id,
            'assigned_at': time.time(),
            'status': 'pending'
        })
        self.current_tasks.append(task)
        self.last_active = time.time()

    def complete_task(self, task_id: str, result: Dict[str, Any]):
        """태스크 완료."""
        for task in self.current_tasks:
            if task.get('task_id') == task_id:
                task.update({
                    'completed_at': time.time(),
                    'status': 'completed',
                    'result': result,
                    'success': result.get('success', False)
                })
                self.completed_tasks.append(task)
                self.current_tasks.remove(task)
                self.last_active = time.time()
                break

    def fail_task(self, task_id: str, error: str):
        """태스크 실패."""
        for task in self.current_tasks:
            if task.get('task_id') == task_id:
                task.update({
                    'completed_at': time.time(),
                    'status': 'failed',
                    'error': error,
                    'success': False
                })
                self.completed_tasks.append(task)
                self.current_tasks.remove(task)
                self.last_active = time.time()
                break


@dataclass
class ExchangeMessage:
    """교환 메시지."""
    message_id: str
    from_agent: str
    to_agent: str
    message_type: str  # "request", "response", "broadcast", "collaboration"
    content: Dict[str, Any]
    format: ExchangeFormat
    timestamp: float
    priority: int = 1  # 1-5, 5가 가장 높음
    metadata: Dict[str, Any] = field(default_factory=dict)
    requires_response: bool = False
    ttl: Optional[float] = None  # Time to live (seconds)

    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        self.timestamp = self.timestamp or time.time()

    @property
    def is_expired(self) -> bool:
        """만료 여부 확인."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            'message_id': self.message_id,
            'from_agent': self.from_agent,
            'to_agent': self.to_agent,
            'message_type': self.message_type,
            'content': self.content,
            'format': self.format.value,
            'timestamp': self.timestamp,
            'priority': self.priority,
            'metadata': self.metadata,
            'requires_response': self.requires_response,
            'ttl': self.ttl
        }


@dataclass
class CollaborationNetwork:
    """협업 네트워크."""
    network_id: str
    root_agent: str
    agents: Dict[str, SubAgentContext] = field(default_factory=dict)
    connections: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))  # agent_id -> connected_agents
    message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    active_exchanges: Dict[str, ExchangeMessage] = field(default_factory=dict)
    collaboration_rules: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add_agent(self, agent: SubAgentContext):
        """에이전트 추가."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Agent {agent.agent_id} ({agent.config.role.value}) added to network {self.network_id}")

    def connect_agents(self, agent1_id: str, agent2_id: str, bidirectional: bool = True):
        """에이전트 연결."""
        self.connections[agent1_id].add(agent2_id)
        if bidirectional:
            self.connections[agent2_id].add(agent1_id)

        logger.debug(f"Connected agents: {agent1_id} <-> {agent2_id}")

    def get_connected_agents(self, agent_id: str) -> Set[str]:
        """연결된 에이전트들 반환."""
        return self.connections.get(agent_id, set())

    async def broadcast_message(self, message: ExchangeMessage):
        """메시지 브로드캐스트."""
        connected_agents = self.get_connected_agents(message.from_agent)

        for agent_id in connected_agents:
            if agent_id != message.to_agent:  # 이미 지정된 대상 제외
                broadcast_msg = ExchangeMessage(
                    message_id="",
                    from_agent=message.from_agent,
                    to_agent=agent_id,
                    message_type="broadcast",
                    content=message.content,
                    format=message.format,
                    timestamp=time.time(),
                    priority=message.priority - 1,  # 브로드캐스트는 우선순위 낮춤
                    metadata={**message.metadata, 'original_broadcast': True},
                    requires_response=False
                )
                await self.message_queue.put(broadcast_msg)

    async def send_message(self, message: ExchangeMessage):
        """메시지 전송."""
        await self.message_queue.put(message)

        if message.requires_response:
            self.active_exchanges[message.message_id] = message

        # 브로드캐스트인 경우 연결된 에이전트들에게도 전송
        if message.message_type == "broadcast":
            await self.broadcast_message(message)

        logger.debug(f"Message sent: {message.message_id} from {message.from_agent} to {message.to_agent}")

    async def receive_message(self, agent_id: str) -> Optional[ExchangeMessage]:
        """메시지 수신."""
        try:
            # 큐에서 메시지 가져오기 (타임아웃 1초)
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)

            # 이 에이전트에게 온 메시지인지 확인
            if message.to_agent == agent_id or message.to_agent == "all":
                return message
            else:
                # 다른 에이전트의 메시지면 다시 큐에 넣음
                await self.message_queue.put(message)
                return None

        except asyncio.TimeoutError:
            return None


class SubAgentManager:
    """
    서브 에이전트 설계 및 교환 시스템.

    각기 다른 컨텍스트와 도구를 부여하고, 서로 요약·교환하도록 설계.
    트리형 에이전트 네트워크 구조 구현.
    """

    def __init__(self):
        """초기화."""
        self.networks: Dict[str, CollaborationNetwork] = {}
        self.agent_registry: Dict[str, SubAgentContext] = {}
        self.exchange_handlers: Dict[str, Callable] = {}

        # 기본 교환 핸들러 등록
        self._register_default_handlers()

        logger.info("SubAgentManager initialized")

    def _register_default_handlers(self):
        """기본 교환 핸들러 등록."""
        self.exchange_handlers.update({
            'task_delegation': self._handle_task_delegation,
            'knowledge_sharing': self._handle_knowledge_sharing,
            'result_synthesis': self._handle_result_synthesis,
            'validation_request': self._handle_validation_request,
            'collaboration_proposal': self._handle_collaboration_proposal,
        })

    async def create_network(
        self,
        network_id: str,
        root_agent_config: SubAgentConfig
    ) -> CollaborationNetwork:
        """
        협업 네트워크 생성.

        Args:
            network_id: 네트워크 ID
            root_agent_config: 루트 에이전트 설정

        Returns:
            생성된 네트워크
        """
        network = CollaborationNetwork(
            network_id=network_id,
            root_agent=root_agent_config.name
        )

        # 루트 에이전트 생성 및 추가
        root_agent = SubAgentContext(
            agent_id=root_agent_config.name,
            config=root_agent_config,
            status=SubAgentStatus.ACTIVE
        )

        network.add_agent(root_agent)
        self.agent_registry[root_agent_config.name] = root_agent
        self.networks[network_id] = network

        logger.info(f"Created collaboration network: {network_id} with root agent {root_agent_config.name}")
        return network

    async def add_sub_agent(
        self,
        network_id: str,
        agent_config: SubAgentConfig,
        parent_agent_id: str
    ) -> SubAgentContext:
        """
        서브 에이전트 추가.

        Args:
            network_id: 네트워크 ID
            agent_config: 에이전트 설정
            parent_agent_id: 부모 에이전트 ID

        Returns:
            생성된 서브 에이전트
        """
        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")

        network = self.networks[network_id]

        # 에이전트 생성
        agent = SubAgentContext(
            agent_id=agent_config.name,
            config=agent_config,
            status=SubAgentStatus.ACTIVE
        )

        # 네트워크에 추가 및 연결
        network.add_agent(agent)
        network.connect_agents(parent_agent_id, agent_config.name)

        # 레지스트리에 등록
        self.agent_registry[agent_config.name] = agent

        logger.info(f"Added sub-agent {agent_config.name} ({agent_config.role.value}) "
                   f"to network {network_id}, connected to {parent_agent_id}")

        return agent

    async def delegate_task(
        self,
        network_id: str,
        from_agent: str,
        to_agent: str,
        task: Dict[str, Any]
    ) -> bool:
        """
        태스크 위임.

        Args:
            network_id: 네트워크 ID
            from_agent: 보내는 에이전트
            to_agent: 받는 에이전트
            task: 위임할 태스크

        Returns:
            위임 성공 여부
        """
        if network_id not in self.networks:
            return False

        network = self.networks[network_id]
        if to_agent not in network.agents:
            return False

        # 태스크 위임 메시지 생성
        message = ExchangeMessage(
            message_id="",
            from_agent=from_agent,
            to_agent=to_agent,
            message_type="request",
            content={
                'action': 'task_delegation',
                'task': task
            },
            format=ExchangeFormat.JSON,
            timestamp=time.time(),
            priority=3,
            requires_response=True,
            ttl=300  # 5분 TTL
        )

        await network.send_message(message)

        # 대상 에이전트의 태스크에 추가
        target_agent = network.agents[to_agent]
        target_agent.add_task(task)

        logger.info(f"Delegated task from {from_agent} to {to_agent} in network {network_id}")
        return True

    async def share_knowledge(
        self,
        network_id: str,
        from_agent: str,
        knowledge: Dict[str, Any],
        target_agents: Optional[List[str]] = None
    ):
        """
        지식 공유.

        Args:
            network_id: 네트워크 ID
            from_agent: 공유하는 에이전트
            knowledge: 공유할 지식
            target_agents: 대상 에이전트들 (None이면 브로드캐스트)
        """
        if network_id not in self.networks:
            return

        network = self.networks[network_id]

        if target_agents:
            # 특정 에이전트들에게 전송
            for target_agent in target_agents:
                if target_agent in network.agents:
                    message = ExchangeMessage(
                        message_id="",
                        from_agent=from_agent,
                        to_agent=target_agent,
                        message_type="request",
                        content={
                            'action': 'knowledge_sharing',
                            'knowledge': knowledge
                        },
                        format=ExchangeFormat.JSON,
                        timestamp=time.time(),
                        priority=2
                    )
                    await network.send_message(message)
        else:
            # 브로드캐스트
            message = ExchangeMessage(
                message_id="",
                from_agent=from_agent,
                to_agent="all",
                message_type="broadcast",
                content={
                    'action': 'knowledge_sharing',
                    'knowledge': knowledge
                },
                format=ExchangeFormat.JSON,
                timestamp=time.time(),
                priority=2
            )
            await network.send_message(message)

        logger.info(f"Shared knowledge from {from_agent} in network {network_id}")

    async def request_collaboration(
        self,
        network_id: str,
        requesting_agent: str,
        collaboration_type: str,
        requirements: Dict[str, Any]
    ) -> List[str]:
        """
        협업 요청.

        Args:
            network_id: 네트워크 ID
            requesting_agent: 요청하는 에이전트
            collaboration_type: 협업 유형
            requirements: 요구사항

        Returns:
            참여 의사 표시한 에이전트들
        """
        if network_id not in self.networks:
            return []

        network = self.networks[network_id]

        # 협업 제안 메시지 브로드캐스트
        message = ExchangeMessage(
            message_id="",
            from_agent=requesting_agent,
            to_agent="all",
            message_type="broadcast",
            content={
                'action': 'collaboration_proposal',
                'collaboration_type': collaboration_type,
                'requirements': requirements
            },
            format=ExchangeFormat.JSON,
            timestamp=time.time(),
            priority=4,
            requires_response=True,
            ttl=120  # 2분 TTL
        )

        await network.send_message(message)

        # 응답 대기 (간단한 구현)
        interested_agents = []
        timeout = time.time() + 10  # 10초 대기

        while time.time() < timeout:
            try:
                response_msg = await asyncio.wait_for(network.receive_message(requesting_agent), timeout=1.0)
                if response_msg and response_msg.message_type == "response":
                    content = response_msg.content
                    if content.get('action') == 'collaboration_acceptance':
                        interested_agents.append(response_msg.from_agent)
            except asyncio.TimeoutError:
                break

        logger.info(f"Collaboration request from {requesting_agent}: {len(interested_agents)} agents interested")
        return interested_agents

    async def process_exchanges(self, network_id: str):
        """
        교환 메시지 처리.

        Args:
            network_id: 네트워크 ID
        """
        if network_id not in self.networks:
            return

        network = self.networks[network_id]

        # 각 에이전트의 메시지 큐 처리
        for agent_id, agent in network.agents.items():
            if agent.status != SubAgentStatus.ACTIVE:
                continue

            try:
                message = await network.receive_message(agent_id)
                if message:
                    await self._process_message(network, agent, message)
            except Exception as e:
                logger.error(f"Error processing message for agent {agent_id}: {e}")

    async def _process_message(
        self,
        network: CollaborationNetwork,
        agent: SubAgentContext,
        message: ExchangeMessage
    ):
        """메시지 처리."""
        try:
            content = message.content
            action = content.get('action')

            if action in self.exchange_handlers:
                handler = self.exchange_handlers[action]
                response = await handler(network, agent, message)

                # 응답이 필요한 경우
                if message.requires_response and response:
                    response_msg = ExchangeMessage(
                        message_id="",
                        from_agent=agent.agent_id,
                        to_agent=message.from_agent,
                        message_type="response",
                        content=response,
                        format=message.format,
                        timestamp=time.time(),
                        priority=message.priority
                    )
                    await network.send_message(response_msg)

            # 협업 히스토리 기록
            agent.collaboration_history.append({
                'timestamp': time.time(),
                'message_id': message.message_id,
                'from_agent': message.from_agent,
                'action': action,
                'processed': True
            })

        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")

    # 교환 핸들러들
    async def _handle_task_delegation(
        self,
        network: CollaborationNetwork,
        agent: SubAgentContext,
        message: ExchangeMessage
    ) -> Dict[str, Any]:
        """태스크 위임 처리."""
        task = message.content.get('task', {})

        # 에이전트의 전문 분야와 비교
        agent_expertise = agent.config.specialization_area or ""
        task_domain = task.get('domain', '')

        # 수락 여부 결정 (간단한 로직)
        accept = agent_expertise.lower() in task_domain.lower() or not agent_expertise

        if accept:
            agent.add_task(task)
            logger.info(f"Agent {agent.agent_id} accepted task delegation: {task.get('task_id', 'unknown')}")
            return {
                'action': 'task_accepted',
                'task_id': task.get('task_id'),
                'estimated_completion': time.time() + 300  # 5분 예상
            }
        else:
            logger.info(f"Agent {agent.agent_id} declined task delegation: expertise mismatch")
            return {
                'action': 'task_declined',
                'reason': 'expertise_mismatch'
            }

    async def _handle_knowledge_sharing(
        self,
        network: CollaborationNetwork,
        agent: SubAgentContext,
        message: ExchangeMessage
    ) -> Dict[str, Any]:
        """지식 공유 처리."""
        knowledge = message.content.get('knowledge', {})

        # 지식 기반 업데이트
        for key, value in knowledge.items():
            if key not in agent.knowledge_base:
                agent.knowledge_base[key] = []
            agent.knowledge_base[key].append({
                'value': value,
                'source': message.from_agent,
                'timestamp': message.timestamp
            })

        logger.info(f"Agent {agent.agent_id} received knowledge sharing from {message.from_agent}")
        return {'action': 'knowledge_received', 'items_count': len(knowledge)}

    async def _handle_result_synthesis(
        self,
        network: CollaborationNetwork,
        agent: SubAgentContext,
        message: ExchangeMessage
    ) -> Dict[str, Any]:
        """결과 종합 처리."""
        results = message.content.get('results', [])

        # 결과 종합 로직 (간단한 구현)
        synthesized = {
            'total_results': len(results),
            'categories': defaultdict(int),
            'confidence_avg': 0.0
        }

        for result in results:
            category = result.get('category', 'unknown')
            synthesized['categories'][category] += 1
            synthesized['confidence_avg'] += result.get('confidence', 0.0)

        if results:
            synthesized['confidence_avg'] /= len(results)

        logger.info(f"Agent {agent.agent_id} synthesized {len(results)} results")
        return {'action': 'synthesis_completed', 'result': synthesized}

    async def _handle_validation_request(
        self,
        network: CollaborationNetwork,
        agent: SubAgentContext,
        message: ExchangeMessage
    ) -> Dict[str, Any]:
        """검증 요청 처리."""
        data_to_validate = message.content.get('data', {})

        # 검증 로직 (간단한 구현)
        validation_result = {
            'is_valid': True,
            'checks_performed': ['format_check', 'consistency_check'],
            'confidence': 0.85
        }

        logger.info(f"Agent {agent.agent_id} validated data from {message.from_agent}")
        return {'action': 'validation_completed', 'result': validation_result}

    async def _handle_collaboration_proposal(
        self,
        network: CollaborationNetwork,
        agent: SubAgentContext,
        message: ExchangeMessage
    ) -> Dict[str, Any]:
        """협업 제안 처리."""
        collaboration_type = message.content.get('collaboration_type', '')
        requirements = message.content.get('requirements', {})

        # 참여 의사 결정 (역할 기반)
        interested = False
        if collaboration_type == 'research' and agent.config.role in [SubAgentRole.RESEARCHER, SubAgentRole.SPECIALIST]:
            interested = True
        elif collaboration_type == 'analysis' and agent.config.role == SubAgentRole.ANALYZER:
            interested = True
        elif collaboration_type == 'validation' and agent.config.role == SubAgentRole.VALIDATOR:
            interested = True

        if interested:
            logger.info(f"Agent {agent.agent_id} expressed interest in collaboration: {collaboration_type}")
            return {
                'action': 'collaboration_acceptance',
                'collaboration_type': collaboration_type,
                'capabilities': agent.config.capabilities
            }
        else:
            return {'action': 'collaboration_declined', 'reason': 'not_interested'}

    def get_network_stats(self, network_id: str) -> Dict[str, Any]:
        """네트워크 통계."""
        if network_id not in self.networks:
            return {}

        network = self.networks[network_id]

        stats = {
            'network_id': network_id,
            'total_agents': len(network.agents),
            'connections': sum(len(conns) for conns in network.connections.values()) // 2,  # 양방향이므로 2로 나눔
            'active_exchanges': len(network.active_exchanges),
            'agent_roles': defaultdict(int),
            'agent_statuses': defaultdict(int)
        }

        for agent in network.agents.values():
            stats['agent_roles'][agent.config.role.value] += 1
            stats['agent_statuses'][agent.status.value] += 1

        return dict(stats)

    def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """에이전트 성능 통계."""
        if agent_id not in self.agent_registry:
            return {}

        agent = self.agent_registry[agent_id]

        return {
            'agent_id': agent_id,
            'role': agent.config.role.value,
            'status': agent.status.value,
            'success_rate': agent.success_rate,
            'total_tasks': len(agent.completed_tasks),
            'active_tasks': agent.active_task_count,
            'knowledge_items': len(agent.knowledge_base),
            'collaborations': len(agent.collaboration_history),
            'uptime': time.time() - agent.created_at
        }


# 전역 서브 에이전트 매니저 인스턴스
_sub_agent_manager = None

def get_sub_agent_manager() -> SubAgentManager:
    """전역 서브 에이전트 매니저 인스턴스 반환."""
    global _sub_agent_manager
    if _sub_agent_manager is None:
        _sub_agent_manager = SubAgentManager()
    return _sub_agent_manager

def set_sub_agent_manager(manager: SubAgentManager):
    """전역 서브 에이전트 매니저 설정."""
    global _sub_agent_manager
    _sub_agent_manager = manager
