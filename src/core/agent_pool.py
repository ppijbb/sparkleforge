#!/usr/bin/env python3
"""
Agent Pool System for Parallel Agent Execution

재사용 가능한 agent 인스턴스 풀 관리 시스템.
병렬 실행을 위해 agent를 효율적으로 관리하고 재사용합니다.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class AgentPoolItem:
    """Agent 풀 아이템."""
    agent_id: str
    agent_type: str
    agent_instance: Any
    created_at: datetime
    last_used_at: Optional[datetime] = None
    usage_count: int = 0
    is_busy: bool = False
    
    def mark_used(self):
        """사용 표시."""
        self.last_used_at = datetime.now()
        self.usage_count += 1
        self.is_busy = True
    
    def mark_available(self):
        """사용 가능 표시."""
        self.is_busy = False


class AgentPool:
    """Agent 인스턴스 풀 관리 시스템."""
    
    def __init__(self, max_pool_size: int = 10):
        """초기화."""
        self.max_pool_size = max_pool_size
        self.agents: Dict[str, Dict[str, AgentPoolItem]] = defaultdict(dict)  # {agent_type: {agent_id: AgentPoolItem}}
        self.available_agents: Dict[str, List[str]] = defaultdict(list)  # {agent_type: [agent_id, ...]}
        self.busy_agents: Dict[str, Set[str]] = defaultdict(set)  # {agent_type: {agent_id, ...}}
        self._lock = asyncio.Lock()
        
        logger.info(f"AgentPool initialized with max_pool_size={max_pool_size}")
    
    async def get_agent(self, agent_type: str, agent_factory: Optional[callable] = None) -> Optional[Any]:
        """사용 가능한 agent 가져오기 또는 생성."""
        async with self._lock:
            # 사용 가능한 agent 확인
            if agent_type in self.available_agents and self.available_agents[agent_type]:
                agent_id = self.available_agents[agent_type].pop(0)
                agent_item = self.agents[agent_type][agent_id]
                agent_item.mark_used()
                self.busy_agents[agent_type].add(agent_id)
                logger.debug(f"Reusing agent {agent_id} of type {agent_type}")
                return agent_item.agent_instance
            
            # 풀 크기 확인
            total_agents = len(self.agents.get(agent_type, {}))
            if total_agents >= self.max_pool_size:
                logger.warning(f"Agent pool for {agent_type} is full ({total_agents}/{self.max_pool_size})")
                return None
            
            # 새 agent 생성
            if agent_factory:
                try:
                    agent_instance = await self._create_agent(agent_factory, agent_type)
                    agent_id = str(uuid.uuid4())
                    
                    agent_item = AgentPoolItem(
                        agent_id=agent_id,
                        agent_type=agent_type,
                        agent_instance=agent_instance,
                        created_at=datetime.now()
                    )
                    agent_item.mark_used()
                    
                    self.agents[agent_type][agent_id] = agent_item
                    self.busy_agents[agent_type].add(agent_id)
                    
                    logger.info(f"Created new agent {agent_id} of type {agent_type}")
                    return agent_instance
                except Exception as e:
                    logger.error(f"Failed to create agent of type {agent_type}: {e}")
                    return None
            else:
                logger.warning(f"No agent factory provided for type {agent_type}")
                return None
    
    async def _create_agent(self, agent_factory: callable, agent_type: str) -> Any:
        """Agent 생성 (비동기 지원)."""
        if asyncio.iscoroutinefunction(agent_factory):
            return await agent_factory(agent_type)
        else:
            return agent_factory(agent_type)
    
    async def return_agent(self, agent_type: str, agent_instance: Any) -> bool:
        """Agent 반환."""
        async with self._lock:
            # 해당 agent 찾기
            for agent_id, agent_item in self.agents.get(agent_type, {}).items():
                if agent_item.agent_instance == agent_instance:
                    agent_item.mark_available()
                    if agent_id in self.busy_agents[agent_type]:
                        self.busy_agents[agent_type].remove(agent_id)
                    self.available_agents[agent_type].append(agent_id)
                    logger.debug(f"Returned agent {agent_id} of type {agent_type} to pool")
                    return True
            
            logger.warning(f"Agent instance not found in pool for type {agent_type}")
            return False
    
    async def create_agent_pool(self, agent_types: List[str], pool_size_per_type: int = 3, agent_factory: Optional[callable] = None) -> Dict[str, int]:
        """Agent 풀 사전 생성."""
        created_counts = {}
        
        for agent_type in agent_types:
            created = 0
            for _ in range(pool_size_per_type):
                agent = await self.get_agent(agent_type, agent_factory)
                if agent:
                    created += 1
                    # 즉시 반환하여 풀에 추가
                    await self.return_agent(agent_type, agent)
            
            created_counts[agent_type] = created
            logger.info(f"Pre-created {created} agents of type {agent_type}")
        
        return created_counts
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """풀 통계 반환."""
        stats = {
            'total_agent_types': len(self.agents),
            'agent_types': {}
        }
        
        for agent_type, agents in self.agents.items():
            total = len(agents)
            available = len(self.available_agents.get(agent_type, []))
            busy = len(self.busy_agents.get(agent_type, set()))
            
            stats['agent_types'][agent_type] = {
                'total': total,
                'available': available,
                'busy': busy,
                'utilization': (busy / total * 100) if total > 0 else 0.0
            }
        
        return stats
    
    async def cleanup(self) -> None:
        """풀 정리."""
        async with self._lock:
            total_cleaned = 0
            for agent_type in list(self.agents.keys()):
                # 사용 가능한 agent는 유지, 사용 중인 agent는 대기
                available = self.available_agents.get(agent_type, [])
                if available:
                    # 사용 가능한 agent 제거 (선택적)
                    for agent_id in available:
                        if agent_id in self.agents[agent_type]:
                            del self.agents[agent_type][agent_id]
                            total_cleaned += 1
                    
                    self.available_agents[agent_type] = []
            
            logger.info(f"Cleaned up {total_cleaned} agents from pool")

