"""
CLI Agent Manager - CLI 기반 에이전트들의 중앙 관리 시스템

다양한 CLI 에이전트들을 통합 관리하고, LLM Manager와의 인터페이스를 제공
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Type
from .base_cli_agent import BaseCLIAgent, CLIAgentConfig
from .claude_code_agent import ClaudeCodeAgent
from .open_code_agent import OpenCodeAgent
from .gemini_cli_agent import GeminiCLIAgent
from .cline_cli_agent import ClineCLIAgent

logger = logging.getLogger(__name__)


class CLIAgentManager:
    """
    CLI 에이전트 관리자

    사용 가능한 CLI 에이전트들을 등록하고 관리하며,
    LLM Manager와의 인터페이스를 제공
    """

    def __init__(self):
        self.agents: Dict[str, BaseCLIAgent] = {}
        self.agent_configs: Dict[str, Dict[str, Any]] = {}

        # 기본 에이전트 등록
        self._register_default_agents()

    def _register_default_agents(self):
        """기본 CLI 에이전트들을 등록"""
        default_agents = {
            'claude_code': ClaudeCodeAgent,
            'open_code': OpenCodeAgent,
            'gemini_cli': GeminiCLIAgent,
            'cline_cli': ClineCLIAgent
        }

        for agent_name, agent_class in default_agents.items():
            self.register_agent_class(agent_name, agent_class)

    def register_agent_class(self, name: str, agent_class: Type[BaseCLIAgent],
                           config: Optional[Dict[str, Any]] = None):
        """
        CLI 에이전트 클래스를 등록

        Args:
            name: 에이전트 이름
            agent_class: 에이전트 클래스
            config: 에이전트 설정
        """
        self.agent_configs[name] = config or {}
        logger.info(f"Registered CLI agent class: {name}")

    def register_agent_instance(self, name: str, agent: BaseCLIAgent):
        """
        CLI 에이전트 인스턴스를 등록

        Args:
            name: 에이전트 이름
            agent: 에이전트 인스턴스
        """
        self.agents[name] = agent
        logger.info(f"Registered CLI agent instance: {name}")

    def create_agent(self, name: str, **kwargs) -> Optional[BaseCLIAgent]:
        """
        CLI 에이전트 인스턴스 생성

        Args:
            name: 에이전트 이름
            **kwargs: 에이전트 생성 파라미터

        Returns:
            생성된 에이전트 인스턴스 또는 None
        """
        # 이미 인스턴스가 있는 경우 반환
        if name in self.agents:
            return self.agents[name]

        # 설정에서 클래스 찾기
        agent_class = self._get_agent_class(name)
        if not agent_class:
            logger.error(f"Unknown CLI agent: {name}")
            return None

        try:
            # 설정과 kwargs 병합
            config = self.agent_configs.get(name, {}).copy()
            config.update(kwargs)

            # 인스턴스 생성
            agent = agent_class(**config)
            self.agents[name] = agent

            logger.info(f"Created CLI agent instance: {name}")
            return agent

        except Exception as e:
            logger.error(f"Failed to create CLI agent {name}: {e}")
            return None

    def _get_agent_class(self, name: str) -> Optional[Type[BaseCLIAgent]]:
        """에이전트 이름으로 클래스 찾기"""
        agent_classes = {
            'claude_code': ClaudeCodeAgent,
            'open_code': OpenCodeAgent,
            'gemini_cli': GeminiCLIAgent,
            'cline_cli': ClineCLIAgent
        }
        return agent_classes.get(name)

    async def execute_with_agent(self, agent_name: str, query: str, **kwargs) -> Dict[str, Any]:
        """
        특정 CLI 에이전트로 쿼리 실행

        Args:
            agent_name: 에이전트 이름
            query: 실행할 쿼리
            **kwargs: 추가 파라미터

        Returns:
            실행 결과
        """
        agent = self.create_agent(agent_name)
        if not agent:
            return {
                'success': False,
                'error': f"CLI agent not available: {agent_name}",
                'response': '',
                'confidence': 0.0
            }

        try:
            result = await agent.execute_query(query, **kwargs)
            result['agent_type'] = 'cli'
            result['agent_name'] = agent_name
            return result

        except Exception as e:
            logger.error(f"CLI agent execution failed: {agent_name} - {e}")
            return {
                'success': False,
                'error': str(e),
                'response': '',
                'confidence': 0.0,
                'agent_type': 'cli',
                'agent_name': agent_name
            }

    async def execute_parallel(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        여러 CLI 에이전트에 대해 병렬 실행

        Args:
            queries: [{'agent': 'name', 'query': 'text', **kwargs}, ...]

        Returns:
            실행 결과 리스트
        """
        tasks = []
        for query_info in queries:
            agent_name = query_info.pop('agent')
            query = query_info.pop('query')
            kwargs = query_info

            task = self.execute_with_agent(agent_name, query, **kwargs)
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_available_agents(self) -> List[str]:
        """사용 가능한 CLI 에이전트 목록 반환"""
        return list(self.agent_configs.keys()) + list(self.agents.keys())

    def get_agent_info(self, name: str) -> Optional[Dict[str, Any]]:
        """특정 에이전트 정보 반환"""
        agent = self.agents.get(name)
        if agent:
            return agent.get_info()

        # 설정에서 정보 찾기
        config = self.agent_configs.get(name)
        if config:
            return {
                'name': name,
                'type': 'cli_agent',
                'configured': True,
                'instance': False
            }

        return None

    async def health_check_all(self) -> Dict[str, bool]:
        """모든 CLI 에이전트 상태 확인"""
        results = {}
        for agent_name in self.get_available_agents():
            agent = self.create_agent(agent_name)
            if agent:
                results[agent_name] = await agent.health_check()
            else:
                results[agent_name] = False
        return results

    def configure_agent(self, name: str, config: Dict[str, Any]):
        """
        CLI 에이전트 설정

        Args:
            name: 에이전트 이름
            config: 설정 딕셔너리
        """
        self.agent_configs[name] = config

        # 기존 인스턴스가 있으면 재생성
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Removed existing CLI agent instance: {name} (will be recreated on next use)")

        logger.info(f"Configured CLI agent: {name}")


# 전역 CLI 에이전트 매니저 인스턴스
_cli_agent_manager = None

def get_cli_agent_manager() -> CLIAgentManager:
    """전역 CLI 에이전트 매니저 인스턴스 가져오기"""
    global _cli_agent_manager
    if _cli_agent_manager is None:
        _cli_agent_manager = CLIAgentManager()
    return _cli_agent_manager