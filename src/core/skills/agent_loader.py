"""
YAML 기반 Agent 설정 로더

각 agent의 설정(프롬프트, 능력, 지시사항 등)을 YAML 파일에서 로드하고
프롬프트 템플릿 렌더링을 제공하는 모듈
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from string import Template


@dataclass
class AgentConfig:
    """Agent 설정 데이터 클래스"""
    name: str
    display_name: str
    description: str
    capabilities: List[str]
    instructions: str
    prompts: Dict[str, Any]
    configuration: Dict[str, Any]
    tools: Dict[str, List[str]]


class AgentLoader:
    """YAML 기반 Agent 설정 로더"""

    def __init__(self):
        self._configs: Dict[str, AgentConfig] = {}
        self._agents_dir = Path(__file__).parent / "agents"

    def load_agent_config(self, agent_name: str) -> AgentConfig:
        """Agent YAML 파일 로드"""
        if agent_name in self._configs:
            return self._configs[agent_name]

        yaml_file = self._agents_dir / f"{agent_name}.yaml"
        if not yaml_file.exists():
            raise FileNotFoundError(f"Agent config file not found: {yaml_file}")

        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        config = AgentConfig(
            name=data['agent']['name'],
            display_name=data['agent']['display_name'],
            description=data['agent']['description'],
            capabilities=data.get('capabilities', []),
            instructions=data.get('instructions', ''),
            prompts=data.get('prompts', {}),
            configuration=data.get('configuration', {}),
            tools=data.get('tools', {'required': [], 'optional': []})
        )

        self._configs[agent_name] = config
        return config

    def get_prompt(self, agent_name: str, prompt_name: str, **kwargs) -> str:
        """프롬프트 템플릿 렌더링"""
        config = self.load_agent_config(agent_name)

        if prompt_name not in config.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found for agent '{agent_name}'")

        prompt_config = config.prompts[prompt_name]
        template_str = prompt_config['template']

        # 템플릿 변수 치환
        template = Template(template_str)
        try:
            rendered = template.safe_substitute(**kwargs)
            return rendered
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {e}")

    def get_system_message(self, agent_name: str, prompt_name: str) -> str:
        """시스템 메시지 반환"""
        config = self.load_agent_config(agent_name)

        if prompt_name not in config.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found for agent '{agent_name}'")

        return config.prompts[prompt_name].get('system_message', '')

    def get_agent_capabilities(self, agent_name: str) -> List[str]:
        """Agent 능력 목록 반환"""
        config = self.load_agent_config(agent_name)
        return config.capabilities

    def get_agent_instructions(self, agent_name: str) -> str:
        """Agent 지시사항 반환"""
        config = self.load_agent_config(agent_name)
        return config.instructions

    def get_agent_configuration(self, agent_name: str) -> Dict[str, Any]:
        """Agent 설정 반환"""
        config = self.load_agent_config(agent_name)
        return config.configuration

    def get_required_tools(self, agent_name: str) -> List[str]:
        """필수 도구 목록 반환"""
        config = self.load_agent_config(agent_name)
        return config.tools.get('required', [])

    def get_optional_tools(self, agent_name: str) -> List[str]:
        """선택적 도구 목록 반환"""
        config = self.load_agent_config(agent_name)
        return config.tools.get('optional', [])


# 전역 인스턴스
_loader = None

def get_agent_loader() -> AgentLoader:
    """전역 Agent 로더 인스턴스 반환"""
    global _loader
    if _loader is None:
        _loader = AgentLoader()
    return _loader

def load_agent_config(agent_name: str) -> AgentConfig:
    """편의 함수: Agent 설정 로드"""
    return get_agent_loader().load_agent_config(agent_name)

def get_prompt(agent_name: str, prompt_name: str, **kwargs) -> str:
    """편의 함수: 프롬프트 템플릿 렌더링"""
    return get_agent_loader().get_prompt(agent_name, prompt_name, **kwargs)

def get_system_message(agent_name: str, prompt_name: str) -> str:
    """편의 함수: 시스템 메시지 반환"""
    return get_agent_loader().get_system_message(agent_name, prompt_name)

def get_agent_capabilities(agent_name: str) -> List[str]:
    """편의 함수: Agent 능력 목록 반환"""
    return get_agent_loader().get_agent_capabilities(agent_name)

def get_agent_instructions(agent_name: str) -> str:
    """편의 함수: Agent 지시사항 반환"""
    return get_agent_loader().get_agent_instructions(agent_name)