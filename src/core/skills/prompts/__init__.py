"""
프롬프트 관리 시스템

하드코딩된 프롬프트들을 구조화된 Python 모듈로 관리하는 시스템입니다.
각 agent의 프롬프트가 별도의 모듈로 분리되어 있어 유지보수가 용이합니다.
"""

from .manager import PromptManager, get_prompt, get_system_message
from .orchestrator import planning, task_decomposition, query_generation, verification, report_generation
from .agents import (
    research_agent_prompts,
    evaluation_agent_prompts,
    synthesis_agent_prompts,
    validation_agent_prompts,
    creativity_agent_prompts,
    task_analyzer_prompts
)
from .core import (
    autonomous_orchestrator_prompts,
    result_sharing_prompts,
    context_engineering_prompts
)
from .shared import system_messages

__all__ = [
    # Manager
    'PromptManager',
    'get_prompt',
    'get_system_message',

    # Orchestrator prompts
    'planning',
    'task_decomposition',
    'query_generation',
    'verification',
    'report_generation',

    # Agent prompts
    'research_agent_prompts',
    'evaluation_agent_prompts',
    'synthesis_agent_prompts',
    'validation_agent_prompts',
    'creativity_agent_prompts',
    'task_analyzer_prompts',

    # Core prompts
    'autonomous_orchestrator_prompts',
    'result_sharing_prompts',
    'context_engineering_prompts',

    # Shared
    'system_messages'
]

