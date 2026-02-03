"""
CLI Agents Module

다양한 CLI 기반 에이전트들을 통합 관리하는 모듈
"""

from .base_cli_agent import BaseCLIAgent, CLIAgentConfig, CLIExecutionResult
from .cli_agent_manager import CLIAgentManager, get_cli_agent_manager
from .claude_code_agent import ClaudeCodeAgent
from .open_code_agent import OpenCodeAgent
from .gemini_cli_agent import GeminiCLIAgent
from .cline_cli_agent import ClineCLIAgent

__all__ = [
    'BaseCLIAgent',
    'CLIAgentConfig',
    'CLIExecutionResult',
    'CLIAgentManager',
    'get_cli_agent_manager',
    'ClaudeCodeAgent',
    'OpenCodeAgent',
    'GeminiCLIAgent',
    'ClineCLIAgent'
]