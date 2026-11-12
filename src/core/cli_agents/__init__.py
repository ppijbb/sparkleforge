"""CLI Agents Module

다양한 CLI 기반 에이전트들을 통합 관리하는 모듈
"""

from .base_cli_agent import BaseCLIAgent, CLIAgentConfig, CLIExecutionResult
from .claude_code_agent import ClaudeCodeAgent
from .cli_agent_manager import CLIAgentManager, get_cli_agent_manager
from .cline_cli_agent import ClineCLIAgent
from .sparkle_llm_agent import SparkleLLMAgent
from .sparkleforge_cli_agent import SparkleForgeCLIAgent

__all__ = [
    "BaseCLIAgent",
    "CLIAgentConfig",
    "CLIExecutionResult",
    "CLIAgentManager",
    "get_cli_agent_manager",
    "ClaudeCodeAgent",
    "SparkleLLMAgent",
    "SparkleForgeCLIAgent",
    "ClineCLIAgent",
]
