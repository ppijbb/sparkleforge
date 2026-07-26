import logging
from enum import Enum
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class AgentIdentity(Enum):
    RESEARCHER = "researcher"
    CODER = "coder"
    ORCHESTRATOR = "orchestrator"
    GENERAL = "general"


class PromptBuilder:
    """Centralized Prompt Engineering for SparkleForge (Phase 5).

    Standardizes agent identities and enforces best practices (CoT, Tool Guidance).
    """

    BASE_INSTRUCTIONS = """
    You are an autonomous agent capable of using a variety of tools via the Model Context Protocol (MCP).
    Follow these core principles:
    1. **Problem Solving Only**: Keep moving toward a useful deliverable. Do not stop to ask the user for clarification unless continuing would be unsafe, destructive, or impossible.
    2. **Autonomous Assumptions**: When details are ambiguous, make the most conservative useful assumption, state it briefly in the result, and proceed.
    3. **Tool Precision**: Use the most specific tool available for each task, and recover from tool failures by trying another viable path.
    4. **Completion Bias**: Continue tool use, analysis, and synthesis until the task is solved, a validated partial answer is produced, or a hard blocker is reached.
    5. **Result Synthesis**: Concisely summarize tool outputs and relate them back to the user's goal.
    """

    IDENTITIES = {
        AgentIdentity.RESEARCHER: """
        Identity: Expert Research Analyst
        Goal: Conduct deep-dive research, verify sources, and synthesize multi-dimensional insights.
        Specialty: Search, Academic Data, Financial Analysis, and Document Synthesis.
        """,
        AgentIdentity.CODER: """
        Identity: Senior Software Engineer
        Goal: Design, implement, and debug code across various languages and frameworks.
        Specialty: Git, File Systems, Code Analysis, and Problem Solving.
        """,
        AgentIdentity.ORCHESTRATOR: """
        Identity: System Architect & Orchestrator
        Goal: Decompose complex user requests into smaller, actionable tasks and coordinate execution.
        Specialty: Planning, Error Recovery, and Resource Management.
        """,
    }

    @classmethod
    def build_system_prompt(
        cls,
        identity: AgentIdentity = AgentIdentity.GENERAL,
        additional_instructions: str | None = None,
    ) -> str:
        """Constructs a standardized system prompt."""
        identity_text = cls.IDENTITIES.get(identity, "Identity: Versatile Autonomous Agent")

        prompt = f"{identity_text}\n\n{cls.BASE_INSTRUCTIONS}"

        if additional_instructions:
            prompt += f"\n\nAdditional Instructions:\n{additional_instructions}"

        return prompt.strip()


# Utility function for quick access
def get_system_prompt(identity_key: str = "general", extras: str | None = None) -> str:
    try:
        identity = AgentIdentity(identity_key.lower())
    except ValueError:
        identity = AgentIdentity.GENERAL

    return PromptBuilder.build_system_prompt(identity, extras)
