import logging
from enum import Enum
from typing import Dict, Any, List, Optional

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
    1. **Chain of Thought**: Always analyze the request and plan your actions before calling tools.
    2. **Tool Precision**: Use the most specific tool available for each task.
    3. **Ambiguity Resolution**: If a request is unclear, use tools to explore or ask for clarification within your thoughts.
    4. **Result Synthesis**: Concisely summarize tool outputs and relate them back to the user's goal.
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
        """
    }

    @classmethod
    def build_system_prompt(
        cls, 
        identity: AgentIdentity = AgentIdentity.GENERAL,
        additional_instructions: Optional[str] = None
    ) -> str:
        """Constructs a standardized system prompt."""
        identity_text = cls.IDENTITIES.get(identity, "Identity: Versatile Autonomous Agent")
        
        prompt = f"{identity_text}\n\n{cls.BASE_INSTRUCTIONS}"
        
        if additional_instructions:
            prompt += f"\n\nAdditional Instructions:\n{additional_instructions}"
            
        return prompt.strip()

# Utility function for quick access
def get_system_prompt(identity_key: str = "general", extras: str = None) -> str:
    try:
        identity = AgentIdentity(identity_key.lower())
    except ValueError:
        identity = AgentIdentity.GENERAL
        
    return PromptBuilder.build_system_prompt(identity, extras)
