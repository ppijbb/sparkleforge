"""에이전트별 MCP 도구 선택 및 할당 메커니즘."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.tools import BaseTool

from .mcp_tool_loader import ToolInfo


class AgentType(Enum):
    """에이전트 타입 정의."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    GENERATOR = "generator"


class ToolCategory(Enum):
    """도구 카테고리 정의."""
    SEARCH = "search"
    DATA = "data"
    CODE = "code"
    ACADEMIC = "academic"
    BUSINESS = "business"
    UTILITY = "utility"
    PLANNING = "planning"
    VERIFICATION = "verification"
    GENERATION = "generation"


@dataclass
class AgentToolAssignment:
    """에이전트별 도구 할당 결과."""
    agent_type: AgentType
    tools: list[BaseTool]
    tool_infos: list[ToolInfo]
    server_sources: dict[str, list[str]]  # server_name -> [tool_names]


class AgentToolSelector:
    """에이전트별 MCP 도구 선택 및 할당."""

    def __init__(self):
        # 에이전트 타입별 선호 카테고리 매핑
        self.agent_category_preferences: dict[AgentType, list[ToolCategory]] = {
            AgentType.PLANNER: [
                ToolCategory.PLANNING,
                ToolCategory.SEARCH,
                ToolCategory.UTILITY
            ],
            AgentType.EXECUTOR: [
                ToolCategory.SEARCH,
                ToolCategory.DATA,
                ToolCategory.ACADEMIC,
                ToolCategory.BUSINESS,
                ToolCategory.CODE
            ],
            AgentType.VERIFIER: [
                ToolCategory.VERIFICATION,
                ToolCategory.SEARCH,
                ToolCategory.DATA,
                ToolCategory.ACADEMIC
            ],
            AgentType.GENERATOR: [
                ToolCategory.GENERATION,
                ToolCategory.UTILITY,
                ToolCategory.SEARCH
            ]
        }

        # 도구 이름/설명 기반 카테고리 추론 규칙
        self.category_keywords: dict[ToolCategory, list[str]] = {
            ToolCategory.SEARCH: [
                "search", "find", "query", "lookup", "retrieve", "discover",
                "google", "web", "internet", "browse"
            ],
            ToolCategory.DATA: [
                "data", "database", "api", "fetch", "get", "retrieve",
                "json", "xml", "csv", "dataset"
            ],
            ToolCategory.CODE: [
                "code", "programming", "script", "execute", "run", "compile",
                "python", "javascript", "shell", "terminal"
            ],
            ToolCategory.ACADEMIC: [
                "academic", "research", "paper", "scholar", "pubmed", "arxiv",
                "citation", "reference", "journal"
            ],
            ToolCategory.BUSINESS: [
                "business", "company", "finance", "stock", "market", "news",
                "corporate", "industry", "economic"
            ],
            ToolCategory.UTILITY: [
                "utility", "tool", "helper", "convert", "format", "parse",
                "calculate", "math", "utility"
            ],
            ToolCategory.PLANNING: [
                "plan", "organize", "schedule", "task", "workflow", "strategy",
                "analyze", "breakdown", "structure"
            ],
            ToolCategory.VERIFICATION: [
                "verify", "check", "validate", "test", "confirm", "audit",
                "quality", "accuracy", "consistency"
            ],
            ToolCategory.GENERATION: [
                "generate", "create", "write", "summarize", "synthesize",
                "report", "content", "draft", "compose"
            ]
        }

    def _infer_tool_category(self, tool_info: ToolInfo) -> ToolCategory:
        """도구 이름과 설명으로 카테고리를 추론."""
        text_to_check = f"{tool_info.name} {tool_info.description}".lower()

        # 각 카테고리에 대해 키워드 매칭 수행
        for category, keywords in self.category_keywords.items():
            if any(keyword in text_to_check for keyword in keywords):
                return category

        # 기본 카테고리
        return ToolCategory.UTILITY

    def _group_tools_by_server(self, tools: list[BaseTool], tool_infos: list[ToolInfo]) -> dict[str, list[tuple[BaseTool, ToolInfo]]]:
        """서버별로 도구 그룹핑."""
        server_groups: dict[str, list[tuple[BaseTool, ToolInfo]]] = {}

        for tool, info in zip(tools, tool_infos):
            server = info.server_guess or "unknown"
            if server not in server_groups:
                server_groups[server] = []
            server_groups[server].append((tool, info))

        return server_groups

    def _select_tools_for_agent(
        self,
        agent_type: AgentType,
        all_tools: list[BaseTool],
        all_tool_infos: list[ToolInfo],
        server_filter: list[str] | None = None,
        max_tools_per_category: int = 3
    ) -> AgentToolAssignment:
        """특정 에이전트 타입에 맞는 도구 선택."""

        # 서버별로 도구 그룹핑
        server_groups = self._group_tools_by_server(all_tools, all_tool_infos)

        # 서버 필터 적용
        if server_filter:
            server_groups = {k: v for k, v in server_groups.items() if k in server_filter}

        selected_tools: list[BaseTool] = []
        selected_infos: list[ToolInfo] = []
        server_sources: dict[str, list[str]] = {}

        # 에이전트의 선호 카테고리별로 도구 선택
        preferred_categories = self.agent_category_preferences[agent_type]

        for category in preferred_categories:
            category_tools: list[tuple[BaseTool, ToolInfo, str]] = []

            # 각 서버에서 해당 카테고리의 도구 수집
            for server_name, tool_pairs in server_groups.items():
                for tool, info in tool_pairs:
                    if self._infer_tool_category(info) == category:
                        category_tools.append((tool, info, server_name))

            # 우선순위에 따라 도구 선택 (최대 개수 제한)
            category_tools.sort(key=lambda x: len(x[1].description), reverse=True)  # 설명이 긴 도구 우선

            for tool, info, server_name in category_tools[:max_tools_per_category]:
                selected_tools.append(tool)
                selected_infos.append(info)

                if server_name not in server_sources:
                    server_sources[server_name] = []
                server_sources[server_name].append(info.name)

        return AgentToolAssignment(
            agent_type=agent_type,
            tools=selected_tools,
            tool_infos=selected_infos,
            server_sources=server_sources
        )

    def select_tools_for_all_agents(
        self,
        all_tools: list[BaseTool],
        all_tool_infos: list[ToolInfo],
        server_filter: dict[AgentType, list[str]] | None = None,
        max_tools_per_agent: dict[AgentType, int] | None = None
    ) -> dict[AgentType, AgentToolAssignment]:
        """모든 에이전트 타입에 대한 도구 선택."""

        assignments: dict[AgentType, AgentToolAssignment] = {}

        for agent_type in AgentType:
            server_filt = server_filter.get(agent_type) if server_filter else None
            max_tools = max_tools_per_agent.get(agent_type, 5) if max_tools_per_agent else 5

            assignment = self._select_tools_for_agent(
                agent_type=agent_type,
                all_tools=all_tools,
                all_tool_infos=all_tool_infos,
                server_filter=server_filt,
                max_tools_per_category=max_tools
            )
            assignments[agent_type] = assignment

        return assignments

    def get_agent_tool_summary(self, assignment: AgentToolAssignment) -> str:
        """에이전트의 도구 할당 요약 생성."""
        summary_parts = [
            f"Agent Type: {assignment.agent_type.value}",
            f"Total Tools: {len(assignment.tools)}",
            f"Server Sources: {', '.join(assignment.server_sources.keys())}"
        ]

        if assignment.tool_infos:
            tool_names = [info.name for info in assignment.tool_infos]
            summary_parts.append(f"Tools: {', '.join(tool_names)}")

        return " | ".join(summary_parts)
