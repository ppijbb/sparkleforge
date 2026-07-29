"""Forge Master Router - Capability Matrix & Tool Goal Assigner

SparkleForge 중앙 하네스가 작업의 특성과 토큰 예산을 평가하여
최적의 외부 CLI 도구(Claude Code, Codex, Gemini CLI, Hermes 등)를 선택하고
도구별 맞춤 Goal을 부여하는 동적 라우팅 시스템
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolGoalAssignment:
    """도구별 맞춤 목표 부여 정보"""

    agent_name: str
    assigned_goal: str
    capability_reason: str
    expected_output_format: str = "json"
    fallback_agents: List[str] = field(default_factory=list)
    max_token_budget: int = 4000


class ForgeMasterRouter:
    """Forge Master 라우터 및 역량 매트릭스 엔진"""

    # 역량 매트릭스: 도구별 강점 및 특화 영역
    CAPABILITY_MATRIX = {
        "claude_code": {
            "strengths": ["refactoring", "complex_bugfix", "architecture", "diff_slicing"],
            "score": 0.95,
            "cost_tier": "high",
        },
        "codex": {
            "strengths": ["code_generation", "syntax_repair", "snippet_synthesis", "python"],
            "score": 0.90,
            "cost_tier": "medium",
        },
        "gemini_cli": {
            "strengths": ["large_context", "doc_search", "multimodal", "broad_synthesis"],
            "score": 0.88,
            "cost_tier": "low",
        },
        "hermes": {
            "strengths": ["agentic_workflow", "custom_tools", "domain_task", "reasoning"],
            "score": 0.87,
            "cost_tier": "medium",
        },
        "open_code": {
            "strengths": ["local_llm", "offline", "general_code"],
            "score": 0.75,
            "cost_tier": "minimal",
        },
        "cline_cli": {
            "strengths": ["task_automation", "tool_use"],
            "score": 0.75,
            "cost_tier": "medium",
        },
    }

    def route_task(
        self,
        task_description: str,
        required_capabilities: Optional[List[str]] = None,
        preferred_agent: Optional[str] = None,
        available_agents: Optional[List[str]] = None,
    ) -> ToolGoalAssignment:
        """작업 요구사항에 따라 최적의 외부 CLI 에이전트 선택 및 Goal 부여

        Args:
            task_description: 작업 내용
            required_capabilities: 필요 역량 목록 (예: ['refactoring', 'syntax_repair'])
            preferred_agent: 선호 에이전트
            available_agents: 실행 가능 상태인 에이전트 목록

        Returns:
            ToolGoalAssignment (선정된 에이전트 및 맞춤 Goal)
        """
        required_caps = [c.lower() for c in (required_capabilities or [])]

        # 기본 에이전트 풀
        pool = available_agents or list(self.CAPABILITY_MATRIX.keys())

        # 실제 역량/키워드 관련성 점수 (기본 score는 동점 처리용 보조 지표일 뿐,
        # 이것만으로는 어떤 에이전트도 선택/폴백 후보가 될 수 없음)
        relevance = self._score_relevance(task_description, required_caps, pool)

        # 선호 에이전트가 존재하고 사용 가능한 경우 1순위 고려
        if preferred_agent and preferred_agent in pool:
            selected = preferred_agent
            reason = f"Explicitly preferred agent: {preferred_agent}"
        else:
            selected, reason = self._pick_best_agent(relevance, pool)

        # 폴백은 이 작업과 실제로 관련 있다고 판단된 에이전트에만 한정
        # (관련성 매치가 하나도 없으면 불필요하게 다른 유료 에이전트로 확산시키지 않음)
        fallbacks = self._relevant_fallbacks(selected, relevance)

        # 맞춤 Goal 부여 생성
        goal_text = self._build_tool_specific_goal(selected, task_description)

        return ToolGoalAssignment(
            agent_name=selected,
            assigned_goal=goal_text,
            capability_reason=reason,
            fallback_agents=fallbacks,
        )

    # 관련성 매치가 하나도 없는 에이전트를 동점 처리용 보조 점수만으로
    # 선택/폴백 후보에 끼워주지 않기 위한 문턱값
    _RELEVANCE_THRESHOLD = 1.0

    def _score_relevance(
        self, description: str, caps: List[str], pool: List[str]
    ) -> Dict[str, float]:
        """역량/키워드 매칭에 기반한 실제 작업 관련성 점수 계산

        CAPABILITY_MATRIX의 정적 `score`는 매치가 전혀 없을 때의 동점 처리용
        보조 지표(가중치 0.01)로만 반영한다. 그렇지 않으면 claude_code/codex처럼
        기본 score가 높은 도구가 관련성과 무관하게 항상 선택/폴백되어 버린다.
        """
        desc_lower = description.lower()
        relevance: Dict[str, float] = {}

        for agent in pool:
            info = self.CAPABILITY_MATRIX.get(agent, {"strengths": [], "score": 0.5})
            match = 0.0

            # 역량 매칭 (요청된 역량 하나당 1점)
            for cap in caps:
                if cap in info["strengths"]:
                    match += 1.0

            # 키워드 헤비스틱 매칭
            if any(k in desc_lower for k in ["refactor", "architecture", "rewrite"]) and agent == "claude_code":
                match += 1.0
            elif any(k in desc_lower for k in ["generate", "snippet", "function", "fix syntax"]) and agent == "codex":
                match += 1.0
            elif any(k in desc_lower for k in ["search", "document", "explain", "large"]) and agent == "gemini_cli":
                match += 1.0
            elif any(k in desc_lower for k in ["workflow", "agentic", "hermes", "domain"]) and agent == "hermes":
                match += 1.0

            relevance[agent] = match + info["score"] * 0.01

        return relevance

    def _pick_best_agent(self, relevance: Dict[str, float], pool: List[str]) -> tuple[str, str]:
        """관련성 점수 기준 최적 에이전트 선택"""
        sorted_agents = sorted(relevance.items(), key=lambda x: x[1], reverse=True)
        best_agent, best_score = sorted_agents[0]

        if best_score < self._RELEVANCE_THRESHOLD:
            reason = f"No specific capability/keyword match; defaulting to broadest-capability agent in pool {pool}"
        else:
            reason = f"Best task-relevance match (score {best_score:.2f}) in pool {pool}"

        return best_agent, reason

    def _relevant_fallbacks(self, selected: str, relevance: Dict[str, float]) -> List[str]:
        """실제로 이 작업과 관련 있다고 판단된 에이전트만 관련도 순으로 폴백 후보에 남김"""
        candidates = [
            (agent, score)
            for agent, score in relevance.items()
            if agent != selected and score >= self._RELEVANCE_THRESHOLD
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [agent for agent, _ in candidates]

    def _build_tool_specific_goal(self, agent_name: str, task_description: str) -> str:
        """각 CLI 에이전트의 강점에 맞춘 맞춤형 Goal 지시문 빌드"""
        if agent_name == "claude_code":
            return f"[Claude Code Dedicated Goal] Focus strictly on code refactoring and diff integrity: {task_description}"
        elif agent_name == "codex":
            return f"[Codex Dedicated Goal] Generate clean, concise code with syntax precision: {task_description}"
        elif agent_name == "gemini_cli":
            return f"[Gemini CLI Dedicated Goal] Synthesize wide context and document insights efficiently: {task_description}"
        elif agent_name == "hermes":
            return f"[Hermes Dedicated Goal] Execute autonomous workflow step-by-step: {task_description}"
        else:
            return f"[{agent_name} Dedicated Goal] {task_description}"
