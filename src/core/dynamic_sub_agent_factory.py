"""Dynamic Sub-Agent Factory - Planner 분석 결과 기반 서브 에이전트 동적 생성.

SubAgentManager와 연동하여 연구 계획의 태스크별로 서브 에이전트를 자율 정의하고,
SkillRetriever + SkillTree 기반으로 task-specific 스킬을 할당합니다.
SubAgentPerformanceStore에서 고성과 에이전트 템플릿을 우선 재활용합니다.
"""

import logging
from typing import Any, Dict, List

from src.core.sub_agent_manager import (
    CollaborationNetwork,
    SubAgentConfig,
    SubAgentContext,
    SubAgentManager,
    SubAgentPerformanceStore,
    SubAgentRole,
    SubAgentStatus,
    get_sub_agent_manager,
)
from src.core.skills_manager import get_skill_manager
from src.core.skill_tree import (
    HotSkillCache,
    SkillRetriever,
    SkillTree,
    get_skill_performance_tracker,
)

logger = logging.getLogger(__name__)


def _infer_role(task: Dict[str, Any]) -> SubAgentRole:
    """태스크 설명/메타데이터에서 서브 에이전트 역할 추론."""
    desc = (task.get("description") or "").lower()
    objectives = " ".join(
        str(o).lower() for o in task.get("objectives", [])
    )
    text = f"{desc} {objectives}"

    if any(
        k in text
        for k in (
            "verify",
            "validate",
            "검증",
            "평가",
            "확인",
            "quality",
            "fact_check",
        )
    ):
        return SubAgentRole.VALIDATOR
    if any(
        k in text
        for k in (
            "analyze",
            "분석",
            "compare",
            "비교",
            "benchmark",
        )
    ):
        return SubAgentRole.ANALYZER
    if any(
        k in text
        for k in (
            "synthesize",
            "summarize",
            "종합",
            "요약",
            "report",
            "리포트",
        )
    ):
        return SubAgentRole.SYNTHESIZER
    if any(
        k in text
        for k in (
            "search",
            "find",
            "gather",
            "research",
            "investigate",
            "검색",
            "수집",
            "연구",
            "조사",
        )
    ):
        return SubAgentRole.RESEARCHER
    return SubAgentRole.SPECIALIST


class DynamicSubAgentFactory:
    """Planner 분석 결과를 기반으로 서브 에이전트를 동적 생성."""

    def __init__(
        self,
        sub_agent_manager: SubAgentManager | None = None,
        skill_manager: Any = None,
    ):
        self.sam = sub_agent_manager or get_sub_agent_manager()
        self._skill_manager = skill_manager or get_skill_manager()
        self._retriever: SkillRetriever | None = None
        self._hot_cache: HotSkillCache | None = None

    def _get_retriever(self) -> SkillRetriever:
        if self._retriever is None:
            tracker = get_skill_performance_tracker()
            self._hot_cache = HotSkillCache(self._skill_manager, max_size=50)
            self._hot_cache.refresh(tracker)
            self._retriever = SkillRetriever(
                self._skill_manager,
                hot_cache=self._hot_cache,
                tracker=tracker,
            )
        return self._retriever

    async def ensure_network(
        self, network_id: str, coordinator_name: str = "planner"
    ) -> CollaborationNetwork:
        """협업 네트워크가 없으면 coordinator를 루트로 생성."""
        if network_id in self.sam.networks:
            return self.sam.networks[network_id]
        root_config = SubAgentConfig(
            role=SubAgentRole.COORDINATOR,
            name=coordinator_name,
            capabilities=["planning", "delegation", "coordination"],
        )
        return await self.sam.create_network(network_id, root_config)

    async def create_agents_from_plan(
        self,
        plan: Dict[str, Any],
        network_id: str,
        parent_agent_id: str,
        coordinator_name: str = "planner",
    ) -> List[SubAgentContext]:
        """Planner 출력의 task 분석에서 서브 에이전트 자동 정의 및 네트워크에 추가.

        Args:
            plan: research_plan 텍스트 또는 dict. dict인 경우 "tasks" 리스트 필드 사용.
            network_id: 협업 네트워크 ID (보통 session_id).
            parent_agent_id: 부모 에이전트 ID (coordinator).
            coordinator_name: 루트 에이전트 이름 (네트워크 생성 시 사용).

        Returns:
            생성된 서브 에이전트 컨텍스트 목록.
        """
        await self.ensure_network(network_id, coordinator_name=coordinator_name)

        if isinstance(plan, dict):
            tasks = plan.get("tasks", [])
        else:
            tasks = []

        if not tasks:
            logger.warning(
                "DynamicSubAgentFactory: no tasks in plan, returning empty list"
            )
            return []

        retriever = self._get_retriever()
        store = SubAgentPerformanceStore()
        agents: List[SubAgentContext] = []

        for task in tasks:
            task_id = task.get("task_id", "unknown")
            description = task.get("description", "")
            role = _infer_role(task)
            name = f"{role.value}_{task_id}".replace(" ", "_")

            capabilities = list(task.get("required_capabilities", []))
            if not capabilities:
                capabilities = [role.value, "research", "execute"]

            domain = task.get("domain") or task.get("name") or description[:80]
            if isinstance(domain, list):
                domain = " ".join(str(d) for d in domain[:3])
            domain_str = domain[:200] if domain else None

            # 고성과 에이전트 템플릿 우선 재활용
            config: SubAgentConfig | None = None
            top = store.get_top_agents(role.value, limit=1)
            if top and top[0].success_rate >= 0.8:
                template = store.get_agent_template(top[0].agent_name)
                if template:
                    config = SubAgentConfig(
                        role=template.role,
                        name=name,
                        capabilities=template.capabilities or capabilities,
                        specialization_area=template.specialization_area or domain_str,
                        max_concurrent_tasks=3,
                    )
            if config is None:
                config = SubAgentConfig(
                    role=role,
                    name=name,
                    capabilities=capabilities,
                    specialization_area=domain_str,
                    max_concurrent_tasks=3,
                )

            try:
                agent = await self.sam.add_sub_agent(
                    network_id, config, parent_agent_id
                )
                if agent:
                    query = description or str(domain or "")
                    skill_matches = await retriever.retrieve(
                        query, agent_skill_tree=None, top_k=3
                    )
                    assigned_ids = [m.skill_id for m in skill_matches]
                    agent.knowledge_base["assigned_skills"] = assigned_ids
                    agent.knowledge_base["task_id"] = task_id
                    agent.knowledge_base["task_description"] = description

                    skill_tree = SkillTree(agent_id=agent.agent_id)
                    for match in skill_matches:
                        cat = getattr(match.metadata, "category", "general")
                        skill_tree.add_skill(match.skill_id, [cat])
                    agent.skill_tree = skill_tree

                    agent.status = SubAgentStatus.ACTIVE
                    agents.append(agent)
                    logger.info(
                        f"DynamicSubAgentFactory: added sub-agent {name} (role={role.value}) for task {task_id}"
                    )
            except Exception as e:
                logger.warning(
                    f"DynamicSubAgentFactory: failed to add sub-agent for task {task_id}: {e}"
                )

        return agents
