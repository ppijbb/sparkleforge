"""동적 방법 탐색기 - 필요한 도구가 없을 때 대체 방법을 표준 체인으로 탐색 (M5)."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ResolutionStrategy(Enum):
    """방법 탐색 체인의 각 단계."""

    REGISTERED_HANDLER = "registered_handler"  # 핸들러 레지스트리에 이미 존재
    SKILL_REPOSITORY = "skill_repository"  # 이전에 만들어 저장한 스킬 재사용
    TOOL_BUILDER = "tool_builder"  # 도구 자체 제작 (MCP Server Auto-Builder 연계)
    ALTERNATIVE_PROCESS = "alternative_process"  # 등록된 대체 프로세스
    UNRESOLVED = "unresolved"  # 모든 단계 실패


# 도구 제작기: (capability, context) -> 실행 가능한 핸들러 또는 스킬 코드 문자열
ToolBuilder = Callable[
    [str, Dict[str, Any]], Union[Callable, str, Awaitable[Union[Callable, str]]]
]


@dataclass
class ResolutionAttempt:
    """탐색 체인 한 단계의 시도 기록."""

    strategy: ResolutionStrategy
    succeeded: bool
    detail: str = ""


@dataclass
class ResolvedMethod:
    """capability에 대해 확보된 실행 방법."""

    capability: str
    strategy: ResolutionStrategy
    handler: Optional[Callable] = None
    attempts: List[ResolutionAttempt] = field(default_factory=list)
    resolved_at: float = field(default_factory=time.time)

    @property
    def resolved(self) -> bool:
        return self.strategy != ResolutionStrategy.UNRESOLVED and self.handler is not None


class MethodResolver:
    """필요한 능력(capability)에 대한 실행 방법을 표준 체인으로 탐색한다.

    탐색 순서:
        1. 핸들러 레지스트리 — 이미 등록된 핸들러
        2. 스킬 저장소 — 과거 세션/태스크에서 제작해 보존한 스킬
        3. 도구 제작기 — 주입된 빌더(예: MCPServerBuilder)로 즉석 제작,
           성공 시 스킬 저장소에 보존해 다음부터 2단계에서 재사용
        4. 대체 프로세스 — capability별로 등록된 우회 방법

    모든 시도는 ResolvedMethod.attempts에 기록되어 어떤 경로로
    방법을 확보했는지(또는 왜 실패했는지) 추적할 수 있다.
    """

    def __init__(
        self,
        handler_registry: Optional[Dict[str, Callable]] = None,
        skill_repository: Any = None,
        tool_builder: Optional[ToolBuilder] = None,
    ):
        self.handler_registry = handler_registry or {}
        self.skill_repository = skill_repository
        self.tool_builder = tool_builder
        self.alternatives: Dict[str, Callable] = {}

    def register_alternative(self, capability: str, handler: Callable) -> None:
        """capability에 대한 대체 프로세스 등록."""
        self.alternatives[capability] = handler
        logger.debug("Alternative process registered for '%s'", capability)

    async def resolve(
        self, capability: str, context: Optional[Dict[str, Any]] = None
    ) -> ResolvedMethod:
        """capability 실행 방법을 체인 순서대로 탐색."""
        context = context or {}
        attempts: List[ResolutionAttempt] = []

        for strategy, finder in (
            (ResolutionStrategy.REGISTERED_HANDLER, self._from_registry),
            (ResolutionStrategy.SKILL_REPOSITORY, self._from_skills),
            (ResolutionStrategy.TOOL_BUILDER, self._from_builder),
            (ResolutionStrategy.ALTERNATIVE_PROCESS, self._from_alternatives),
        ):
            try:
                handler = await finder(capability, context)
            except Exception as e:
                attempts.append(
                    ResolutionAttempt(strategy=strategy, succeeded=False, detail=str(e))
                )
                logger.warning(
                    "Resolution step %s failed for '%s': %s", strategy.value, capability, e
                )
                continue

            if handler is not None:
                attempts.append(ResolutionAttempt(strategy=strategy, succeeded=True))
                logger.info(
                    "Capability '%s' resolved via %s", capability, strategy.value
                )
                return ResolvedMethod(
                    capability=capability,
                    strategy=strategy,
                    handler=handler,
                    attempts=attempts,
                )
            attempts.append(
                ResolutionAttempt(
                    strategy=strategy, succeeded=False, detail="not available"
                )
            )

        logger.warning("Capability '%s' could not be resolved", capability)
        return ResolvedMethod(
            capability=capability,
            strategy=ResolutionStrategy.UNRESOLVED,
            attempts=attempts,
        )

    async def _from_registry(
        self, capability: str, context: Dict[str, Any]
    ) -> Optional[Callable]:
        return self.handler_registry.get(capability)

    async def _from_skills(
        self, capability: str, context: Dict[str, Any]
    ) -> Optional[Callable]:
        if self.skill_repository is None:
            return None
        skill = self.skill_repository.get_skill(capability)
        if skill is None:
            return None
        return lambda *args, **kwargs: self.skill_repository.execute_skill(
            capability, *args, **kwargs
        )

    async def _from_builder(
        self, capability: str, context: Dict[str, Any]
    ) -> Optional[Callable]:
        if self.tool_builder is None:
            return None
        built = self.tool_builder(capability, context)
        if asyncio.iscoroutine(built):
            built = await built
        if built is None:
            return None

        # 코드 문자열이면 스킬로 보존 후 실행 핸들러 생성 (재사용 확보)
        if isinstance(built, str):
            if self.skill_repository is None:
                logger.warning(
                    "Builder returned code for '%s' but no skill repository to store it",
                    capability,
                )
                return None
            self.skill_repository.save_skill(
                capability,
                built,
                description=f"Auto-built tool for capability '{capability}'",
                metadata={"source": "tool_builder"},
            )
            return lambda *args, **kwargs: self.skill_repository.execute_skill(
                capability, *args, **kwargs
            )

        if callable(built):
            return built
        return None

    async def _from_alternatives(
        self, capability: str, context: Dict[str, Any]
    ) -> Optional[Callable]:
        return self.alternatives.get(capability)
