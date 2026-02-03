"""Cross-Agent 통신 도구 - 에이전트를 다른 에이전트의 도구로 노출."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

import anyio


# -----------------------------
# Public API surface
# -----------------------------


@dataclass(frozen=True)
class CrossAgent:
    """도구로 노출할 피어 에이전트 메타데이터.

    Attributes:
        agent: 에이전트 그래프 (LangGraph Runnable)
        description: 호출 에이전트가 언제 위임을 결정할지 도와주는 설명
    """
    agent: Runnable[Any, Any]
    description: str = ""


def make_cross_agent_tools(
    peers: Mapping[str, CrossAgent],
    *,
    tool_name_prefix: str = "ask_agent_",
    include_broadcast: bool = True,
) -> list[BaseTool]:
    """Cross-Agent 통신을 위한 LangChain 도구 생성.

    각 피어에 대해 f"{tool_name_prefix}{peer_name}" 이름의 도구를 생성하고,
    선택적으로 여러 피어에게 병렬 질의를 위한 broadcast 도구를 추가합니다.

    Args:
        peers: 피어 이름 -> CrossAgent 매핑
        tool_name_prefix: 각 피어 도구의 접두사 (기본값: "ask_agent_")
        include_broadcast: broadcast 도구 포함 여부 (기본값: True)

    Returns:
        생성된 도구 리스트
    """
    out: list[BaseTool] = []

    # 각 피어에 대한 개별 ask 도구 생성
    for name, peer in peers.items():
        tool_name = f"{tool_name_prefix}{name}"
        out.append(_AskAgentTool(
            name=tool_name,
            description=f"질문을 '{name}' 피어 에이전트에게 전달하고 최종 답변을 반환합니다. {peer.description}",
            agent=peer.agent,
            peer_name=name,
        ))

    # Broadcast 도구 추가 (선택적)
    if include_broadcast:
        out.append(_BroadcastTool(
            peers=peers,
            timeout_s=30.0,  # 기본 타임아웃
        ))

    return out


# -----------------------------
# Tool Implementations
# -----------------------------


class _AskAgentTool(BaseTool):
    """단일 피어 에이전트에게 질의를 전달하는 도구."""

    name: str
    description: str

    _agent: Runnable[Any, Any] = PrivateAttr()
    _peer_name: str = PrivateAttr()

    def __init__(
        self,
        *,
        name: str,
        description: str,
        agent: Runnable[Any, Any],
        peer_name: str,
    ) -> None:
        super().__init__(name=name, description=description)
        self._agent = agent
        self._peer_name = peer_name

    async def _arun(self, message: str, context: str | None = None, timeout_s: float | None = None) -> str:
        """피어 에이전트를 호출하고 결과를 추출."""
        # 메시지 포맷팅
        full_message = message
        if context:
            full_message = f"Context: {context}\n\nQuestion: {message}"

        # 에이전트 입력 구성
        agent_input = {"messages": [{"role": "user", "content": full_message}]}

        # 타임아웃 적용 (선택적)
        if timeout_s:
            with anyio.move_on_after(timeout_s) as cancel_scope:
                result = await self._agent.ainvoke(agent_input)
                if cancel_scope.cancelled_caught:
                    return f"피어 '{self._peer_name}'이(가) 타임아웃되었습니다."
        else:
            result = await self._agent.ainvoke(agent_input)

        # 결과에서 최종 답변 추출
        return _extract_final_answer(result)

    def _run(self, message: str, context: str | None = None, timeout_s: float | None = None) -> str:
        """동기 실행 (anyio 사용)."""
        return anyio.run(lambda: self._arun(message=message, context=context, timeout_s=timeout_s))


class _BroadcastTool(BaseTool):
    """여러 피어 에이전트에게 병렬 질의를 수행하는 도구."""

    name: str = "broadcast_to_agents"
    description: str = "지정된 피어 에이전트들에게 동일한 질문을 병렬로 전달하고 각자의 답변을 수집합니다."

    _peers: Mapping[str, CrossAgent] = PrivateAttr()
    _timeout_s: float = PrivateAttr()

    def __init__(self, *, peers: Mapping[str, CrossAgent], timeout_s: float = 30.0) -> None:
        super().__init__()
        self._peers = peers
        self._timeout_s = timeout_s

    async def _arun(self, message: str, peers: list[str] | None = None, timeout_s: float | None = None) -> dict[str, str]:
        """여러 피어 에이전트를 병렬 호출."""
        # 사용할 피어 선택 (지정되지 않은 경우 모든 피어)
        target_peers = peers or list(self._peers.keys())
        timeout = timeout_s or self._timeout_s

        # 존재하는 피어만 필터링
        valid_peers = {name: self._peers[name] for name in target_peers if name in self._peers}

        if not valid_peers:
            return {"error": f"유효한 피어 에이전트를 찾을 수 없습니다: {target_peers}"}

        # 병렬 호출 준비
        async def call_peer(peer_name: str, peer: CrossAgent) -> tuple[str, str]:
            try:
                # 개별 ask 도구처럼 호출
                ask_tool = _AskAgentTool(
                    name=f"ask_{peer_name}",
                    description="",
                    agent=peer.agent,
                    peer_name=peer_name
                )
                result = await ask_tool._arun(message, timeout_s=timeout)
                return peer_name, result
            except Exception as e:
                return peer_name, f"에러: {e}"

        # 병렬 실행
        tasks = [call_peer(name, peer) for name, peer in valid_peers.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 집계
        responses: dict[str, str] = {}
        for result in results:
            if isinstance(result, Exception):
                responses["error"] = f"병렬 호출 중 예외 발생: {result}"
            else:
                peer_name, response = result
                responses[peer_name] = response

        return responses

    def _run(self, message: str, peers: list[str] | None = None, timeout_s: float | None = None) -> dict[str, str]:
        """동기 실행 (anyio 사용)."""
        return anyio.run(lambda: self._arun(message=message, peers=peers, timeout_s=timeout_s))


# -----------------------------
# Helper Functions
# -----------------------------


def _extract_final_answer(result: Any) -> str:
    """에이전트 결과에서 최종 답변 텍스트를 추출."""
    # LangGraph/ReAct 결과에서 최종 답변 추출
    if isinstance(result, dict):
        # messages에서 마지막 AI/assistant 메시지 찾기
        messages = result.get("messages", [])
        for message in reversed(messages):
            # LangChain 메시지 객체인 경우
            if hasattr(message, "content"):
                return message.content
            # 딕셔너리 형태인 경우
            elif isinstance(message, dict):
                content = message.get("content")
                if content:
                    return content

        # 다른 가능한 필드들 확인
        for key in ["output", "result", "answer", "response", "final_answer"]:
            if key in result:
                value = result[key]
                if isinstance(value, str):
                    return value

    # 리스트인 경우 (메시지 리스트)
    elif isinstance(result, list):
        for item in reversed(result):
            if isinstance(item, dict) and "content" in item:
                return item["content"]
            elif hasattr(item, "content"):
                return item.content

    # 기본적으로 문자열 변환
    return str(result)


# -----------------------------
# Integration Helper
# -----------------------------


def integrate_cross_agent_tools(
    orchestrator: Any,
    peer_agents: Mapping[str, CrossAgent] | None = None
) -> list[BaseTool]:
    """AgentOrchestrator에 Cross-Agent 도구를 통합.

    Args:
        orchestrator: AgentOrchestrator 인스턴스
        peer_agents: 추가 피어 에이전트 (없으면 orchestrator의 에이전트 사용)

    Returns:
        생성된 Cross-Agent 도구 리스트
    """
    if peer_agents is None:
        # 기본적으로 orchestrator의 에이전트들을 피어로 사용
        peer_agents = {}
        if hasattr(orchestrator, 'planner') and orchestrator.planner:
            peer_agents['planner'] = CrossAgent(
                agent=orchestrator.planner,
                description="연구 계획 수립 전문가"
            )
        if hasattr(orchestrator, 'executor') and orchestrator.executor:
            peer_agents['executor'] = CrossAgent(
                agent=orchestrator.executor,
                description="연구 실행 전문가"
            )
        if hasattr(orchestrator, 'verifier') and orchestrator.verifier:
            peer_agents['verifier'] = CrossAgent(
                agent=orchestrator.verifier,
                description="결과 검증 전문가"
            )
        if hasattr(orchestrator, 'generator') and orchestrator.generator:
            peer_agents['generator'] = CrossAgent(
                agent=orchestrator.generator,
                description="보고서 생성 전문가"
            )

    return make_cross_agent_tools(peer_agents)
