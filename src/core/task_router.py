"""Task Router for Agent Harness — LLM-Powered Intent Classifier.

이 모듈은 사용자의 입력(User Query) 또는 동적 생성된 Task를 LLM으로 분석하여
가장 적합한 Agent 또는 Workflow Path로 라우팅합니다.

핵심 원칙:
- 하드코딩된 키워드 매핑 ZERO (LLM 실패 시에만 폴백으로 사용)
- LLM이 의도(Intent)를 분석하여 경로 결정 (async/await)
- 태스크별 최적 에이전트 동적 할당
"""

import asyncio
import logging
import re
import json
from typing import Dict, Any, List, Optional
from enum import Enum

from src.core.harness_state import HarnessState, TaskState

logger = logging.getLogger(__name__)


class RoutePath(Enum):
    """실행 경로 옵션"""
    SINGLE_AGENT = "single_agent"
    PLANNER_PARALLEL = "planner_parallel"
    FINANCIAL_PIPELINE = "financial_pipeline"
    CODEBASE_AGENT = "codebase_agent"
    CREATIVITY_AGENT = "creativity_agent"
    DOCUMENT_PIPELINE = "document_pipeline"


# LLM 라우팅 판단용 프롬프트
_ROUTE_CLASSIFY_PROMPT = """You are an intelligent task router for an AI agent system.
Analyze the user's request and determine the MOST appropriate pipeline.

Available pipelines:
- "codebase_agent": Software development tasks — writing code, building systems, implementing applications, debugging, architecture design, APIs, services, tools. Use this when the user wants to BUILD or CREATE software.
- "financial_pipeline": Financial/economic research — stock analysis, market data, investment strategies, economic indicators.
- "creativity_agent": Creative writing, story generation, design ideation, brainstorming.
- "document_pipeline": Document reading and extraction — specifically for PDF, DOCX, PPTX, XLSX files or URLs. Use this when the user wants to READ, ANALYZE, or EXTRACT info from a document.
- "planner_parallel": Complex multi-step research requiring synthesis across multiple sources — scientific research, technical deep-dives, comparative analysis.
- "single_agent": Simple, direct questions that require a single focused answer.

User request: "{query}"

Think step by step:
1. What is the user's PRIMARY goal? (build software / research information / analyze data / create content)
2. What is the expected OUTPUT? (code files / research report / financial chart / creative text)
3. Which pipeline best serves that goal?

Respond with ONLY a JSON object:
{{"route": "<pipeline_name>", "reason": "<one sentence explanation>"}}"""

# 태스크별 에이전트 할당 프롬프트
_AGENT_ASSIGN_PROMPT = """You are an AI task dispatcher.
Given a sub-task description, assign the MOST appropriate specialist agent.

Available agents:
- "code_architect_agent": Designs system architecture, defines interfaces, data models, module structure.
- "code_implementor_agent": Writes actual implementation code for a specific module or feature.
- "code_reviewer_agent": Reviews code for correctness, security, performance.
- "researcher_agent": Gathers technical information, specifications, best practices.
- "analyzer_agent": Analyzes requirements, compares approaches, benchmarks solutions.
- "validator_agent": Tests, verifies, and fact-checks results.
- "synthesizer_agent": Combines multiple results into a coherent final output.

Sub-task description: "{description}"

Respond with ONLY a JSON object:
{{"agent": "<agent_name>", "reason": "<one sentence>"}}"""


class TaskRouter:
    """LLM 기반 지능형 태스크 라우터.

    사용자 요청의 의도(Intent)를 LLM이 직접 분석하여
    최적의 파이프라인과 에이전트를 동적으로 결정합니다.
    - determine_route(): async — LLM이 파이프라인 결정 (retry with backoff)
    - assign_agent_for_task(): async — LLM이 에이전트 결정 (retry with backoff)
    """

    MAX_RETRIES = 3
    BASE_BACKOFF = 1.0  # 오

    def __init__(self):
        pass  # LLM orchestrator는 호출 시점에 lazy하게 가져옴

    def _extract_json(self, text: str) -> dict:
        """LLM 응답에서 JSON을 추출합니다."""
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    async def determine_route(self, query: str) -> RoutePath:
        """LLM이 쿼리의 의도를 분석하여 실행 경로를 결정합니다.

        최대 MAX_RETRIES회 재시도하며, 모든 시도가 실패하면 예외를 발생시킵니다.
        """
        from src.core.llm_manager import get_llm_orchestrator, TaskType

        route_map = {
            "codebase_agent": RoutePath.CODEBASE_AGENT,
            "financial_pipeline": RoutePath.FINANCIAL_PIPELINE,
            "creativity_agent": RoutePath.CREATIVITY_AGENT,
            "document_pipeline": RoutePath.DOCUMENT_PIPELINE,
            "planner_parallel": RoutePath.PLANNER_PARALLEL,
            "single_agent": RoutePath.SINGLE_AGENT,
        }

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                orchestrator = get_llm_orchestrator()
                prompt = _ROUTE_CLASSIFY_PROMPT.format(query=query)

                result = await orchestrator.execute_with_model(
                    prompt=prompt,
                    task_type=TaskType.ANALYSIS,
                    use_cascade=False,
                )
                parsed = self._extract_json(result.content)
                route_str = parsed.get("route", "").lower().strip()
                reason = parsed.get("reason", "")

                if route_str in route_map:
                    route = route_map[route_str]
                    logger.info(f"TaskRouter [LLM]: Route → {route.name} | Reason: {reason}")
                    return route
                else:
                    last_error = f"Unrecognized route: '{route_str}'"
                    logger.warning(
                        f"TaskRouter: attempt {attempt + 1}/{self.MAX_RETRIES} — {last_error}"
                    )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"TaskRouter: attempt {attempt + 1}/{self.MAX_RETRIES} failed — {e}"
                )

            if attempt < self.MAX_RETRIES - 1:
                backoff = self.BASE_BACKOFF * (2 ** attempt)
                await asyncio.sleep(backoff)

        raise RuntimeError(
            f"TaskRouter: 모든 {self.MAX_RETRIES}회 라우팅 시도 실패. 마지막 오류: {last_error}"
        )

    async def assign_agent_for_task(self, task: TaskState) -> str:
        """LLM이 서브 태스크를 분석하여 최적 에이전트를 결정합니다.

        최대 MAX_RETRIES회 재시도하며, 모든 시도가 실패하면 예외를 발생시킵니다.
        """
        description = task.get("description", "")
        from src.core.llm_manager import get_llm_orchestrator, TaskType

        valid_agents = {
            "code_architect_agent", "code_implementor_agent", "code_reviewer_agent",
            "researcher_agent", "analyzer_agent", "validator_agent", "synthesizer_agent",
        }

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                orchestrator = get_llm_orchestrator()
                prompt = _AGENT_ASSIGN_PROMPT.format(description=description[:500])

                result = await orchestrator.execute_with_model(
                    prompt=prompt,
                    task_type=TaskType.ANALYSIS,
                    use_cascade=False,
                )
                parsed = self._extract_json(result.content)
                agent = parsed.get("agent", "").lower().strip()
                reason = parsed.get("reason", "")

                if agent in valid_agents:
                    logger.info(f"TaskRouter [LLM]: Agent → {agent} | Reason: {reason}")
                    return agent
                else:
                    last_error = f"Unrecognized agent: '{agent}'"
                    logger.warning(
                        f"TaskRouter: agent assign attempt {attempt + 1}/{self.MAX_RETRIES} — {last_error}"
                    )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"TaskRouter: agent assign attempt {attempt + 1}/{self.MAX_RETRIES} failed — {e}"
                )

            if attempt < self.MAX_RETRIES - 1:
                backoff = self.BASE_BACKOFF * (2 ** attempt)
                await asyncio.sleep(backoff)

        raise RuntimeError(
            f"TaskRouter: 모든 {self.MAX_RETRIES}회 에이전트 할당 시도 실패. 마지막 오류: {last_error}"
        )

    def update_state_for_route(self, state: HarnessState, route: RoutePath) -> HarnessState:
        """결정된 라우트에 따라 파이프라인의 초기 상태 플래그를 설정합니다."""
        if route == RoutePath.FINANCIAL_PIPELINE:
            state["governance"]["is_economic_request"] = True
        state["workflow"]["phase"] = "classify"
        return state
