"""Task Router for Agent Harness — LLM-Powered Intent Classifier.

이 모듈은 사용자의 입력(User Query) 또는 동적 생성된 Task를 LLM으로 분석하여
가장 적합한 Agent 또는 Workflow Path로 라우팅합니다.

핵심 원칙:
- 하드코딩된 키워드 매핑 ZERO (LLM 실패 시에만 폴백으로 사용)
- LLM이 의도(Intent)를 분석하여 경로 결정 (async/await)
- 태스크별 최적 에이전트 동적 할당
"""

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
    - determine_route(): async — LLM이 파이프라인 결정
    - assign_agent_for_task(): async — LLM이 에이전트 결정
    """

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
        """LLM이 쿼리의 의도를 분석하여 실행 경로를 결정합니다 (async)."""
        try:
            from src.core.llm_manager import get_llm_orchestrator, TaskType
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

            route_map = {
                "codebase_agent": RoutePath.CODEBASE_AGENT,
                "financial_pipeline": RoutePath.FINANCIAL_PIPELINE,
                "creativity_agent": RoutePath.CREATIVITY_AGENT,
                "document_pipeline": RoutePath.DOCUMENT_PIPELINE,
                "planner_parallel": RoutePath.PLANNER_PARALLEL,
                "single_agent": RoutePath.SINGLE_AGENT,
            }

            if route_str in route_map:
                route = route_map[route_str]
                logger.info(f"TaskRouter [LLM]: Route → {route.name} | Reason: {reason}")
                return route
            else:
                logger.warning(
                    f"TaskRouter [LLM]: Unrecognized route '{route_str}', using heuristic"
                )

        except Exception as e:
            logger.warning(f"TaskRouter [LLM]: routing failed ({e}), using heuristic")

        # === Fallback: 키워드 휴리스틱 (LLM 실패 시에만) ===
        return self._heuristic_route(query)

    def _heuristic_route(self, query: str) -> RoutePath:
        """LLM 실패 시 사용하는 키워드 기반 폴백 라우터."""
        q = query.lower()
        financial_kw = [
            "주식", "주가", "환율", "금리", "stock", "finance", "market", "invest", "trading",
        ]
        code_kw = [
            "코드", "code", "구현", "서비스", "앱", "application", "app", "api", "server",
            "client", "함수", "클래스", "python", "javascript", "개발", "implement", "build",
            "작성", "기능", "화상통화", "video", "stream", "socket", "프로토콜", "protocol",
        ]
        creative_kw = ["소설", "시나리오", "창작", "story", "creative", "novel", "design"]
        doc_kw = ["문서", "파일", "pdf", "docx", "pptx", "xlsx", "읽어줘", "분석해줘", "document", "extract", "parse"]

        if any(k in q for k in financial_kw):
            logger.info("TaskRouter [Heuristic]: FINANCIAL_PIPELINE")
            return RoutePath.FINANCIAL_PIPELINE
        if any(k in q for k in code_kw):
            logger.info("TaskRouter [Heuristic]: CODEBASE_AGENT")
            return RoutePath.CODEBASE_AGENT
        if any(k in q for k in creative_kw):
            logger.info("TaskRouter [Heuristic]: CREATIVITY_AGENT")
            return RoutePath.CREATIVITY_AGENT
        if any(k in q for k in doc_kw) or q.endswith(".pdf") or q.endswith(".docx"):
            logger.info("TaskRouter [Heuristic]: DOCUMENT_PIPELINE")
            return RoutePath.DOCUMENT_PIPELINE

        is_complex = len(query) > 80 or any(
            k in q for k in ["비교", "분석", "연구", "compare", "analyze", "research"]
        )
        if is_complex:
            logger.info("TaskRouter [Heuristic]: PLANNER_PARALLEL")
            return RoutePath.PLANNER_PARALLEL

        logger.info("TaskRouter [Heuristic]: SINGLE_AGENT")
        return RoutePath.SINGLE_AGENT

    async def assign_agent_for_task(self, task: TaskState) -> str:
        """LLM이 서브 태스크를 분석하여 최적 에이전트를 결정합니다 (async)."""
        description = task.get("description", "")
        try:
            from src.core.llm_manager import get_llm_orchestrator, TaskType
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

            valid_agents = {
                "code_architect_agent", "code_implementor_agent", "code_reviewer_agent",
                "researcher_agent", "analyzer_agent", "validator_agent", "synthesizer_agent",
            }

            if agent in valid_agents:
                logger.info(f"TaskRouter [LLM]: Agent → {agent} | Reason: {reason}")
                return agent
            else:
                logger.warning(f"TaskRouter [LLM]: Unrecognized agent '{agent}', using heuristic")

        except Exception as e:
            logger.warning(f"TaskRouter [LLM]: agent assignment failed ({e}), using heuristic")

        # === Fallback 휴리스틱 ===
        return self._heuristic_assign(description)

    def _heuristic_assign(self, description: str) -> str:
        """LLM 실패 시 키워드 기반 에이전트 폴백 할당."""
        d = description.lower()
        if any(k in d for k in ["verify", "validate", "검증", "평가", "fact_check", "test"]):
            return "validator_agent"
        if any(k in d for k in ["architecture", "설계", "아키텍처", "structure", "interface", "모듈"]):
            return "code_architect_agent"
        if any(k in d for k in ["implement", "구현", "write code", "코드 작성", "develop", "개발", "build"]):
            return "code_implementor_agent"
        if any(k in d for k in ["review", "리뷰", "검토", "audit"]):
            return "code_reviewer_agent"
        if any(k in d for k in ["analyze", "분석", "compare", "비교", "benchmark"]):
            return "analyzer_agent"
        if any(k in d for k in ["synthesize", "summarize", "종합", "요약", "report", "보고"]):
            return "synthesizer_agent"
        return "researcher_agent"

    def update_state_for_route(self, state: HarnessState, route: RoutePath) -> HarnessState:
        """결정된 라우트에 따라 파이프라인의 초기 상태 플래그를 설정합니다."""
        if route == RoutePath.FINANCIAL_PIPELINE:
            state["governance"]["is_economic_request"] = True
        state["workflow"]["phase"] = "classify"
        return state
