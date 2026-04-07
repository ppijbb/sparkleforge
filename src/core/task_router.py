"""Task Router for Agent Harness.

이 모듈은 사용자의 입력(User Query) 또는 동적 생성된 Task를 분석하여
가장 적합한 Agent 또는 Workflow Path로 라우팅합니다.
"""

import logging
import re
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


class TaskRouter:
    """사용자 요청을 적절한 워크플로우 경로로 라우팅합니다."""
    
    def __init__(self):
        # 라우팅을 위한 키워드 매핑
        self.financial_keywords = [
            "주식", "종목", "주가", "환율", "금리", "시장", "경제", "투자", 
            "stock", "price", "market", "finance", "economy", "invest", "trading", "ticker"
        ]
        self.codebase_keywords = [
            "코드", "리팩터링", "버그", "함수", "클래스", "python", "javascript", 
            "bug", "code", "refactor", "function", "class", "github", "repo", "repository"
        ]
        self.creativity_keywords = [
            "소설", "시나리오", "이야기", "창작", "디자인", "아이디어", "브레인스토밍",
            "story", "novel", "creative", "design", "idea", "brainstorm"
        ]
        
    def determine_route(self, query: str) -> RoutePath:
        """입력된 쿼리에 기반하여 실행 경로를 결정합니다."""
        query_lower = query.lower()
        
        # 1. 재무/경제 파이프라인 체크
        if any(kw in query_lower for kw in self.financial_keywords):
            logger.info(f"TaskRouter: Route mapped to FINANCIAL_PIPELINE for query: '{query[:30]}...'")
            return RoutePath.FINANCIAL_PIPELINE
            
        # 2. 코드 생성/수정 체크
        if any(kw in query_lower for kw in self.codebase_keywords):
            logger.info(f"TaskRouter: Route mapped to CODEBASE_AGENT for query: '{query[:30]}...'")
            return RoutePath.CODEBASE_AGENT
            
        # 3. 창의적 작업 체크
        if any(kw in query_lower for kw in self.creativity_keywords):
            logger.info(f"TaskRouter: Route mapped to CREATIVITY_AGENT for query: '{query[:30]}...'")
            return RoutePath.CREATIVITY_AGENT
            
        # 4. 단순 vs 복합 쿼리 체크 (길이나 특정 단어로 휴리스틱)
        is_complex = len(query) > 100 or any(kw in query_lower for kw in ["비교", "분석", "연구", "리서치", "compare", "analyze", "research"])
        
        if is_complex:
            logger.info(f"TaskRouter: Route mapped to PLANNER_PARALLEL for complex query")
            return RoutePath.PLANNER_PARALLEL
        else:
            logger.info(f"TaskRouter: Route mapped to SINGLE_AGENT for simple query")
            return RoutePath.SINGLE_AGENT

    def assign_agent_for_task(self, task: TaskState) -> str:
        """분할된 서브 태스크에 적합한 특화 에이전트 역할을 할당합니다."""
        desc = task.get("description", "").lower()
        title = task.get("description", "").split('\n')[0].lower() # 간이 타이틀
        
        if any(k in desc for k in ["verify", "validate", "검증", "평가", "확인", "fact_check"]):
            return "validator_agent"
        if any(k in desc for k in ["analyze", "분석", "compare", "비교", "benchmark"]):
            return "analyzer_agent"
        if any(k in desc for k in ["synthesize", "summarize", "종합", "요약", "report"]):
            return "synthesizer_agent"
        
        # 기본값
        return "researcher_agent"

    def update_state_for_route(self, state: HarnessState, route: RoutePath) -> HarnessState:
        """결정된 라우트에 따라 파이프라인의 초기 상태 플래그를 설정합니다."""
        if route == RoutePath.FINANCIAL_PIPELINE:
            state["governance"]["is_economic_request"] = True
            
        state["workflow"]["phase"] = "classify"
        return state
