"""SparkleForge Agent Harness State Models.

이 모듈은 LangGraph 기반 Agent Harness에서 사용되는 계층화된 상태(State) 모델을 정의합니다.
기존의 단일 AgentState 객체를 논리적인 계층(Task, Workflow, Context, Governance, Meta)으로 분리하여
상태 전이 추적 및 데이터 오염을 방지합니다.
"""

from typing import Annotated, Any, Dict, List, Optional, Set
from typing_extensions import TypedDict
import operator

def override_reducer(a: Any, b: Any) -> Any:
    """새로운 값으로 덮어쓰는 리듀서"""
    return b

def add_messages(left: list, right: list) -> list:
    """메시지 리스트 병합 (LangGraph의 표준 동작 모사)"""
    return left + right


class TaskState(TypedDict):
    """개별 단위 Task 상태"""
    task_id: str
    description: str
    status: str  # pending | running | completed | failed
    assigned_agent: Optional[str]
    tool_used: Optional[str]
    result: Optional[Any]
    error: Optional[str]
    execution_time: float


class WorkflowState(TypedDict):
    """전체 워크플로우 추적 상태"""
    session_id: str
    user_query: str
    phase: str  # start | classify | plan | execute | verify | synthesize | output
    plan: str
    tasks: List[TaskState]
    completed_task_ids: Set[str]
    failed_task_ids: Set[str]
    final_output: Optional[str]


class ContextState(TypedDict):
    """도메인 분석 및 검색 결과 등 컨텍스트 상태"""
    domain: Optional[str]
    domain_analysis: Optional[Dict[str, Any]]
    financial_analysis: Optional[Dict[str, Any]]
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    synthesized_insights: List[str]


class GovernanceState(TypedDict):
    """보안 및 거버넌스 관련 상태"""
    trust_level: str
    is_economic_request: bool
    requires_approval: bool
    approved: bool
    tool_calls_count: int
    tool_failures: int


class MetaState(TypedDict):
    """실행 메타데이터 (성능, 토큰, 루프 카운트 등)"""
    iteration_count: int
    max_iterations: int
    start_time: float
    total_tokens_used: int
    warnings: List[str]
    current_agent: str


class HarnessState(TypedDict):
    """Harness의 최상위 루트 상태

    LangGraph StateGraph에서 사용되는 메인 스키마.
    """
    # LangGraph 표준 메시지 추적
    messages: Annotated[list, add_messages]
    
    # 계층화된 하위 상태들 (덮어쓰기 방식으로 갱신)
    workflow: Annotated[WorkflowState, override_reducer]
    context: Annotated[ContextState, override_reducer]
    governance: Annotated[GovernanceState, override_reducer]
    meta: Annotated[MetaState, override_reducer]


def create_initial_harness_state(
    session_id: str, 
    user_query: str, 
    max_iterations: int = 10
) -> HarnessState:
    """초기 HarnessState 객체를 생성합니다."""
    import time
    
    return {
        "messages": [],
        "workflow": {
            "session_id": session_id,
            "user_query": user_query,
            "phase": "start",
            "plan": "",
            "tasks": [],
            "completed_task_ids": set(),
            "failed_task_ids": set(),
            "final_output": None
        },
        "context": {
            "domain": None,
            "domain_analysis": None,
            "financial_analysis": None,
            "search_queries": [],
            "search_results": [],
            "synthesized_insights": []
        },
        "governance": {
            "trust_level": "medium",
            "is_economic_request": False,
            "requires_approval": False,
            "approved": False,
            "tool_calls_count": 0,
            "tool_failures": 0
        },
        "meta": {
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "start_time": time.time(),
            "total_tokens_used": 0,
            "warnings": [],
            "current_agent": "system"
        }
    }
