from typing import Annotated, Any, Dict, List, TypedDict, Final

from langchain_core.messages import BaseMessage


def override_reducer(a: Any, b: Any) -> Any:
    """새로운 값으로 덮어쓰는 리듀서"""
    return b


def add_messages(left: list, right: list) -> list:
    """메시지 리스트 병합 (LangGraph의 표준 동작 모사)"""
    return left + right


class ResearchState(TypedDict):
    """LangGraph 연구 워크플로우 상태 정의 (8대 혁신 통합)."""

    # Input
    user_request: Final[str]
    context: Final[Dict[str, Any] | None]
    objective_id: Final[str]

    # Adaptive Supervisor (혁신 1)
    complexity_score: float
    allocated_researchers: int
    priority_queue: List[Dict[str, Any]]
    quality_threshold: float

    # Analysis
    analyzed_objectives: List[Dict[str, Any]]
    intent_analysis: Dict[str, Any]
    domain_analysis: Dict[str, Any]
    scope_analysis: Dict[str, Any]

    # Planning Agent (새 필드)
    preliminary_research: Dict[str, Any]
    planned_tasks: List[Dict[str, Any]]
    agent_assignments: Dict[str, List[str]]
    execution_plan: Dict[str, Any]
    plan_approved: bool
    plan_feedback: str | None
    plan_iteration: int

    # Execution (Universal MCP Hub + Streaming Pipeline)
    execution_results: List[Dict[str, Any]]
    agent_status: Dict[str, Any]
    execution_metadata: Dict[str, Any]
    streaming_data: List[Dict[str, Any]]
    streaming_events: List[Dict[str, Any]]

    # Hierarchical Compression (혁신 2)
    compression_results: List[Dict[str, Any]]
    compression_metadata: Dict[str, Any]

    # Continuous Verification (혁신 4)
    verification_results: Dict[str, Any]
    confidence_scores: Dict[str, float]
    verification_stages: List[Dict[str, Any]]

    # Evaluation
    evaluation_results: Dict[str, Any]
    quality_metrics: Dict[str, float]
    improvement_areas: List[str]

    # Validation
    validation_results: Dict[str, Any]
    validation_score: float
    missing_elements: List[str]

    # Synthesis (Adaptive Context Window)
    final_synthesis: Dict[str, Any]
    deliverable_path: str | None
    synthesis_metadata: Dict[str, Any]

    # Human-in-the-loop 관련 필드
    pending_questions: List[Dict[str, Any]]
    user_responses: Dict[str, Any]
    clarification_context: Dict[str, Any]
    waiting_for_user: bool
    autopilot_mode: bool
    context_window_usage: Dict[str, Any]

    # Greedy Overseer 필드
    overseer_iterations: int
    overseer_requirements: List[Dict[str, Any]]
    overseer_evaluations: List[Dict[str, Any]]
    completeness_scores: Dict[str, float]
    quality_assessments: Dict[str, Dict[str, float]]
    overseer_decision: str | None

    # Control Flow
    current_step: str
    iteration: int
    max_iterations: int
    should_continue: bool
    error_message: str | None

    # Runtime sub-agent delegation (Anvil Phase Σ-2, issue #495/#509).
    # Mirrors the overseer_iterations/max_iterations guard pattern above:
    # delegate_to_agent() increments delegation_depth for the call and
    # refuses to delegate once it reaches max_delegation_depth.
    delegation_depth: int
    max_delegation_depth: int

    # Innovation Stats
    innovation_stats: Dict[str, Any]

    # Messages for LangGraph
    messages: Annotated[List[BaseMessage], "Messages in the conversation"]
