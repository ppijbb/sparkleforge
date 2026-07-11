import logging

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from src.core.orchestrator.analysis import AnalysisNode
from src.core.orchestrator.compression import CompressionNode
from src.core.orchestrator.execution import ExecutionNode
from src.core.orchestrator.planning import PlanningNode
from src.core.orchestrator.state import ResearchState
from src.core.orchestrator.synthesis import SynthesisNode
from src.core.orchestrator.verification import VerificationNode

logger = logging.getLogger(__name__)


def create_orchestrator_graph(
    creativity_agent,
    context_manager,
    streaming_manager,
    hybrid_storage,
    context_loader,
    research_depth,
    llm_config,
    agent_config,
):
    """Orchestrator LangGraph 워크플로우 구축."""
    # 노드 초기화
    analysis = AnalysisNode(creativity_agent, context_manager, streaming_manager, hybrid_storage)
    planning = PlanningNode(
        context_manager, context_loader, research_depth, hybrid_storage, streaming_manager
    )
    execution = ExecutionNode(llm_config, agent_config, research_depth, streaming_manager)
    verification = VerificationNode(agent_config)
    compression = CompressionNode(context_manager)
    synthesis = SynthesisNode(context_manager, creativity_agent, hybrid_storage)

    workflow = StateGraph(ResearchState)

    # 노드 등록
    workflow.add_node("analyze_objectives", analysis.analyze_objectives)
    workflow.add_node("planning_agent", planning.planning_agent)
    workflow.add_node("verify_plan", verification.verify_plan)
    workflow.add_node("overseer_initial_review", verification.overseer_initial_review)
    workflow.add_node("adaptive_supervisor", execution.adaptive_supervisor)
    workflow.add_node("execute_research", execution.execute_research)
    workflow.add_node("hierarchical_compression", compression.hierarchical_compression)
    workflow.add_node("continuous_verification", verification.continuous_verification)
    workflow.add_node("overseer_evaluation", verification.overseer_evaluation)
    workflow.add_node("evaluate_results", synthesis.evaluate_results)
    workflow.add_node("validate_results", synthesis.validate_results)
    workflow.add_node("synthesize_deliverable", synthesis.synthesize_deliverable)

    # 엣지 정의
    workflow.set_entry_point("analyze_objectives")
    workflow.add_edge("analyze_objectives", "planning_agent")

    workflow.add_conditional_edges(
        "planning_agent",
        lambda state: (
            "waiting_for_clarification" if state.get("waiting_for_user", False) else "verify_plan"
        ),
        {
            "waiting_for_clarification": "planning_agent",
            "verify_plan": "verify_plan",
        },
    )

    workflow.add_conditional_edges(
        "verify_plan",
        lambda state: ("approved" if state.get("plan_approved", False) else "planning_agent"),
        {"approved": "overseer_initial_review", "planning_agent": "planning_agent"},
    )

    workflow.add_edge("overseer_initial_review", "adaptive_supervisor")
    workflow.add_edge("adaptive_supervisor", "execute_research")

    workflow.add_conditional_edges(
        "execute_research",
        execution.decide_next_step_based_on_context,
        {
            "continue_research": "execute_research",
            "compress": "hierarchical_compression",
            "verify": "continuous_verification",
        },
    )

    workflow.add_edge("hierarchical_compression", "continuous_verification")
    workflow.add_edge("continuous_verification", "overseer_evaluation")

    workflow.add_conditional_edges(
        "overseer_evaluation",
        verification.overseer_decision_router,
        {
            "retry": "execute_research",
            "waiting_for_clarification": "planning_agent",
            "proceed": "evaluate_results",
        },
    )

    workflow.add_edge("evaluate_results", "validate_results")
    workflow.add_edge("validate_results", "synthesize_deliverable")
    workflow.add_edge("synthesize_deliverable", END)

    checkpointer = AsyncSqliteSaver.from_conn_string("data/orchestrator_checkpoints.db")
    return workflow.compile(checkpointer=checkpointer)
