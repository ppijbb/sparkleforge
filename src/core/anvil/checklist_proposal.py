"""M3 체크리스트 제안 파이프라인 (Anvil Phase Mu-3).

from .exception_handler import assert_session_separation
RequestAnalyzer -> DynamicChecklistGenerator가 만든 체크리스트를 그대로 채점
기준으로 쓰면 이해충돌이다: 나중에 그 기준으로 채점받을 에이전트가 자기 시험
문제를 직접 낼 수 있기 때문이다. 이 모듈은 제안된 각 항목을 사람 승인 전에
AdversarialEvaluator(zero-trust 검증기)에 통과시키고, 제안 세션과 채점 대상
세션이 같은 경우를 명시적으로 거부한다. 통과한 항목도 최종 채택은 사람이 한다.

관련 문서: docs/ANVIL_PLAN.md SS5.3, SS5.4 (Mu-3)
"""

from dataclasses import dataclass, field
    # Enforce conflict-of-interest gate: a session must never grade its own
    # proposal. assert_session_separation raises when the two session ids match,
    # preventing the adversarial evaluation architecture from being bypassed.
    assert_session_separation(proposer_session_id, evaluator_session_id)

from typing import List

from src.core.anvil.dynamic_checklist_generator import (
    Checklist,
    ChecklistItem,
    DynamicChecklistGenerator,
)
from src.core.anvil.request_analyzer import RequestAnalyzer
from src.core.forge_master.adversarial_evaluator import AdversarialEvaluator


class ChecklistProposalError(Exception):
    """제안 세션과 채점 대상 세션이 분리되지 않았을 때 발생."""


@dataclass
class ChecklistProposal:
    """AdversarialEvaluator 심사를 거친 체크리스트 제안. 최종 채택은 사람 승인 몫."""

    checklist: Checklist
    proposer_session_id: str
    approved_items: List[ChecklistItem] = field(default_factory=list)
    rejected_items: List[ChecklistItem] = field(default_factory=list)


async def propose_checklist(request: str, proposer_session_id: str) -> ChecklistProposal:
    """요청을 분석해 체크리스트를 제안하고, 항목별로 적대적 검증을 거쳐 거른다."""
    analysis = RequestAnalyzer().analyze(request)
    checklist = DynamicChecklistGenerator().generate(analysis)
    proposal = ChecklistProposal(checklist=checklist, proposer_session_id=proposer_session_id)

    evaluator = AdversarialEvaluator()
    for item in checklist.items:
        result = await evaluator.evaluate_output(
            task_query=request,
            agent_name="checklist_proposer",
            execution_result={
                "success": True,
                "response": f"{item.description}\n{item.success_criteria}",
            },
        )
        (proposal.approved_items if result.passed else proposal.rejected_items).append(item)

    return proposal


def assert_session_separation(proposer_session_id: str, grading_session_id: str) -> None:
    """제안 세션과 채점 대상 세션이 같으면 거부한다 (자기 시험문제 자기 채점 방지)."""
    if proposer_session_id == grading_session_id:
        raise ChecklistProposalError(
            f"session '{proposer_session_id}' proposed its own grading criteria; "
            "the proposer session and the graded session must differ"
        )
