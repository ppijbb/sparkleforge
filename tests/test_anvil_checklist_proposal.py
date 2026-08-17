"""Tests for the Anvil Phase Mu-3 checklist-proposal pipeline (src/core/anvil/checklist_proposal.py).

Covers the conflict-of-interest safeguard from docs/ANVIL_PLAN.md SS5.3: proposed
grading checks must pass an adversarial (zero-trust) review, and the proposer
session must never equal the session later graded by those checks.
"""

import pytest

from src.core.anvil.checklist_proposal import (
    ChecklistProposalError,
    assert_session_separation,
    propose_checklist,
)


class TestProposeChecklist:
    @pytest.mark.asyncio
    async def test_normal_request_yields_approved_items(self):
        proposal = await propose_checklist(
            "새 로그인 함수를 구현하고 관련 테스트를 통과시켜줘.",
            proposer_session_id="session-a",
        )

        assert proposal.checklist.items
        assert proposal.approved_items, "well-formed requirements should clear the adversarial gate"
        assert proposal.proposer_session_id == "session-a"

    @pytest.mark.asyncio
    async def test_every_item_is_either_approved_or_rejected(self):
        proposal = await propose_checklist("파일 정리해줘", proposer_session_id="session-b")

        classified = set(id(i) for i in proposal.approved_items + proposal.rejected_items)
        assert classified == set(id(i) for i in proposal.checklist.items)


class TestSessionSeparation:
    def test_same_session_is_rejected(self):
        with pytest.raises(ChecklistProposalError):
            assert_session_separation("session-a", "session-a")

    def test_different_sessions_pass(self):
        assert_session_separation("session-a", "session-b") is None
