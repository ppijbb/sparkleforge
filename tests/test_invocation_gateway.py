"""Issue #568: agent-to-agent syscall boundary formalization.

Agent delegation (orchestrator/delegation.py::delegate_to_agent) and MCP
tool execution (mcp_integration/hub_mixins/execution.py::execute_tool) each
had to remember to add their own IntentGuardrail/CapabilityManager checks by
hand -- and neither one actually had. InvocationGateway gives both a single
mandatory pass-through so a new call path can't silently skip the check, and
every decision lands in ActionJournal regardless of outcome.
"""

from types import SimpleNamespace

import pytest

from src.core.guard.action_journal import ActionJournal
from src.core.guard.capability_manager import CapabilityManager
from src.core.guard.invocation_gateway import (
    SYSTEM_ACTOR,
    InvocationGateway,
    InvocationKind,
)


@pytest.fixture
def isolated_gateway(tmp_path):
    """A gateway backed by throwaway CapabilityManager/ActionJournal state,
    so tests don't read/write the real data/ directory or leak grants
    between tests (both are process-wide singletons otherwise)."""
    cm = CapabilityManager.__new__(CapabilityManager)
    cm._initialized = False
    CapabilityManager.__init__(cm, state_path=str(tmp_path / "capability_grants.json"))

    journal = ActionJournal.__new__(ActionJournal)
    journal._initialized = False
    ActionJournal.__init__(journal, journal_path=str(tmp_path / "action_journal.jsonl"))

    return InvocationGateway(capability_manager=cm, action_journal=journal)


def test_system_actor_gets_bootstrap_capabilities(isolated_gateway):
    for capability in ("execute_shell", "write_file", "read_file", "network_request"):
        assert isolated_gateway.capability_manager.agent_has(SYSTEM_ACTOR, capability)


def test_authorize_allows_when_no_capability_required(isolated_gateway):
    decision = isolated_gateway.authorize(
        kind=InvocationKind.AGENT_DELEGATION,
        actor="orchestrator",
        target="research_agent",
        description="delegate research task",
    )

    assert decision.allowed is True
    assert decision.reason == "authorized"


def test_authorize_denies_when_actor_lacks_required_capability(isolated_gateway):
    decision = isolated_gateway.authorize(
        kind=InvocationKind.MCP_TOOL,
        actor="untrusted_agent",
        target="shell_exec",
        description="run rm -rf",
        required_capability="execute_shell",
    )

    assert decision.allowed is False
    assert "execute_shell" in decision.reason


def test_authorize_allows_system_actor_for_bootstrapped_capability(isolated_gateway):
    decision = isolated_gateway.authorize(
        kind=InvocationKind.MCP_TOOL,
        actor=SYSTEM_ACTOR,
        target="shell_exec",
        description="run ls",
        required_capability="execute_shell",
    )

    assert decision.allowed is True


def test_authorize_denies_on_intent_drift(isolated_gateway):
    drifted_guardrail = SimpleNamespace(
        evaluate=lambda description: SimpleNamespace(aligned=False, drift_score=0.9)
    )

    decision = isolated_gateway.authorize(
        kind=InvocationKind.AGENT_DELEGATION,
        actor="orchestrator",
        target="research_agent",
        description="delegate something unrelated",
        intent_guardrail=drifted_guardrail,
    )

    assert decision.allowed is False
    assert "drift" in decision.reason


def test_authorize_tolerates_intent_guardrail_failure(isolated_gateway):
    def _raise(_description):
        raise RuntimeError("boom")

    broken_guardrail = SimpleNamespace(evaluate=_raise)

    decision = isolated_gateway.authorize(
        kind=InvocationKind.AGENT_DELEGATION,
        actor="orchestrator",
        target="research_agent",
        description="delegate",
        intent_guardrail=broken_guardrail,
    )

    assert decision.allowed is True  # fails safe: a broken guardrail doesn't block real work


def test_every_decision_is_journaled_regardless_of_outcome(isolated_gateway):
    isolated_gateway.authorize(
        kind=InvocationKind.MCP_TOOL,
        actor=SYSTEM_ACTOR,
        target="shell_exec",
        description="allowed call",
        required_capability="execute_shell",
    )
    isolated_gateway.authorize(
        kind=InvocationKind.MCP_TOOL,
        actor="untrusted_agent",
        target="shell_exec",
        description="denied call",
        required_capability="execute_shell",
    )

    entries = isolated_gateway.action_journal._entries
    gateway_entries = [e for e in entries if e.action.startswith("invocation_gateway:")]
    assert len(gateway_entries) == 2
    assert any("ALLOWED" in e.description for e in gateway_entries)
    assert any("DENIED" in e.description for e in gateway_entries)


class TestInferRequiredCapability:
    def test_shell_tools_map_to_execute_shell(self):
        from src.core.mcp_integration.hub_mixins.execution import _infer_required_capability

        assert _infer_required_capability("shell_exec") == "execute_shell"
        assert _infer_required_capability("run_shell_command") == "execute_shell"

    def test_write_tools_map_to_write_file(self):
        from src.core.mcp_integration.hub_mixins.execution import _infer_required_capability

        assert _infer_required_capability("write_file") == "write_file"
        assert _infer_required_capability("edit_file") == "write_file"
        assert _infer_required_capability("delete_file") == "write_file"

    def test_read_tools_map_to_read_file(self):
        from src.core.mcp_integration.hub_mixins.execution import _infer_required_capability

        assert _infer_required_capability("read_file") == "read_file"
        assert _infer_required_capability("list_files") == "read_file"

    def test_browser_tools_map_to_network_request(self):
        from src.core.mcp_integration.hub_mixins.execution import _infer_required_capability

        assert _infer_required_capability("browser_navigate") == "network_request"

    def test_unmapped_tools_return_none(self):
        from src.core.mcp_integration.hub_mixins.execution import _infer_required_capability

        assert _infer_required_capability("search_academic_papers") is None
