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
def isolated_identity_manager(tmp_path):
    """Issue #614: an AgentIdentityManager backed by throwaway vault/registry
    files, for testing InvocationGateway's mandate-based authorization."""
    from src.core.guard.agent_identity import AgentIdentityManager
    from src.core.guard.credential_vault import CredentialVault

    vault = CredentialVault.__new__(CredentialVault)
    vault._initialized = False
    CredentialVault.__init__(vault, fallback_path=str(tmp_path / ".credential_store"))

    identity_manager = AgentIdentityManager.__new__(AgentIdentityManager)
    identity_manager._initialized = False
    AgentIdentityManager.__init__(
        identity_manager, vault=vault, registry_path=str(tmp_path / "pubkeys.json")
    )
    return identity_manager


@pytest.fixture
def isolated_gateway(tmp_path, isolated_identity_manager):
    """A gateway backed by throwaway CapabilityManager/ActionJournal/identity
    registry state, so tests don't read/write the real data/ directory or
    leak grants between tests (all three are process-wide singletons
    otherwise)."""
    cm = CapabilityManager.__new__(CapabilityManager)
    cm._initialized = False
    CapabilityManager.__init__(cm, state_path=str(tmp_path / "capability_grants.json"))

    journal = ActionJournal.__new__(ActionJournal)
    journal._initialized = False
    ActionJournal.__init__(journal, journal_path=str(tmp_path / "action_journal.jsonl"))

    return InvocationGateway(
        capability_manager=cm, action_journal=journal, identity_manager=isolated_identity_manager
    )


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


def test_authorize_fails_closed_on_intent_guardrail_failure(isolated_gateway):
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

    # fail closed: a guardrail that can't render a verdict must not be treated
    # as a pass -- it's the only real check in front of dangerous tool calls.
    assert decision.allowed is False
    assert "guardrail_evaluation_failed" in decision.reason


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


class TestMandateAuthorization:
    """Issue #614: a valid signed mandate should authorize its subject even
    with no pre-existing local capability grant, and reflect that as a real
    CapabilityManager grant so agent_has() checks elsewhere see it too."""

    def test_valid_mandate_authorizes_subject_with_no_prior_grant(
        self, isolated_gateway, isolated_identity_manager
    ):
        from src.core.guard.agent_identity import issue_mandate

        issuer = isolated_identity_manager.get_or_create_identity("human_operator")
        mandate = issue_mandate(issuer, subject="remote_agent", scope=["execute_shell"], ttl_seconds=60)

        assert isolated_gateway.capability_manager.agent_has("remote_agent", "execute_shell") is False

        decision = isolated_gateway.authorize(
            kind=InvocationKind.MCP_TOOL,
            actor="remote_agent",
            target="shell_exec",
            description="run a build command",
            required_capability="execute_shell",
            mandate=mandate,
        )

        assert decision.allowed is True
        assert isolated_gateway.capability_manager.agent_has("remote_agent", "execute_shell") is True

    def test_mandate_with_wrong_scope_is_denied(self, isolated_gateway, isolated_identity_manager):
        from src.core.guard.agent_identity import issue_mandate

        issuer = isolated_identity_manager.get_or_create_identity("human_operator")
        mandate = issue_mandate(issuer, subject="remote_agent", scope=["read_file"], ttl_seconds=60)

        decision = isolated_gateway.authorize(
            kind=InvocationKind.MCP_TOOL,
            actor="remote_agent",
            target="shell_exec",
            description="try shell without scope",
            required_capability="execute_shell",
            mandate=mandate,
        )

        assert decision.allowed is False
        assert "does not cover" in decision.reason
        assert isolated_gateway.capability_manager.agent_has("remote_agent", "execute_shell") is False

    def test_impersonated_issuer_is_denied(self, isolated_gateway, isolated_identity_manager):
        """Issue #798 vuln 2: an attacker with their own real keypair signs a
        mandate that *claims* issuer="human_operator", then presents it. The
        gateway must verify against human_operator's *registered* public key
        -- never a key the caller supplies -- so a signature made with the
        impostor's own key is rejected rather than trusted at face value."""
        from src.core.guard.agent_identity import issue_mandate

        isolated_identity_manager.get_or_create_identity("human_operator")
        impostor = isolated_identity_manager.get_or_create_identity("impostor")

        forged_mandate = issue_mandate(
            impostor, subject="remote_agent", scope=["execute_shell"], ttl_seconds=60
        )
        forged_mandate.issuer = "human_operator"  # claim an identity that isn't theirs

        decision = isolated_gateway.authorize(
            kind=InvocationKind.MCP_TOOL,
            actor="remote_agent",
            target="shell_exec",
            description="forged mandate attempt",
            required_capability="execute_shell",
            mandate=forged_mandate,
        )

        assert decision.allowed is False
        assert "invalid" in decision.reason
        assert isolated_gateway.capability_manager.agent_has("remote_agent", "execute_shell") is False

    def test_mandate_with_unregistered_issuer_is_denied(self, isolated_gateway, isolated_identity_manager):
        """Issue #798 vuln 2: an issuer identity that was never created via
        AgentIdentityManager has no registered public key, so the gateway
        must reject the mandate rather than trust a caller-supplied key."""
        from src.core.guard.agent_identity import issue_mandate

        ghost_issuer = isolated_identity_manager.get_or_create_identity("ghost_operator")
        mandate = issue_mandate(ghost_issuer, subject="remote_agent", scope=["execute_shell"], ttl_seconds=60)
        # Simulate an issuer identity absent from this gateway's registry.
        mandate.issuer = "never_registered"

        decision = isolated_gateway.authorize(
            kind=InvocationKind.MCP_TOOL,
            actor="remote_agent",
            target="shell_exec",
            description="unregistered issuer attempt",
            required_capability="execute_shell",
            mandate=mandate,
        )

        assert decision.allowed is False
        assert "not a registered identity" in decision.reason

    def test_expired_mandate_is_denied(self, isolated_gateway, isolated_identity_manager):
        from src.core.guard.agent_identity import issue_mandate

        issuer = isolated_identity_manager.get_or_create_identity("human_operator")
        mandate = issue_mandate(issuer, subject="remote_agent", scope=["execute_shell"], ttl_seconds=-10)

        decision = isolated_gateway.authorize(
            kind=InvocationKind.MCP_TOOL,
            actor="remote_agent",
            target="shell_exec",
            description="expired mandate attempt",
            required_capability="execute_shell",
            mandate=mandate,
        )

        assert decision.allowed is False
        assert "expired" in decision.reason

    def test_mandate_verification_result_is_journaled(self, isolated_gateway, isolated_identity_manager):
        from src.core.guard.agent_identity import issue_mandate

        issuer = isolated_identity_manager.get_or_create_identity("human_operator")
        mandate = issue_mandate(issuer, subject="remote_agent", scope=["execute_shell"], ttl_seconds=60)

        isolated_gateway.authorize(
            kind=InvocationKind.MCP_TOOL,
            actor="remote_agent",
            target="shell_exec",
            description="journaled mandate call",
            required_capability="execute_shell",
            mandate=mandate,
        )

        entries = isolated_gateway.action_journal._entries
        gateway_entries = [e for e in entries if e.action.startswith("invocation_gateway:")]
        assert len(gateway_entries) == 1
        assert gateway_entries[0].metadata["mandate"]["valid"] is True
        assert gateway_entries[0].metadata["mandate"]["subject"] == "remote_agent"

    def test_mandate_grant_expires_with_mandate(self, isolated_gateway, isolated_identity_manager, monkeypatch):
        """Issue #798 vuln 1: the capability grant reflected from a mandate
        must stop counting once mandate.not_after passes, instead of
        persisting in CapabilityManager indefinitely."""
        import time as time_module

        from src.core.guard.agent_identity import issue_mandate

        issuer = isolated_identity_manager.get_or_create_identity("human_operator")
        mandate = issue_mandate(issuer, subject="remote_agent", scope=["execute_shell"], ttl_seconds=60)

        decision = isolated_gateway.authorize(
            kind=InvocationKind.MCP_TOOL,
            actor="remote_agent",
            target="shell_exec",
            description="run a build command",
            required_capability="execute_shell",
            mandate=mandate,
        )
        assert decision.allowed is True
        assert isolated_gateway.capability_manager.agent_has("remote_agent", "execute_shell") is True

        monkeypatch.setattr(time_module, "time", lambda: mandate.not_after + 1)
        assert isolated_gateway.capability_manager.agent_has("remote_agent", "execute_shell") is False


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
