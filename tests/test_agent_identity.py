"""Issue #614: verifiable agent identity & delegated mandates.

CapabilityManager grants were local trust records with no signature and no
way for an external process to verify them. This gives each agent instance
a real Ed25519 keypair (AgentIdentityManager) and a signed mandate format
(Mandate) with issuer/subject/scope/not-before/not-after fields modeled on
W3C Verifiable Credentials / AP2 mandates. verify_mandate() takes only a
public key -- no vault access -- so a mandate's authenticity can be checked
by a completely separate process that only has the issuer's public key.
"""

import base64
import time

import pytest

from src.core.guard.agent_identity import (
    AgentIdentityManager,
    Mandate,
    issue_mandate,
    mandate_covers_capability,
    verify_mandate,
)
from src.core.guard.credential_vault import CredentialVault


@pytest.fixture
def isolated_manager(tmp_path):
    """AgentIdentityManager backed by throwaway vault/registry files, not the
    real singletons (both are process-wide otherwise and would leak keys
    between tests)."""
    vault = CredentialVault.__new__(CredentialVault)
    vault._initialized = False
    CredentialVault.__init__(vault, fallback_path=str(tmp_path / ".credential_store"))

    return AgentIdentityManager(vault=vault, registry_path=str(tmp_path / "pubkeys.json"))


def test_get_or_create_identity_is_stable_across_calls(isolated_manager):
    first = isolated_manager.get_or_create_identity("agent_a")
    second = isolated_manager.get_or_create_identity("agent_a")

    assert first.public_key_b64() == second.public_key_b64()


def test_different_agents_get_different_keys(isolated_manager):
    a = isolated_manager.get_or_create_identity("agent_a")
    b = isolated_manager.get_or_create_identity("agent_b")

    assert a.public_key_b64() != b.public_key_b64()


def test_public_key_registry_is_queryable_without_private_key_access(isolated_manager):
    identity = isolated_manager.get_or_create_identity("agent_a")

    assert isolated_manager.get_public_key_b64("agent_a") == identity.public_key_b64()
    assert isolated_manager.get_public_key_b64("nonexistent_agent") is None


def test_issue_and_verify_mandate_round_trip(isolated_manager):
    issuer = isolated_manager.get_or_create_identity("human_operator")

    mandate = issue_mandate(issuer, subject="remote_agent", scope=["execute_shell"], ttl_seconds=60)
    valid, reason = verify_mandate(mandate, issuer.public_key_b64())

    assert valid is True
    assert reason == "valid"


def test_verify_mandate_works_with_only_the_public_key(isolated_manager):
    """The issue's success criterion: verification must work in a process
    that only has the public key, no vault/manager access at all."""
    issuer = isolated_manager.get_or_create_identity("human_operator")
    mandate = issue_mandate(issuer, subject="remote_agent", scope=["read_file"], ttl_seconds=60)
    public_key_only = issuer.public_key_b64()  # simulate handing this to a separate process

    # No AgentIdentityManager, no CredentialVault, no isolated_manager -- just the bytes.
    valid, reason = verify_mandate(mandate, public_key_only)

    assert valid is True


def test_verify_mandate_rejects_wrong_issuer_key(isolated_manager):
    issuer = isolated_manager.get_or_create_identity("human_operator")
    impostor = isolated_manager.get_or_create_identity("impostor")
    mandate = issue_mandate(issuer, subject="remote_agent", scope=["execute_shell"], ttl_seconds=60)

    valid, reason = verify_mandate(mandate, impostor.public_key_b64())

    assert valid is False
    assert "signature" in reason


def test_verify_mandate_rejects_tampered_scope(isolated_manager):
    issuer = isolated_manager.get_or_create_identity("human_operator")
    mandate = issue_mandate(issuer, subject="remote_agent", scope=["read_file"], ttl_seconds=60)

    mandate.scope.append("execute_shell")  # widen scope after signing

    valid, reason = verify_mandate(mandate, issuer.public_key_b64())

    assert valid is False
    assert "signature" in reason


def test_verify_mandate_rejects_tampered_subject(isolated_manager):
    issuer = isolated_manager.get_or_create_identity("human_operator")
    mandate = issue_mandate(issuer, subject="remote_agent", scope=["read_file"], ttl_seconds=60)

    mandate.subject = "different_agent"

    valid, reason = verify_mandate(mandate, issuer.public_key_b64())

    assert valid is False


def test_verify_mandate_rejects_expired_mandate(isolated_manager):
    issuer = isolated_manager.get_or_create_identity("human_operator")
    mandate = issue_mandate(issuer, subject="remote_agent", scope=["read_file"], ttl_seconds=-10)

    valid, reason = verify_mandate(mandate, issuer.public_key_b64())

    assert valid is False
    assert "expired" in reason


def test_verify_mandate_rejects_not_yet_valid_mandate(isolated_manager):
    issuer = isolated_manager.get_or_create_identity("human_operator")
    mandate = Mandate(
        issuer=issuer.agent_id,
        subject="remote_agent",
        scope=["read_file"],
        not_before=time.time() + 3600,
        not_after=time.time() + 7200,
    )
    signature = issuer.private_key.sign(mandate.canonical_payload())
    mandate.signature = base64.b64encode(signature).decode("ascii")

    valid, reason = verify_mandate(mandate, issuer.public_key_b64())

    assert valid is False
    assert "not yet valid" in reason


def test_mandate_covers_capability():
    mandate = Mandate(
        issuer="a", subject="b", scope=["execute_shell", "read_file"],
        not_before=0, not_after=0,
    )

    assert mandate_covers_capability(mandate, "execute_shell") is True
    assert mandate_covers_capability(mandate, "write_file") is False


def test_mandate_serialization_round_trip(isolated_manager):
    issuer = isolated_manager.get_or_create_identity("human_operator")
    mandate = issue_mandate(issuer, subject="remote_agent", scope=["execute_shell"], ttl_seconds=60)

    restored = Mandate.from_dict(mandate.to_dict())
    valid, _ = verify_mandate(restored, issuer.public_key_b64())

    assert valid is True
    assert restored.scope == mandate.scope
