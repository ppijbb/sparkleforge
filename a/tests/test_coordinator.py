"""Tests for CoordinatorNode credential delegation."""

from unittest.mock import MagicMock

from src.core.session.coordinator import CoordinatorNode
from src.core.session.remote_session import RemoteSession


def test_delegate_credential_uses_shared_vault_without_explicit_argument():
    vault = MagicMock()
    vault.retrieve.return_value = "secret-value"
    coordinator = CoordinatorNode(vault=vault)

    session = MagicMock(spec=RemoteSession)
    coordinator.register_worker("worker-1", session)

    import asyncio

    ok = asyncio.get_event_loop().run_until_complete(
        coordinator.delegate_credential("api-key", "worker-1")
    )

    assert ok is True
    vault.retrieve.assert_called_once_with("api-key")


def test_delegate_credential_fails_when_credential_missing():
    vault = MagicMock()
    vault.retrieve.return_value = None
    coordinator = CoordinatorNode(vault=vault)

    import asyncio

    ok = asyncio.get_event_loop().run_until_complete(
        coordinator.delegate_credential("missing-key", "worker-1")
    )

    assert ok is False
"""Tests for CoordinatorNode credential delegation."""

from unittest.mock import MagicMock

from src.core.session.coordinator import CoordinatorNode
from src.core.session.remote_session import RemoteSession


def test_delegate_credential_uses_shared_vault_without_explicit_argument():
    vault = MagicMock()
    vault.retrieve.return_value = "secret-value"
    coordinator = CoordinatorNode(vault=vault)

    session = MagicMock(spec=RemoteSession)
    coordinator.register_worker("worker-1", session)

    import asyncio

    ok = asyncio.get_event_loop().run_until_complete(
        coordinator.delegate_credential("api-key", "worker-1")
    )

    assert ok is True
    vault.retrieve.assert_called_once_with("api-key")


def test_delegate_credential_fails_when_credential_missing():
    vault = MagicMock()
    vault.retrieve.return_value = None
    coordinator = CoordinatorNode(vault=vault)

    import asyncio

    ok = asyncio.get_event_loop().run_until_complete(
        coordinator.delegate_credential("missing-key", "worker-1")
    )

    assert ok is False
