"""Agent identity and signed mandates (issue #614, Anvil Phase Chi).

SparkleForge's guard plane had no cryptographically verifiable way for an
agent to prove "I was delegated scope Y by human/agent X until time Z" --
`CapabilityManager` grants are local trust records with no signature, and
`CredentialVault` stores secrets but issues nothing that an external process
could verify. This gives every agent instance a real Ed25519 keypair
(`AgentIdentityManager`) and a signed mandate format (`Mandate`) modeled on
the field structure of W3C Verifiable Credentials / Google AP2 mandates
(issuer, subject, scope, not-before/not-after, signature) -- full format
compatibility with those specs is a later step; the field structure and a
working sign/verify round-trip are real now.

`verify_mandate()` takes only a public key (base64-encoded raw Ed25519
bytes) and the mandate itself -- no vault or private-key access required --
so a mandate's authenticity can be checked by a completely separate process
that only has the issuer's public key, satisfying the issue's success
criterion that verification must work outside this process.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from src.core.guard.credential_vault import CredentialVault

logger = logging.getLogger(__name__)

_PRIVATE_KEY_PREFIX = "agent_identity_private_key:"
_DEFAULT_REGISTRY_PATH = os.path.join("data", "agent_identity_public_keys.json")


@dataclass
class AgentIdentity:
    """An agent instance's real keypair. private_key never leaves this process."""

    agent_id: str
    private_key: ed25519.Ed25519PrivateKey
    public_key: ed25519.Ed25519PublicKey

    def public_key_b64(self) -> str:
        raw = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return base64.b64encode(raw).decode("ascii")


@dataclass
class Mandate:
    """A signed delegation: issuer grants subject `scope` until not_after.

    Field names deliberately mirror W3C Verifiable Credentials / AP2 mandate
    vocabulary (issuer, subject, scope, not-before/not-after) so a later
    step can map this onto those formats directly instead of renaming
    fields retroactively.
    """

    issuer: str
    subject: str
    scope: List[str]
    not_before: float
    not_after: float
    nonce: str = field(default_factory=lambda: uuid.uuid4().hex)
    signature: str = ""  # base64 Ed25519 signature; empty until issue_mandate() signs it

    def canonical_payload(self) -> bytes:
        """Deterministic byte representation that gets signed/verified.

        Excludes `signature` itself; scope is sorted so semantically
        identical mandates always produce the same bytes regardless of
        list ordering.
        """
        payload = {
            "issuer": self.issuer,
            "subject": self.subject,
            "scope": sorted(self.scope),
            "not_before": self.not_before,
            "not_after": self.not_after,
            "nonce": self.nonce,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def to_dict(self) -> Dict[str, object]:
        return {
            "issuer": self.issuer,
            "subject": self.subject,
            "scope": self.scope,
            "not_before": self.not_before,
            "not_after": self.not_after,
            "nonce": self.nonce,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Mandate":
        return cls(
            issuer=data["issuer"],
            subject=data["subject"],
            scope=list(data["scope"]),
            not_before=data["not_before"],
            not_after=data["not_after"],
            nonce=data["nonce"],
            signature=data.get("signature", ""),
        )


class AgentIdentityManager:
    """Issues and persists per-agent Ed25519 keypairs.

    Private keys are stored through CredentialVault (AES-GCM-encrypted at
    rest, same mechanism the rest of the guard plane already trusts for
    secrets). Public keys are kept in a small local JSON registry so other
    agents/processes can be handed a specific agent_id's public key to
    verify mandates it issued, without needing vault access.
    """

    _instance: Optional["AgentIdentityManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "AgentIdentityManager":
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(
        self,
        vault: Optional[CredentialVault] = None,
        registry_path: Optional[str] = None,
    ) -> None:
        # Guard against the attribute missing entirely if a prior
        # initialization attempt raised before `_initialized` was set.
        if getattr(self, "_initialized", False):
            return
        self.vault = vault or CredentialVault()
        self._registry_path = registry_path or _DEFAULT_REGISTRY_PATH
        self._public_keys: Dict[str, str] = {}
        self._lock_data = threading.RLock()
        try:
            self._load_registry()
        finally:
            # Only mark initialized after fallible work completes (or re-raises).
            self._initialized = True

    def _load_registry(self) -> None:
        if os.path.exists(self._registry_path):
            try:
                with open(self._registry_path, "r") as f:
                    self._public_keys = json.load(f)
            except Exception as e:
                logger.warning("Failed to load agent identity public key registry: %s", e)

    def _save_registry(self) -> None:
        directory = os.path.dirname(self._registry_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            with open(self._registry_path, "w") as f:
                json.dump(self._public_keys, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save agent identity public key registry: %s", e)

    def get_or_create_identity(self, agent_id: str) -> AgentIdentity:
        with self._lock_data:
            stored = self.vault.retrieve(f"{_PRIVATE_KEY_PREFIX}{agent_id}")
            if stored:
                private_key = ed25519.Ed25519PrivateKey.from_private_bytes(base64.b64decode(stored))
            else:
                private_key = ed25519.Ed25519PrivateKey.generate()
                raw = private_key.private_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PrivateFormat.Raw,
                    encryption_algorithm=serialization.NoEncryption(),
                )
                self.vault.store(f"{_PRIVATE_KEY_PREFIX}{agent_id}", base64.b64encode(raw).decode("ascii"))
                logger.info("Issued new Ed25519 identity for agent '%s'", agent_id)

            identity = AgentIdentity(
                agent_id=agent_id, private_key=private_key, public_key=private_key.public_key()
            )
            if self._public_keys.get(agent_id) != identity.public_key_b64():
                self._public_keys[agent_id] = identity.public_key_b64()
                self._save_registry()
            return identity

    def get_public_key_b64(self, agent_id: str) -> Optional[str]:
        with self._lock_data:
            return self._public_keys.get(agent_id)


_agent_identity_manager: Optional[AgentIdentityManager] = None


def get_agent_identity_manager() -> AgentIdentityManager:
    global _agent_identity_manager
    if _agent_identity_manager is None:
        _agent_identity_manager = AgentIdentityManager()
    return _agent_identity_manager


def issue_mandate(
    issuer_identity: AgentIdentity, subject: str, scope: List[str], ttl_seconds: float
) -> Mandate:
    """Issue a mandate delegating `scope` to `subject`, signed by issuer_identity, valid for ttl_seconds."""
    now = time.time()
    mandate = Mandate(
        issuer=issuer_identity.agent_id,
        subject=subject,
        scope=list(scope),
        not_before=now,
        not_after=now + ttl_seconds,
    )
    signature = issuer_identity.private_key.sign(mandate.canonical_payload())
    mandate.signature = base64.b64encode(signature).decode("ascii")
    return mandate


def verify_mandate(mandate: Mandate, issuer_public_key_b64: str) -> Tuple[bool, str]:
    """Verify a mandate's signature and validity window using only the issuer's public key.

    Deliberately takes no CredentialVault/AgentIdentityManager dependency --
    this must be callable from a separate process/agent that only has the
    issuer's public key, per the issue's success criterion.
    """
    now = time.time()
    if now < mandate.not_before:
        return False, f"mandate not yet valid (not_before={mandate.not_before}, now={now})"
    if now > mandate.not_after:
        return False, f"mandate expired (not_after={mandate.not_after}, now={now})"

    try:
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(issuer_public_key_b64))
        public_key.verify(base64.b64decode(mandate.signature), mandate.canonical_payload())
    except InvalidSignature:
        return False, "invalid signature"
    except Exception as e:
        return False, f"mandate verification error: {e}"

    return True, "valid"


def mandate_covers_capability(mandate: Mandate, required_capability: str) -> bool:
    return required_capability in mandate.scope
