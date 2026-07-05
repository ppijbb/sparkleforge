"""AES-256-GCM envelope encryption for credential handoff between nodes.

Credentials delegated from a CoordinatorNode to a WorkerNode must never cross
the transport in plaintext. Both sides derive the same AES key from a shared
secret established at pairing time; the credential key is bound into the
ciphertext as associated data so an envelope cannot be replayed under a
different credential name.
"""
import base64
import json
import logging
import os
from typing import Any, Dict, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)

_HKDF_INFO = b"sparkleforge-credential-delegation"
_NONCE_SIZE = 12


def _derive_key(shared_secret: str) -> bytes:
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_HKDF_INFO)
    return hkdf.derive(shared_secret.encode("utf-8"))


def encrypt_credential_envelope(
    shared_secret: str,
    credential_key: str,
    value: str,
    expires_at: float,
) -> str:
    """Seal a credential value and its expiry into a base64 envelope."""
    key = _derive_key(shared_secret)
    nonce = os.urandom(_NONCE_SIZE)
    plaintext = json.dumps({"value": value, "expires_at": expires_at}).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, credential_key.encode("utf-8"))
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_credential_envelope(
    shared_secret: str,
    credential_key: str,
    envelope: str,
) -> Optional[Dict[str, Any]]:
    """Open a sealed envelope; returns {"value", "expires_at"} or None if invalid."""
    try:
        raw = base64.b64decode(envelope.encode("ascii"), validate=True)
        if len(raw) <= _NONCE_SIZE:
            raise ValueError("envelope too short")
        nonce, ciphertext = raw[:_NONCE_SIZE], raw[_NONCE_SIZE:]
        key = _derive_key(shared_secret)
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, credential_key.encode("utf-8"))
        opened = json.loads(plaintext)
        if "value" not in opened or "expires_at" not in opened:
            raise ValueError("envelope missing required fields")
        return opened
    except (InvalidTag, ValueError, TypeError) as e:
        logger.warning(f"Failed to open credential envelope for '{credential_key}': {e}")
        return None
