"""
credential_vault.py — Secure credential storage using OS keyring with fallback to encrypted file.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Dict, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

# Anchor state files to the SparkleForge install root, never the runtime cwd,
# so coworker sessions don't leak the credential vault key into the target
# repo (issue #1331).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_SERVICE_NAME = "sparkleforge-anvil"

# Try OS keyring first
try:
    import keyring
    # Check if DBus is available on Linux to avoid crashes
    if os.name == "posix" and "DBUS_SESSION_BUS_ADDRESS" not in os.environ:
        _KEYRING_AVAILABLE = False
        logger.info("keyring available but no DBUS_SESSION_BUS_ADDRESS — using encrypted file fallback")
    else:
        _KEYRING_AVAILABLE = True
except ImportError:
    _KEYRING_AVAILABLE = False
    logger.info("keyring not available — using encrypted file fallback")


def _resolve_secret_seed(fallback_path: str) -> bytes:
    """Resolve the key material for the fallback file's AES-GCM encryption.

    Prefers SPARKLEFORGE_SECRET_SEED if set. Otherwise generates a random,
    per-installation key once and persists it next to the fallback store —
    never a fixed, source-visible default, since that would make the
    "encryption" purely cosmetic.
    """
    env_seed = os.environ.get("SPARKLEFORGE_SECRET_SEED")
    if env_seed:
        return env_seed.encode("utf-8")

    key_dir = os.path.dirname(fallback_path) or "."
    key_path = os.path.join(key_dir, ".vault_key")
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            return f.read()

    os.makedirs(key_dir, exist_ok=True)
    key = os.urandom(32)
    with open(key_path, "wb") as f:
        f.write(key)
    os.chmod(key_path, 0o600)
    return key


class CredentialVault:
    """
    Secure credential vault. Prefers OS keyring; falls back to XOR-obfuscated
    JSON file (not production-grade encryption, but better than plaintext).

    In production, replace the fallback with a proper AES-256-GCM implementation
    or integrate with 1Password CLI / HashiCorp Vault.
    """

    _instance: Optional["CredentialVault"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, fallback_path: Optional[str] = None) -> "CredentialVault":
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self, fallback_path: Optional[str] = None) -> None:
        if self._initialized:
            return
        self._initialized = True
        default_fallback = str(_PROJECT_ROOT / "data" / ".credential_store")
        self._fallback_path = fallback_path or default_fallback
        self._secret_seed = _resolve_secret_seed(self._fallback_path)
        self._cache: Dict[str, str] = {}
        self._lock_data = threading.RLock()

    def _encrypt(self, value: str) -> str:
        """AES-GCM encryption for fallback storage."""
        key = hashlib.sha256(self._secret_seed).digest()
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, value.encode("utf-8"), None)
        return base64.b64encode(nonce + ciphertext).decode()

    def _decrypt(self, value: str) -> str:
        key = hashlib.sha256(self._secret_seed).digest()
        aesgcm = AESGCM(key)
        data = base64.b64decode(value.encode())
        return aesgcm.decrypt(data[:12], data[12:], None).decode("utf-8")

    def _load_fallback(self) -> Dict[str, str]:
        if not os.path.exists(self._fallback_path):
            return {}
        try:
            with open(self._fallback_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load credential store: %s", e)
            return {}

    def _save_fallback(self, store: Dict[str, str]) -> None:
        os.makedirs(
            os.path.dirname(self._fallback_path) if os.path.dirname(self._fallback_path) else ".",
            exist_ok=True,
        )
        try:
            # Restrict permissions to owner-only
            with open(self._fallback_path, "w") as f:
                json.dump(store, f)
            os.chmod(self._fallback_path, 0o600)
        except Exception as e:
            logger.error("Failed to save credential store: %s", e)

    def store(self, key: str, value: str) -> bool:
        """Store a credential securely."""
        with self._lock_data:
            if _KEYRING_AVAILABLE:
                try:
                    keyring.set_password(_SERVICE_NAME, key, value)
                    self._cache[key] = value
                    logger.info("Stored credential '%s' in OS keyring", key)
                    return True
                except Exception as e:
                    logger.warning("Keyring store failed: %s — using fallback", e)

            # Fallback: obfuscated JSON file
            store = self._load_fallback()
            store[key] = self._encrypt(value)
            self._save_fallback(store)
            self._cache[key] = value
            logger.info("Stored credential '%s' in fallback store", key)
            return True

    def retrieve(self, key: str) -> Optional[str]:
        """Retrieve a stored credential."""
        with self._lock_data:
            if key in self._cache:
                return self._cache[key]

            if _KEYRING_AVAILABLE:
                try:
                    val = keyring.get_password(_SERVICE_NAME, key)
                    if val is not None:
                        self._cache[key] = val
                        return val
                except Exception as e:
                    logger.warning("Keyring retrieve failed: %s — using fallback", e)

            # Fallback
            store = self._load_fallback()
            if key in store:
                try:
                    val = self._decrypt(store[key])
                    self._cache[key] = val
                    return val
                except Exception as e:
                    logger.error("Failed to deobfuscate credential '%s': %s", key, e)

            return None

    def delete(self, key: str) -> bool:
        """Delete a stored credential."""
        with self._lock_data:
            self._cache.pop(key, None)

            if _KEYRING_AVAILABLE:
                try:
                    keyring.delete_password(_SERVICE_NAME, key)
                    return True
                except Exception:
                    pass

            store = self._load_fallback()
            if key in store:
                del store[key]
                self._save_fallback(store)
                return True
            return False

    def list_keys(self) -> list:
        """List stored credential keys (not values)."""
        keys = set(self._cache.keys())
        store = self._load_fallback()
        keys.update(store.keys())
        return sorted(keys)

    def reset(self) -> None:
        """Clear cache (for testing). Does NOT delete from OS keyring."""
        with self._lock_data:
            self._cache.clear()
