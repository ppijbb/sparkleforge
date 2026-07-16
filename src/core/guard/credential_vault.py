"""
credential_vault.py — Secure credential storage using OS keyring with fallback to encrypted file.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

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

_SERVICE_NAME = "sparkleforge-anvil"


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
        self._fallback_path = fallback_path or os.path.join("data", ".credential_store")
        self._cache: Dict[str, str] = {}
        self._lock_data = threading.RLock()

    def _obfuscate(self, value: str) -> str:
        """Simple XOR obfuscation for fallback storage (not cryptographic)."""
        key = hashlib.sha256(_SERVICE_NAME.encode()).digest()
        raw = value.encode("utf-8")
        obfuscated = bytes(b ^ key[i % len(key)] for i, b in enumerate(raw))
        return base64.b64encode(obfuscated).decode()

    def _deobfuscate(self, value: str) -> str:
        key = hashlib.sha256(_SERVICE_NAME.encode()).digest()
        raw = base64.b64decode(value.encode())
        original = bytes(b ^ key[i % len(key)] for i, b in enumerate(raw))
        return original.decode("utf-8")

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
            store[key] = self._obfuscate(value)
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
                    val = self._deobfuscate(store[key])
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
