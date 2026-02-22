"""Prompt security layer: input validation/sanitization and output validation.

OWASP LLM Prompt Injection Prevention patterns and control-data separation support.
Used at orchestrator entry points and (optionally) for external content before LLM context.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment (no hardcoded secrets)
# ---------------------------------------------------------------------------
def _env_bool(key: str, default: bool = True) -> bool:
    return os.getenv(key, "true" if default else "false").lower() in ("true", "1", "yes")


def _env_int(key: str, default: int, min_val: int = 1, max_val: int = 1_000_000) -> int:
    try:
        v = int(os.getenv(key, str(default)))
        return max(min_val, min(max_val, v))
    except ValueError:
        return default


ENABLE_PROMPT_SECURITY = _env_bool("ENABLE_PROMPT_SECURITY", True)
ENABLE_OUTPUT_VALIDATION = _env_bool("ENABLE_OUTPUT_VALIDATION", True)
MAX_USER_INPUT_LENGTH = _env_int("MAX_USER_INPUT_LENGTH", 20_000, 1000, 500_000)
MAX_OUTPUT_LENGTH = _env_int("MAX_OUTPUT_LENGTH", 50_000, 1000, 1_000_000)

# Rejection message shown to user (do not include request details)
REJECTION_MESSAGE = "요청을 처리할 수 없습니다. 입력 내용을 확인해 주세요."

# ---------------------------------------------------------------------------
# Injection patterns (aligned with memory_validation + OWASP cheat sheet)
# ---------------------------------------------------------------------------
INJECTION_PATTERNS: List[Tuple[str, str]] = [
    # Direct prompt injection
    (r"ignore\s+(previous|all|above)\s+instructions?", "prompt_injection"),
    (r"forget\s+(everything|all|previous)", "prompt_injection"),
    (r"reveal\s+(your\s+)?(system\s+)?prompt", "prompt_extraction"),
    (r"you\s+are\s+now\s+(in\s+)?(developer\s+mode|a\s+|an\s+)", "role_hijacking"),
    (r"system\s*:\s*", "system_prompt_injection"),
    (r"<\|(system|assistant|user)\|>", "token_injection"),
    (r"new\s+instructions?\s*:", "instruction_override"),
    (r"override\s+(instructions?|rules?|system)", "instruction_override"),
    (r"bypass\s+(all\s+)?(safety|restrictions?)", "bypass"),
    (r"disregard\s+(all\s+)?(previous|above)", "prompt_injection"),
    (r"what\s+were\s+your\s+(exact\s+)?instructions?", "prompt_extraction"),
    (r"repeat\s+(the\s+)?(text\s+)?above\s+starting", "prompt_extraction"),
    # Data manipulation
    (r"delete\s+(all|everything|memories?|data)", "data_manipulation"),
    (r"clear\s+(all|everything|memories?)", "data_manipulation"),
    (r"reset\s+(all|everything|memories?)", "data_manipulation"),
]

# Exported for reuse (e.g. memory_validation); (regex_str, name)
PUBLIC_INJECTION_PATTERNS: List[Tuple[str, str]] = list(INJECTION_PATTERNS)

# Typoglycemia-style fuzzy target words (first/last letter match)
FUZZY_DANGER_WORDS = [
    "ignore", "bypass", "override", "reveal", "delete", "system", "prompt",
]


@dataclass
class InputValidationResult:
    """Result of user input validation."""

    is_safe: bool
    rejection_reason: Optional[str] = None  # Log-only; not shown to user
    sanitized_text: str = ""

    def __post_init__(self) -> None:
        if self.sanitized_text is None:
            self.sanitized_text = ""


class PromptInjectionFilter:
    """OWASP-style input filter for prompt injection and abuse."""

    def __init__(
        self,
        max_length: int = MAX_USER_INPUT_LENGTH,
        enable_fuzzy: bool = True,
        patterns: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.max_length = max_length
        self.enable_fuzzy = enable_fuzzy
        self._patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in (patterns or INJECTION_PATTERNS)
        ]

    def detect_injection(self, text: str) -> Tuple[bool, Optional[str]]:
        """Returns (True, reason) if injection detected, else (False, None)."""
        if not text or not text.strip():
            return False, None
        sample = text[: self.max_length * 2]  # Scan a bit more for pattern at boundary
        for pattern, name in self._patterns:
            if pattern.search(sample):
                return True, name
        if self.enable_fuzzy:
            reason = self._fuzzy_match(sample)
            if reason:
                return True, reason
        # Check decoded content (base64/hex) for embedded injection
        decoded = self._decode_and_scan(sample)
        if decoded:
            for pattern, name in self._patterns:
                if pattern.search(decoded):
                    return True, f"encoded_{name}"
        return False, None

    def _fuzzy_match(self, text: str) -> Optional[str]:
        """Typoglycemia-style: same first/last letter, scrambled middle."""
        words = re.findall(r"\b\w+\b", text.lower())
        for word in words:
            if len(word) < 4:
                continue
            for target in FUZZY_DANGER_WORDS:
                if len(word) != len(target):
                    continue
                if word[0] == target[0] and word[-1] == target[-1]:
                    if sorted(word[1:-1]) == sorted(target[1:-1]):
                        return "typoglycemia"
        return None

    def _decode_and_scan(self, text: str) -> Optional[str]:
        """If text looks like base64 or hex, decode and return for pattern scan."""
        stripped = text.strip()
        # Base64-ish (alphanumeric + /+=)
        if len(stripped) >= 20 and re.match(r"^[A-Za-z0-9+/=]+$", stripped):
            try:
                raw = base64.b64decode(stripped, validate=True)
                return raw.decode("utf-8", errors="replace")
            except (binascii.Error, ValueError):
                pass
        # Hex
        if len(stripped) >= 24 and re.match(r"^[0-9a-fA-F]+$", stripped):
            try:
                raw = bytes.fromhex(stripped)
                return raw.decode("utf-8", errors="replace")
            except (ValueError, UnicodeDecodeError):
                pass
        return None

    def sanitize(self, text: str) -> str:
        """Remove or replace injection patterns and enforce length."""
        if not text:
            return ""
        out = text
        for pattern, _ in self._patterns:
            out = pattern.sub("[REMOVED]", out)
        out = re.sub(r"\[REMOVED\](?:\s*\[REMOVED\])*", "[REMOVED]", out)
        out = re.sub(r"\s+", " ", out).strip()
        return out[: self.max_length]


class OutputValidator:
    """Validates LLM output for prompt leakage and sensitive patterns."""

    def __init__(self, max_length: int = MAX_OUTPUT_LENGTH) -> None:
        self.max_length = max_length
        self._leak_patterns = [
            re.compile(r"SYSTEM\s*[:]\s*You\s+are", re.IGNORECASE),
            re.compile(r"API[_\s]?KEY\s*[:=]\s*\S+", re.IGNORECASE),
            re.compile(r"instructions?\s*[:]\s*\d+\.", re.IGNORECASE),
            re.compile(r"secret[_\s]?key\s*[:=]\s*\S+", re.IGNORECASE),
        ]

    def validate_output(self, text: str) -> Tuple[bool, str]:
        """Returns (is_acceptable, filtered_or_original_text)."""
        if not text:
            return True, ""
        out = text
        if len(out) > self.max_length:
            out = out[: self.max_length] + "\n[Output truncated for length.]"
        for pattern in self._leak_patterns:
            if pattern.search(out):
                logger.warning("Output validation failed: suspicious pattern detected")
                return False, REJECTION_MESSAGE
        return True, out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_input_filter: Optional[PromptInjectionFilter] = None
_output_validator: Optional[OutputValidator] = None


def get_input_filter() -> PromptInjectionFilter:
    global _input_filter
    if _input_filter is None:
        _input_filter = PromptInjectionFilter()
    return _input_filter


def get_output_validator() -> OutputValidator:
    global _output_validator
    if _output_validator is None:
        _output_validator = OutputValidator()
    return _output_validator


def validate_user_input(text: str) -> InputValidationResult:
    """Validate and optionally sanitize user input. Safe to call with empty or None."""
    if not ENABLE_PROMPT_SECURITY:
        return InputValidationResult(
            is_safe=True,
            sanitized_text=(text or "")[:MAX_USER_INPUT_LENGTH],
        )
    if text is None:
        text = ""
    text = text.strip()
    if not text:
        return InputValidationResult(is_safe=True, sanitized_text="")
    flt = get_input_filter()
    detected, reason = flt.detect_injection(text)
    if detected:
        logger.warning("Prompt security: input rejected (reason=%s)", reason)
        return InputValidationResult(
            is_safe=False,
            rejection_reason=reason,
            sanitized_text="",
        )
    sanitized = flt.sanitize(text)
    return InputValidationResult(is_safe=True, sanitized_text=sanitized)


def validate_llm_output(text: str) -> Tuple[bool, str]:
    """Validate LLM output. Returns (ok, final_text)."""
    if not ENABLE_OUTPUT_VALIDATION:
        return True, (text or "")[:MAX_OUTPUT_LENGTH]
    if text is None:
        text = ""
    return get_output_validator().validate_output(text)
