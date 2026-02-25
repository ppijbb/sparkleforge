"""Per-agent security enforcement engine.

Production-grade guardrails: input/output validation, MVI context scoping,
topic rails, tool access control, PII redaction, rate limiting, and audit
logging — all configurable per agent type via AgentSecurityConfig.
"""

from __future__ import annotations

import contextvars
import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from src.core.prompt_security import (
    REJECTION_MESSAGE,
    PromptInjectionFilter,
    get_input_filter,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII patterns (KISA/GDPR aligned)
# ---------------------------------------------------------------------------
_PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{6}[-\s]?\d{7}\b"), "[주민번호-REDACTED]"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3,4}[-.\s]?\d{4}\b"), "[전화번호-REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[이메일-REDACTED]"),
    (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[카드번호-REDACTED]"),
    (re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"), "[여권번호-REDACTED]"),
    (re.compile(r"(?i)(?:password|passwd|pwd)\s*[:=]\s*\S+"), "[PASSWORD-REDACTED]"),
    (re.compile(r"(?i)(?:secret|token|api[_\s]?key)\s*[:=]\s*\S+"), "[SECRET-REDACTED]"),
]


@dataclass
class SecurityViolation:
    """단일 보안 위반 레코드."""

    agent_name: str
    violation_type: str
    detail: str
    timestamp: float = field(default_factory=time.time)
    severity: str = "warning"


@dataclass
class AgentSecurityResult:
    """에이전트 보안 검증 결과."""

    is_allowed: bool
    violations: List[SecurityViolation] = field(default_factory=list)
    filtered_text: str = ""
    filtered_state: Optional[Dict[str, Any]] = None


class AgentSecurityManager:
    """에이전트별 보안 정책 시행 엔진."""

    def __init__(self) -> None:
        self._call_counters: Dict[str, int] = {}
        self._audit_log: List[SecurityViolation] = []
        self._compiled_output_patterns: Dict[str, List[re.Pattern]] = {}

    # ------------------------------------------------------------------
    # Config helpers (lazy import to avoid circular dependency)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_config():
        from src.core.researcher_config import get_agent_security_config
        try:
            return get_agent_security_config()
        except RuntimeError:
            return None

    @staticmethod
    def _get_policy(agent_name: str):
        from src.core.researcher_config import get_agent_security_policy
        try:
            return get_agent_security_policy(agent_name)
        except RuntimeError:
            from src.core.researcher_config import AgentSecurityPolicyEntry
            return AgentSecurityPolicyEntry()

    def _is_enabled(self) -> bool:
        cfg = self._get_config()
        return cfg.enabled if cfg else False

    # ------------------------------------------------------------------
    # Input enforcement
    # ------------------------------------------------------------------
    def enforce_input(
        self, agent_name: str, text: str
    ) -> AgentSecurityResult:
        if not self._is_enabled():
            return AgentSecurityResult(is_allowed=True, filtered_text=text)

        policy = self._get_policy(agent_name)
        violations: List[SecurityViolation] = []
        out = (text or "").strip()

        if len(out) > policy.input_max_length:
            out = out[: policy.input_max_length]
            violations.append(SecurityViolation(
                agent_name=agent_name,
                violation_type="input_truncated",
                detail=f"Input truncated to {policy.input_max_length} chars",
                severity="info",
            ))

        if policy.enable_injection_scan:
            flt = get_input_filter()
            detected, reason = flt.detect_injection(out)
            if detected:
                self._record(SecurityViolation(
                    agent_name=agent_name,
                    violation_type="injection_detected",
                    detail=f"Injection pattern: {reason}",
                    severity="critical",
                ))
                return AgentSecurityResult(
                    is_allowed=False,
                    violations=self._audit_log[-1:],
                    filtered_text="",
                )
            out = flt.sanitize(out)

        if policy.allowed_topics:
            if not self._check_topic_rail(out, policy.allowed_topics):
                v = SecurityViolation(
                    agent_name=agent_name,
                    violation_type="topic_violation",
                    detail=f"Content does not match allowed topics: {policy.allowed_topics}",
                    severity="warning",
                )
                violations.append(v)

        if policy.enable_pii_redaction:
            out = self._redact_pii(out)

        for v in violations:
            self._record(v)

        return AgentSecurityResult(
            is_allowed=True, violations=violations, filtered_text=out
        )

    # ------------------------------------------------------------------
    # Output enforcement
    # ------------------------------------------------------------------
    def enforce_output(
        self, agent_name: str, text: str
    ) -> AgentSecurityResult:
        if not self._is_enabled():
            return AgentSecurityResult(is_allowed=True, filtered_text=text)

        policy = self._get_policy(agent_name)
        violations: List[SecurityViolation] = []
        out = text or ""

        if len(out) > policy.output_max_length:
            out = out[: policy.output_max_length]
            violations.append(SecurityViolation(
                agent_name=agent_name,
                violation_type="output_truncated",
                detail=f"Output truncated to {policy.output_max_length} chars",
                severity="info",
            ))

        patterns = self._get_compiled_output_patterns(agent_name, policy.blocked_output_patterns)
        for pat in patterns:
            if pat.search(out):
                v = SecurityViolation(
                    agent_name=agent_name,
                    violation_type="blocked_output_pattern",
                    detail=f"Output matched blocked pattern: {pat.pattern}",
                    severity="critical",
                )
                violations.append(v)
                out = pat.sub("[BLOCKED]", out)

        if policy.enable_pii_redaction:
            out = self._redact_pii(out)

        for v in violations:
            self._record(v)

        return AgentSecurityResult(
            is_allowed=len([v for v in violations if v.severity == "critical"]) == 0,
            violations=violations,
            filtered_text=out,
        )

    # ------------------------------------------------------------------
    # MVI context scoping
    # ------------------------------------------------------------------
    def filter_state_mvi(
        self, agent_name: str, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self._is_enabled():
            return state

        policy = self._get_policy(agent_name)
        if not policy.context_scope:
            return state

        allowed: Set[str] = set(policy.context_scope)
        filtered = {}
        for key, value in state.items():
            if key in allowed:
                filtered[key] = value

        denied_keys = set(state.keys()) - allowed
        if denied_keys:
            logger.debug(
                "[%s] MVI: denied access to state keys: %s",
                agent_name,
                denied_keys,
            )
            if self._get_config() and self._get_config().audit_logging:
                self._record(SecurityViolation(
                    agent_name=agent_name,
                    violation_type="mvi_context_filter",
                    detail=f"Filtered state keys: {sorted(denied_keys)}",
                    severity="info",
                ))

        return filtered

    # ------------------------------------------------------------------
    # Tool access control
    # ------------------------------------------------------------------
    def check_tool_permission(
        self, agent_name: str, tool_name: str, tool_category: str = ""
    ) -> bool:
        if not self._is_enabled():
            return True

        policy = self._get_policy(agent_name)
        if not policy.allowed_tool_categories:
            return True

        cat_lower = (tool_category or "").lower()
        allowed = {c.lower() for c in policy.allowed_tool_categories}
        if cat_lower and cat_lower not in allowed:
            self._record(SecurityViolation(
                agent_name=agent_name,
                violation_type="tool_access_denied",
                detail=f"Tool '{tool_name}' (category: {cat_lower}) not in allowed: {sorted(allowed)}",
                severity="warning",
            ))
            return False
        return True

    # ------------------------------------------------------------------
    # Rate limiting (LLM calls per execution)
    # ------------------------------------------------------------------
    def check_rate_limit(self, agent_name: str) -> bool:
        if not self._is_enabled():
            return True

        policy = self._get_policy(agent_name)
        count = self._call_counters.get(agent_name, 0)
        if count >= policy.max_llm_calls_per_execution:
            self._record(SecurityViolation(
                agent_name=agent_name,
                violation_type="rate_limit_exceeded",
                detail=f"LLM calls ({count}) >= limit ({policy.max_llm_calls_per_execution})",
                severity="critical",
            ))
            return False
        self._call_counters[agent_name] = count + 1
        return True

    def reset_rate_limit(self, agent_name: str) -> None:
        self._call_counters.pop(agent_name, None)

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------
    def get_audit_log(
        self, agent_name: str | None = None, limit: int = 100
    ) -> List[SecurityViolation]:
        if agent_name:
            entries = [v for v in self._audit_log if v.agent_name == agent_name]
        else:
            entries = list(self._audit_log)
        return entries[-limit:]

    def get_security_summary(self) -> Dict[str, Any]:
        if not self._audit_log:
            return {"total_violations": 0, "by_agent": {}, "by_severity": {}}

        by_agent: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for v in self._audit_log:
            by_agent[v.agent_name] = by_agent.get(v.agent_name, 0) + 1
            by_severity[v.severity] = by_severity.get(v.severity, 0) + 1

        return {
            "total_violations": len(self._audit_log),
            "by_agent": by_agent,
            "by_severity": by_severity,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record(self, violation: SecurityViolation) -> None:
        self._audit_log.append(violation)
        cfg = self._get_config()
        if cfg and cfg.audit_logging:
            log_fn = logger.warning if violation.severity in ("warning", "critical") else logger.info
            log_fn(
                "[SECURITY:%s] %s — %s (severity=%s)",
                violation.agent_name,
                violation.violation_type,
                violation.detail,
                violation.severity,
            )

    def _check_topic_rail(self, text: str, allowed_topics: List[str]) -> bool:
        if not allowed_topics:
            return True
        text_lower = text.lower()
        return any(topic.lower() in text_lower for topic in allowed_topics)

    @staticmethod
    def _redact_pii(text: str) -> str:
        out = text
        for pattern, replacement in _PII_PATTERNS:
            out = pattern.sub(replacement, out)
        return out

    def _get_compiled_output_patterns(
        self, agent_name: str, raw_patterns: List[str]
    ) -> List[re.Pattern]:
        if agent_name in self._compiled_output_patterns:
            return self._compiled_output_patterns[agent_name]
        compiled = []
        for p in raw_patterns:
            try:
                compiled.append(re.compile(p, re.IGNORECASE))
            except re.error as e:
                logger.warning("Invalid blocked_output_pattern '%s': %s", p, e)
        self._compiled_output_patterns[agent_name] = compiled
        return compiled


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_manager: Optional[AgentSecurityManager] = None


def get_agent_security_manager() -> AgentSecurityManager:
    global _manager
    if _manager is None:
        _manager = AgentSecurityManager()
    return _manager


# ---------------------------------------------------------------------------
# Context-propagated agent name (avoids modifying every execute_llm_task call)
# ---------------------------------------------------------------------------
_current_agent_name: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_agent_name", default=None
)


def get_current_agent_name() -> str | None:
    return _current_agent_name.get()


@contextmanager
def agent_security_context(agent_name: str) -> Generator[None, None, None]:
    """Context manager that sets the active agent name for downstream
    security checks (rate limit, output enforcement) via contextvars."""
    token = _current_agent_name.set(agent_name)
    try:
        yield
    finally:
        _current_agent_name.reset(token)
