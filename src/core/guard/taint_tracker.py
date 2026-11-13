"""
Behavioral Semantic Firewall: Taint Tracker and Intent Drift Middleware.
Tags data originating from untrusted web/MCP sources and intercepts tool calls
where tainted values flow into sensitive destinations (.env, credentials, outbound writes).
"""

import time
import re
import logging
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger("sparkleforge.core.guard.taint_tracker")


class TaintedValue:
    """Wrapper for values tagged as untrusted/tainted from external sources."""
    def __init__(self, value: Any, source: str = "untrusted_web_mcp"):
        self.value = value
        self.source = source
        self.timestamp = time.time()

    def __repr__(self) -> str:
        return f"TaintedValue({repr(self.value)}, source={repr(self.source)})"

    def __str__(self) -> str:
        return str(self.value)


class TaintTracker:
    """
    Tracks taint flows and validates tool invocations against data exfiltration
    and prompt injection risks.
    """
    def __init__(self):
        # Patterns indicating sensitive operations
        self.sensitive_destinations = [
            ".env",
            "credential",
            "secret",
            "token",
            "api_key",
            "password",
            "config",
        ]
        # Outbound network or write actions
        self.outbound_actions = [
            "curl",
            "fetch",
            "http",
            "requests",
            "write",
            "upload",
            "send",
            "post",
            "put",
            "bash",
            "shell",
            "exec",
        ]

    def tag(self, data: Any, source: str = "untrusted_web_mcp") -> Any:
        """Recursively or directly tag untrusted data as tainted."""
        if isinstance(data, TaintedValue):
            return data
        if isinstance(data, str):
            return TaintedValue(data, source=source)
        elif isinstance(data, dict):
            return {k: self.tag(v, source) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.tag(v, source) for v in data]
        return data

    def is_tainted(self, obj: Any) -> bool:
        """Check if an object or any of its nested contents are tainted."""
        if isinstance(obj, TaintedValue):
            return True
        if isinstance(obj, dict):
            return any(self.is_tainted(v) for v in obj.values())
        if isinstance(obj, list):
            return any(self.is_tainted(item) for item in obj)
        if isinstance(obj, str):
            pass
        return False

    def inspect_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inspects a tool call before execution.
        If tainted data flows into sensitive destinations or outbound network/write calls,
        raises a SecurityError or blocks/requests approval.
        
        Quantified target: Per-tool-call inspection latency <= 10 ms.
        """
        start_time = time.time()
        try:
            tool_lower = tool_name.lower()
            
            # Check if any argument is tainted
            has_taint = self.is_tainted(arguments)
            
            if has_taint:
                # Check if tool is writing or sending outbound or touching sensitive files
                is_outbound = any(action in tool_lower for action in self.outbound_actions)
                
                # Check arguments for sensitive paths (.env, secrets)
                args_str = str(arguments).lower()
                touches_sensitive = any(dest in args_str for dest in self.sensitive_destinations)
                
                if is_outbound or touches_sensitive:
                    logger.warning(
                        f"[TaintTracker] BLOCKED: Tainted data from untrusted source attempted flow into tool '{tool_name}' with args {arguments}"
                    )
                    raise PermissionError(
                        f"Behavioral Semantic Firewall: Blocked potential indirect prompt injection / "
                        f"exfiltration via tool '{tool_name}'. Tainted data detected in sensitive context."
                    )
            
            # Also check for embedded exfiltration patterns in string arguments even if untagged explicitly
            args_str_raw = str(arguments)
            if self._detect_exfiltration_intent(args_str_raw):
                logger.warning(f"[TaintTracker] BLOCKED: Intent drift detected in tool '{tool_name}'")
                raise PermissionError(
                    f"Behavioral Semantic Firewall: Blocked tool '{tool_name}' due to detected intent drift / exfiltration pattern."
                )
                
            return {"allowed": True, "tool_name": tool_name}
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > 10.0:
                logger.debug(f"[TaintTracker] Warning: Inspection latency {elapsed_ms:.2f}ms exceeded 10ms target.")

    def _detect_exfiltration_intent(self, text: str) -> bool:
        """Detects common patterns of exfiltrating secrets (e.g. cat .env | curl)."""
        lower = text.lower()
        exfil_patterns = [
            r"cat\s+.*\.env",
            r"printenv",
            r"env\s*\|",
            r"curl\s+.*@.*\.env",
            r"base64\s+.*\.env",
        ]
        for pattern in exfil_patterns:
            if re.search(pattern, lower):
                return True
        return False


# Global singleton instance
taint_tracker = TaintTracker()
