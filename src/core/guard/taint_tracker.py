"""TaintTracker: behavioral semantic firewall for taint flow and intent drift.

Protects against indirect prompt injection and secret exfiltration by tagging
data originating from untrusted web/MCP responses and blocking/prompting when
tainted values flow into .env/credential reads or outbound network/write tool arguments.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

@dataclass
class TaintInfo:
    source: str
    tainted_keys: Set[str] = field(default_factory=set)
    payload_snippets: List[str] = field(default_factory=list)

class TaintTracker:
    """Tracks tainted data flow from untrusted sources (web, MCP fetch, etc.)
    and checks outgoing sensitive actions/arguments against tainted propagation.
    """

    def __init__(self) -> None:
        self._tainted_values: Set[str] = set()
        # Patterns indicating sensitive targets or exfiltration vectors
        self.sensitive_target_patterns = [
            r"\.env",
            r"credentials?",
            r"secret",
            r"password",
            r"api[_-]?key",
            r"token",
        ]
        self.outbound_risk_capabilities = {
            "execute_shell",
            "write_file",
            "network_request",
        }

    def mark_tainted(self, data: Any, source: str = "untrusted_source") -> None:
        """Recursively or stringently tag data/strings as tainted."""
        if data is None:
            return
        if isinstance(data, str):
            if len(data) > 3:
                self._tainted_values.add(data)
                # Also add tokens or substrings if they look like prompt injections or instructions
                for line in data.splitlines():
                    cleaned = line.strip()
                    if len(cleaned) > 5:
                        self._tainted_values.add(cleaned)
        elif isinstance(data, dict):
            for k, v in data.items():
                self.mark_tainted(k, source)
                self.mark_tainted(v, source)
        elif isinstance(data, (list, tuple, set)):
            for item in data:
                self.mark_tainted(item, source)

    def is_tainted(self, value: Any) -> bool:
        """Check if a value or any part of it matches known tainted entries or injection markers."""
        if value is None:
            return False
        val_str = str(value)
        # Direct match or substring containment of significant tainted snippets
        for tainted in self._tainted_values:
            if tainted and tainted in val_str:
                return True
        # Heuristic check for common injection patterns inside untrusted data
        injection_markers = [
            "ignore previous instructions",
            "exfiltrate",
            "curl ",
            "wget ",
            "cat .env",
            "printenv",
        ]
        lower_val = val_str.lower()
        for marker in injection_markers:
            if marker in lower_val and len(self._tainted_values) > 0:
                return True
        return False

    def check_flow(self, target: str, arguments: Any, capability: Optional[str] = None) -> Tuple[bool, str]:
        """Verify whether tainted data is flowing into a sensitive target or outbound tool call.

        Returns:
            (allowed, reason)
        """
        # Check target name for sensitive file/credential access
        target_lower = target.lower()
        is_sensitive_target = any(re.search(pat, target_lower) for pat in self.sensitive_target_patterns)

        # Check if arguments contain tainted values
        has_taint = self.is_tainted(arguments) or self.is_tainted(target)

        if has_taint and (is_sensitive_target or (capability in self.outbound_risk_capabilities)):
            reason = f"TaintTracker firewall block: tainted untrusted data flow detected into target '{target}' with capability '{capability}'"
            logger.warning(reason)
            return False, reason

        return True, "clean"

_taint_tracker: Optional[TaintTracker] = None

def get_taint_tracker() -> TaintTracker:
    global _taint_tracker
    if _taint_tracker is None:
        _taint_tracker = TaintTracker()
    return _taint_tracker
