"""SessionStats — per-session context consumption and savings.

Tracks bytes returned to context vs indexed/sandboxed.
"""

import time
from typing import Any, Dict

_session_stats: "SessionStats | None" = None


class SessionStats:
    """Per-session stats: calls, bytes returned, bytes indexed, bytes sandboxed."""

    def __init__(self) -> None:
        self.calls: Dict[str, int] = {}
        self.bytes_returned: Dict[str, int] = {}
        self.bytes_indexed: int = 0
        self.bytes_sandboxed: int = 0
        self.session_start: float = time.time()

    def track_response(self, tool_name: str, bytes_count: int) -> None:
        self.calls[tool_name] = self.calls.get(tool_name, 0) + 1
        self.bytes_returned[tool_name] = self.bytes_returned.get(tool_name, 0) + bytes_count

    def track_indexed(self, bytes_count: int) -> None:
        self.bytes_indexed += bytes_count

    def track_sandboxed(self, bytes_count: int) -> None:
        self.bytes_sandboxed += bytes_count

    def total_bytes_returned(self) -> int:
        return sum(self.bytes_returned.values())

    def total_calls(self) -> int:
        return sum(self.calls.values())

    def kept_out(self) -> int:
        """Total bytes kept out of context (indexed + sandboxed)."""
        return self.bytes_indexed + self.bytes_sandboxed

    def total_processed(self) -> int:
        return self.kept_out() + self.total_bytes_returned()

    def savings_ratio(self) -> float:
        total = self.total_processed()
        returned = self.total_bytes_returned()
        if returned <= 0:
            return 0.0
        return total / returned

    def reduction_percent(self) -> str:
        total = self.total_processed()
        if total <= 0:
            return "0"
        returned = self.total_bytes_returned()
        pct = (1 - returned / total) * 100
        return f"{pct:.0f}"

    def estimated_tokens_returned(self) -> int:
        return round(self.total_bytes_returned() / 4)

    def format_summary(self) -> str:
        """Human-readable summary table."""
        lines = [
            "## context-mode session stats",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Session | {(time.time() - self.session_start) / 60:.1f} min |",
            f"| Tool calls | {self.total_calls()} |",
            f"| Total data processed | **{self._kb(self.total_processed())}** |",
            f"| Kept in sandbox | **{self._kb(self.kept_out())}** |",
            f"| Entered context | {self._kb(self.total_bytes_returned())} |",
            f"| Tokens consumed | ~{self.estimated_tokens_returned():,} |",
            f"| **Context savings** | **{self.savings_ratio():.1f}x ({self.reduction_percent()}% reduction)** |",
        ]
        all_tools = sorted(set(self.calls) | set(self.bytes_returned))
        if all_tools:
            lines.extend(["", "| Tool | Calls | Context | Tokens |", "|------|------:|--------:|-------:|"])
            for tool in all_tools:
                calls = self.calls.get(tool, 0)
                bytes_ = self.bytes_returned.get(tool, 0)
                tokens = round(bytes_ / 4)
                lines.append(f"| {tool} | {calls} | {self._kb(bytes_)} | ~{tokens:,} |")
            lines.append(
                f"| **Total** | **{self.total_calls()}** | **{self._kb(self.total_bytes_returned())}** | "
                f"**~{self.estimated_tokens_returned():,}** |"
            )
        return "\n".join(lines)

    @staticmethod
    def _kb(b: int) -> str:
        if b >= 1024 * 1024:
            return f"{b / 1024 / 1024:.1f}MB"
        return f"{b / 1024:.1f}KB"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": dict(self.calls),
            "bytes_returned": dict(self.bytes_returned),
            "bytes_indexed": self.bytes_indexed,
            "bytes_sandboxed": self.bytes_sandboxed,
            "session_start": self.session_start,
            "total_bytes_returned": self.total_bytes_returned(),
            "total_calls": self.total_calls(),
            "kept_out": self.kept_out(),
            "savings_ratio": self.savings_ratio(),
            "reduction_percent": self.reduction_percent(),
        }


def get_session_stats() -> SessionStats:
    """Global session stats singleton."""
    global _session_stats
    if _session_stats is None:
        _session_stats = SessionStats()
    return _session_stats


def reset_session_stats() -> None:
    """Reset global stats (e.g. for tests)."""
    global _session_stats
    _session_stats = SessionStats()
