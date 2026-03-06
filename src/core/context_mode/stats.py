"""SessionStats — per-session context consumption and savings.

Tracks bytes returned to context vs indexed/sandboxed.
Supports snapshot/delta for per-task savings reporting.
"""

import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, List

_session_stats: "SessionStats | None" = None

# Per-turn tool context savings (raw vs returned bytes) for task-level reporting
_current_turn_tool_savings: ContextVar[List[Dict[str, Any]]] = ContextVar(
    "turn_tool_savings", default=[]
)


def record_tool_context_savings(
    tool_name: str,
    raw_bytes: int,
    returned_bytes: int,
    turn_id: str | None = None,
) -> None:
    """Record one tool call's context-mode savings for the current turn."""
    try:
        records = list(_current_turn_tool_savings.get())
    except LookupError:
        records = []
    kept_out = max(0, raw_bytes - returned_bytes)
    records.append({
        "tool_name": tool_name,
        "raw_bytes": raw_bytes,
        "returned_bytes": returned_bytes,
        "kept_out_bytes": kept_out,
        "turn_id": turn_id,
    })
    _current_turn_tool_savings.set(records)


def get_current_turn_tool_savings() -> List[Dict[str, Any]]:
    """Return list of tool context savings for the current turn."""
    try:
        return list(_current_turn_tool_savings.get())
    except LookupError:
        return []


def clear_current_turn_tool_savings() -> None:
    """Clear recorded tool savings (call at end of turn)."""
    try:
        _current_turn_tool_savings.set([])
    except LookupError:
        pass


@dataclass
class SessionStatsSnapshot:
    """Point-in-time snapshot for computing task-level delta."""

    total_bytes_returned: int
    kept_out: int
    total_processed: int
    total_calls: int
    timestamp: float


@dataclass
class SessionStatsDelta:
    """Delta between two snapshots (e.g. before/after a task)."""

    bytes_returned_delta: int
    kept_out_delta: int
    processed_delta: int
    calls_delta: int
    savings_ratio: float
    reduction_percent: float


class SessionStats:
    """Per-session stats: calls, bytes returned, bytes indexed, bytes sandboxed."""

    def __init__(self) -> None:
        self.calls: Dict[str, int] = {}
        self.bytes_returned: Dict[str, int] = {}
        self.bytes_indexed: int = 0
        self.bytes_sandboxed: int = 0
        self.session_start: float = time.time()

    def snapshot(self) -> SessionStatsSnapshot:
        """Capture current state for later delta calculation."""
        return SessionStatsSnapshot(
            total_bytes_returned=self.total_bytes_returned(),
            kept_out=self.kept_out(),
            total_processed=self.total_processed(),
            total_calls=self.total_calls(),
            timestamp=time.time(),
        )

    def delta(self, from_snapshot: SessionStatsSnapshot) -> SessionStatsDelta:
        """Compute delta from a previous snapshot."""
        now_returned = self.total_bytes_returned()
        now_kept = self.kept_out()
        now_processed = self.total_processed()
        now_calls = self.total_calls()
        bytes_delta = now_returned - from_snapshot.total_bytes_returned
        kept_delta = now_kept - from_snapshot.kept_out
        processed_delta = now_processed - from_snapshot.total_processed
        calls_delta = now_calls - from_snapshot.total_calls
        if now_processed > 0:
            ratio = now_processed / now_returned if now_returned > 0 else 0.0
            reduction = (1 - now_returned / now_processed) * 100
        else:
            ratio = 0.0
            reduction = 0.0
        return SessionStatsDelta(
            bytes_returned_delta=bytes_delta,
            kept_out_delta=kept_delta,
            processed_delta=processed_delta,
            calls_delta=calls_delta,
            savings_ratio=ratio,
            reduction_percent=reduction,
        )

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
