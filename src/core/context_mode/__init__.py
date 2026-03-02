"""Context Mode — 95%+ token reduction for all LLM context usage.

Ported from claude-context-mode: FTS5 BM25 knowledge base, sandboxed execution,
and tool output interception so large data stays out of context.
"""

from src.core.context_mode.store import ContentStore, cleanup_stale_dbs, get_store
from src.core.context_mode.stats import SessionStats, get_session_stats

__all__ = [
    "ContentStore",
    "cleanup_stale_dbs",
    "get_store",
    "SessionStats",
    "get_session_stats",
]
