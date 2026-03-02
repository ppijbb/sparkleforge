"""ToolOutputInterceptor — reduce tool output before it enters LLM context.

When output exceeds threshold, index into FTS5 and return summary/section previews.
Achieves 95%+ token reduction for all LLMs (Gemini, OpenRouter, Groq, etc.).
"""

import logging
from typing import Any, Dict, List, Optional

from src.core.context_mode.snippet import extract_snippet
from src.core.context_mode.stats import get_session_stats

logger = logging.getLogger(__name__)

INTENT_SEARCH_THRESHOLD_BYTES = 5_000  # ~80–100 lines
def _get_store():
    """Lazy ContentStore singleton (avoids FTS5 overhead until first index/search)."""
    from src.core.context_mode.store import get_store
    return get_store()


def _content_from_result(result: Dict[str, Any]) -> tuple[str, int]:
    """Extract single text from MCP tool result. Returns (text, byte_count)."""
    content = result.get("content") or []
    if not isinstance(content, list):
        text = str(content)
        return text, len(text.encode("utf-8", errors="replace"))
    parts: List[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
        elif isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    text = "\n".join(parts)
    return text, len(text.encode("utf-8", errors="replace"))


def _replace_result_content(result: Dict[str, Any], new_text: str) -> Dict[str, Any]:
    """Return a copy of result with content set to new_text."""
    out = dict(result)
    out["content"] = [{"type": "text", "text": new_text}]
    return out


def _intent_search(stdout: str, intent: str, source: str, max_results: int = 5) -> str:
    """Index stdout into store, search by intent, return section titles + previews."""
    store = _get_store()
    stats = get_session_stats()
    stats.track_indexed(len(stdout.encode("utf-8", errors="replace")))
    indexed = store.index_plain_text(stdout, source)
    results = store.search_with_fallback(intent, max_results, source)
    distinctive = store.get_distinctive_terms(indexed.source_id)
    total_lines = len(stdout.splitlines())
    total_kb = len(stdout.encode("utf-8", errors="replace")) / 1024
    if not results:
        lines = [
            f'Indexed {indexed.total_chunks} sections from "{source}" into knowledge base.',
            f'No sections matched intent "{intent}" in {total_lines}-line output ({total_kb:.1f}KB).',
        ]
        if distinctive:
            lines.extend(["", f"Searchable terms: {', '.join(distinctive)}"])
        lines.extend(["", "Use search() to explore the indexed content."])
        return "\n".join(lines)
    lines = [
        f'Indexed {indexed.total_chunks} sections from "{source}" into knowledge base.',
        f'{len(results)} sections matched "{intent}" ({total_lines} lines, {total_kb:.1f}KB):',
        "",
    ]
    for r in results:
        preview = (r.content.split("\n")[0][:120]) if r.content else ""
        lines.append(f"  - {r.title}: {preview}")
    if distinctive:
        lines.extend(["", f"Searchable terms: {', '.join(distinctive)}"])
    lines.extend(["", "Use search(queries: [...]) to retrieve full content of any section."])
    return "\n".join(lines)


def process(
    tool_name: str,
    result: Dict[str, Any],
    intent: Optional[str] = None,
    threshold_bytes: int = INTENT_SEARCH_THRESHOLD_BYTES,
) -> Dict[str, Any]:
    """Intercept tool result: if large, index and return summary. Track stats."""
    stats = get_session_stats()
    text, byte_count = _content_from_result(result)
    if byte_count <= threshold_bytes:
        stats.track_response(tool_name, byte_count)
        return result
    intent = (intent or "").strip()
    source = f"intercept:{tool_name}"
    try:
        if intent:
            summary = _intent_search(text, intent, source)
        else:
            store = _get_store()
            stats.track_indexed(byte_count)
            indexed = store.index_plain_text(text, source)
            total_kb = byte_count / 1024
            lines = [
                f'Indexed {indexed.total_chunks} sections from "{source}" ({total_kb:.1f}KB) into knowledge base.',
                f'Use search(queries: ["..."]) to query this content. Use source: "{source}" to scope results.',
            ]
            summary = "\n".join(lines)
        stats.track_response(tool_name, len(summary.encode("utf-8", errors="replace")))
        return _replace_result_content(result, summary)
    except Exception as e:
        logger.warning("Context-mode interceptor failed: %s", e)
        stats.track_response(tool_name, byte_count)
        return result


def get_store_for_tests():
    """Return the lazy store (for tests that need to reset or inspect)."""
    from src.core.context_mode.store import _store_instance
    return _store_instance
