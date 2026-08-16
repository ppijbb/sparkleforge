"""Semantic file search tool integration.

Wraps the SemanticFS embedding similarity search function as an
agent-callable tool (`semantic_file_search`) and registers it in the
centralized tool registry so the AgentHarness can use it for
context-aware document sorting and retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.core.tools.registry import ToolCategory, ToolMetadata, registry

logger = logging.getLogger(__name__)

# In-memory cache of indexed directories: maps resolved directory path to a
# tuple of (directory mtime, list of candidate file entries). Avoids redundant
# filesystem scans during continuous research loops.
_DIRECTORY_INDEX_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

def _keyword_score(query: str, content: str) -> float:
    """Simple fallback similarity score based on term overlap."""
    query_terms = {term for term in query.lower().split() if term}
    if not query_terms:
        return 0.0
    content_lower = content.lower()
    hits = sum(1 for term in query_terms if term in content_lower)
    return hits / len(query_terms)


def _get_indexed_candidates(
    base: Path, extensions: tuple[str, ...]
) -> List[Dict[str, Any]]:
    """Return candidate file entries for ``base``, caching the index.

    The cache is keyed by the resolved directory path and invalidated when
    the directory's modification time changes, so repeated searches over the
    same stable tree reuse the in-memory index instead of rescanning disk.
    """
    cache_key = str(base)
    try:
        dir_mtime = base.stat().st_mtime
    except OSError:
        dir_mtime = 0.0

    cached = _DIRECTORY_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] == dir_mtime:
        return cached[1]

    candidates: List[Dict[str, Any]] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix and extensions and path.suffix.lower() not in extensions:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        candidates.append({"path": str(path), "content": text})

    _DIRECTORY_INDEX_CACHE[cache_key] = (dir_mtime, candidates)
    return candidates


def _semantic_search(
    query: str,
    directory: str = ".",
    top_k: int = 10,
    file_extensions: List[str] | None = None,
) -> Dict[str, Any]:
    """Search files by semantic similarity to ``query``.

    Falls back to keyword matching when embeddings are unavailable so the
    tool remains useful in offline/test environments.
    """
    base = Path(directory).expanduser().resolve()
    if not base.exists():
        return {"success": False, "error": f"Directory not found: {directory}", "matches": []}

    extensions = tuple(file_extensions) if file_extensions else ()
    candidates = _get_indexed_candidates(base, extensions)

    try:
        from src.core.actuate.semantic_fs import SemanticFS

        fs = SemanticFS()
        scored = []
        for candidate in candidates:
            try:
                score = fs.similarity(query, candidate["content"])
            except Exception:
                score = _keyword_score(query, candidate["content"])
            scored.append({**candidate, "score": float(score)})
    except Exception as e:
        logger.debug(f"[semantic_file_search] embedding search unavailable: {e}")
        scored = [
            {**candidate, "score": _keyword_score(query, candidate["content"])}
            for candidate in candidates
        ]

    scored.sort(key=lambda item: item["score"], reverse=True)
    matches = [
        {"path": item["path"], "score": round(item["score"], 4)}
        for item in scored[:top_k]
    ]
    return {"success": True, "query": query, "directory": str(base), "matches": matches}


async def _semantic_file_search(
    query: str,
    directory: str = ".",
    top_k: int = 10,
    file_extensions: List[str] | None = None,
) -> Dict[str, Any]:
    """Async wrapper exposed as the agent-callable tool executor."""
    return _semantic_search(query, directory, top_k, file_extensions)


SEMANTIC_FILE_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Natural-language description of the content to find.",
        },
        "directory": {
            "type": "string",
            "description": "Root directory to search recursively.",
            "default": ".",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of matches to return.",
            "default": 10,
        },
        "file_extensions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional list of file extensions to filter (e.g. ['.txt', '.md']).",
        },
    },
    "required": ["query"],
}


def register_semantic_file_search_tool() -> None:
    """Register the ``semantic_file_search`` tool in the global registry."""
    metadata = ToolMetadata(
        name="semantic_file_search",
        description=(
            "Search files by semantic similarity to a natural-language query. "
            "Useful for organizing receipts and documents by content meaning."
        ),
        parameters=SEMANTIC_FILE_SEARCH_PARAMETERS,
        category=ToolCategory.SEARCH,
        tags=["semantic", "file", "search", "document", "retrieval"],
        source="local",
    )
    registry.register(metadata, _semantic_file_search, _semantic_file_search)
