"""Context Mode MCP Server — 95%+ token reduction for all LLMs.

Execute, index, search, fetch_and_index, batch_execute, stats.
Large outputs stay in sandbox; only summaries enter context.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastmcp import FastMCP
    from pydantic import BaseModel, Field
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCP = None
    BaseModel = None
    Field = None

logger = logging.getLogger(__name__)

if FASTMCP_AVAILABLE:
    mcp = FastMCP("context-mode")

# Lazy store and executor
_store = None
_executor = None
_search_call_count = 0
_search_window_start = time.time()
SEARCH_WINDOW_SEC = 60
SEARCH_MAX_RESULTS_AFTER = 3
SEARCH_BLOCK_AFTER = 8
MAX_TOTAL_SEARCH_BYTES = 40 * 1024


def _get_store():
    from src.core.context_mode.store import get_store
    return get_store()


def _get_executor():
    global _executor
    if _executor is None:
        from src.core.context_mode.executor import SandboxedExecutor
        project_root = os.environ.get("SPARKLEFORGE_PROJECT_DIR") or os.getcwd()
        _executor = SandboxedExecutor(project_root=str(project_root))
    return _executor


def _track_response(tool_name: str, text: str) -> None:
    from src.core.context_mode.stats import get_session_stats
    get_session_stats().track_response(tool_name, len(text.encode("utf-8", errors="replace")))


def _track_indexed(byte_count: int) -> None:
    from src.core.context_mode.stats import get_session_stats
    get_session_stats().track_indexed(byte_count)


if FASTMCP_AVAILABLE:

    class ExecuteInput(BaseModel):
        language: str = Field(..., description="Runtime: python, shell, javascript, typescript")
        code: str = Field(..., description="Source code to execute. Use print/echo/console.log for output.")
        timeout: int = Field(default=30000, description="Max execution time in ms")
        intent: Optional[str] = Field(
            default=None,
            description="What you're looking for. When set and output >5KB, returns section titles + previews only.",
        )

    @mcp.tool()
    def execute(language: str, code: str, timeout: int = 30000, intent: Optional[str] = None) -> str:
        """Execute code in sandbox. Only stdout enters context. Prefer over bash for large output."""
        executor = _get_executor()
        result = executor.execute(language=language, code=code, timeout=timeout)
        if result.timed_out:
            out = f"Execution timed out after {timeout}ms\n\nPartial stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            _track_response("execute", out)
            return out
        if result.exit_code != 0:
            out = f"Exit code: {result.exit_code}\n\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            _track_response("execute", out)
            return out
        stdout = result.stdout or "(no output)"
        threshold = 5_000
        if intent and intent.strip() and len(stdout.encode("utf-8", errors="replace")) > threshold:
            from src.core.context_mode.interceptor import _intent_search
            summary = _intent_search(stdout, intent.strip(), f"execute:{language}")
            _track_response("execute", summary)
            return summary
        _track_response("execute", stdout)
        return stdout

    class ExecuteFileInput(BaseModel):
        path: str = Field(..., description="Absolute or project-relative file path")
        language: str = Field(..., description="python, shell, javascript, typescript")
        code: str = Field(..., description="Code to process FILE_CONTENT. Print summary.")
        timeout: int = Field(default=30000, description="Max execution time in ms")
        intent: Optional[str] = Field(default=None, description="When set and output >5KB, returns matching sections only.")

    @mcp.tool()
    def execute_file(path: str, language: str, code: str, timeout: int = 30000, intent: Optional[str] = None) -> str:
        """Read file into FILE_CONTENT in sandbox; only your printed summary enters context."""
        executor = _get_executor()
        result = executor.execute_file(path=path, language=language, code=code, timeout=timeout)
        if result.timed_out:
            out = f"Timed out processing {path} after {timeout}ms"
            _track_response("execute_file", out)
            return out
        if result.exit_code != 0:
            out = f"Error processing {path} (exit {result.exit_code}):\n{result.stderr or result.stdout}"
            _track_response("execute_file", out)
            return out
        stdout = result.stdout or "(no output)"
        threshold = 5_000
        if intent and intent.strip() and len(stdout.encode("utf-8", errors="replace")) > threshold:
            from src.core.context_mode.interceptor import _intent_search
            summary = _intent_search(stdout, intent.strip(), f"file:{path}")
            _track_response("execute_file", summary)
            return summary
        _track_response("execute_file", stdout)
        return stdout

    class IndexInput(BaseModel):
        content: Optional[str] = Field(default=None, description="Raw text/markdown to index. Provide this OR path.")
        path: Optional[str] = Field(default=None, description="File path to read and index.")
        source: Optional[str] = Field(default=None, description="Label for indexed content.")

    @mcp.tool()
    def index(content: Optional[str] = None, path: Optional[str] = None, source: Optional[str] = None) -> str:
        """Index content into searchable BM25 knowledge base. Full content does NOT stay in context."""
        if not content and not path:
            return "Error: Either content or path must be provided"
        try:
            if content:
                _track_indexed(len(content.encode("utf-8", errors="replace")))
            elif path:
                _track_indexed(Path(path).read_bytes().__len__())
            store = _get_store()
            result = store.index(content=content, path=path, source=source)
            out = f"Indexed {result.total_chunks} sections ({result.code_chunks} with code) from: {result.label}\nUse search(queries: [\"...\"]) to query. Use source: \"{result.label}\" to scope."
            _track_response("index", out)
            return out
        except Exception as e:
            msg = f"Index error: {e}"
            _track_response("index", msg)
            return msg

    class SearchInput(BaseModel):
        queries: Optional[List[str]] = Field(default=None, description="Array of search queries. Batch in one call.")
        query: Optional[str] = Field(default=None, description="Single query (alternative to queries).")
        limit: int = Field(default=3, description="Results per query (default 3)")
        source: Optional[str] = Field(default=None, description="Filter to indexed source (partial match).")

    @mcp.tool()
    def search(
        queries: Optional[List[str]] = None,
        query: Optional[str] = None,
        limit: int = 3,
        source: Optional[str] = None,
    ) -> str:
        """Search indexed content. Pass ALL questions as queries array in ONE call."""
        global _search_call_count, _search_window_start
        query_list: List[str] = []
        if queries:
            query_list.extend(queries)
        elif query and query.strip():
            query_list.append(query)
        if not query_list:
            return "Error: provide query or queries."
        store = _get_store()
        now = time.time()
        if now - _search_window_start > SEARCH_WINDOW_SEC:
            _search_call_count = 0
            _search_window_start = now
        _search_call_count += 1
        if _search_call_count > SEARCH_BLOCK_AFTER:
            msg = (
                f"BLOCKED: {_search_call_count} search calls in {int(now - _search_window_start)}s. "
                "Stop making individual search calls. Use batch_execute(commands, queries) next."
            )
            _track_response("search", msg)
            return msg
        effective_limit = 1 if _search_call_count > SEARCH_MAX_RESULTS_AFTER else min(limit, 2)
        from src.core.context_mode.snippet import extract_snippet
        sections: List[str] = []
        total_size = 0
        for q in query_list:
            if total_size > MAX_TOTAL_SEARCH_BYTES:
                sections.append(f"## {q}\n(output cap reached)\n")
                continue
            results = store.search_with_fallback(q, effective_limit, source)
            if not results:
                sections.append(f"## {q}\nNo results found.")
                continue
            parts = []
            for r in results:
                snip = extract_snippet(r.content, q, 1500, r.highlighted if hasattr(r, "highlighted") else None)
                parts.append(f"--- [{r.source}] ---\n### {r.title}\n\n{snip}")
            formatted = "\n\n".join(parts)
            sections.append(f"## {q}\n\n{formatted}")
            total_size += len(formatted.encode("utf-8", errors="replace"))
        output = "\n\n---\n\n".join(sections)
        if _search_call_count >= SEARCH_MAX_RESULTS_AFTER:
            output += f"\n\n⚠ search call #{_search_call_count}/{SEARCH_BLOCK_AFTER} in this window. Batch queries or use batch_execute."
        _track_response("search", output)
        return output

    class FetchAndIndexInput(BaseModel):
        url: str = Field(..., description="URL to fetch and index")
        source: Optional[str] = Field(default=None, description="Label for indexed content.")

    @mcp.tool()
    def fetch_and_index(url: str, source: Optional[str] = None) -> str:
        """Fetch URL, convert HTML to markdown, index. Returns ~3KB preview; use search() for more."""
        try:
            import httpx
            resp = httpx.get(url, follow_redirects=True, timeout=30)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            msg = f"Failed to fetch {url}: {e}"
            _track_response("fetch_and_index", msg)
            return msg
        try:
            from markdownify import markdownify as md
            markdown = md(html)
        except Exception:
            markdown = html
        store = _get_store()
        _track_indexed(len(markdown.encode("utf-8", errors="replace")))
        indexed = store.index(content=markdown, source=source or url)
        preview_len = 3072
        preview = markdown[:preview_len] + "\n\n…[truncated — use search() for full content]" if len(markdown) > preview_len else markdown
        total_kb = len(markdown.encode("utf-8", errors="replace")) / 1024
        out = (
            f"Fetched and indexed **{indexed.total_chunks} sections** ({total_kb:.1f}KB) from: {indexed.label}\n"
            f'Use search(queries: [...], source: "{indexed.label}") for lookups.\n\n---\n\n{preview}'
        )
        _track_response("fetch_and_index", out)
        return out

    class BatchCommand(BaseModel):
        label: str = Field(..., description="Section header for this command's output")
        command: str = Field(..., description="Shell command to execute")

    class BatchExecuteInput(BaseModel):
        commands: List[BatchCommand] = Field(..., description="Commands to run. Output labeled by section.")
        queries: List[str] = Field(..., description="Search queries to run on indexed output. Put ALL questions here.")
        timeout: int = Field(default=60000, description="Max execution time in ms")

    @mcp.tool()
    def batch_execute(commands: List[Any], queries: List[str], timeout: int = 60000) -> str:
        """Execute multiple commands in one call, index output, run search queries. Primary tool for research."""
        from src.core.context_mode.snippet import extract_snippet
        script_lines = []
        for c in commands:
            label = c.get("label", "Section") if isinstance(c, dict) else getattr(c, "label", "Section")
            cmd = c.get("command", "") if isinstance(c, dict) else getattr(c, "command", "")
            safe_label = label.replace("'", "'\"'\"'")
            script_lines.append(f"echo '# {safe_label}'\necho ''\n{cmd} 2>&1\necho ''")
        script = "\n".join(script_lines)
        executor = _get_executor()
        result = executor.execute(language="shell", code=script, timeout=timeout)
        if result.timed_out:
            out = f"Batch timed out after {timeout}ms. Partial:\n{(result.stdout or '')[:2000]}"
            _track_response("batch_execute", out)
            return out
        stdout = result.stdout or "(no output)"
        total_bytes = len(stdout.encode("utf-8", errors="replace"))
        _track_indexed(total_bytes)
        store = _get_store()
        source = "batch:" + ",".join(
            (c.get("label", "") if isinstance(c, dict) else getattr(c, "label", "")) for c in commands
        )[:80]
        indexed = store.index(content=stdout, source=source)
        all_sections = store.get_chunks_by_source(indexed.source_id)
        inventory = ["## Indexed Sections", ""]
        for s in all_sections:
            kb = len(s.content.encode("utf-8", errors="replace")) / 1024
            inventory.append(f"- {s.title} ({kb:.1f}KB)")
        query_results = []
        max_out = 80 * 1024
        out_size = 0
        for q in queries:
            if out_size > max_out:
                query_results.append(f"## {q}\n(output cap reached)\n")
                continue
            results = store.search_with_fallback(q, 3, source)
            if not results:
                results = store.search_with_fallback(q, 3)
            query_results.append(f"## {q}\n")
            query_results.append("")
            if results:
                for r in results:
                    snip = extract_snippet(r.content, q, 1500, getattr(r, "highlighted", None))
                    query_results.append(f"### {r.title}\n{snip}\n")
                    out_size += len(snip.encode("utf-8", errors="replace"))
            else:
                query_results.append("No matching sections found.\n")
        distinctive = store.get_distinctive_terms(indexed.source_id) if hasattr(store, "get_distinctive_terms") else []
        lines = [
            f"Executed {len(commands)} commands ({len(stdout.splitlines())} lines, {total_bytes/1024:.1f}KB). Indexed {indexed.total_chunks} sections. Searched {len(queries)} queries.",
            "",
            *inventory,
            "",
            *query_results,
        ]
        if distinctive:
            lines.append(f"\nSearchable terms for follow-up: {', '.join(distinctive)}")
        out = "\n".join(lines)
        _track_response("batch_execute", out)
        return out

    @mcp.tool()
    def stats() -> str:
        """Session context consumption: bytes returned, indexed, savings ratio."""
        from src.core.context_mode.stats import get_session_stats
        return get_session_stats().format_summary()


def run() -> None:
    """Run the context-mode MCP server."""
    if FASTMCP_AVAILABLE:
        mcp.run(show_banner=False)
    else:
        raise RuntimeError("FastMCP is not available; install fastmcp to run context-mode server.")


if __name__ == "__main__":
    run()
