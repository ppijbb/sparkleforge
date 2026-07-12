"""Academic search tool dispatch (ToolCategory.ACADEMIC): arxiv/scholarly lookups."""
import asyncio
import logging
import time
from typing import Any, Dict

from src.core.tools.registry import ToolResult

logger = logging.getLogger(__name__)


def _execute_academic_tool_sync(tool_name: str, parameters: Dict[str, Any]) -> str:
    """동기 버전 - LangChain Tool에서 호출."""
    try:
        import concurrent.futures

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, _execute_academic_tool(tool_name, parameters)
                    )
                    # timeout 설정으로 무한 대기 방지
                    result = future.result(timeout=300)  # 최대 5분
            else:
                result = loop.run_until_complete(_execute_academic_tool(tool_name, parameters))
        except RuntimeError:
            result = asyncio.run(_execute_academic_tool(tool_name, parameters))

        if result.success:
            import json

            return json.dumps(result.data, ensure_ascii=False, indent=2)
        else:
            raise RuntimeError(result.error or "Tool execution failed")
    except Exception as e:
        raise RuntimeError(f"Tool execution failed: {str(e)}")



async def _execute_academic_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """학술 도구 실행 - src/utils에서 직접 사용."""
    start_time = time.time()
    query = parameters.get("query", "")
    max_results = parameters.get("max_results", 10) or parameters.get("num_results", 10)

    # src/utils에서 직접 사용 (MCP 서버로 실행하지 않음)
    if tool_name == "arxiv":
        try:
            from src.utils.academic_utils import search_arxiv

            if not query:
                return ToolResult(
                    success=False,
                    data=None,
                    error="query parameter is required",
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                )

            # src/utils의 search_arxiv 직접 호출
            result = await search_arxiv(query, max_results)

            if result.get("success"):
                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "results": result.get("results", []),
                        "total_results": result.get("total_results", 0),
                        "count": result.get("count", 0),
                        "source": "embedded_arxiv",
                    },
                    execution_time=time.time() - start_time,
                    confidence=0.95,
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=result.get("error", "arXiv search failed"),
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                )
        except ImportError:
            logger.debug("src.utils.academic_utils not available, using existing logic")
        except Exception as e:
            logger.warning(f"Embedded arXiv search failed: {e}, falling back to existing logic")

    # 기존 로직 (src/utils 실패 시 또는 다른 tool_name)
    try:
        if tool_name == "arxiv":
            # arXiv API (100% 무료)
            import arxiv

            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            results = []
            for paper in client.results(search):
                results.append(
                    {
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "abstract": paper.summary,
                        "url": paper.entry_id,
                        "published": paper.published.isoformat(),
                        "pdf_url": paper.pdf_url,
                    }
                )

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                    "source": "arxiv",
                },
                execution_time=time.time() - start_time,
                confidence=0.95,
            )

        elif tool_name == "scholar":
            # Google Scholar (무료, rate limit 있음)
            from scholarly import scholarly

            search_query = scholarly.search_pubs(query)
            results = []

            for i, pub in enumerate(search_query):
                if i >= max_results:
                    break

                results.append(
                    {
                        "title": pub.get("bib", {}).get("title", ""),
                        "authors": pub.get("bib", {}).get("author", ""),
                        "abstract": pub.get("bib", {}).get("abstract", ""),
                        "url": pub.get("pub_url", ""),
                        "year": pub.get("bib", {}).get("pub_year", ""),
                        "citations": pub.get("num_citations", 0),
                    }
                )

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                    "source": "scholar",
                },
                execution_time=time.time() - start_time,
                confidence=0.8,
            )

        else:
            raise ValueError(f"Unknown academic tool: {tool_name}")

    except Exception as e:
        logger.error(f"Academic tool execution failed: {tool_name} - {e}")
        return ToolResult(
            success=False,
            data=None,
            error=f"Academic tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0,
        )
