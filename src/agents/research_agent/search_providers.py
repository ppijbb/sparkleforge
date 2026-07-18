"""Web/academic search provider backends for ResearchAgent.

Split out of the former monolithic research_agent.py (issue #582, mirroring
the Sigma-1 split of mcp_integration.py -- module by responsibility, facade
re-export kept at src/agents/research_agent/__init__.py).
"""

import asyncio
import os
from datetime import datetime
from typing import Any, Dict

import requests

from src.utils.logger import setup_logger

logger = setup_logger("research_agent", log_level="INFO")


class SearchProvidersMixin:
    """Web/academic search backends (Tavily, Exa, Brave, Serper, DDG, etc.)."""

    async def _perform_web_search(self, query: str) -> Dict[str, Any]:
        """Perform web search using MCP tools."""
        try:
            # Use MCP tools for web search
            from src.core.mcp_integration import execute_tool

            # Try MCP search tools in priority order
            mcp_tools = ["g-search", "tavily", "exa"]

            for tool in mcp_tools:
                try:
                    result = await execute_tool(tool, {"query": query, "max_results": 10})

                    if result.get("success", False):
                        results = result.get("data", {}).get("results", [])
                        if isinstance(results, list):
                            logger.info(
                                f"Search succeeded with MCP tool {tool}: {len(results)} results"
                            )
                            return {
                                "success": True,
                                "query": query,
                                "results": results,
                                "tool_used": tool,
                            }
                except Exception as e:
                    logger.warning(f"MCP tool {tool} failed: {e}")
                    continue

            # If all MCP tools fail, raise error
            logger.error("All MCP search tools failed")
            raise RuntimeError(
                f"All MCP search tools failed for query: {query}. No fallback available."
            )

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {
                "success": False,
                "method": "web_search",
                "query": query,
                "results": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


    async def _search_with_tavily(self, query: str) -> Dict[str, Any]:
        """Search using Tavily API (best quality)."""
        try:
            tavily_key = os.getenv("TAVILY_API_KEY")
            if not tavily_key:
                return {"success": False, "error": "Tavily API key not configured"}

            from tavily import TavilyClient

            client = TavilyClient(api_key=tavily_key)
            response = client.search(query, max_results=5, search_depth="advanced")

            formatted_results = []
            results = response.get("results", [])
            if not isinstance(results, list):
                results = []

            for result in results:
                if isinstance(result, dict):
                    formatted_results.append(
                        {
                            "title": result.get("title", ""),
                            "snippet": result.get("content", ""),
                            "url": result.get("url", ""),
                            "source": "tavily",
                            "score": result.get("score", 0),
                        }
                    )

            return {
                "success": True,
                "method": "tavily",
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return {"success": False, "error": str(e)}


    async def _search_with_exa(self, query: str) -> Dict[str, Any]:
        """Search using Exa API (neural search)."""
        try:
            exa_key = os.getenv("EXA_API_KEY")
            if not exa_key:
                return {"success": False, "error": "Exa API key not configured"}

            from exa_py import Exa

            client = Exa(api_key=exa_key)
            response = client.search_and_contents(query, num_results=5, text=True, highlights=True)

            formatted_results = []
            for result in response.results:
                formatted_results.append(
                    {
                        "title": result.title,
                        "snippet": result.text[:500] if result.text else "",
                        "url": result.url,
                        "source": "exa",
                        "score": result.score if hasattr(result, "score") else 0,
                    }
                )

            return {
                "success": True,
                "method": "exa",
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"Exa search failed: {e}")
            return {"success": False, "error": str(e)}


    async def _search_with_brave(self, query: str) -> Dict[str, Any]:
        """Search using Brave Search API."""
        try:
            brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
            if not brave_key:
                return {"success": False, "error": "Brave API key not configured"}

            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {"Accept": "application/json", "X-Subscription-Token": brave_key}
            params = {"q": query, "count": 5}

            # 블로킹 I/O는 스레드로 위임하고, 단일 계층 타임아웃으로 보호
            response = await asyncio.wait_for(
                asyncio.to_thread(requests.get, url, headers=headers, params=params, timeout=10),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            formatted_results = []
            for result in data.get("web", {}).get("results", [])[:5]:
                formatted_results.append(
                    {
                        "title": result.get("title", ""),
                        "snippet": result.get("description", ""),
                        "url": result.get("url", ""),
                        "source": "brave",
                    }
                )

            return {
                "success": True,
                "method": "brave",
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"Brave search failed: {e}")
            return {"success": False, "error": str(e)}


    async def _search_with_serper(self, query: str) -> Dict[str, Any]:
        """Search using Serper API (Google Search)."""
        try:
            serper_key = os.getenv("SERPER_API_KEY")
            if not serper_key:
                return {"success": False, "error": "Serper API key not configured"}

            url = "https://google.serper.dev/search"
            headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
            payload = {"q": query, "num": 5, "gl": "kr", "hl": "ko"}

            response = await asyncio.wait_for(
                asyncio.to_thread(requests.post, url, headers=headers, json=payload, timeout=10),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            formatted_results = []
            organic_results = data.get("organic", [])
            if not isinstance(organic_results, list):
                organic_results = []

            for result in organic_results[:5]:
                if isinstance(result, dict):
                    formatted_results.append(
                        {
                            "title": result.get("title", ""),
                            "snippet": result.get("snippet", ""),
                            "url": result.get("link", ""),
                            "source": "serper",
                        }
                    )

            return {
                "success": True,
                "method": "serper",
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"Serper search failed: {e}")
            return {"success": False, "error": str(e)}


    async def _search_with_duckduckgo(self, query: str) -> Dict[str, Any]:
        """Search using DuckDuckGo (no API key required)."""
        try:
            import duckduckgo_search

            # Search for results
            results = duckduckgo_search.DDGS().text(query, max_results=5)

            formatted_results = []
            if not isinstance(results, list):
                results = []

            for result in results:
                if isinstance(result, dict):
                    formatted_results.append(
                        {
                            "title": result.get("title", ""),
                            "snippet": result.get("body", ""),
                            "url": result.get("href", ""),
                            "source": "duckduckgo",
                        }
                    )

            return {
                "success": True,
                "method": "duckduckgo",
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return {"success": False, "error": str(e)}


    async def _search_with_google_custom(self, query: str) -> Dict[str, Any]:
        """Search using Google Custom Search API."""
        try:
            # This would require Google Custom Search API key
            # For now, return failure to try other methods
            return {"success": False, "error": "Google Custom Search not configured"}

        except Exception as e:
            logger.warning(f"Google Custom Search failed: {e}")
            return {"success": False, "error": str(e)}


    async def _search_with_bing(self, query: str) -> Dict[str, Any]:
        """Search using Bing Search API."""
        try:
            # This would require Bing Search API key
            # For now, return failure to try other methods
            return {"success": False, "error": "Bing Search not configured"}

        except Exception as e:
            logger.warning(f"Bing search failed: {e}")
            return {"success": False, "error": str(e)}

    # LLM fallback method removed - no fallback responses allowed


    async def _perform_academic_search(self, query: str) -> Dict[str, Any]:
        """Perform academic search using real academic APIs."""
        try:
            from src.research.tools.academic_search import AcademicSearchTool

            # Initialize academic search tool
            academic_config = {
                "max_results": 10,
                "timeout": 30,
                "primary_provider": "arxiv",
                "fallback_providers": ["scholar", "pubmed"],
            }

            academic_tool = AcademicSearchTool(academic_config)
            result = await academic_tool.arun(query)

            if result["success"]:
                # Convert academic results to standard format
                academic_results = []
                for item in result["results"]:
                    academic_results.append(
                        {
                            "title": item.get("title", ""),
                            "authors": item.get("authors", []),
                            "journal": item.get("journal", ""),
                            "year": item.get("published", "")[:4] if item.get("published") else "",
                            "abstract": item.get("abstract", ""),
                            "source": item.get("source", "academic_database"),
                            "url": item.get("url", ""),
                            "pdf_url": item.get("pdf_url", ""),
                            "arxiv_id": item.get("arxiv_id", ""),
                            "pmid": item.get("pmid", ""),
                            "doi": item.get("doi", ""),
                        }
                    )

                return {
                    "method": "academic_search",
                    "query": query,
                    "results": academic_results,
                    "total_results": len(academic_results),
                    "provider": result.get("provider", "academic"),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                logger.warning(f"Academic search failed: {result.get('error', 'Unknown error')}")
                return {
                    "method": "academic_search",
                    "query": query,
                    "results": [],
                    "total_results": 0,
                    "error": result.get("error", "Academic search failed"),
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Academic search failed: {e}")
            return {
                "method": "academic_search",
                "query": query,
                "results": [],
                "total_results": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


