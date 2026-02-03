"""
Search MCP Server - Embedded replacement for external tavily-mcp and exa-mcp.

Provides comprehensive web search capabilities using multiple search providers:
- DuckDuckGo (primary, no API key required)
- SerpAPI (optional, for Google search)
- Custom search API support
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    from fastmcp import FastMCP
    from pydantic import BaseModel, Field
    import httpx
    FASTMCP_AVAILABLE = True
except ImportError as e:
    FASTMCP_AVAILABLE = False
    FastMCP = None
    BaseModel = None
    Field = None
    httpx = None

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("search")


class SearchInput(BaseModel):
    """Input schema for web search."""
    query: str = Field(
        ...,
        description="Search query",
        min_length=1,
        max_length=500
    )
    num_results: int = Field(
        default=10,
        description="Number of results to return",
        ge=1,
        le=50
    )
    timeout: int = Field(
        default=30,
        description="Search timeout in seconds",
        ge=5,
        le=120
    )
    safe_search: bool = Field(
        default=True,
        description="Enable safe search filtering"
    )


class MultiSearchInput(BaseModel):
    """Input schema for multiple searches at once."""
    queries: List[str] = Field(
        ...,
        description="List of search queries",
        min_length=1,
        max_length=10
    )
    num_results: int = Field(
        default=5,
        description="Results per query",
        ge=1,
        le=20
    )


class NewsSearchInput(BaseModel):
    """Input schema for news search."""
    query: str = Field(
        ...,
        description="News search query",
        min_length=1,
        max_length=500
    )
    num_results: int = Field(
        default=10,
        description="Number of news results",
        ge=1,
        le=20
    )
    days_back: int = Field(
        default=7,
        description="Search within last N days",
        ge=1,
        le=30
    )


def parse_search_result(html: str, num_results: int = 10) -> List[Dict[str, str]]:
    """Parse search results from DuckDuckGo HTML."""
    results = []
    
    # Pattern for DuckDuckGo result links
    pattern = r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]*)"[^>]*>(.*?)</a>'
    matches = re.findall(pattern, html, re.DOTALL)
    
    for href, title_html in matches[:num_results]:
        title = re.sub(r'<[^>]+>', '', title_html).strip()
        url = href
        
        # Extract snippet (description)
        snippet_pattern = r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*>.*?</a>.*?<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>'
        snippet_match = re.search(snippet_pattern, html, re.DOTALL)
        snippet = ""
        if snippet_match:
            snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()
        
        if url and not url.startswith("/"):
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet[:200] if snippet else ""
            })
    
    return results


async def search_duckduckgo(query: str, num_results: int = 10, 
                            timeout: int = 30) -> Dict[str, Any]:
    """
    Search using DuckDuckGo HTML interface.
    
    Args:
        query: Search query
        num_results: Number of results to return
        timeout: Request timeout
    
    Returns:
        Dict with search results
    """
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query, "kl": "us-en", "kp": "1"}
            )
            response.raise_for_status()
            
            results = parse_search_results(response.text, num_results)
            
            return {
                "success": True,
                "query": query,
                "provider": "duckduckgo",
                "results": results,
                "count": len(results),
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "query": query,
            "provider": "duckduckgo",
            "error": str(e)
        }


async def search_bing(query: str, num_results: int = 10,
                      timeout: int = 30) -> Dict[str, Any]:
    """
    Search using Bing (via HTML interface).
    
    Args:
        query: Search query
        num_results: Number of results
        timeout: Request timeout
    
    Returns:
        Dict with search results
    """
    try:
        api_key = None  # Could be loaded from environment
        if api_key:
            # Use Bing Search API
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    params={"q": query, "count": num_results},
                    headers={"Ocp-Apim-Subscription-Key": api_key}
                )
                response.raise_for_status()
                data = response.json()
                
                results = []
                for item in data.get("webPages", {}).get("value", [])[:num_results]:
                    results.append({
                        "title": item.get("name", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", "")[:200]
                    })
                
                return {
                    "success": True,
                    "query": query,
                    "provider": "bing-api",
                    "results": results,
                    "count": len(results)
                }
        else:
            # Fallback to HTML (limited)
            return {
                "success": False,
                "query": query,
                "provider": "bing",
                "error": "Bing API key not configured"
            }
    
    except Exception as e:
        logger.error(f"Bing search error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "query": query,
            "provider": "bing",
            "error": str(e)
        }


async def search_google_serpapi(query: str, num_results: int = 10,
                                 timeout: int = 30) -> Dict[str, Any]:
    """
    Search using SerpAPI (Google results).
    
    Args:
        query: Search query
        num_results: Number of results
        timeout: Request timeout
    
    Returns:
        Dict with search results
    """
    try:
        api_key = None  # Could be loaded from environment
        if api_key:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    "https://serpapi.com/search",
                    params={
                        "q": query,
                        "num": num_results,
                        "api_key": api_key,
                        "engine": "google"
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                results = []
                for item in data.get("organic_results", [])[:num_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", "")[:200]
                    })
                
                return {
                    "success": True,
                    "query": query,
                    "provider": "google-serpapi",
                    "results": results,
                    "count": len(results)
                }
        else:
            return {
                "success": False,
                "query": query,
                "provider": "serpapi",
                "error": "SerpAPI key not configured"
            }
    
    except Exception as e:
        logger.error(f"SerpAPI search error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "query": query,
            "provider": "serpapi",
            "error": str(e)
        }


async def search_all_providers(query: str, num_results: int = 10) -> Dict[str, Any]:
    """
    Search across all available providers and aggregate results.
    
    Args:
        query: Search query
        num_results: Results per provider
    
    Returns:
        Aggregated results from all providers
    """
    # Run searches in parallel
    tasks = [
        search_duckduckgo(query, num_results),
        search_bing(query, num_results),
        search_google_serpapi(query, num_results)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_results = []
    for result in results:
        if isinstance(result, dict) and result.get("success"):
            successful_results.append(result)
    
    if not successful_results:
        # Return first error
        for result in results:
            if isinstance(result, dict) and not result.get("success"):
                return result
        
        return {
            "success": False,
            "query": query,
            "error": "All search providers failed"
        }
    
    # Aggregate results
    aggregated = {
        "success": True,
        "query": query,
        "providers_searched": len(successful_results),
        "results": [],
        "timestamp": datetime.now().isoformat()
    }
    
    for provider_result in successful_results:
        for item in provider_result.get("results", []):
            # Deduplicate by URL
            if not any(r["url"] == item["url"] for r in aggregated["results"]):
                item["provider"] = provider_result.get("provider", "unknown")
                aggregated["results"].append(item)
    
    # Limit total results
    aggregated["results"] = aggregated["results"][:num_results]
    aggregated["count"] = len(aggregated["results"])
    
    return aggregated


def parse_search_results(html: str, num_results: int = 10) -> List[Dict[str, str]]:
    """Parse search results from DuckDuckGo HTML."""
    results = []
    
    # Pattern for DuckDuckGo result links
    pattern = r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]*)"[^>]*>(.*?)</a>'
    matches = re.findall(pattern, html, re.DOTALL)
    
    for href, title_html in matches[:num_results]:
        title = re.sub(r'<[^>]+>', '', title_html).strip()
        url = href
        
        # Skip internal DuckDuckGo links
        if url.startswith("/"):
            continue
        
        results.append({
            "title": title,
            "url": url,
            "snippet": ""
        })
    
    return results


@mcp.tool()
async def search(input: SearchInput) -> str:
    """
    Perform web search using DuckDuckGo (primary) or fallback providers.
    
    Returns JSON with:
    - success: boolean indicating if search succeeded
    - results: list of search results with title, url, snippet
    - count: number of results returned
    - provider: which search provider was used
    """
    result = await search_duckduckgo(input.query, input.num_results, input.timeout)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def search_all(input: SearchInput) -> str:
    """
    Search across all available providers and aggregate results.
    
    Returns JSON with:
    - success: boolean
    - results: aggregated results from all providers (deduplicated)
    - count: total results
    - providers_searched: number of providers queried
    """
    result = await search_all_providers(input.query, input.num_results)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def multi_search(input: MultiSearchInput) -> str:
    """
    Perform multiple searches at once.
    
    Returns JSON with:
    - success: boolean
    - queries: list of query results
    - total_results: total number of results
    """
    tasks = [
        search_duckduckgo(query, input.num_results)
        for query in input.queries
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    output = {
        "success": True,
        "queries": [],
        "total_results": 0
    }
    
    for i, result in enumerate(results):
        if isinstance(result, dict):
            output["queries"].append({
                "query": input.queries[i],
                "success": result.get("success", False),
                "results": result.get("results", []),
                "count": result.get("count", 0),
                "error": result.get("error")
            })
            if result.get("success"):
                output["total_results"] += result.get("count", 0)
        else:
            output["queries"].append({
                "query": input.queries[i],
                "success": False,
                "error": str(result)
            })
    
    return json.dumps(output, ensure_ascii=False, indent=2)


@mcp.tool()
async def search_news(input: NewsSearchInput) -> str:
    """
    Search for recent news articles.
    
    Note: This is a basic implementation. For better results,
    configure SerpAPI or use a news-specific API.
    
    Returns JSON with news results.
    """
    # Basic implementation using DuckDuckGo with news filter
    query = f"{input.query} news"
    
    result = await search_duckduckgo(query, input.num_results, input.timeout)
    
    # Add news-specific metadata
    result["search_type"] = "news"
    result["days_back"] = input.days_back
    
    return json.dumps(result, ensure_ascii=False, indent=2)


def run():
    """Run the search MCP server."""
    mcp.run()


if __name__ == "__main__":
    run()
