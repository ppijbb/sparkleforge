"""
Web Fetch MCP Server - Embedded replacement for external fetch-mcp.

Provides web content retrieval capabilities using httpx with proper
error handling, content type detection, and response parsing.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional
from pathlib import Path
from urllib.parse import urlparse

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
mcp = FastMCP("web-fetch")


class FetchInput(BaseModel):
    """Input schema for fetch tool."""
    url: str = Field(
        ...,
        description="URL to fetch content from",
        min_length=1,
        max_length=2048
    )
    max_length: int = Field(
        default=50000,
        description="Maximum content length to return",
        ge=100,
        le=500000
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds",
        ge=1,
        le=120
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional headers to include in request"
    )
    follow_redirects: bool = Field(
        default=True,
        description="Follow HTTP redirects"
    )


class SearchInput(BaseModel):
    """Input schema for search tool (simple web search using DuckDuckGo)."""
    query: str = Field(
        ...,
        description="Search query",
        min_length=1,
        max_length=500
    )
    num_results: int = Field(
        default=10,
        description="Number of search results to return",
        ge=1,
        le=20
    )


def detect_content_type(response: httpx.Response) -> str:
    """Detect content type from response headers."""
    content_type = response.headers.get("content-type", "").lower()
    
    if "text/html" in content_type:
        return "html"
    elif "text/plain" in content_type:
        return "text"
    elif "application/json" in content_type:
        return "json"
    elif "application/pdf" in content_type:
        return "pdf"
    elif "text/css" in content_type:
        return "css"
    elif "text/javascript" in content_type or "application/javascript" in content_type:
        return "javascript"
    else:
        return "unknown"


def truncate_text(text: str, max_length: int = 10000) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "...\n[truncated]"


def clean_html_content(html: str, max_length: int) -> str:
    """Extract clean text from HTML content."""
    # Remove script and style elements
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)
    
    # Decode HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#\d+;', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return truncate_text(text, max_length)


async def fetch_url(url: str, max_length: int = 50000, timeout: int = 30,
                    headers: Optional[Dict[str, str]] = None,
                    follow_redirects: bool = True) -> Dict[str, Any]:
    """
    Fetch content from a URL with proper error handling.
    
    Args:
        url: URL to fetch
        max_length: Maximum content length
        timeout: Request timeout in seconds
        headers: Optional headers
        follow_redirects: Whether to follow redirects
    
    Returns:
        Dict with success status, content, and metadata
    """
    # Validate URL
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {
                "success": False,
                "error": "Invalid URL format",
                "url": url
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"URL parsing error: {str(e)}",
            "url": url
        }
    
    # Validate protocol
    if parsed.scheme not in ["http", "https"]:
        return {
            "success": False,
            "error": "Only HTTP and HTTPS protocols are supported",
            "url": url
        }
    
    # Build request headers
    default_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SparkleForge-MCP/1.0; +http://sparkleforge.ai)"
    }
    if headers:
        default_headers.update(headers)
    
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=follow_redirects
        ) as client:
            response = await client.get(url, headers=default_headers)
            response.raise_for_status()
            
            content_type = detect_content_type(response)
            
            if content_type == "html":
                content = clean_html_content(response.text, max_length)
            elif content_type == "text":
                content = truncate_text(response.text, max_length)
            elif content_type == "json":
                content = truncate_text(response.text, max_length)
            else:
                # For binary content, return message
                content = f"[Binary content: {content_type}, {len(response.content)} bytes]"
            
            return {
                "success": True,
                "url": url,
                "content": content,
                "content_type": content_type,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "character_count": len(content)
            }
    
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error fetching {url}: {e.response.status_code}")
        return {
            "success": False,
            "error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
            "url": url,
            "status_code": e.response.status_code
        }
    
    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching {url}")
        return {
            "success": False,
            "error": f"Request timed out after {timeout} seconds",
            "url": url
        }
    
    except httpx.ConnectError as e:
        logger.warning(f"Connection error fetching {url}: {str(e)}")
        return {
            "success": False,
            "error": f"Connection failed: {str(e)}",
            "url": url
        }
    
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "url": url
        }


async def simple_search(query: str, num_results: int = 10) -> Dict[str, Any]:
    """
    Simple web search using DuckDuckGo HTML.
    
    Args:
        query: Search query
        num_results: Number of results to return
    
    Returns:
        Dict with search results
    """
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # DuckDuckGo HTML search
            response = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query, "kl": "us-en", "kp": "1"}
            )
            response.raise_for_status()
            
            # Parse results
            results = []
            # DuckDuckGo HTML pattern
            pattern = r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]*)"[^>]*>(.*?)</a>'
            matches = re.findall(pattern, response.text, re.DOTALL)
            
            for href, title_html in matches[:num_results]:
                # Clean title
                title = re.sub(r'<[^>]+>', '', title_html).strip()
                # Clean URL
                url = href
                if url.startswith("/"):
                    continue
                results.append({
                    "title": title,
                    "url": url
                })
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Search failed: {str(e)}",
            "query": query
        }


@mcp.tool()
async def fetch(input: FetchInput) -> str:
    """
    Fetch content from a web URL.
    
    Returns JSON with:
    - success: boolean indicating if fetch succeeded
    - content: the fetched content (truncated if too long)
    - content_type: type of content (html, text, json, etc.)
    - url: the original URL
    - metadata: additional information like status code
    """
    result = await fetch_url(
        url=input.url,
        max_length=input.max_length,
        timeout=input.timeout,
        headers=input.headers,
        follow_redirects=input.follow_redirects
    )
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def search(input: SearchInput) -> str:
    """
    Perform a simple web search.
    
    Returns JSON with:
    - success: boolean indicating if search succeeded
    - results: list of search results with title and url
    - count: number of results returned
    - query: the original search query
    """
    result = await simple_search(input.query, input.num_results)
    return json.dumps(result, ensure_ascii=False, indent=2)


def run():
    """Run the web fetch MCP server."""
    mcp.run()


if __name__ == "__main__":
    run()
