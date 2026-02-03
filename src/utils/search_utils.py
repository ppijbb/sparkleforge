"""
Search utilities - DuckDuckGo and other search providers.

Extracted from embedded_mcp_servers for direct use in mcp_integration.py.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

logger = logging.getLogger(__name__)


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
    if not HTTPX_AVAILABLE:
        return {
            "success": False,
            "query": query,
            "provider": "duckduckgo",
            "error": "httpx not available"
        }
    
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
