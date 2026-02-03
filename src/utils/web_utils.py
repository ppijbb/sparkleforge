"""
Web utilities - URL fetching and content extraction.

Extracted from embedded_mcp_servers for direct use in mcp_integration.py.
"""

import logging
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

logger = logging.getLogger(__name__)


def detect_content_type(response) -> str:
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


def clean_html_content(html: str, max_length: int = 10000) -> str:
    """Clean and extract text from HTML."""
    # Remove script and style tags
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)
    
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
    if not HTTPX_AVAILABLE:
        return {
            "success": False,
            "error": "httpx not available",
            "url": url
        }
    
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
