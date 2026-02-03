"""
Auto-generated MCP server for web_search_server
Generated at: 2026-01-09T16:26:59.446926
"""

import os
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path

try:
    from fastmcp import FastMCP
    from pydantic import BaseModel, Field
    import httpx
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}. Install with: pip install fastmcp pydantic httpx")

logger = logging.getLogger(__name__)

mcp = FastMCP("web_search_server")

class McpSearchInput(BaseModel):
    """Input schema for mcp_search"""
    query: str = Field(..., description="The search query string. Can include terms in any language.")
    max_results: Optional[str] = Field(default=10, description="The maximum number of search results to return. The underlying API may have its own limits.")

@mcp.tool()
async def mcp_search(input: McpSearchInput) -> str:
    """
    mcp_search tool
    """
    try:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return json.dumps({"error": "API key not found. Set SERPER_API_KEY environment variable."})
        headers = {"Authorization": f"Bearer {api_key}"}

        url = f"https://google.serper.dev/search"
        data = input.model_dump(exclude_none=True)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()

            result = response.json()
            return json.dumps(result, ensure_ascii=False, indent=2)

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
    except Exception as e:
        error_msg = f"Error in mcp_search: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

if __name__ == "__main__":
    mcp.run()
