"""
Auto-generated MCP server for web_search_service
Generated at: 2026-01-09T16:23:54.130461
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

mcp = FastMCP("web_search_service")

class McpSearchInput(BaseModel):
    """Input schema for mcp_search"""
    query: str = Field(..., description="The search query to execute. Should be a detailed, specific question or topic.")
    max_results: Optional[int] = Field(default=10, description="The maximum number of search results to return.")

@mcp.tool()
async def mcp_search(input: McpSearchInput) -> str:
    """
    mcp_search tool
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return json.dumps({"error": "API key not found. Set TAVILY_API_KEY environment variable."})
        headers = {"Authorization": f"Bearer {api_key}"}

        url = f"https://api.tavily.com/search"
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
