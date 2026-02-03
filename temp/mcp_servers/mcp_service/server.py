"""
Auto-generated MCP server for mcp_service
Generated at: 2026-01-09T16:46:00.900064
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

mcp = FastMCP("mcp_service")

class McpSearchInput(BaseModel):
    """Input schema for mcp_search"""
    query: str = Field(..., description="Search query")
    max_results: Optional[int] = Field(default=10, description="Maximum number of search results")

@mcp.tool()
async def mcp_search(input: McpSearchInput) -> str:
    """
    mcp_search tool
    """
    try:
        api_key = os.getenv("MCP_API_KEY")
        if not api_key:
            return json.dumps({"error": "API key not found. Set MCP_API_KEY environment variable."})
        headers = {"Authorization": f"Bearer {api_key}"}

        url = f"https://api.mcp-service.com/search"
        params = input.model_dump(exclude_none=True)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params, headers=headers)
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
