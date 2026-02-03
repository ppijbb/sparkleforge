"""
Auto-generated MCP server for google_custom_search
Generated at: 2026-01-09T16:27:47.284432
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

mcp = FastMCP("google_custom_search")

class McpSearchInput(BaseModel):
    """Input schema for mcp_search"""
    query: str = Field(..., description="The search term or query to look up.")
    max_results: Optional[str] = Field(default=10, description="The maximum number of search results to return. Must be between 1 and 10.")
    cx: str = Field(..., description="The Programmable Search Engine ID (cx) to use for the search. This should be stored in an environment variable like GOOGLE_CSE_ID and passed to the tool.")

@mcp.tool()
async def mcp_search(input: McpSearchInput) -> str:
    """
    mcp_search tool
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return json.dumps({"error": "API key not found. Set GOOGLE_API_KEY environment variable."})
        headers = {"Authorization": f"Bearer {api_key}"}

        url = f"https://www.googleapis.com/customsearch/v1"
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
