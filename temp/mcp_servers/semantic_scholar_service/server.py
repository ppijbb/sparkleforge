"""
Auto-generated MCP server for semantic_scholar_service
Generated at: 2026-01-15T10:01:39.274756
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

mcp = FastMCP("semantic_scholar_service")

class SemanticScholar::Papers-Search-BasicInput(BaseModel):
    """Input schema for semantic_scholar::papers-search-basic"""
    query: str = Field(..., description="Search query")
    max_results: Optional[int] = Field(default=10, description="Maximum number of search results")

@mcp.tool()
async def semantic_scholar::papers-search-basic(input: SemanticScholar::Papers-Search-BasicInput) -> str:
    """
    semantic_scholar::papers-search-basic tool
    """
    try:
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if not api_key:
            return json.dumps({"error": "API key not found. Set SEMANTIC_SCHOLAR_API_KEY environment variable."})
        headers = {"Authorization": f"Bearer {api_key}"}

        url = f"https://api.semanticscholar.org/v1/papers/search"
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
        error_msg = f"Error in semantic_scholar::papers-search-basic: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

if __name__ == "__main__":
    mcp.run()
