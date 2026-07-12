"""Optional-dependency runtime flags and client types shared across
``UniversalMCPHub`` and its mixins (Anvil Phase Sigma, issue #507).

Centralizes the MCP/FastMCP/LangChain "is this optional dependency
installed" try/except cascades so ``hub.py`` and each mixin module under
``src/core/mcp_integration/hub_mixins/`` can import the same flags/types
without hub.py and its mixins importing from each other (which would be
circular, since hub.py composes ``UniversalMCPHub`` from the mixins).

Also defines ``project_root``, used by a couple of mixins to locate
``configs/*.json``. It must be computed from a module that lives in the
same directory the original monolithic ``hub.py`` did (``src/core/
mcp_integration/``), not from a mixin's own file under ``hub_mixins/``,
since ``Path(__file__).parent.parent.parent`` is directory-depth-relative
-- computing it from a deeper file would silently point somewhere else.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import httpx
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.shared.exceptions import McpError
    from mcp.types import TextContent

    MCP_AVAILABLE = True
    HTTP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    HTTP_CLIENT_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    TextContent = None
    McpError = Exception  # Fallback
    httpx = None

# FastMCP imports
try:
    import logging as fastmcp_logging

    from fastmcp import Client as FastMCPClient

    # FastMCP 로거 레벨을 warning으로 설정
    fastmcp_logger = fastmcp_logging.getLogger("fastmcp")
    fastmcp_logger.setLevel(fastmcp_logging.WARNING)
    # fastmcp 관련 모든 로거에 대해 warning 레벨 적용
    for _logger_name in ["fastmcp", "fastmcp.client", "fastmcp.runner"]:
        _logger_instance = fastmcp_logging.getLogger(_logger_name)
        _logger_instance.setLevel(fastmcp_logging.WARNING)

    # MCP 클라이언트 로거도 필터링 (heartbeat 오류 방지)
    for _logger_name in ["mcp", "mcp.client", "mcp.client.streamable_http", "Runner"]:
        _logger_instance = fastmcp_logging.getLogger(_logger_name)
        _logger_instance.setLevel(fastmcp_logging.WARNING)

        class HeartbeatFilter(fastmcp_logging.Filter):
            def filter(self, record):
                msg = record.getMessage()
                # heartbeat 관련 오류 메시지 필터링
                if "heartbeat" in msg.lower() or "invalid_token" in msg.lower():
                    return False
                return True

        _logger_instance.addFilter(HeartbeatFilter())

    FASTMCP_AVAILABLE = True
except ImportError:
    FastMCPClient = None
    FASTMCP_AVAILABLE = False

# LangChain imports
try:
    from langchain_core.tools import BaseTool, StructuredTool

    # Pydantic v2 호환성 - 최신 LangChain은 pydantic v2 사용
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        try:
            from pydantic.v1 import BaseModel, Field
        except ImportError:
            from langchain_core.pydantic_v1 import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = None
    StructuredTool = None
    BaseModel = None
    Field = None
