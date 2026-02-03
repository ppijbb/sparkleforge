"""
Base MCP Server - Base class for all embedded MCP servers.

This module provides a foundation for creating FastMCP-based servers
with consistent error handling, logging, and tool registration patterns.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path

try:
    from fastmcp import FastMCP
    from pydantic import BaseModel, Field
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCP = None
    BaseModel = None
    Field = None

logger = logging.getLogger(__name__)


class BaseMCPServer(ABC):
    """Base class for all embedded MCP servers."""
    
    def __init__(self, server_name: str, description: str = ""):
        self.server_name = server_name
        self.description = description
        self.server = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the server."""
        # Suppress verbose MCP logging
        logging.getLogger("fastmcp").setLevel(logging.WARNING)
        logging.getLogger("mcp").setLevel(logging.WARNING)
    
    @abstractmethod
    def register_tools(self):
        """Register all tools with the server. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_input_models(self) -> Dict[str, type]:
        """Return dict of tool_name -> Pydantic model class. Must be implemented by subclasses."""
        pass
    
    def create_server(self) -> FastMCP:
        """Create and configure the FastMCP server."""
        if not FASTMCP_AVAILABLE:
            raise RuntimeError("FastMCP is not installed. Install with: pip install fastmcp pydantic httpx")
        
        self.server = FastMCP(self.server_name)
        self.register_tools()
        return self.server
    
    def run(self, transport: str = "stdio"):
        """Run the MCP server with specified transport."""
        server = self.create_server()
        server.run(transport=transport)
    
    async def cleanup(self):
        """Cleanup resources. Override in subclasses if needed."""
        pass


def create_server_class(server_name: str, description: str = ""):
    """Factory function to create a base server class with common functionality."""
    
    class ServerClass(BaseMCPServer):
        def __init__(self):
            super().__init__(server_name, description)
        
        def register_tools(self):
            raise NotImplementedError("Subclasses must implement register_tools")
        
        def get_input_models(self) -> Dict[str, type]:
            raise NotImplementedError("Subclasses must implement get_input_models")
    
    return ServerClass


# Error handling utilities
class MCPError(Exception):
    """Base error class for MCP server operations."""
    
    def __init__(self, message: str, code: str = "UNKNOWN_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.message,
            "code": self.code,
            "details": self.details
        }


class ValidationError(MCPError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = "", value: Any = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details={"field": field, "value": value}
        )


class ConnectionError(MCPError):
    """Raised when external service connection fails."""
    
    def __init__(self, message: str, service: str = "", retryable: bool = True):
        super().__init__(
            message=message,
            code="CONNECTION_ERROR",
            details={"service": service, "retryable": retryable}
        )


class RateLimitError(MCPError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(
            message=message,
            code="RATE_LIMIT_ERROR",
            details={"retry_after": retry_after}
        )


# Tool execution wrapper
async def run_with_error_handling(func, *args, **kwargs) -> str:
    """Execute a tool function with consistent error handling."""
    try:
        result = await func(*args, **kwargs)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
    except ValidationError as e:
        logger.warning(f"Validation error: {e.message}")
        return json.dumps(e.to_dict(), ensure_ascii=False)
    except RateLimitError as e:
        logger.warning(f"Rate limit error: {e.message}")
        return json.dumps(e.to_dict(), ensure_ascii=False)
    except ConnectionError as e:
        logger.warning(f"Connection error: {e.message}")
        return json.dumps(e.to_dict(), ensure_ascii=False)
    except Exception as e:
        logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
        return json.dumps({
            "error": str(e),
            "code": "INTERNAL_ERROR",
            "details": {"function": func.__name__}
        }, ensure_ascii=False, indent=2)


# Utility functions
def sanitize_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """Sanitize and validate a file path."""
    if ".." in path or path.startswith("/"):
        raise ValidationError("Invalid path", field="path", value=path)
    
    full_path = Path(path)
    if base_dir:
        full_path = base_dir / path
        # Ensure path is within base_dir
        try:
            full_path.resolve().relative_to(base_dir.resolve())
        except ValueError:
            raise ValidationError("Path escapes base directory", field="path", value=path)
    
    return full_path


def truncate_text(text: str, max_length: int = 10000) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "...\n[truncated]"
