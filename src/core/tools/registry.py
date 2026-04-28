import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class ToolCategory(str, Enum):
    """MCP 도구 카테고리 (Compatibility with legacy system)."""
    SEARCH = "search"
    DATA = "data"
    CODE = "code"
    ACADEMIC = "academic"
    BUSINESS = "business"
    UTILITY = "utility"
    BROWSER = "browser"
    DOCUMENT = "document"
    FILE = "file"
    GIT = "git"
    COMPUTER = "computer"

@dataclass
class ToolInfo:
    """도구 정보 (Legacy compatibility)."""
    name: str
    category: ToolCategory
    description: str
    parameters: Dict[str, Any]
    mcp_server: str

@dataclass
class ToolMetadata:
    """Metadata for a tool (Internal representation)."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    category: Union[str, ToolCategory] = ToolCategory.UTILITY
    mcp_server: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source: str = "local" # 'mcp' or 'local'
    original_name: Optional[str] = None

@dataclass
class ToolResult:
    """도구 실행 결과 (Legacy compatibility)."""
    success: bool
    data: Any = None
    error: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    confidence: float = 0.0
    tool_name: str | None = None
    source: str | None = None

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style accessor for legacy call sites."""
        return self.to_dict().get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation used by older MCP paths."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "confidence": self.confidence,
            "tool_name": self.tool_name,
            "source": self.source,
        }

class ToolRegistry:
    """Centralized Tool Registry for SparkleForge (Phase 2).
    
    This is a singleton registry that stores tools discovered from 
    MCP servers or registered natively.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
            cls._instance.tools: Dict[str, ToolMetadata] = {}
            cls._instance.executors: Dict[str, Any] = {}
            cls._instance.langchain_tools: Dict[str, Any] = {}
            cls._instance.mcp_tool_mapping: Dict[str, Tuple[str, str]] = {}
            cls._instance.tool_sources: Dict[str, str] = {}
        return cls._instance
    
    def register(
        self, 
        metadata: ToolMetadata, 
        executor: Any,
        langchain_tool: Any = None
    ):
        """Registers a tool with its executor."""
        self.tools[metadata.name] = metadata
        self.executors[metadata.name] = executor
        self.tool_sources[metadata.name] = metadata.source
        if langchain_tool:
            self.langchain_tools[metadata.name] = langchain_tool
        
        if metadata.source == "mcp" and metadata.mcp_server and metadata.original_name:
            self.mcp_tool_mapping[metadata.name] = (metadata.mcp_server, metadata.original_name)
            
        logger.debug(f"Registered tool: {metadata.name} (source: {metadata.source})")

    # --- Legacy Compatibility Methods ---
    
    def register_mcp_tool(self, server_name: str, tool_obj: Any, tool_def: Any = None):
        """Bridge for UniversalMCPHub legacy registration."""
        if isinstance(tool_obj, str):
            tool_name = tool_obj
        else:
            tool_name = tool_obj.name if hasattr(tool_obj, "name") else str(tool_obj)

        registered_name = f"{server_name}::{tool_name}"
        
        description = "Tool from MCP server"
        input_schema = {}
        if tool_def:
            description = getattr(tool_def, "description", description)
            input_schema = getattr(tool_def, "inputSchema", {})

        # Simple category inference
        category = ToolCategory.UTILITY
        if "search" in tool_name.lower(): category = ToolCategory.SEARCH
        
        metadata = ToolMetadata(
            name=registered_name,
            description=description,
            parameters=input_schema,
            category=category,
            mcp_server=server_name,
            source="mcp",
            original_name=tool_name
        )
        
        # In legacy mode, 'tool_obj' might be a callable or a session-bound proxy
        self.register(metadata, tool_obj)

    def register_local_tool(self, tool_info: ToolInfo, langchain_tool: Any):
        """Bridge for UniversalMCPHub legacy registration."""
        metadata = ToolMetadata(
            name=tool_info.name,
            description=tool_info.description,
            parameters=tool_info.parameters,
            category=tool_info.category,
            mcp_server=tool_info.mcp_server,
            source="local"
        )
        self.register(metadata, langchain_tool, langchain_tool)

    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """Convert ToolMetadata back to ToolInfo for legacy consumers."""
        meta = self.tools.get(tool_name)
        if not meta: return None
        return ToolInfo(
            name=meta.name,
            category=meta.category if isinstance(meta.category, ToolCategory) else ToolCategory.UTILITY,
            description=meta.description,
            parameters=meta.parameters,
            mcp_server=meta.mcp_server or ""
        )

    def get_langchain_tool(self, tool_name: str) -> Any:
        return self.langchain_tools.get(tool_name)

    def get_all_langchain_tools(self) -> List[Any]:
        return list(self.langchain_tools.values())

    def get_all_tool_names(self) -> List[str]:
        return list(self.tools.keys())

    def remove_tool(self, tool_name: str) -> None:
        self.tools.pop(tool_name, None)
        self.executors.pop(tool_name, None)
        self.langchain_tools.pop(tool_name, None)
        self.mcp_tool_mapping.pop(tool_name, None)
        self.tool_sources.pop(tool_name, None)

    def is_mcp_tool(self, tool_name: str) -> bool:
        meta = self.tools.get(tool_name)
        return meta.source == "mcp" if meta else False

    def get_mcp_server_info(self, tool_name: str) -> Optional[Tuple[str, str]]:
        return self.mcp_tool_mapping.get(tool_name)

    # --- Execution Logic ---

    async def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        executor = self.executors.get(name)
        if not executor:
            raise ValueError(f"Tool executor not found: {name}")
            
        try:
            # Check if it's a LangChain tool-like object
            if hasattr(executor, "_arun"):
                return await executor._arun(**arguments)
            elif asyncio.iscoroutinefunction(executor):
                return await executor(**arguments)
            elif callable(executor):
                return executor(**arguments)
            else:
                return executor # Might be a pre-computed value or proxy
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            raise

# Global registry instance
registry = ToolRegistry()


def tool(
    *,
    name: str | None = None,
    description: str = "",
    parameters: Dict[str, Any] | None = None,
    category: ToolCategory | str = ToolCategory.UTILITY,
    tags: List[str] | None = None,
):
    """Register a native callable while preserving the original function."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        metadata = ToolMetadata(
            name=tool_name,
            description=description or (func.__doc__ or "").strip() or tool_name,
            parameters=parameters or {},
            category=category,
            tags=tags or [],
            source="local",
        )
        registry.register(metadata, func, func)
        return func

    return decorator
