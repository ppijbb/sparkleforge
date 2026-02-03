"""
Embedded MCP Server Registry - Central registry for all embedded MCP servers.

This module provides:
- Server discovery and loading
- Configuration generation
- Process management for embedded servers
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Server definitions
EMBEDDED_SERVERS = {
    "web-fetch": {
        "name": "web-fetch",
        "module": "embedded_mcp_servers.web_fetch_server",
        "description": "Web content fetching and search",
        "tools": ["fetch", "search"],
        "type": "embedded",
        "transport": "stdio"
    },
    "search": {
        "name": "search",
        "module": "embedded_mcp_servers.search_server",
        "description": "Web search with multiple providers",
        "tools": ["search", "search_all", "multi_search", "search_news"],
        "type": "embedded",
        "transport": "stdio"
    },
    "arxiv": {
        "name": "arxiv",
        "module": "embedded_mcp_servers.arxiv_server",
        "description": "Academic paper search from arXiv",
        "tools": ["search", "get_paper_details", "list_categories", "search_by_category", "search_recent_papers"],
        "type": "embedded",
        "transport": "stdio"
    },
    "memory": {
        "name": "memory",
        "module": "embedded_mcp_servers.memory_server",
        "description": "Persistent memory storage",
        "tools": ["store_memory", "recall_memories", "get_memory", "update_memory", "delete_memory", "list_all_memories", "clear_memories", "get_memory_stats"],
        "type": "embedded",
        "transport": "stdio"
    },
    "filesystem": {
        "name": "filesystem",
        "module": "embedded_mcp_servers.filesystem_server",
        "description": "File system operations",
        "tools": ["read_file", "write_file", "list_directory", "glob_files", "get_file_stats", "search_content", "create_directory", "remove_path", "copy_path"],
        "type": "embedded",
        "transport": "stdio"
    },
    "github": {
        "name": "github",
        "module": "embedded_mcp_servers.github_server",
        "description": "GitHub API operations",
        "tools": ["search_repositories", "get_repository", "list_issues", "get_issue", "create_issue", "list_pull_requests", "get_file_contents", "search_code", "get_user_info", "list_user_repos"],
        "type": "embedded",
        "transport": "stdio"
    }
}


def get_embedded_server_config(server_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for an embedded server."""
    if server_name not in EMBEDDED_SERVERS:
        return None
    
    server = EMBEDDED_SERVERS[server_name]
    
    # Get Python executable
    python_path = os.getenv("MCP_PYTHON_PATH", "python")
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    return {
        "command": python_path,
        "args": ["-m", server["module"]],
        "env": {
            "PYTHONPATH": str(project_root)
        },
        "type": "stdio",
        "server_name": server_name,
        "description": server["description"],
        "tools": server["tools"]
    }


def generate_mcp_config() -> Dict[str, Any]:
    """Generate MCP configuration for all embedded servers."""
    config = {
        "mcpServers": {}
    }
    
    for server_name in EMBEDDED_SERVERS:
        server_config = get_embedded_server_config(server_name)
        if server_config:
            config["mcpServers"][server_name] = {
                "command": server_config["command"],
                "args": server_config["args"],
                "env": server_config["env"]
            }
    
    return config


def get_all_tools() -> Dict[str, List[str]]:
    """Get all tools from all embedded servers."""
    tools = {}
    for server_name, server in EMBEDDED_SERVERS.items():
        tools[server_name] = server["tools"]
    return tools


def get_server_info(server_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific server."""
    return EMBEDDED_SERVERS.get(server_name)


def list_servers() -> List[Dict[str, Any]]:
    """List all embedded servers with their info."""
    return [
        {
            "name": name,
            "description": info["description"],
            "tools": info["tools"],
            "type": info["type"],
            "transport": info["transport"]
        }
        for name, info in EMBEDDED_SERVERS.items()
    ]


def validate_server_name(server_name: str) -> bool:
    """Check if a server name is valid."""
    return server_name in EMBEDDED_SERVERS


def get_server_command(server_name: str) -> List[str]:
    """Get the command to run a server."""
    config = get_embedded_server_config(server_name)
    if config:
        return [config["command"]] + config["args"]
    return []


# Fallback mapping for external MCP servers
EXTERNAL_TO_EMBEDDED_FALLBACK = {
    "fetch-mcp": "web-fetch",
    "tavily": "search",
    "exa": "search",
    "@modelcontextprotocol/server-arxiv": "arxiv",
    "@modelcontextprotocol/server-filesystem": "filesystem",
    "@modelcontextprotocol/server-github": "github"
}


def get_fallback_server(external_server: str) -> Optional[str]:
    """Get the embedded fallback for an external server."""
    return EXTERNAL_TO_EMBEDDED_FALLBACK.get(external_server)


def create_unified_config(external_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a unified MCP configuration that uses embedded servers
    with fallbacks to external servers.
    """
    unified = {
        "mcpServers": {},
        "_metadata": {
            "generated_by": "embedded_server_registry",
            "embedded_servers": list(EMBEDDED_SERVERS.keys()),
            "external_fallbacks": EXTERNAL_TO_EMBEDDED_FALLBACK
        }
    }
    
    # Add all embedded servers
    for server_name, server_config in generate_mcp_config()["mcpServers"].items():
        unified["mcpServers"][server_name] = server_config
    
    # Add external servers as fallbacks (only if no embedded version)
    if external_config:
        for server_name, server_config in external_config.get("mcpServers", {}).items():
            # Skip if we have an embedded version
            if validate_server_name(server_name):
                logger.info(f"Skipping external {server_name} - using embedded version")
                continue
            
            # Add as fallback
            unified["mcpServers"][server_name] = server_config
    
    return unified


if __name__ == "__main__":
    # Print all embedded servers
    servers = list_servers()
    print("Embedded MCP Servers:")
    print("-" * 60)
    for server in servers:
        print(f"\n{server['name']}:")
        print(f"  Description: {server['description']}")
        print(f"  Type: {server['type']}")
        print(f"  Transport: {server['transport']}")
        print(f"  Tools: {', '.join(server['tools'])}")
