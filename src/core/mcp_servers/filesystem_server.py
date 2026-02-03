"""
Filesystem MCP Server - Embedded replacement for @modelcontextprotocol/server-filesystem.

Provides secure file system operations with path validation and access control.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    from fastmcp import FastMCP
    from pydantic import BaseModel, Field
    FASTMCP_AVAILABLE = True
except ImportError as e:
    FASTMCP_AVAILABLE = False
    FastMCP = None
    BaseModel = None
    Field = None

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("filesystem")


class ReadInput(BaseModel):
    """Input for reading a file."""
    path: str = Field(..., description="File path to read", min_length=1)
    encoding: str = Field(default="utf-8", description="File encoding")
    max_length: int = Field(default=100000, description="Max characters to read", ge=0)


class WriteInput(BaseModel):
    """Input for writing a file."""
    path: str = Field(..., description="File path to write", min_length=1)
    content: str = Field(..., description="Content to write")
    encoding: str = Field(default="utf-8", description="File encoding")


class ListInput(BaseModel):
    """Input for listing directory contents."""
    path: str = Field(default=".", description="Directory path to list")
    recursive: bool = Field(default=False, description="List recursively")


class GlobInput(BaseModel):
    """Input for glob pattern matching."""
    pattern: str = Field(..., description="Glob pattern (e.g., '**/*.py')")
    path: str = Field(default=".", description="Base path for search")
    max_results: int = Field(default=100, description="Max results", ge=1, le=1000)


class StatsInput(BaseModel):
    """Input for getting file stats."""
    path: str = Field(..., description="File or directory path")


class SearchInput(BaseModel):
    """Input for searching file contents."""
    query: str = Field(..., description="Text to search for")
    path: str = Field(default=".", description="Base path for search")
    file_pattern: str = Field(default="*", description="File pattern to match")
    case_sensitive: bool = Field(default=False, description="Case sensitive search")


class MakeDirInput(BaseModel):
    """Input for creating a directory."""
    path: str = Field(..., description="Directory path to create")
    parents: bool = Field(default=True, description="Create parent directories")


class RemoveInput(BaseModel):
    """Input for removing a file or directory."""
    path: str = Field(..., description="Path to remove")
    recursive: bool = Field(default=False, description="Remove recursively")


class CopyInput(BaseModel):
    """Input for copying a file or directory."""
    source: str = Field(..., description="Source path")
    destination: str = Field(..., description="Destination path")


# Configuration
ALLOWED_PATHS: List[str] = []
DENIED_PATHS: List[str] = []
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def load_config():
    """Load filesystem configuration."""
    global ALLOWED_PATHS, DENIED_PATHS, MAX_FILE_SIZE
    
    # Load from environment
    allowed_env = os.getenv("MCP_FILESYSTEM_ALLOWED_PATHS", "")
    denied_env = os.getenv("MCP_FILESYSTEM_DENIED_PATHS", "")
    
    if allowed_env:
        ALLOWED_PATHS = [p.strip() for p in allowed_env.split(",")]
    else:
        # Default: allow project root and temp directories
        ALLOWED_PATHS = [
            str(Path(__file__).parent.parent.parent),
            "/tmp",
            str(Path.home())
        ]
    
    if denied_env:
        DENIED_PATHS = [p.strip() for p in denied_env.split(",")]


def validate_path(path: str, operation: str = "access") -> Dict[str, Any]:
    """
    Validate and sanitize a file path.
    
    Returns dict with:
    - valid: bool
    - path: sanitized Path object or None
    - error: error message if invalid
    """
    try:
        # Normalize path
        path_obj = Path(path).resolve()
        
        # Check for path traversal
        if ".." in path:
            return {"valid": False, "error": "Path traversal not allowed", "path": None}
        
        # Check allowed paths
        if ALLOWED_PATHS:
            is_allowed = False
            for allowed in ALLOWED_PATHS:
                allowed_resolved = Path(allowed).resolve()
                try:
                    path_obj.relative_to(allowed_resolved)
                    is_allowed = True
                    break
                except ValueError:
                    continue
            
            if not is_allowed:
                return {
                    "valid": False,
                    "error": f"Path not in allowed directories: {ALLOWED_PATHS}",
                    "path": None
                }
        
        # Check denied paths
        for denied in DENIED_PATHS:
            denied_resolved = Path(denied).resolve()
            try:
                path_obj.relative_to(denied_resolved)
                return {"valid": False, "error": "Path is in denied directory", "path": None}
            except ValueError:
                continue
        
        return {"valid": True, "path": path_obj, "error": None}
    
    except Exception as e:
        return {"valid": False, "error": f"Invalid path: {str(e)}", "path": None}


@mcp.tool()
async def read_file(input: ReadInput) -> str:
    """
    Read the contents of a file.
    
    Input:
    - path: File path to read
    - encoding: File encoding (default: utf-8)
    - max_length: Maximum characters to read
    
    Returns JSON with file contents and metadata.
    """
    validation = validate_path(input.path, "read")
    
    if not validation["valid"]:
        return json.dumps({
            "success": False,
            "error": validation["error"],
            "path": input.path
        }, ensure_ascii=False, indent=2)
    
    path = validation["path"]
    
    # Check if file exists
    if not path.exists():
        return json.dumps({
            "success": False,
            "error": "File not found",
            "path": str(path)
        }, ensure_ascii=False, indent=2)
    
    # Check if it's a file
    if not path.is_file():
        return json.dumps({
            "success": False,
            "error": "Path is not a file",
            "path": str(path)
        }, ensure_ascii=False, indent=2)
    
    try:
        # Check file size
        size = path.stat().st_size
        if size > MAX_FILE_SIZE:
            return json.dumps({
                "success": False,
                "error": f"File too large ({size} bytes). Max size: {MAX_FILE_SIZE} bytes",
                "path": str(path)
            }, ensure_ascii=False, indent=2)
        
        # Read file
        with open(path, "r", encoding=input.encoding, errors="replace") as f:
            content = f.read(input.max_length)
        
        return json.dumps({
            "success": True,
            "path": str(path),
            "content": content,
            "size": size,
            "encoding": input.encoding,
            "truncated": len(content) >= input.max_length
        }, ensure_ascii=False, indent=2)
    
    except PermissionError:
        return json.dumps({
            "success": False,
            "error": "Permission denied",
            "path": str(path)
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "path": str(path)
        }, ensure_ascii=False, indent=2)


@mcp.tool()
async def write_file(input: WriteInput) -> str:
    """
    Write content to a file.
    
    Input:
    - path: File path to write
    - content: Content to write
    - encoding: File encoding (default: utf-8)
    
    Returns JSON with success status and file info.
    """
    validation = validate_path(input.path, "write")
    
    if not validation["valid"]:
        return json.dumps({
            "success": False,
            "error": validation["error"],
            "path": input.path
        }, ensure_ascii=False, indent=2)
    
    path = validation["path"]
    
    try:
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(path, "w", encoding=input.encoding) as f:
            f.write(input.content)
        
        return json.dumps({
            "success": True,
            "path": str(path),
            "size": len(input.content),
            "created": not path.exists() or False  # This is before write, so check stat
        }, ensure_ascii=False, indent=2)
    
    except PermissionError:
        return json.dumps({
            "success": False,
            "error": "Permission denied",
            "path": str(path)
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logger.error(f"Error writing file: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "path": str(path)
        }, ensure_ascii=False, indent=2)


@mcp.tool()
async def list_directory(input: ListInput) -> str:
    """
    List contents of a directory.
    
    Input:
    - path: Directory path
    - recursive: List subdirectories recursively
    
    Returns JSON with directory contents.
    """
    validation = validate_path(input.path, "list")
    
    if not validation["valid"]:
        return json.dumps({
            "success": False,
            "error": validation["error"],
            "path": input.path
        }, ensure_ascii=False, indent=2)
    
    path = validation["path"]
    
    if not path.exists():
        return json.dumps({
            "success": False,
            "error": "Directory not found",
            "path": str(path)
        }, ensure_ascii=False, indent=2)
    
    if not path.is_dir():
        return json.dumps({
            "success": False,
            "error": "Path is not a directory",
            "path": str(path)
        }, ensure_ascii=False, indent=2)
    
    try:
        items = []
        
        if input.recursive:
            for item in path.rglob("*"):
                if item.is_file():
                    stat = item.stat()
                    items.append({
                        "name": item.name,
                        "path": str(item),
                        "type": "file",
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        else:
            for item in path.iterdir():
                if item.is_file():
                    stat = item.stat()
                    items.append({
                        "name": item.name,
                        "path": str(item),
                        "type": "file",
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                elif item.is_dir():
                    items.append({
                        "name": item.name,
                        "path": str(item),
                        "type": "directory"
                    })
        
        return json.dumps({
            "success": True,
            "path": str(path),
            "items": items,
            "count": len(items)
        }, ensure_ascii=False, indent=2)
    
    except PermissionError:
        return json.dumps({
            "success": False,
            "error": "Permission denied",
            "path": str(path)
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logger.error(f"Error listing directory: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "path": str(path)
        }, ensure_ascii=False, indent=2)


@mcp.tool()
async def glob_files(input: GlobInput) -> str:
    """
    Find files matching a glob pattern.
    
    Input:
    - pattern: Glob pattern (e.g., '**/*.py')
    - path: Base path for search
    - max_results: Maximum results to return
    
    Returns JSON with matching files.
    """
    validation = validate_path(input.path, "search")
    
    if not validation["valid"]:
        return json.dumps({
            "success": False,
            "error": validation["error"],
            "path": input.path
        }, ensure_ascii=False, indent=2)
    
    path = validation["path"]
    
    try:
        matches = list(path.glob(input.pattern))
        files = [
            {
                "path": str(m),
                "name": m.name,
                "is_file": m.is_file(),
                "is_dir": m.is_dir()
            }
            for m in matches[:input.max_results]
        ]
        
        return json.dumps({
            "success": True,
            "pattern": input.pattern,
            "path": str(path),
            "files": files,
            "count": len(files)
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logger.error(f"Error in glob: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "pattern": input.pattern
        }, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_file_stats(input: StatsInput) -> str:
    """
    Get information about a file or directory.
    
    Input:
    - path: File or directory path
    
    Returns JSON with file/directory info.
    """
    validation = validate_path(input.path, "stats")
    
    if not validation["valid"]:
        return json.dumps({
            "success": False,
            "error": validation["error"],
            "path": input.path
        }, ensure_ascii=False, indent=2)
    
    path = validation["path"]
    
    if not path.exists():
        return json.dumps({
            "success": False,
            "error": "Path does not exist",
            "path": str(path)
        }, ensure_ascii=False, indent=2)
    
    try:
        stat = path.stat()
        result = {
            "success": True,
            "path": str(path),
            "exists": True,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:]
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "path": str(path)
        }, ensure_ascii=False, indent=2)


@mcp.tool()
async def search_content(input: SearchInput) -> str:
    """
    Search for text in files.
    
    Input:
    - query: Text to search for
    - path: Base path for search
    - file_pattern: File pattern to match
    - case_sensitive: Case sensitive search
    
    Returns JSON with search results.
    """
    validation = validate_path(input.path, "search")
    
    if not validation["valid"]:
        return json.dumps({
            "success": False,
            "error": validation["error"],
            "path": input.path
        }, ensure_ascii=False, indent=2)
    
    path = validation["path"]
    
    try:
        results = []
        flags = 0 if not input.case_sensitive else re.IGNORECASE
        
        for file_path in path.glob(input.file_pattern):
            if not file_path.is_file():
                continue
            
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    if re.search(input.query, line, flags):
                        results.append({
                            "file": str(file_path),
                            "line": i,
                            "content": line.strip()
                        })
                
            except Exception as e:
                logger.debug(f"Error reading {file_path}: {e}")
                continue
        
        return json.dumps({
            "success": True,
            "query": input.query,
            "path": str(path),
            "results": results,
            "count": len(results)
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logger.error(f"Error searching: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "query": input.query
        }, ensure_ascii=False, indent=2)


@mcp.tool()
async def create_directory(input: MakeDirInput) -> str:
    """
    Create a directory.
    
    Input:
    - path: Directory path to create
    - parents: Create parent directories
    
    Returns JSON with success status.
    """
    validation = validate_path(input.path, "create")
    
    if not validation["valid"]:
        return json.dumps({
            "success": False,
            "error": validation["error"],
            "path": input.path
        }, ensure_ascii=False, indent=2)
    
    path = validation["path"]
    
    try:
        path.mkdir(parents=input.parents, exist_ok=True)
        
        return json.dumps({
            "success": True,
            "path": str(path)
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logger.error(f"Error creating directory: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "path": str(path)
        }, ensure_ascii=False, indent=2)


@mcp.tool()
async def remove_path(input: RemoveInput) -> str:
    """
    Remove a file or directory.
    
    Input:
    - path: Path to remove
    - recursive: Remove recursively (for directories)
    
    Returns JSON with success status.
    """
    validation = validate_path(input.path, "remove")
    
    if not validation["valid"]:
        return json.dumps({
            "success": False,
            "error": validation["error"],
            "path": input.path
        }, ensure_ascii=False, indent=2)
    
    path = validation["path"]
    
    if not path.exists():
        return json.dumps({
            "success": False,
            "error": "Path does not exist",
            "path": str(path)
        }, ensure_ascii=False, indent=2)
    
    try:
        if path.is_dir() and input.recursive:
            shutil.rmtree(path)
        elif path.is_dir():
            path.rmdir()
        else:
            path.unlink()
        
        return json.dumps({
            "success": True,
            "path": str(path)
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logger.error(f"Error removing: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "path": str(path)
        }, ensure_ascii=False, indent=2)


@mcp.tool()
async def copy_path(input: CopyInput) -> str:
    """
    Copy a file or directory.
    
    Input:
    - source: Source path
    - destination: Destination path
    
    Returns JSON with success status.
    """
    src_validation = validate_path(input.source, "copy")
    dst_validation = validate_path(input.destination, "copy")
    
    if not src_validation["valid"]:
        return json.dumps({
            "success": False,
            "error": src_validation["error"],
            "source": input.source
        }, ensure_ascii=False, indent=2)
    
    if not dst_validation["valid"]:
        return json.dumps({
            "success": False,
            "error": dst_validation["error"],
            "destination": input.destination
        }, ensure_ascii=False, indent=2)
    
    source = src_validation["path"]
    destination = dst_validation["path"]
    
    if not source.exists():
        return json.dumps({
            "success": False,
            "error": "Source does not exist",
            "source": str(source)
        }, ensure_ascii=False, indent=2)
    
    try:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        
        return json.dumps({
            "success": True,
            "source": str(source),
            "destination": str(destination)
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logger.error(f"Error copying: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "source": str(source),
            "destination": str(destination)
        }, ensure_ascii=False, indent=2)


def run():
    """Run the filesystem MCP server."""
    load_config()
    mcp.run()


if __name__ == "__main__":
    run()
