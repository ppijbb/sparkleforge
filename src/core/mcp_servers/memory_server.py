"""
Memory MCP Server - Persistent memory storage for SparkleForge.

Provides persistent memory capabilities using file-based storage
with optional ChromaDB integration for vector search.
"""

import asyncio
import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

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
mcp = FastMCP("memory")


class MemoryType(str, Enum):
    """Types of memory."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"


class MemoryItem(BaseModel):
    """A single memory item."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="Memory content")
    memory_type: MemoryType = Field(default=MemoryType.SHORT_TERM)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = Field(default=0)
    importance: float = Field(default=0.5, ge=0, le=1)


class StoreMemoryInput(BaseModel):
    """Input for storing a memory."""
    content: str = Field(..., description="Content to store", min_length=1)
    memory_type: str = Field(default="short_term", description="Type of memory")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    importance: float = Field(default=0.5, ge=0, le=1, description="Importance score")


class RecallInput(BaseModel):
    """Input for recalling memories."""
    query: str = Field(..., description="Query to search memories")
    memory_type: Optional[str] = Field(default=None, description="Filter by memory type")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    max_results: int = Field(default=5, ge=1, le=100, description="Max results")
    min_importance: float = Field(default=0.0, ge=0, le=1, description="Min importance")


class UpdateMemoryInput(BaseModel):
    """Input for updating a memory."""
    memory_id: str = Field(..., description="Memory ID to update")
    content: Optional[str] = Field(default=None, description="New content")
    tags: Optional[List[str]] = Field(default=None, description="New tags")
    importance: Optional[float] = Field(default=None, ge=0, le=1, description="New importance")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="New metadata")


class MemoryManager:
    """Manages persistent memory storage."""
    
    def __init__(self, storage_dir: str = "data/memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.memories_file = self.storage_dir / "memories.json"
        self.index_file = self.storage_dir / "index.json"
        
        # In-memory cache
        self._memories: Dict[str, Dict] = {}
        self._index: Dict[str, List[str]] = {}  # tag -> memory_ids
        
        # Load existing memories
        self._load_memories()
    
    def _load_memories(self):
        """Load memories from disk."""
        if self.memories_file.exists():
            try:
                with open(self.memories_file, 'r', encoding='utf-8') as f:
                    self._memories = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load memories: {e}")
                self._memories = {}
        
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self._index = {}
    
    def _save_memories(self):
        """Save memories to disk."""
        try:
            with open(self.memories_file, 'w', encoding='utf-8') as f:
                json.dump(self._memories, f, indent=2, ensure_ascii=False)
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    def _update_index(self, memory_id: str, tags: List[str], remove: bool = False):
        """Update the tag index."""
        for tag in tags:
            tag = tag.lower().strip()
            if not tag:
                continue
            
            if remove:
                if tag in self._index and memory_id in self._index[tag]:
                    self._index[tag].remove(memory_id)
                    if not self._index[tag]:
                        del self._index[tag]
            else:
                if tag not in self._index:
                    self._index[tag] = []
                if memory_id not in self._index[tag]:
                    self._index[tag].append(memory_id)
    
    def store(
        self,
        content: str,
        memory_type: str = "short_term",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        importance: float = 0.5
    ) -> Dict[str, Any]:
        """Store a new memory."""
        memory_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        memory = {
            "id": memory_id,
            "content": content,
            "memory_type": memory_type,
            "tags": [t.lower().strip() for t in tags] if tags else [],
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
            "access_count": 0,
            "importance": importance
        }
        
        self._memories[memory_id] = memory
        self._update_index(memory_id, memory["tags"])
        self._save_memories()
        
        return memory
    
    def recall(
        self,
        query: str,
        memory_type: str = None,
        tags: List[str] = None,
        max_results: int = 5,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Recall memories matching query."""
        results = []
        query_lower = query.lower()
        
        for memory_id, memory in self._memories.items():
            # Filter by memory type
            if memory_type and memory["memory_type"] != memory_type:
                continue
            
            # Filter by tags
            if tags:
                memory_tags = set(memory["tags"])
                if not any(t.lower() in memory_tags for t in tags):
                    continue
            
            # Filter by importance
            if memory["importance"] < min_importance:
                continue
            
            # Score by content match
            score = 0
            query_words = set(query_lower.split())
            content_words = set(memory["content"].lower().split())
            
            # Exact word matches
            score += len(query_words & content_words) * 2
            
            # Partial matches
            for qw in query_words:
                if qw in memory["content"].lower():
                    score += 1
            
            # Boost by importance and recency
            score += memory["importance"] * 2
            
            if score > 0:
                # Increment access count
                memory["access_count"] += 1
                memory["updated_at"] = datetime.now().isoformat()
                
                results.append({
                    **memory,
                    "score": score
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        self._save_memories()
        
        return results[:max_results]
    
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID."""
        memory = self._memories.get(memory_id)
        if memory:
            memory["access_count"] += 1
            memory["updated_at"] = datetime.now().isoformat()
            self._save_memories()
        return memory
    
    def update(
        self,
        memory_id: str,
        content: str = None,
        tags: List[str] = None,
        importance: float = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Update an existing memory."""
        if memory_id not in self._memories:
            return None
        
        memory = self._memories[memory_id]
        now = datetime.now().isoformat()
        
        # Update fields
        if content is not None:
            memory["content"] = content
        
        if tags is not None:
            # Remove old tags from index
            self._update_index(memory_id, memory["tags"], remove=True)
            memory["tags"] = [t.lower().strip() for t in tags]
            self._update_index(memory_id, memory["tags"])
        
        if importance is not None:
            memory["importance"] = importance
        
        if metadata is not None:
            memory["metadata"] = {**memory["metadata"], **metadata}
        
        memory["updated_at"] = now
        self._save_memories()
        
        return memory
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id not in self._memories:
            return False
        
        memory = self._memories[memory_id]
        self._update_index(memory_id, memory["tags"], remove=True)
        
        del self._memories[memory_id]
        self._save_memories()
        
        return True
    
    def list_memories(
        self,
        memory_type: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all memories, optionally filtered by type."""
        memories = list(self._memories.values())
        
        if memory_type:
            memories = [m for m in memories if m["memory_type"] == memory_type]
        
        # Sort by updated_at descending
        memories.sort(key=lambda x: x["updated_at"], reverse=True)
        
        return memories[offset:offset + limit]
    
    def clear(self, memory_type: str = None) -> int:
        """Clear memories, optionally by type."""
        if memory_type:
            # Get IDs to delete
            to_delete = [
                mid for mid, m in self._memories.items()
                if m["memory_type"] == memory_type
            ]
            for mid in to_delete:
                self._update_index(mid, self._memories[mid]["tags"], remove=True)
                del self._memories[mid]
        else:
            # Clear all
            self._memories = {}
            self._index = {}
        
        self._save_memories()
        
        return len(to_delete) if memory_type else len(self._memories)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        by_type = {}
        for memory in self._memories.values():
            mtype = memory["memory_type"]
            if mtype not in by_type:
                by_type[mtype] = 0
            by_type[mtype] += 1
        
        return {
            "total_memories": len(self._memories),
            "by_type": by_type,
            "unique_tags": len(self._index),
            "total_accesses": sum(m["access_count"] for m in self._memories.values())
        }


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager."""
    global _memory_manager
    if _memory_manager is None:
        storage_dir = os.getenv("MEMORY_STORAGE_DIR", "data/memory")
        _memory_manager = MemoryManager(storage_dir)
    return _memory_manager


@mcp.tool()
async def store_memory(input: StoreMemoryInput) -> str:
    """
    Store a new memory.
    
    Input:
    - content: What to remember
    - memory_type: short_term, long_term, episodic, semantic, working
    - tags: Optional tags for categorization
    - importance: 0-1 score for memory importance
    - metadata: Additional metadata
    
    Returns JSON with the stored memory.
    """
    manager = get_memory_manager()
    
    try:
        mem_type = MemoryType(input.memory_type)
    except ValueError:
        mem_type = MemoryType.SHORT_TERM
    
    memory = manager.store(
        content=input.content,
        memory_type=mem_type.value,
        tags=input.tags,
        metadata=input.metadata,
        importance=input.importance
    )
    
    return json.dumps({
        "success": True,
        "memory": memory
    }, ensure_ascii=False, indent=2)


@mcp.tool()
async def recall_memories(input: RecallInput) -> str:
    """
    Recall memories matching a query.
    
    Input:
    - query: Text to search for in memories
    - memory_type: Optional filter by memory type
    - tags: Optional filter by tags
    - max_results: Maximum memories to return
    - min_importance: Minimum importance score
    
    Returns JSON with matching memories.
    """
    manager = get_memory_manager()
    
    memories = manager.recall(
        query=input.query,
        memory_type=input.memory_type,
        tags=input.tags,
        max_results=input.max_results,
        min_importance=input.min_importance
    )
    
    return json.dumps({
        "success": True,
        "query": input.query,
        "results": memories,
        "count": len(memories)
    }, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_memory(memory_id: str = Field(..., description="Memory ID")) -> str:
    """
    Get a specific memory by ID.
    
    Input:
    - memory_id: The memory ID
    
    Returns JSON with the memory.
    """
    manager = get_memory_manager()
    memory = manager.get(memory_id)
    
    if memory:
        return json.dumps({
            "success": True,
            "memory": memory
        }, ensure_ascii=False, indent=2)
    else:
        return json.dumps({
            "success": False,
            "error": "Memory not found",
            "memory_id": memory_id
        }, ensure_ascii=False, indent=2)


@mcp.tool()
async def update_memory(input: UpdateMemoryInput) -> str:
    """
    Update an existing memory.
    
    Input:
    - memory_id: ID of memory to update
    - content: New content (optional)
    - tags: New tags (optional)
    - importance: New importance (optional)
    - metadata: New metadata (optional)
    
    Returns JSON with updated memory.
    """
    manager = get_memory_manager()
    
    memory = manager.update(
        memory_id=input.memory_id,
        content=input.content,
        tags=input.tags,
        importance=input.importance,
        metadata=input.metadata
    )
    
    if memory:
        return json.dumps({
            "success": True,
            "memory": memory
        }, ensure_ascii=False, indent=2)
    else:
        return json.dumps({
            "success": False,
            "error": "Memory not found",
            "memory_id": input.memory_id
        }, ensure_ascii=False, indent=2)


@mcp.tool()
async def delete_memory(memory_id: str = Field(..., description="Memory ID to delete")) -> str:
    """
    Delete a memory.
    
    Input:
    - memory_id: ID of memory to delete
    
    Returns JSON with success status.
    """
    manager = get_memory_manager()
    success = manager.delete(memory_id)
    
    return json.dumps({
        "success": success,
        "memory_id": memory_id
    }, ensure_ascii=False, indent=2)


@mcp.tool()
async def list_all_memories(
    memory_type: Optional[str] = Field(default=None, description="Filter by type"),
    limit: int = Field(default=100, ge=1, le=1000, description="Max results"),
    offset: int = Field(default=0, ge=0, description="Skip first N results")
) -> str:
    """
    List all memories.
    
    Input:
    - memory_type: Optional filter by type
    - limit: Maximum results
    - offset: Skip first N results
    
    Returns JSON with memories list.
    """
    manager = get_memory_manager()
    memories = manager.list_memories(memory_type, limit, offset)
    
    return json.dumps({
        "success": True,
        "memories": memories,
        "count": len(memories)
    }, ensure_ascii=False, indent=2)


@mcp.tool()
async def clear_memories(memory_type: Optional[str] = Field(default=None, description="Type to clear (all if not specified)")) -> str:
    """
    Clear memories.
    
    Input:
    - memory_type: Type to clear (optional, clears all if not specified)
    
    Returns JSON with count of cleared memories.
    """
    manager = get_memory_manager()
    count = manager.clear(memory_type)
    
    return json.dumps({
        "success": True,
        "cleared_count": count,
        "memory_type": memory_type or "all"
    }, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_memory_stats() -> str:
    """
    Get memory statistics.
    
    Returns JSON with memory statistics.
    """
    manager = get_memory_manager()
    stats = manager.get_stats()
    
    return json.dumps({
        "success": True,
        "stats": stats
    }, ensure_ascii=False, indent=2)


def run():
    """Run the memory MCP server."""
    mcp.run()


if __name__ == "__main__":
    run()
