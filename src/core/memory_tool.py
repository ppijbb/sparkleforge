"""
Memory-as-a-Tool

백서 요구사항: 메모리를 도구로 노출하여 에이전트가 직접 조회/업데이트 가능
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.memory_types import BaseMemory, MemoryType, create_memory_from_dict
from src.core.memory_access_control import get_memory_access_control, AccessLevel

logger = logging.getLogger(__name__)


class MemoryTool:
    """
    Memory-as-a-Tool 구현.
    
    백서 요구사항: 메모리를 도구로 노출하여 에이전트가 직접 사용 가능
    """
    
    def __init__(self):
        """초기화."""
        self.access_control = get_memory_access_control()
        # TODO: 실제 메모리 저장소 연결
        # Issue: High priority - 실제 메모리 저장소(벡터 DB 등) 연결 필요
        # 현재는 임시 인메모리 저장소 사용
        # 향후 HybridStorage 또는 벡터 DB와 통합 필요
        self.memory_storage: Dict[str, BaseMemory] = {}  # 임시 저장소
        
        logger.info("MemoryTool initialized")
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """
        도구 스키마 반환 (MCP 도구 형식).
        
        Returns:
            도구 스키마
        """
        return {
            "name": "memory_tool",
            "description": "Access and manage memories (Semantic, Episodic, Procedural). Allows agents to read, write, and search memories.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write", "search", "update", "delete"],
                        "description": "Action to perform on memories"
                    },
                    "memory_id": {
                        "type": "string",
                        "description": "Memory ID (for read, update, delete actions)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for search action)"
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["semantic", "episodic", "procedural"],
                        "description": "Type of memory to create (for write action)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Memory content (for write, update actions)"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata (for write, update actions)"
                    }
                },
                "required": ["action"]
            }
        }
    
    async def execute(
        self,
        action: str,
        user_id: str,
        memory_id: Optional[str] = None,
        query: Optional[str] = None,
        memory_type: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        메모리 도구 실행.
        
        Args:
            action: 동작 (read, write, search, update, delete)
            user_id: 사용자 ID
            memory_id: 메모리 ID
            query: 검색 쿼리
            memory_type: 메모리 타입
            content: 메모리 내용
            metadata: 메타데이터
            
        Returns:
            실행 결과
        """
        try:
            if action == "read":
                return await self._read_memory(memory_id, user_id)
            elif action == "write":
                return await self._write_memory(user_id, memory_type, content, metadata)
            elif action == "search":
                return await self._search_memories(user_id, query)
            elif action == "update":
                return await self._update_memory(memory_id, user_id, content, metadata)
            elif action == "delete":
                return await self._delete_memory(memory_id, user_id)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
        except Exception as e:
            logger.error(f"Memory tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _read_memory(self, memory_id: str, user_id: str) -> Dict[str, Any]:
        """메모리 읽기."""
        # 접근 권한 확인
        if not self.access_control.check_access(user_id, memory_id, AccessLevel.READ):
            return {
                "success": False,
                "error": "Access denied"
            }
        
        # 메모리 조회
        memory = self.memory_storage.get(memory_id)
        if not memory:
            return {
                "success": False,
                "error": "Memory not found"
            }
        
        # Provenance 기록
        from src.core.memory_provenance import get_provenance_tracker
        provenance_tracker = get_provenance_tracker()
        provenance_tracker.record_retrieval(memory_id, session_id=None, agent_id=None)
        
        return {
            "success": True,
            "memory": memory.to_dict()
        }
    
    async def _write_memory(
        self,
        user_id: str,
        memory_type: Optional[str],
        content: str,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """메모리 쓰기."""
        # 메모리 생성 권한 확인
        if not self.access_control.can_generate_memory(user_id):
            return {
                "success": False,
                "error": "Memory generation not allowed for this user"
            }
        
        # 메모리 타입 결정
        if not memory_type:
            memory_type = "semantic"  # 기본값
        
        # 메모리 생성
        import uuid
        memory_id = f"mem_{user_id}_{uuid.uuid4().hex[:12]}"
        
        try:
            mem_type = MemoryType(memory_type)
            
            if mem_type == MemoryType.SEMANTIC:
                from src.core.memory_types import SemanticMemory
                memory = SemanticMemory(
                    memory_id=memory_id,
                    content=content,
                    user_id=user_id,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    metadata=metadata or {}
                )
            elif mem_type == MemoryType.EPISODIC:
                from src.core.memory_types import EpisodicMemory
                memory = EpisodicMemory(
                    memory_id=memory_id,
                    content=content,
                    user_id=user_id,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    metadata=metadata or {}
                )
            elif mem_type == MemoryType.PROCEDURAL:
                from src.core.memory_types import ProceduralMemory
                memory = ProceduralMemory(
                    memory_id=memory_id,
                    content=content,
                    user_id=user_id,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    metadata=metadata or {}
                )
            else:
                return {
                    "success": False,
                    "error": f"Unknown memory type: {memory_type}"
                }
            
            # 검증 (Memory Poisoning 방지)
            from src.core.memory_validation import get_memory_validator
            validator = get_memory_validator()
            validation_result = await validator.validate_memory(memory, user_id)
            
            if not validation_result.is_valid:
                return {
                    "success": False,
                    "error": f"Memory validation failed: {', '.join(validation_result.issues)}",
                    "sanitized_content": validation_result.sanitized_content
                }
            
            # 저장
            self.memory_storage[memory_id] = memory
            
            # Provenance 기록
            from src.core.memory_provenance import get_provenance_tracker
            provenance_tracker = get_provenance_tracker()
            provenance_tracker.record_extraction(
                memory_id=memory_id,
                source_session_id=metadata.get("session_id") if metadata else None,
                agent_id=metadata.get("agent_id") if metadata else None
            )
            
            return {
                "success": True,
                "memory_id": memory_id,
                "memory": memory.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to write memory: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _search_memories(self, user_id: str, query: str) -> Dict[str, Any]:
        """메모리 검색."""
        # 사용자의 메모리만 검색
        user_memories = [
            mem for mem in self.memory_storage.values()
            if mem.user_id == user_id
        ]
        
        # 간단한 키워드 검색
        matching_memories = []
        query_lower = query.lower()
        
        for memory in user_memories:
            if query_lower in memory.content.lower():
                matching_memories.append(memory.to_dict())
        
        return {
            "success": True,
            "query": query,
            "results": matching_memories,
            "count": len(matching_memories)
        }
    
    async def _update_memory(
        self,
        memory_id: str,
        user_id: str,
        content: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """메모리 업데이트."""
        # 접근 권한 확인
        if not self.access_control.check_access(user_id, memory_id, AccessLevel.WRITE):
            return {
                "success": False,
                "error": "Access denied"
            }
        
        memory = self.memory_storage.get(memory_id)
        if not memory:
            return {
                "success": False,
                "error": "Memory not found"
            }
        
        # 업데이트
        if content:
            memory.content = content
        if metadata:
            memory.metadata.update(metadata)
        memory.last_accessed = datetime.now()
        
        # Provenance 기록
        from src.core.memory_provenance import get_provenance_tracker
        provenance_tracker = get_provenance_tracker()
        provenance_tracker.record_update(memory_id, agent_id=None)
        
        return {
            "success": True,
            "memory": memory.to_dict()
        }
    
    async def _delete_memory(self, memory_id: str, user_id: str) -> Dict[str, Any]:
        """메모리 삭제."""
        # 접근 권한 확인 (ADMIN 필요)
        if not self.access_control.check_access(user_id, memory_id, AccessLevel.ADMIN):
            return {
                "success": False,
                "error": "Access denied (admin required)"
            }
        
        if memory_id not in self.memory_storage:
            return {
                "success": False,
                "error": "Memory not found"
            }
        
        del self.memory_storage[memory_id]
        
        return {
            "success": True,
            "message": f"Memory {memory_id} deleted"
        }


# 전역 인스턴스
_memory_tool: Optional[MemoryTool] = None


def get_memory_tool() -> MemoryTool:
    """전역 메모리 도구 인스턴스 반환."""
    global _memory_tool
    if _memory_tool is None:
        _memory_tool = MemoryTool()
    return _memory_tool

