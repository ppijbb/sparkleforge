"""
LLM-driven Memory Extraction & Consolidation

백서 요구사항: Memory는 LLM-driven ETL pipeline
- Extraction: 대화에서 핵심 정보 추출
- Consolidation: 기존 메모리와 통합, 충돌 해결, 중복 제거
"""

import logging
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.llm_manager import execute_llm_task, TaskType
from src.core.memory_types import (
    MemoryType, BaseMemory, SemanticMemory, EpisodicMemory, ProceduralMemory,
    create_memory_from_dict
)
from src.core.memory_provenance import get_provenance_tracker, ProvenanceEventType

logger = logging.getLogger(__name__)


# Extraction 프롬프트 템플릿
EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction system. Your task is to extract key information from conversation history and convert it into structured memories.

Memory Types:
1. Semantic Memory: Factual knowledge about the user or topics (e.g., "User is a Python developer", "User prefers data analysis")
2. Episodic Memory: Specific events or experiences (e.g., "User completed a machine learning project in January 2024")
3. Procedural Memory: How-to knowledge or tool usage patterns (e.g., "User knows how to use pandas for data analysis")

Extract memories that are:
- Important and relevant for future interactions
- Specific and actionable
- Not redundant with existing information

Return a JSON array of memory objects, each with:
- memory_type: "semantic", "episodic", or "procedural"
- content: The memory content (concise, factual)
- importance: 0.0-1.0 (how important this memory is)
- For semantic: include "facts" (list of extracted facts) and optional "domain"
- For episodic: include "event_timestamp" (ISO format) and "context" (event details)
- For procedural: include "procedure_steps" (list), optional "tool_name", and "examples" (list)
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """Extract memories from the following conversation history:

{conversation_history}

Existing memories (for reference, avoid duplicates):
{existing_memories}

Return only new or updated memories as a JSON array.
"""


# Consolidation 프롬프트 템플릿
CONSOLIDATION_SYSTEM_PROMPT = """You are a memory consolidation system. Your task is to merge and consolidate memories, resolving conflicts and removing duplicates.

Rules:
1. If memories contain the same information, merge them into one
2. If memories conflict, keep the most recent or most reliable one
3. Update importance scores based on multiple sources
4. Preserve provenance information (source memory IDs)

Return a JSON object with:
- consolidated_memories: Array of merged/updated memories
- deleted_memory_ids: Array of IDs that were merged/deleted
- conflicts_resolved: Array of conflict resolution decisions
"""

CONSOLIDATION_USER_PROMPT_TEMPLATE = """Consolidate the following memories:

New memories:
{new_memories}

Existing memories:
{existing_memories}

Merge duplicates, resolve conflicts, and return the consolidated set.
"""


class MemoryExtractor:
    """
    LLM-driven Memory Extraction 시스템.
    
    백서 요구사항: LLM을 사용하여 대화에서 핵심 정보를 추출합니다.
    """
    
    def __init__(self):
        """초기화."""
        self.provenance_tracker = get_provenance_tracker()
        logger.info("MemoryExtractor initialized")
    
    async def extract_memories(
        self,
        conversation_history: List[Dict[str, Any]],
        session_id: str,
        user_id: str,
        existing_memories: Optional[List[BaseMemory]] = None,
        turn: Optional[int] = None
    ) -> List[BaseMemory]:
        """
        대화에서 메모리 추출 (LLM-driven).
        
        Args:
            conversation_history: 대화 히스토리 (messages)
            session_id: 세션 ID
            user_id: 사용자 ID
            existing_memories: 기존 메모리 (중복 방지용)
            turn: 턴 번호
            
        Returns:
            추출된 메모리 목록
        """
        try:
            # 대화 히스토리 포맷팅
            history_text = self._format_conversation_history(conversation_history)
            
            # 기존 메모리 포맷팅
            existing_text = ""
            if existing_memories:
                existing_text = "\n".join([
                    f"- {mem.content} (type: {mem.memory_type.value})"
                    for mem in existing_memories[:10]  # 최근 10개만
                ])
            
            # Extraction 프롬프트 구성
            user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
                conversation_history=history_text,
                existing_memories=existing_text or "None"
            )
            
            # LLM 호출
            result = await execute_llm_task(
                prompt=user_prompt,
                task_type=TaskType.MEMORY_EXTRACTION,
                system_message=EXTRACTION_SYSTEM_PROMPT
            )
            
            # 결과 파싱
            extracted_memories = self._parse_extraction_result(
                result.content,
                user_id=user_id,
                session_id=session_id,
                turn=turn
            )
            
            # Provenance 기록
            for memory in extracted_memories:
                self.provenance_tracker.record_extraction(
                    memory_id=memory.memory_id,
                    source_session_id=session_id,
                    source_turn=turn,
                    metadata={"extraction_model": result.model_used}
                )
            
            logger.info(f"Extracted {len(extracted_memories)} memories from session {session_id}")
            return extracted_memories
            
        except Exception as e:
            logger.error(f"Failed to extract memories: {e}", exc_info=True)
            return []
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """대화 히스토리를 텍스트로 포맷팅."""
        formatted = []
        for i, msg in enumerate(history):
            role = msg.get("role", msg.get("type", "unknown"))
            content = msg.get("content", msg.get("text", ""))
            formatted.append(f"Turn {i+1} [{role}]: {content[:500]}")  # 최대 500자
        return "\n".join(formatted)
    
    def _parse_extraction_result(
        self,
        llm_output: str,
        user_id: str,
        session_id: str,
        turn: Optional[int] = None
    ) -> List[BaseMemory]:
        """LLM 출력을 메모리 객체로 파싱."""
        memories = []
        
        try:
            # JSON 추출 (코드 블록 제거)
            json_text = llm_output.strip()
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()
            
            memory_data_list = json.loads(json_text)
            
            if not isinstance(memory_data_list, list):
                memory_data_list = [memory_data_list]
            
            for mem_data in memory_data_list:
                try:
                    # 메모리 ID 생성
                    memory_id = f"mem_{uuid.uuid4().hex[:12]}"
                    
                    # 공통 필드
                    base_fields = {
                        "memory_id": memory_id,
                        "content": mem_data.get("content", ""),
                        "user_id": user_id,
                        "created_at": datetime.now(),
                        "last_accessed": datetime.now(),
                        "importance": float(mem_data.get("importance", 0.5)),
                        "metadata": {
                            "session_id": session_id,
                            "turn": turn,
                            "extraction_timestamp": datetime.now().isoformat()
                        }
                    }
                    
                    # 타입별 메모리 생성
                    memory_type = MemoryType(mem_data.get("memory_type", "semantic"))
                    
                    if memory_type == MemoryType.SEMANTIC:
                        memory = SemanticMemory(
                            **base_fields,
                            facts=mem_data.get("facts", []),
                            domain=mem_data.get("domain")
                        )
                    elif memory_type == MemoryType.EPISODIC:
                        event_ts = None
                        if mem_data.get("event_timestamp"):
                            try:
                                event_ts = datetime.fromisoformat(mem_data["event_timestamp"])
                            except:
                                pass
                        memory = EpisodicMemory(
                            **base_fields,
                            event_timestamp=event_ts,
                            session_id=session_id,
                            context=mem_data.get("context", {})
                        )
                    elif memory_type == MemoryType.PROCEDURAL:
                        memory = ProceduralMemory(
                            **base_fields,
                            procedure_steps=mem_data.get("procedure_steps", []),
                            tool_name=mem_data.get("tool_name"),
                            success_rate=float(mem_data.get("success_rate", 0.0)),
                            examples=mem_data.get("examples", [])
                        )
                    else:
                        # 기본적으로 Semantic으로 처리
                        memory = SemanticMemory(**base_fields)
                    
                    memories.append(memory)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse memory entry: {e}, data: {mem_data}")
                    continue
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM output as JSON: {e}, output: {llm_output[:500]}")
        except Exception as e:
            logger.error(f"Failed to parse extraction result: {e}", exc_info=True)
        
        return memories


class MemoryConsolidator:
    """
    LLM-driven Memory Consolidation 시스템.
    
    백서 요구사항: 기존 메모리와 통합, 충돌 해결, 중복 제거.
    """
    
    def __init__(self):
        """초기화."""
        self.provenance_tracker = get_provenance_tracker()
        logger.info("MemoryConsolidator initialized")
    
    async def consolidate_memories(
        self,
        new_memories: List[BaseMemory],
        existing_memories: List[BaseMemory],
        user_id: str
    ) -> Dict[str, Any]:
        """
        메모리 통합 (LLM-driven).
        
        Args:
            new_memories: 새로 추출된 메모리
            existing_memories: 기존 메모리
            user_id: 사용자 ID
            
        Returns:
            {
                "consolidated": List[BaseMemory],  # 통합된 메모리
                "deleted_ids": List[str],  # 삭제된 메모리 ID
                "conflicts": List[Dict]  # 충돌 해결 정보
            }
        """
        try:
            if not new_memories:
                return {
                    "consolidated": existing_memories,
                    "deleted_ids": [],
                    "conflicts": []
                }
            
            # 메모리를 JSON으로 변환
            new_memories_json = [mem.to_dict() for mem in new_memories]
            existing_memories_json = [mem.to_dict() for mem in existing_memories]
            
            # Consolidation 프롬프트 구성
            user_prompt = CONSOLIDATION_USER_PROMPT_TEMPLATE.format(
                new_memories=json.dumps(new_memories_json, ensure_ascii=False, indent=2),
                existing_memories=json.dumps(existing_memories_json, ensure_ascii=False, indent=2)
            )
            
            # LLM 호출
            result = await execute_llm_task(
                prompt=user_prompt,
                task_type=TaskType.MEMORY_CONSOLIDATION,
                system_message=CONSOLIDATION_SYSTEM_PROMPT
            )
            
            # 결과 파싱
            consolidation_result = self._parse_consolidation_result(
                result.content,
                user_id=user_id
            )
            
            # Provenance 기록
            for consolidated_mem in consolidation_result["consolidated"]:
                # 원본 메모리 ID 찾기
                source_ids = self._find_source_memory_ids(
                    consolidated_mem,
                    new_memories + existing_memories
                )
                
                if len(source_ids) > 1:
                    # 통합 이벤트 기록
                    self.provenance_tracker.record_consolidation(
                        memory_id=consolidated_mem.memory_id,
                        source_memory_ids=source_ids,
                        metadata={"consolidation_model": result.model_used}
                    )
            
            logger.info(f"Consolidated {len(new_memories)} new memories with {len(existing_memories)} existing")
            return consolidation_result
            
        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}", exc_info=True)
            # 실패 시 새 메모리만 반환
            return {
                "consolidated": existing_memories + new_memories,
                "deleted_ids": [],
                "conflicts": []
            }
    
    def _parse_consolidation_result(
        self,
        llm_output: str,
        user_id: str
    ) -> Dict[str, Any]:
        """LLM 출력을 통합 결과로 파싱."""
        try:
            # JSON 추출
            json_text = llm_output.strip()
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()
            
            result_data = json.loads(json_text)
            
            # 통합된 메모리 파싱
            consolidated = []
            for mem_data in result_data.get("consolidated_memories", []):
                try:
                    # 메모리 ID가 없으면 새로 생성
                    if "memory_id" not in mem_data:
                        mem_data["memory_id"] = f"mem_{uuid.uuid4().hex[:12]}"
                    
                    # created_at이 없으면 현재 시간
                    if "created_at" not in mem_data:
                        mem_data["created_at"] = datetime.now().isoformat()
                    
                    memory = create_memory_from_dict(mem_data)
                    consolidated.append(memory)
                except Exception as e:
                    logger.warning(f"Failed to parse consolidated memory: {e}")
                    continue
            
            return {
                "consolidated": consolidated,
                "deleted_ids": result_data.get("deleted_memory_ids", []),
                "conflicts": result_data.get("conflicts_resolved", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to parse consolidation result: {e}, output: {llm_output[:500]}")
            return {
                "consolidated": [],
                "deleted_ids": [],
                "conflicts": []
            }
    
    def _find_source_memory_ids(
        self,
        consolidated_memory: BaseMemory,
        all_memories: List[BaseMemory]
    ) -> List[str]:
        """통합된 메모리의 원본 메모리 ID 찾기."""
        source_ids = []
        
        # 내용 유사도 기반으로 원본 찾기 (간단한 휴리스틱)
        consolidated_content = consolidated_memory.content.lower()
        
        for mem in all_memories:
            if mem.memory_id == consolidated_memory.memory_id:
                continue
            
            mem_content = mem.content.lower()
            # 내용이 유사하면 원본으로 간주
            if (consolidated_content in mem_content or 
                mem_content in consolidated_content or
                self._similarity_score(consolidated_content, mem_content) > 0.7):
                source_ids.append(mem.memory_id)
        
        return source_ids
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """간단한 유사도 점수 (0.0 ~ 1.0)."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0


# 전역 인스턴스
_memory_extractor: Optional[MemoryExtractor] = None
_memory_consolidator: Optional[MemoryConsolidator] = None


def get_memory_extractor() -> MemoryExtractor:
    """전역 메모리 추출기 인스턴스 반환."""
    global _memory_extractor
    if _memory_extractor is None:
        _memory_extractor = MemoryExtractor()
    return _memory_extractor


def get_memory_consolidator() -> MemoryConsolidator:
    """전역 메모리 통합기 인스턴스 반환."""
    global _memory_consolidator
    if _memory_consolidator is None:
        _memory_consolidator = MemoryConsolidator()
    return _memory_consolidator

