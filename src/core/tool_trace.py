#!/usr/bin/env python3
"""
ToolTrace 추적 시스템 (9대 혁신)

모든 도구 호출을 추적하여 디버깅, 분석, 리포트 생성 지원.
MCP 도구 통합 추적 + Shared Memory 연동 + 실시간 스트리밍 지원.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Default max size for raw_answer (50KB)
DEFAULT_RAW_ANSWER_MAX_SIZE = 50 * 1024


@dataclass
class ToolTrace:
    """
    도구 호출 추적 정보.
    
    도구 호출의 완전한 루프를 기록합니다.
    """
    tool_id: str  # 고유 식별자 (e.g., "tool_1", "tool_2")
    citation_id: str  # Citation ID (e.g., "CIT-1-01", "PLAN-01")
    tool_type: str  # 도구 타입 (rag_naive, web_search, mcp_tool 등)
    query: str  # 발행된 쿼리
    raw_answer: str  # 도구가 반환한 원시 상세 결과 (50KB 제한)
    summary: str  # Note Agent가 생성한 핵심 요약
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_answer_truncated: bool = False  # raw_answer가 잘렸는지 여부
    raw_answer_original_size: int = 0  # 잘리기 전 원본 크기
    
    # MCP 통합 필드
    mcp_server: Optional[str] = None  # MCP 서버 이름 (MCP 도구인 경우)
    mcp_tool_name: Optional[str] = None  # MCP 도구 이름 (MCP 도구인 경우)
    
    # Shared Memory 연동 필드
    shared_memory_key: Optional[str] = None  # Shared Memory에 저장된 키
    
    # 실시간 스트리밍 필드
    streamed: bool = False  # 스트리밍되었는지 여부
    
    def __post_init__(self):
        """초기화 후 처리: raw_answer 크기 제한."""
        if self.raw_answer_original_size == 0:
            self.raw_answer_original_size = len(self.raw_answer)
        
        # 크기 제한 적용
        if len(self.raw_answer) > DEFAULT_RAW_ANSWER_MAX_SIZE:
            self.raw_answer = self._truncate_raw_answer(self.raw_answer, DEFAULT_RAW_ANSWER_MAX_SIZE)
            self.raw_answer_truncated = True
    
    @staticmethod
    def _truncate_raw_answer(raw_answer: str, max_size: int) -> str:
        """
        raw_answer를 스마트하게 잘라냅니다 (유효한 JSON 구조 보존 시도).
        
        Args:
            raw_answer: 원본 raw answer 문자열
            max_size: 최대 크기 (바이트)
        
        Returns:
            잘린 문자열
        """
        if len(raw_answer) <= max_size:
            return raw_answer
        
        # JSON으로 파싱 시도하고 지능적으로 잘라내기
        try:
            data = json.loads(raw_answer)
            
            # dict인 경우 일반적인 RAG 응답 필드를 잘라냄
            if isinstance(data, dict):
                content_fields = ["answer", "content", "text", "chunks", "documents", "results"]
                for field_name in content_fields:
                    if field_name in data:
                        if isinstance(data[field_name], str) and len(data[field_name]) > max_size // 2:
                            data[field_name] = data[field_name][:max_size // 2] + "... [truncated]"
                        elif isinstance(data[field_name], list):
                            # 처음 몇 개 항목만 유지
                            data[field_name] = data[field_name][:3]
                            if data[field_name]:
                                data[field_name].append({"note": "... additional items truncated"})
                
                truncated = json.dumps(data, ensure_ascii=False)
                if len(truncated) <= max_size:
                    return truncated
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Fallback: 간단한 잘라내기
        truncation_marker = f"\n... [content truncated, original size: {len(raw_answer)} bytes]"
        return raw_answer[:max_size - len(truncation_marker)] + truncation_marker
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolTrace":
        """딕셔너리에서 생성."""
        # 하위 호환성: old data에는 새 필드가 없을 수 있음
        defaults = {
            "raw_answer_truncated": False,
            "raw_answer_original_size": len(data.get("raw_answer", "")),
            "mcp_server": None,
            "mcp_tool_name": None,
            "shared_memory_key": None,
            "streamed": False,
        }
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
        return cls(**data)
    
    @classmethod
    def create_with_size_limit(
        cls,
        tool_id: str,
        citation_id: str,
        tool_type: str,
        query: str,
        raw_answer: str,
        summary: str,
        max_size: int = DEFAULT_RAW_ANSWER_MAX_SIZE,
        mcp_server: Optional[str] = None,
        mcp_tool_name: Optional[str] = None,
    ) -> "ToolTrace":
        """
        명시적 크기 제한으로 ToolTrace 생성.
        
        Args:
            tool_id: Tool ID
            citation_id: Citation ID
            tool_type: Tool type
            query: Query string
            raw_answer: Raw answer (필요시 잘림)
            summary: Summary
            max_size: raw_answer의 최대 크기
            mcp_server: MCP 서버 이름 (optional)
            mcp_tool_name: MCP 도구 이름 (optional)
        
        Returns:
            ToolTrace 인스턴스
        """
        original_size = len(raw_answer)
        truncated = len(raw_answer) > max_size
        
        if truncated:
            raw_answer = cls._truncate_raw_answer(raw_answer, max_size)
        
        return cls(
            tool_id=tool_id,
            citation_id=citation_id,
            tool_type=tool_type,
            query=query,
            raw_answer=raw_answer,
            summary=summary,
            raw_answer_truncated=truncated,
            raw_answer_original_size=original_size,
            mcp_server=mcp_server,
            mcp_tool_name=mcp_tool_name,
        )


class ToolTraceManager:
    """
    ToolTrace 관리자.
    
    ToolTrace를 수집, 저장, 조회하는 기능을 제공합니다.
    """
    
    def __init__(self, research_id: Optional[str] = None):
        """
        초기화.
        
        Args:
            research_id: 연구 작업 ID (optional)
        """
        self.research_id = research_id or f"research_{int(datetime.now().timestamp())}"
        self.traces: List[ToolTrace] = []
        self.trace_by_id: Dict[str, ToolTrace] = {}  # tool_id -> ToolTrace
        self.trace_by_citation: Dict[str, List[ToolTrace]] = {}  # citation_id -> List[ToolTrace]
    
    def add_trace(self, trace: ToolTrace) -> None:
        """
        ToolTrace 추가.
        
        Args:
            trace: ToolTrace 객체
        """
        self.traces.append(trace)
        self.trace_by_id[trace.tool_id] = trace
        
        if trace.citation_id not in self.trace_by_citation:
            self.trace_by_citation[trace.citation_id] = []
        self.trace_by_citation[trace.citation_id].append(trace)
        
        logger.debug(f"ToolTrace added: {trace.tool_id} ({trace.tool_type})")
    
    def get_trace(self, tool_id: str) -> Optional[ToolTrace]:
        """
        Tool ID로 ToolTrace 조회.
        
        Args:
            tool_id: Tool ID
        
        Returns:
            ToolTrace 객체, 없으면 None
        """
        return self.trace_by_id.get(tool_id)
    
    def get_traces_by_citation(self, citation_id: str) -> List[ToolTrace]:
        """
        Citation ID로 ToolTrace 목록 조회.
        
        Args:
            citation_id: Citation ID
        
        Returns:
            ToolTrace 목록
        """
        return self.trace_by_citation.get(citation_id, [])
    
    def get_all_traces(self) -> List[ToolTrace]:
        """모든 ToolTrace 반환."""
        return self.traces.copy()
    
    def get_traces_by_type(self, tool_type: str) -> List[ToolTrace]:
        """
        도구 타입으로 ToolTrace 목록 조회.
        
        Args:
            tool_type: 도구 타입
        
        Returns:
            ToolTrace 목록
        """
        return [t for t in self.traces if t.tool_type == tool_type]
    
    def get_statistics(self) -> Dict[str, Any]:
        """ToolTrace 통계 반환."""
        tool_type_counts = {}
        for trace in self.traces:
            tool_type_counts[trace.tool_type] = tool_type_counts.get(trace.tool_type, 0) + 1
        
        total_size = sum(t.raw_answer_original_size for t in self.traces)
        truncated_count = sum(1 for t in self.traces if t.raw_answer_truncated)
        
        return {
            "total_traces": len(self.traces),
            "tool_type_counts": tool_type_counts,
            "total_size_bytes": total_size,
            "truncated_count": truncated_count,
            "unique_citations": len(self.trace_by_citation),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "research_id": self.research_id,
            "traces": [t.to_dict() for t in self.traces],
            "statistics": self.get_statistics(),
        }
    
    def save_to_json(self, filepath: str) -> None:
        """JSON 파일로 저장."""
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> "ToolTraceManager":
        """JSON 파일에서 로드."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        
        manager = cls(research_id=data.get("research_id"))
        for trace_data in data.get("traces", []):
            trace = ToolTrace.from_dict(trace_data)
            manager.add_trace(trace)
        
        return manager

