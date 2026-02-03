"""
Recursive Context Manager

RLM의 재귀적 컨텍스트 관리 아이디어를 sparkleforge 연구 워크플로우에 통합.
각 연구 단계에서 context를 재귀적으로 확장하고 활용하여 완전 자동형 처리를 수행.

ROMA의 ExecutionContext 아이디어를 참고하여 contextvars 기반 컨텍스트 전파 추가.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# ContextVar for execution-scoped context propagation (ROMA-style)
_execution_context: ContextVar[Optional['ExecutionContext']] = ContextVar(
    "execution_context", default=None
)


@dataclass
class RecursiveContext:
    """재귀적 컨텍스트 데이터 구조."""
    context_id: str
    depth: int
    parent_context_id: Optional[str]
    context_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "context_id": self.context_id,
            "depth": self.depth,
            "parent_context_id": self.parent_context_id,
            "context_data": self.context_data,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecursiveContext':
        """딕셔너리에서 생성."""
        return cls(
            context_id=data["context_id"],
            depth=data["depth"],
            parent_context_id=data.get("parent_context_id"),
            context_data=data["context_data"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now()),
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else data.get("updated_at", datetime.now())
        )


class ExecutionContext:
    """
    실행 컨텍스트 (ROMA 스타일, contextvars 기반).
    
    실행별 고유 컨텍스트를 제공하며, thread-safe, async-safe하게 전파됩니다.
    명시적 파라미터 전달 없이 현재 실행 컨텍스트에 접근할 수 있습니다.
    """
    
    def __init__(self, execution_id: str, context_manager: 'RecursiveContextManager'):
        """
        초기화.
        
        Args:
            execution_id: 실행 고유 ID
            context_manager: RecursiveContextManager 인스턴스
        """
        self.execution_id = execution_id
        self.context_manager = context_manager
        self.created_at = datetime.now()
    
    @classmethod
    def set(cls, execution_id: str, context_manager: 'RecursiveContextManager') -> 'ContextVar.Token':
        """
        실행 컨텍스트 설정.
        
        Args:
            execution_id: 실행 고유 ID
            context_manager: RecursiveContextManager 인스턴스
        
        Returns:
            ContextVar 토큰 (reset에 사용)
        """
        ctx = cls(execution_id, context_manager)
        token = _execution_context.set(ctx)
        logger.debug(f"ExecutionContext set for execution: {execution_id}")
        return token
    
    @classmethod
    def get(cls) -> Optional['ExecutionContext']:
        """
        현재 실행 컨텍스트 반환.
        
        Returns:
            ExecutionContext 인스턴스 또는 None
        """
        return _execution_context.get()
    
    @classmethod
    def reset(cls, token: 'ContextVar.Token'):
        """
        실행 컨텍스트 리셋.
        
        Args:
            token: set()에서 반환된 토큰
        """
        _execution_context.reset(token)
        logger.debug("ExecutionContext reset")


class RecursiveContextManager:
    """
    재귀적 컨텍스트 관리 시스템.
    
    각 연구 단계에서 context를 재귀적으로 확장하고 활용하여
    완전 자동형 처리를 수행합니다.
    
    ExecutionContext와 통합하여 contextvars 기반 컨텍스트 전파를 지원합니다.
    """
    
    def __init__(self, max_depth: int = 10):
        """
        초기화.
        
        Args:
            max_depth: 최대 컨텍스트 깊이
        """
        self.context_stack: deque = deque()
        self.context_cache: Dict[str, RecursiveContext] = {}
        self.max_depth = max_depth
        self.current_context_id: Optional[str] = None
        
        logger.info(f"RecursiveContextManager initialized (max_depth: {max_depth})")
    
    def get_current_execution_context(self) -> Optional[ExecutionContext]:
        """
        현재 실행 컨텍스트 반환 (contextvars 기반).
        
        Returns:
            ExecutionContext 인스턴스 또는 None
        """
        return ExecutionContext.get()
    
    def push_context(
        self,
        context_data: Dict[str, Any],
        depth: int = 0,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        컨텍스트 스택에 추가.
        
        Args:
            context_data: 컨텍스트 데이터
            depth: 컨텍스트 깊이
            parent_id: 부모 컨텍스트 ID
            metadata: 추가 메타데이터
        
        Returns:
            컨텍스트 ID
        """
        if depth > self.max_depth:
            logger.warning(f"Context depth {depth} exceeds max_depth {self.max_depth}")
            depth = self.max_depth
        
        context_id = f"ctx_{uuid.uuid4().hex[:12]}"
        
        # 부모 컨텍스트 데이터 병합
        merged_data = context_data.copy()
        if parent_id and parent_id in self.context_cache:
            parent_context = self.context_cache[parent_id]
            # 부모 컨텍스트 데이터를 우선순위로 병합
            merged_data = {**parent_context.context_data, **merged_data}
        
        context = RecursiveContext(
            context_id=context_id,
            depth=depth,
            parent_context_id=parent_id,
            context_data=merged_data,
            metadata=metadata or {}
        )
        
        self.context_stack.append(context)
        self.context_cache[context_id] = context
        self.current_context_id = context_id
        
        logger.debug(f"Context pushed: {context_id} (depth: {depth})")
        return context_id
    
    def pop_context(self) -> Optional[RecursiveContext]:
        """
        컨텍스트 스택에서 제거.
        
        Returns:
            제거된 컨텍스트 또는 None
        """
        if not self.context_stack:
            return None
        
        context = self.context_stack.pop()
        
        # 현재 컨텍스트 업데이트
        if self.context_stack:
            self.current_context_id = self.context_stack[-1].context_id
        else:
            self.current_context_id = None
        
        logger.debug(f"Context popped: {context.context_id}")
        return context
    
    def get_current_context(self) -> Optional[RecursiveContext]:
        """
        현재 컨텍스트 반환.
        
        Returns:
            현재 컨텍스트 또는 None
        """
        if self.current_context_id and self.current_context_id in self.context_cache:
            return self.context_cache[self.current_context_id]
        
        if self.context_stack:
            return self.context_stack[-1]
        
        return None
    
    def get_context(self, context_id: str) -> Optional[RecursiveContext]:
        """
        특정 컨텍스트 반환.
        
        Args:
            context_id: 컨텍스트 ID
        
        Returns:
            컨텍스트 또는 None
        """
        return self.context_cache.get(context_id)
    
    def extend_context(
        self,
        context_id: str,
        additional_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[RecursiveContext]:
        """
        컨텍스트 확장.
        
        Args:
            context_id: 확장할 컨텍스트 ID
            additional_data: 추가할 데이터
            metadata: 추가 메타데이터
        
        Returns:
            확장된 컨텍스트 또는 None
        """
        if context_id not in self.context_cache:
            logger.warning(f"Context not found: {context_id}")
            return None
        
        context = self.context_cache[context_id]
        
        # 데이터 병합 (새 데이터가 우선)
        context.context_data = {**context.context_data, **additional_data}
        
        # 메타데이터 업데이트
        if metadata:
            context.metadata = {**context.metadata, **metadata}
        
        context.updated_at = datetime.now()
        
        logger.debug(f"Context extended: {context_id}")
        return context
    
    def merge_contexts(
        self,
        context_ids: List[str],
        merge_strategy: str = "priority"  # "priority", "union", "intersection"
    ) -> Optional[RecursiveContext]:
        """
        여러 컨텍스트 병합.
        
        Args:
            context_ids: 병합할 컨텍스트 ID 목록
            merge_strategy: 병합 전략 ("priority", "union", "intersection")
        
        Returns:
            병합된 컨텍스트 또는 None
        """
        if not context_ids:
            return None
        
        # 모든 컨텍스트 가져오기
        contexts = []
        for ctx_id in context_ids:
            if ctx_id in self.context_cache:
                contexts.append(self.context_cache[ctx_id])
            else:
                logger.warning(f"Context not found for merge: {ctx_id}")
        
        if not contexts:
            return None
        
        # 병합 전략에 따라 데이터 병합
        merged_data = {}
        merged_metadata = {}
        
        if merge_strategy == "priority":
            # 나중에 오는 컨텍스트가 우선순위 높음
            for context in contexts:
                merged_data = {**merged_data, **context.context_data}
                merged_metadata = {**merged_metadata, **context.metadata}
        elif merge_strategy == "union":
            # 모든 키의 합집합
            for context in contexts:
                merged_data.update(context.context_data)
                merged_metadata.update(context.metadata)
        elif merge_strategy == "intersection":
            # 공통 키만 유지
            if contexts:
                common_keys = set(contexts[0].context_data.keys())
                for context in contexts[1:]:
                    common_keys &= set(context.context_data.keys())
                
                for key in common_keys:
                    # 마지막 컨텍스트의 값 사용
                    merged_data[key] = contexts[-1].context_data[key]
        
        # 최대 깊이 계산
        max_depth = max(ctx.depth for ctx in contexts)
        
        # 병합된 컨텍스트 생성
        merged_context_id = f"ctx_merged_{uuid.uuid4().hex[:12]}"
        merged_context = RecursiveContext(
            context_id=merged_context_id,
            depth=max_depth,
            parent_context_id=None,  # 병합된 컨텍스트는 부모 없음
            context_data=merged_data,
            metadata={
                "merge_strategy": merge_strategy,
                "source_contexts": context_ids,
                "merged_at": datetime.now().isoformat()
            }
        )
        
        self.context_cache[merged_context_id] = merged_context
        logger.debug(f"Contexts merged: {len(context_ids)} contexts -> {merged_context_id}")
        
        return merged_context
    
    def get_context_for_depth(self, depth: int) -> Optional[RecursiveContext]:
        """
        특정 깊이의 컨텍스트 반환.
        
        Args:
            depth: 컨텍스트 깊이
        
        Returns:
            해당 깊이의 컨텍스트 또는 None
        """
        for context in reversed(self.context_stack):
            if context.depth == depth:
                return context
        
        # 캐시에서 검색
        for context in self.context_cache.values():
            if context.depth == depth:
                return context
        
        return None
    
    def get_context_chain(self, context_id: str) -> List[RecursiveContext]:
        """
        컨텍스트 체인 반환 (부모부터 루트까지).
        
        Args:
            context_id: 시작 컨텍스트 ID
        
        Returns:
            컨텍스트 체인 (부모부터 루트까지)
        """
        chain = []
        current_id = context_id
        
        while current_id:
            if current_id in self.context_cache:
                context = self.context_cache[current_id]
                chain.append(context)
                current_id = context.parent_context_id
            else:
                break
        
        return chain
    
    def clear_context(self, context_id: Optional[str] = None):
        """
        컨텍스트 제거.
        
        Args:
            context_id: 제거할 컨텍스트 ID (None이면 현재 컨텍스트)
        """
        if context_id is None:
            context_id = self.current_context_id
        
        if context_id and context_id in self.context_cache:
            del self.context_cache[context_id]
            
            # 스택에서도 제거
            self.context_stack = deque(
                ctx for ctx in self.context_stack if ctx.context_id != context_id
            )
            
            # 현재 컨텍스트 업데이트
            if self.current_context_id == context_id:
                if self.context_stack:
                    self.current_context_id = self.context_stack[-1].context_id
                else:
                    self.current_context_id = None
            
            logger.debug(f"Context cleared: {context_id}")
    
    def clear_all(self):
        """모든 컨텍스트 제거."""
        self.context_stack.clear()
        self.context_cache.clear()
        self.current_context_id = None
        logger.debug("All contexts cleared")
    
    def get_context_stats(self) -> Dict[str, Any]:
        """컨텍스트 통계 반환."""
        return {
            "stack_size": len(self.context_stack),
            "cache_size": len(self.context_cache),
            "current_context_id": self.current_context_id,
            "max_depth": self.max_depth,
            "depths": [ctx.depth for ctx in self.context_cache.values()]
        }
    
    def evaluate_context_completeness(self, context_id: Optional[str] = None) -> float:
        """
        컨텍스트 완전성 평가.
        
        Args:
            context_id: 평가할 컨텍스트 ID (None이면 현재 컨텍스트)
        
        Returns:
            완전성 점수 (0.0 ~ 1.0)
        """
        if context_id is None:
            context_id = self.current_context_id
        
        if not context_id or context_id not in self.context_cache:
            return 0.0
        
        context = self.context_cache[context_id]
        
        # 필수 필드 체크
        required_fields = [
            "user_request",
            "intent_analysis",
            "planned_tasks",
            "execution_results"
        ]
        
        present_fields = sum(1 for field in required_fields if field in context.context_data)
        completeness = present_fields / len(required_fields) if required_fields else 1.0
        
        # 데이터 품질 평가
        data_quality = 1.0
        for key, value in context.context_data.items():
            if value is None or (isinstance(value, (list, dict)) and len(value) == 0):
                data_quality *= 0.9  # 빈 값이 있으면 약간 감점
        
        final_score = completeness * 0.7 + data_quality * 0.3
        
        return min(1.0, max(0.0, final_score))


# 전역 인스턴스
_context_manager: Optional[RecursiveContextManager] = None


def get_recursive_context_manager() -> RecursiveContextManager:
    """전역 RecursiveContextManager 인스턴스 반환."""
    global _context_manager
    if _context_manager is None:
        _context_manager = RecursiveContextManager()
    return _context_manager

