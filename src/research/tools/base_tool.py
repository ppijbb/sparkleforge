"""
Base Research Tool (v2.0 - 8대 혁신 통합)

Universal MCP Hub, Production-Grade Reliability, Multi-Model Orchestration을
통합한 고도화된 연구 도구 베이스 클래스.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.core.researcher_config import get_mcp_config, get_reliability_config
from src.core.reliability import execute_with_reliability
from src.core.mcp_integration import execute_tool, get_best_tool_for_task, ToolCategory

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """도구 유형."""
    SEARCH = "search"
    DATA = "data"
    CODE = "code"
    ACADEMIC = "academic"
    BUSINESS = "business"
    UTILITY = "utility"


@dataclass
class ToolResult:
    """도구 실행 결과."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    confidence: float = 0.0
    
    def get(self, key: str, default: Any = None) -> Any:
        """딕셔너리처럼 접근 가능."""
        if isinstance(self.data, dict):
            return self.data.get(key, default)
        return default


class ToolStatus(Enum):
    """도구 상태."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class SearchResult:
    """검색 결과 데이터 구조 (8대 혁신 통합)."""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[datetime] = None
    relevance_score: float = 0.0
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 8대 혁신 관련 필드
    mcp_tool_used: Optional[str] = None
    execution_time: float = 0.0
    compression_applied: bool = False
    verification_score: float = 0.0
    
    def __post_init__(self):
        """Initialize default values for dataclass fields."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ToolPerformance:
    """도구 성능 메트릭."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None
    error_count: int = 0
    circuit_breaker_status: bool = False


class BaseResearchTool(ABC):
    """8대 혁신을 통합한 고도화된 연구 도구 베이스 클래스."""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화."""
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.enabled = config.get('enabled', True)
        self.timeout = config.get('timeout', 30)
        self.max_results = config.get('max_results', 10)
        self.tool_type = ToolType(config.get('tool_type', 'search'))
        
        # 8대 혁신 통합
        self.mcp_config = get_mcp_config()
        self.reliability_config = get_reliability_config()
        
        # 성능 추적
        self.performance = ToolPerformance()
        self.status = ToolStatus.ACTIVE
        
        # MCP 도구 매핑
        self.mcp_tool_mapping = self._initialize_mcp_mapping()
        
        logger.info(f"Initialized research tool: {self.name} with 8 core innovations")
    
    def _initialize_mcp_mapping(self) -> Dict[str, str]:
        """MCP 도구 매핑 초기화."""
        return {
            'web_search': 'g-search',
            'academic_search': 'arxiv',
            'data_fetch': 'fetch',
            'file_operation': 'filesystem',
            'code_execution': 'python_coder',
            'code_analysis': 'code_interpreter'
        }
    
    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """검색 작업 수행 (Universal MCP Hub 통합)."""
        pass
    
    @abstractmethod
    async def get_content(self, url: str) -> Optional[str]:
        """URL에서 콘텐츠 가져오기."""
        pass
    
    async def execute_with_mcp(
        self,
        operation: str,
        parameters: Dict[str, Any],
        mcp_tool: str = None,
        alternative_tools: List[str] = None
    ) -> Dict[str, Any]:
        """MCP 전용 실행 (대체 도구 지원)."""
        if not self.mcp_config.enabled:
            raise RuntimeError("MCP is disabled. This tool requires MCP to function.")
        
        if not mcp_tool:
            mcp_tool = self.mcp_tool_mapping.get(operation, operation)
        
        # 기본 도구 시도
        tools_to_try = [mcp_tool]
        if alternative_tools:
            tools_to_try.extend(alternative_tools)
        
        last_error = None
        
        for tool in tools_to_try:
            try:
                result = await execute_tool(tool, parameters)
                if result.success:
                    self.performance.successful_requests += 1
                    self.performance.last_used = datetime.now()
                    return {
                        'success': True,
                        'data': result.data,
                        'source': 'mcp',
                        'tool_used': tool,
                        'execution_time': result.execution_time,
                        'confidence': result.confidence
                    }
                else:
                    logger.warning(f"MCP tool {tool} failed: {result.error}")
                    last_error = result.error
                    self.performance.failed_requests += 1
            except Exception as e:
                logger.warning(f"MCP tool {tool} error: {e}")
                last_error = str(e)
                self.performance.failed_requests += 1
        
        # 모든 도구 실패
        self.performance.failed_requests += 1
        raise RuntimeError(f"All MCP tools failed for operation {operation}. Last error: {last_error}")
    
    
    async def search_with_reliability(
        self,
        query: str,
        **kwargs
    ) -> List[SearchResult]:
        """Production-Grade Reliability로 검색 실행."""
        return await execute_with_reliability(
            self._execute_search,
            query,
            **kwargs,
            component_name=f"search_tool_{self.name}",
            save_state=True
        )
    
    async def _execute_search(self, query: str, **kwargs) -> List[SearchResult]:
        """검색 실행 (내부 메서드)."""
        start_time = datetime.now()
        
        try:
            # 쿼리 검증
            if not await self.validate_query(query):
                raise ValueError(f"Invalid query: {query}")
            
            # 검색 실행
            results = await self.search(query, **kwargs)
            
            # 성능 업데이트
            execution_time = (datetime.now() - start_time).total_seconds()
            self.performance.total_requests += 1
            self.performance.successful_requests += 1
            self.performance.average_response_time = (
                (self.performance.average_response_time * (self.performance.total_requests - 1) + execution_time) /
                self.performance.total_requests
            )
            self.performance.success_rate = self.performance.successful_requests / self.performance.total_requests
            
            return results
            
        except Exception as e:
            # 실패 처리
            self.performance.total_requests += 1
            self.performance.failed_requests += 1
            self.performance.error_count += 1
            self.performance.success_rate = self.performance.successful_requests / self.performance.total_requests
            
            logger.error(f"Search failed for {self.name}: {e}")
            raise
    
    def is_enabled(self) -> bool:
        """도구 활성화 상태 확인."""
        return self.enabled and self.status == ToolStatus.ACTIVE
    
    def get_config(self) -> Dict[str, Any]:
        """도구 설정 반환."""
        return self.config
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환."""
        return {
            'name': self.name,
            'tool_type': self.tool_type.value,
            'status': self.status.value,
            'total_requests': self.performance.total_requests,
            'successful_requests': self.performance.successful_requests,
            'failed_requests': self.performance.failed_requests,
            'success_rate': self.performance.success_rate,
            'average_response_time': self.performance.average_response_time,
            'error_count': self.performance.error_count,
            'circuit_breaker_status': self.performance.circuit_breaker_status,
            'last_used': self.performance.last_used.isoformat() if self.performance.last_used else None
        }
    
    async def validate_query(self, query: str) -> bool:
        """검색 쿼리 검증."""
        if not query or not query.strip():
            return False
        
        if len(query.strip()) < 2:
            return False
        
        # 추가 검증 로직
        if len(query) > 1000:  # 너무 긴 쿼리
            return False
        
        # 금지된 문자 확인
        forbidden_chars = ['<', '>', '"', "'", '&', '|', ';', '`']
        if any(char in query for char in forbidden_chars):
            return False
        
        return True
    
    def _calculate_relevance_score(self, query: str, title: str, snippet: str) -> float:
        """관련성 점수 계산 (개선된 알고리즘)."""
        query_words = set(query.lower().split())
        title_words = set(title.lower().split())
        snippet_words = set(snippet.lower().split())
        
        # 단어 겹침 계산
        title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
        snippet_overlap = len(query_words.intersection(snippet_words)) / len(query_words) if query_words else 0
        
        # 제목을 스니펫보다 더 중요하게 가중치 적용
        score = (title_overlap * 0.7) + (snippet_overlap * 0.3)
        
        # 정확한 매치 보너스
        if query.lower() in title.lower():
            score += 0.2
        if query.lower() in snippet.lower():
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_confidence_score(self, result: Dict[str, Any]) -> float:
        """신뢰도 점수 계산."""
        confidence = 0.8  # 기본 신뢰도
        
        # 소스 신뢰도
        source = result.get('source', '').lower()
        if 'academic' in source or 'scholar' in source:
            confidence += 0.1
        elif 'news' in source or 'blog' in source:
            confidence -= 0.1
        
        # 메타데이터 품질
        metadata = result.get('metadata', {})
        if metadata.get('peer_reviewed'):
            confidence += 0.1
        if metadata.get('citations', 0) > 10:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    async def _make_request(
        self,
        url: str,
        headers: Dict[str, str] = None,
        method: str = 'GET',
        data: Dict[str, Any] = None
    ) -> Optional[Dict]:
        """HTTP 요청 실행 (MCP 도구 사용)."""
        try:
            # Use MCP tools for HTTP requests instead of direct aiohttp
            from src.core.mcp_integration import execute_tool
            
            result = await execute_tool("fetch", {
                "url": url,
                "method": method,
                "headers": headers or {},
                "data": data or {}
            })
            
            if result.get('success', False):
                return result.get('data', {})
            else:
                logger.error(f"MCP fetch failed for {url}: {result.get('error', 'Unknown error')}")
                return None
                        
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def update_status(self, status: ToolStatus):
        """도구 상태 업데이트."""
        self.status = status
        logger.info(f"Tool {self.name} status updated to {status.value}")
    
    def reset_performance(self):
        """성능 통계 리셋."""
        self.performance = ToolPerformance()
        logger.info(f"Performance stats reset for {self.name}")
    
    def __str__(self) -> str:
        """Return string representation of the tool."""
        return f"{self.__class__.__name__}(name={self.name}, type={self.tool_type.value}, status={self.status.value})"
    
    def __repr__(self) -> str:
        """Return detailed string representation of the tool."""
        return self.__str__()