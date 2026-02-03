"""
MCP Manager Agent - Dynamic Tool Discovery and Management

DeepResearchAgent 영감을 받은 동적 MCP 도구 관리 에이전트.
런타임에 도구를 동적으로 발견, 등록, 관리하고 태스크에 맞는 도구를 추천.

핵심 특징:
- Dynamic tool discovery from MCP servers
- Runtime server addition/removal
- Task-based tool recommendation
- Automatic reconnection and failover
- Tool health monitoring
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MCPServerStatus(Enum):
    """MCP 서버 상태."""
    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    RECONNECTING = "reconnecting"


class ToolCategory(Enum):
    """도구 카테고리."""
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
    API = "api"
    MEMORY = "memory"
    OTHER = "other"


class MCPServerInfo(BaseModel):
    """MCP 서버 정보."""
    server_id: str = Field(description="서버 고유 ID")
    name: str = Field(description="서버 이름")
    command: str = Field(description="실행 명령어")
    args: List[str] = Field(default_factory=list, description="명령어 인자")
    env: Dict[str, str] = Field(default_factory=dict, description="환경 변수")
    
    # 상태
    status: MCPServerStatus = Field(default=MCPServerStatus.UNKNOWN, description="연결 상태")
    last_connected: Optional[datetime] = Field(default=None, description="마지막 연결 시간")
    last_error: Optional[str] = Field(default=None, description="마지막 오류")
    
    # 통계
    connection_count: int = Field(default=0, description="연결 횟수")
    failure_count: int = Field(default=0, description="실패 횟수")
    total_tool_calls: int = Field(default=0, description="총 도구 호출 수")
    
    # 메타데이터
    categories: List[ToolCategory] = Field(default_factory=list, description="서버 카테고리")
    priority: int = Field(default=5, ge=1, le=10, description="우선순위 (1=최고)")
    
    class Config:
        arbitrary_types_allowed = True


class ToolInfo(BaseModel):
    """도구 정보."""
    tool_id: str = Field(description="도구 고유 ID")
    name: str = Field(description="도구 이름")
    description: str = Field(default="", description="도구 설명")
    server_id: str = Field(description="소속 서버 ID")
    
    # 스키마
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="입력 스키마")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="출력 스키마")
    
    # 카테고리 및 태그
    category: ToolCategory = Field(default=ToolCategory.OTHER, description="카테고리")
    tags: List[str] = Field(default_factory=list, description="태그")
    keywords: List[str] = Field(default_factory=list, description="검색 키워드")
    
    # 사용 통계
    call_count: int = Field(default=0, description="호출 횟수")
    success_count: int = Field(default=0, description="성공 횟수")
    avg_response_time: float = Field(default=0.0, description="평균 응답 시간 (초)")
    
    # 상태
    is_available: bool = Field(default=True, description="사용 가능 여부")
    last_used: Optional[datetime] = Field(default=None, description="마지막 사용 시간")
    
    @property
    def success_rate(self) -> float:
        """성공률."""
        if self.call_count == 0:
            return 1.0
        return self.success_count / self.call_count
    
    class Config:
        arbitrary_types_allowed = True


class ToolRecommendation(BaseModel):
    """도구 추천."""
    tool: ToolInfo
    score: float = Field(description="추천 점수")
    reason: str = Field(default="", description="추천 이유")


class MCPManagerAgent:
    """
    MCP Manager Agent - 동적 도구 관리 에이전트.
    
    MCP 서버와 도구를 동적으로 관리하고 태스크에 맞는 도구를 추천.
    """
    
    def __init__(
        self,
        auto_reconnect: bool = True,
        reconnect_interval_seconds: float = 30.0,
        max_reconnect_attempts: int = 5,
        health_check_interval_seconds: float = 60.0
    ):
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval_seconds
        self.max_reconnect_attempts = max_reconnect_attempts
        self.health_check_interval = health_check_interval_seconds
        
        # 서버 및 도구 저장소
        self.servers: Dict[str, MCPServerInfo] = {}
        self.tools: Dict[str, ToolInfo] = {}
        
        # 인덱스
        self.category_index: Dict[ToolCategory, Set[str]] = defaultdict(set)
        self.server_tools: Dict[str, Set[str]] = defaultdict(set)
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 재연결 상태
        self.reconnect_attempts: Dict[str, int] = defaultdict(int)
        
        # 콜백
        self.on_server_connected: Optional[Callable] = None
        self.on_server_disconnected: Optional[Callable] = None
        self.on_tool_discovered: Optional[Callable] = None
        
        # 모니터링 상태
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"MCPManagerAgent initialized: auto_reconnect={auto_reconnect}, "
            f"reconnect_interval={reconnect_interval_seconds}s"
        )
    
    async def register_server(
        self,
        server_id: str,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        categories: Optional[List[ToolCategory]] = None,
        priority: int = 5,
        connect_now: bool = True
    ) -> MCPServerInfo:
        """
        MCP 서버 등록.
        
        Args:
            server_id: 서버 ID
            name: 서버 이름
            command: 실행 명령어
            args: 명령어 인자
            env: 환경 변수
            categories: 서버 카테고리
            priority: 우선순위
            connect_now: 즉시 연결 여부
        """
        server = MCPServerInfo(
            server_id=server_id,
            name=name,
            command=command,
            args=args or [],
            env=env or {},
            categories=categories or [],
            priority=priority
        )
        
        self.servers[server_id] = server
        
        logger.info(f"Registered MCP server: {name} ({server_id})")
        
        if connect_now:
            await self.connect_server(server_id)
        
        return server
    
    async def connect_server(self, server_id: str) -> bool:
        """서버 연결."""
        if server_id not in self.servers:
            logger.error(f"Server {server_id} not registered")
            return False
        
        server = self.servers[server_id]
        server.status = MCPServerStatus.CONNECTING
        
        try:
            # 실제 MCP 연결 (mcp_auto_discovery 통합)
            # 여기서는 시뮬레이션 - 실제 구현에서는 FastMCPMulti 사용
            
            # 도구 발견 시뮬레이션
            discovered_tools = await self._discover_tools(server)
            
            for tool in discovered_tools:
                self._register_tool(tool)
            
            server.status = MCPServerStatus.CONNECTED
            server.last_connected = datetime.now()
            server.connection_count += 1
            self.reconnect_attempts[server_id] = 0
            
            logger.info(
                f"Connected to {server.name}: {len(discovered_tools)} tools discovered"
            )
            
            if self.on_server_connected:
                await self._safe_callback(self.on_server_connected, server)
            
            return True
            
        except Exception as e:
            server.status = MCPServerStatus.FAILED
            server.last_error = str(e)
            server.failure_count += 1
            
            logger.error(f"Failed to connect to {server.name}: {e}")
            return False
    
    async def disconnect_server(self, server_id: str):
        """서버 연결 해제."""
        if server_id not in self.servers:
            return
        
        server = self.servers[server_id]
        
        # 서버의 도구들 비활성화
        for tool_id in self.server_tools.get(server_id, set()):
            if tool_id in self.tools:
                self.tools[tool_id].is_available = False
        
        server.status = MCPServerStatus.DISCONNECTED
        
        logger.info(f"Disconnected from {server.name}")
        
        if self.on_server_disconnected:
            await self._safe_callback(self.on_server_disconnected, server)
    
    async def remove_server(self, server_id: str):
        """서버 제거."""
        await self.disconnect_server(server_id)
        
        # 서버의 도구들 제거
        tool_ids = list(self.server_tools.get(server_id, set()))
        for tool_id in tool_ids:
            self._unregister_tool(tool_id)
        
        self.servers.pop(server_id, None)
        self.server_tools.pop(server_id, None)
        
        logger.info(f"Removed server {server_id}")
    
    async def _discover_tools(self, server: MCPServerInfo) -> List[ToolInfo]:
        """
        서버에서 도구 발견.
        
        실제 구현에서는 MCP 프로토콜을 통해 도구 목록을 가져옴.
        """
        # Placeholder - 실제 구현에서는 MCP 서버와 통신
        # 여기서는 서버 이름 기반으로 카테고리별 샘플 도구 생성
        
        tools = []
        
        for category in server.categories:
            tool = ToolInfo(
                tool_id=f"{server.server_id}_{category.value}_tool",
                name=f"{server.name}_{category.value}",
                description=f"{category.value} tool from {server.name}",
                server_id=server.server_id,
                category=category,
                tags=[category.value],
                keywords=[category.value, server.name.lower()]
            )
            tools.append(tool)
        
        return tools
    
    def _register_tool(self, tool: ToolInfo):
        """도구 등록."""
        self.tools[tool.tool_id] = tool
        self.server_tools[tool.server_id].add(tool.tool_id)
        self.category_index[tool.category].add(tool.tool_id)
        
        # 키워드 인덱스
        for keyword in tool.keywords:
            self.keyword_index[keyword.lower()].add(tool.tool_id)
        
        for tag in tool.tags:
            self.keyword_index[tag.lower()].add(tool.tool_id)
        
        logger.debug(f"Registered tool: {tool.name}")
        
        if self.on_tool_discovered:
            asyncio.create_task(self._safe_callback(self.on_tool_discovered, tool))
    
    def _unregister_tool(self, tool_id: str):
        """도구 등록 해제."""
        if tool_id not in self.tools:
            return
        
        tool = self.tools[tool_id]
        
        # 인덱스에서 제거
        self.category_index[tool.category].discard(tool_id)
        
        for keyword in tool.keywords:
            self.keyword_index[keyword.lower()].discard(tool_id)
        
        del self.tools[tool_id]
    
    def recommend_tools(
        self,
        task_description: str,
        category: Optional[ToolCategory] = None,
        top_k: int = 5,
        only_available: bool = True
    ) -> List[ToolRecommendation]:
        """
        태스크에 맞는 도구 추천.
        
        Args:
            task_description: 태스크 설명
            category: 카테고리 필터
            top_k: 최대 추천 개수
            only_available: 사용 가능한 도구만 포함
        """
        keywords = self._extract_keywords(task_description)
        
        # 후보 도구 수집
        candidate_ids: Set[str] = set()
        
        # 키워드 기반 검색
        for keyword in keywords:
            if keyword.lower() in self.keyword_index:
                candidate_ids.update(self.keyword_index[keyword.lower()])
        
        # 카테고리 필터
        if category:
            candidate_ids &= self.category_index.get(category, set())
        
        # 점수 계산
        recommendations = []
        
        for tool_id in candidate_ids:
            tool = self.tools.get(tool_id)
            if not tool:
                continue
            
            if only_available and not tool.is_available:
                continue
            
            # 점수 계산
            score, reason = self._calculate_recommendation_score(
                tool, keywords, task_description
            )
            
            recommendations.append(ToolRecommendation(
                tool=tool,
                score=score,
                reason=reason
            ))
        
        # 점수 기준 정렬
        recommendations.sort(key=lambda r: r.score, reverse=True)
        
        return recommendations[:top_k]
    
    def _calculate_recommendation_score(
        self,
        tool: ToolInfo,
        keywords: List[str],
        task_description: str
    ) -> Tuple[float, str]:
        """추천 점수 계산."""
        score = 0.0
        reasons = []
        
        # 키워드 매칭
        tool_keywords = set(k.lower() for k in tool.keywords + tool.tags)
        query_keywords = set(k.lower() for k in keywords)
        
        keyword_match = len(tool_keywords & query_keywords)
        if keyword_match > 0:
            keyword_score = keyword_match / max(len(query_keywords), 1)
            score += keyword_score * 0.4
            reasons.append(f"{keyword_match} keyword matches")
        
        # 성공률
        if tool.success_rate > 0.8:
            score += 0.2
            reasons.append(f"High success rate ({tool.success_rate:.0%})")
        
        # 사용 빈도 (인기도)
        if tool.call_count > 10:
            score += min(0.2, tool.call_count / 100)
            reasons.append(f"Popular ({tool.call_count} calls)")
        
        # 서버 우선순위
        if tool.server_id in self.servers:
            server = self.servers[tool.server_id]
            priority_bonus = (10 - server.priority) / 10 * 0.1
            score += priority_bonus
        
        # 응답 시간
        if tool.avg_response_time > 0 and tool.avg_response_time < 5:
            score += 0.1
            reasons.append(f"Fast ({tool.avg_response_time:.1f}s)")
        
        reason = "; ".join(reasons) if reasons else "General match"
        
        return min(1.0, score), reason
    
    def get_tools_by_category(
        self,
        category: ToolCategory,
        only_available: bool = True
    ) -> List[ToolInfo]:
        """카테고리별 도구 조회."""
        tool_ids = self.category_index.get(category, set())
        tools = []
        
        for tool_id in tool_ids:
            tool = self.tools.get(tool_id)
            if tool and (not only_available or tool.is_available):
                tools.append(tool)
        
        return tools
    
    def record_tool_usage(
        self,
        tool_id: str,
        success: bool,
        response_time: float
    ):
        """도구 사용 기록."""
        if tool_id not in self.tools:
            return
        
        tool = self.tools[tool_id]
        tool.call_count += 1
        tool.last_used = datetime.now()
        
        if success:
            tool.success_count += 1
        
        # 이동 평균 응답 시간
        if tool.avg_response_time == 0:
            tool.avg_response_time = response_time
        else:
            tool.avg_response_time = (
                0.9 * tool.avg_response_time + 0.1 * response_time
            )
        
        # 서버 통계 업데이트
        if tool.server_id in self.servers:
            self.servers[tool.server_id].total_tool_calls += 1
    
    async def reconnect_failed_servers(self):
        """실패한 서버들 재연결."""
        for server_id, server in self.servers.items():
            if server.status == MCPServerStatus.FAILED:
                if self.reconnect_attempts[server_id] < self.max_reconnect_attempts:
                    server.status = MCPServerStatus.RECONNECTING
                    self.reconnect_attempts[server_id] += 1
                    
                    logger.info(
                        f"Attempting to reconnect {server.name} "
                        f"(attempt {self.reconnect_attempts[server_id]})"
                    )
                    
                    success = await self.connect_server(server_id)
                    
                    if not success:
                        await asyncio.sleep(self.reconnect_interval)
    
    async def start_monitoring(self):
        """모니터링 시작."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("MCP Manager monitoring started")
    
    async def stop_monitoring(self):
        """모니터링 중지."""
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("MCP Manager monitoring stopped")
    
    async def _monitoring_loop(self):
        """모니터링 루프."""
        while self._monitoring:
            try:
                # 자동 재연결
                if self.auto_reconnect:
                    await self.reconnect_failed_servers()
                
                # 헬스 체크
                await self._health_check()
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _health_check(self):
        """서버 헬스 체크."""
        for server_id, server in self.servers.items():
            if server.status == MCPServerStatus.CONNECTED:
                # 연결 상태 확인 (실제 구현에서는 ping 등)
                # Placeholder
                pass
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "can", "to", "of", "in", "for", "on",
            "with", "at", "by", "from", "as", "and", "or", "but", "if",
            "i", "want", "need", "please", "help", "me", "how"
        }
        
        words = text.lower().split()
        keywords = []
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word and clean_word not in stopwords and len(clean_word) > 2:
                keywords.append(clean_word)
        
        return keywords
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 반환."""
        status_counts = defaultdict(int)
        for server in self.servers.values():
            status_counts[server.status.value] += 1
        
        category_counts = {
            cat.value: len(ids) 
            for cat, ids in self.category_index.items()
        }
        
        return {
            "total_servers": len(self.servers),
            "server_status": dict(status_counts),
            "total_tools": len(self.tools),
            "available_tools": sum(1 for t in self.tools.values() if t.is_available),
            "tool_categories": category_counts,
            "total_tool_calls": sum(t.call_count for t in self.tools.values()),
            "auto_reconnect": self.auto_reconnect,
            "monitoring": self._monitoring
        }
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """안전한 콜백 실행."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Callback error (non-fatal): {e}")


# Singleton instance
_mcp_manager_agent: Optional[MCPManagerAgent] = None


def get_mcp_manager_agent(
    auto_reconnect: bool = True,
    reconnect_interval_seconds: float = 30.0
) -> MCPManagerAgent:
    """MCPManagerAgent 싱글톤 인스턴스 반환."""
    global _mcp_manager_agent
    
    if _mcp_manager_agent is None:
        _mcp_manager_agent = MCPManagerAgent(
            auto_reconnect=auto_reconnect,
            reconnect_interval_seconds=reconnect_interval_seconds
        )
    
    return _mcp_manager_agent
