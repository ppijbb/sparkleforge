"""
MCP 의존성 감소 및 직접 API 활용

MCP 없이 직접 API 호출하는 시스템, 도구 직접 등록 및 실행 메커니즘,
MCP와 직접 API 병행 지원 (선택 가능), 직접 API를 통한 도구 호출 최적화,
MCP 의존성 제거 옵션 제공, 유연한 아키텍처로 전환
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import aiohttp

logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """API 제공자."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    TOGETHER = "together"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class AuthenticationType(Enum):
    """인증 유형."""
    API_KEY = "api_key"
    OAUTH = "oauth"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    NONE = "none"


@dataclass
class APIEndpoint:
    """API 엔드포인트."""
    url: str
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    retries: int = 3
    rate_limit: Optional[int] = None  # requests per minute


@dataclass
class APICredentials:
    """API 인증 정보."""
    provider: APIProvider
    auth_type: AuthenticationType
    api_key: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expiry: Optional[float] = None

    def is_expired(self) -> bool:
        """토큰 만료 여부 확인."""
        if self.token_expiry:
            return time.time() > self.token_expiry
        return False


@dataclass
class DirectTool:
    """직접 API 도구."""
    tool_id: str
    name: str
    description: str
    provider: APIProvider
    endpoint: APIEndpoint
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    credentials: APICredentials
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    usage_count: int = 0
    success_count: int = 0

    @property
    def success_rate(self) -> float:
        """성공률."""
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0


class APIAdapter(ABC):
    """API 어댑터 추상 클래스."""

    @abstractmethod
    async def execute_tool(self, tool: DirectTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """도구 실행."""
        pass

    @abstractmethod
    def format_request(self, tool: DirectTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """요청 포맷팅."""
        pass

    @abstractmethod
    def parse_response(self, response: Dict[str, Any], tool: DirectTool) -> Dict[str, Any]:
        """응답 파싱."""
        pass


class OpenAIAdapter(APIAdapter):
    """OpenAI API 어댑터."""

    async def execute_tool(self, tool: DirectTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """도구 실행."""
        request_data = self.format_request(tool, parameters)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                tool.endpoint.url,
                json=request_data,
                headers=tool.endpoint.headers,
                timeout=aiohttp.ClientTimeout(total=tool.endpoint.timeout)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self.parse_response(result, tool)
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error {response.status}: {error_text}")

    def format_request(self, tool: DirectTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """요청 포맷팅."""
        return {
            "model": parameters.get("model", "gpt-3.5-turbo"),
            "messages": parameters.get("messages", []),
            "temperature": parameters.get("temperature", 0.7),
            "max_tokens": parameters.get("max_tokens", 1000)
        }

    def parse_response(self, response: Dict[str, Any], tool: DirectTool) -> Dict[str, Any]:
        """응답 파싱."""
        return {
            "success": True,
            "response": response.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "usage": response.get("usage", {}),
            "model": response.get("model", "")
        }


class AnthropicAdapter(APIAdapter):
    """Anthropic API 어댑터."""

    async def execute_tool(self, tool: DirectTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """도구 실행."""
        request_data = self.format_request(tool, parameters)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                tool.endpoint.url,
                json=request_data,
                headers=tool.endpoint.headers,
                timeout=aiohttp.ClientTimeout(total=tool.endpoint.timeout)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self.parse_response(result, tool)
                else:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error {response.status}: {error_text}")

    def format_request(self, tool: DirectTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """요청 포맷팅."""
        return {
            "model": parameters.get("model", "claude-3-sonnet-20240229"),
            "max_tokens": parameters.get("max_tokens", 1000),
            "messages": parameters.get("messages", [])
        }

    def parse_response(self, response: Dict[str, Any], tool: DirectTool) -> Dict[str, Any]:
        """응답 파싱."""
        return {
            "success": True,
            "response": response.get("content", [{}])[0].get("text", ""),
            "usage": response.get("usage", {}),
            "model": response.get("model", "")
        }


class SearchAPIAdapter(APIAdapter):
    """검색 API 어댑터."""

    async def execute_tool(self, tool: DirectTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """도구 실행."""
        if tool.provider == APIProvider.GOOGLE:
            return await self._execute_google_search(tool, parameters)
        elif tool.provider == APIProvider.TOGETHER:
            return await self._execute_together_search(tool, parameters)
        else:
            raise NotImplementedError(f"Search provider {tool.provider} not implemented")

    async def _execute_google_search(self, tool: DirectTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Google 검색 실행."""
        query = parameters.get("query", "")
        num_results = parameters.get("num_results", 10)

        # Google Custom Search API 사용 (실제 구현에서는 API 키 필요)
        search_url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": tool.credentials.api_key,
            "cx": parameters.get("search_engine_id", ""),
            "q": query,
            "num": min(num_results, 10)
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_google_results(data)
                else:
                    error_text = await response.text()
                    raise Exception(f"Google Search API error {response.status}: {error_text}")

    def _parse_google_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Google 검색 결과 파싱."""
        items = data.get("items", [])
        results = []

        for item in items:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "display_link": item.get("displayLink", "")
            })

        return {
            "success": True,
            "results": results,
            "total_results": len(results)
        }

    async def _execute_together_search(self, tool: DirectTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Together AI 검색 실행."""
        # 실제 구현에서는 Together AI의 검색 API 사용
        # 여기서는 플레이스홀더
        return {
            "success": True,
            "results": [],
            "note": "Together AI search not fully implemented"
        }

    def format_request(self, tool: DirectTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """요청 포맷팅."""
        return parameters

    def parse_response(self, response: Dict[str, Any], tool: DirectTool) -> Dict[str, Any]:
        """응답 파싱."""
        return response


class RateLimiter:
    """속도 제한기."""

    def __init__(self, requests_per_minute: int):
        """초기화."""
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """토큰 획득."""
        async with self.lock:
            now = time.time()
            # 1분 이전 요청들 제거
            self.requests = [req for req in self.requests if now - req < 60]

            if len(self.requests) >= self.requests_per_minute:
                # 대기 시간 계산
                oldest_request = min(self.requests)
                wait_time = 60 - (now - oldest_request)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            self.requests.append(now)


class DirectAPIManager:
    """
    직접 API 관리자.

    MCP 없이 직접 API를 호출하고 도구를 관리하는 시스템.
    """

    def __init__(self):
        """초기화."""
        self.tools: Dict[str, DirectTool] = {}
        self.adapters: Dict[APIProvider, APIAdapter] = {}
        self.credentials: Dict[APIProvider, APICredentials] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}

        # 기본 어댑터 등록
        self._register_default_adapters()

        logger.info("DirectAPIManager initialized")

    def _register_default_adapters(self):
        """기본 어댑터 등록."""
        self.adapters[APIProvider.OPENAI] = OpenAIAdapter()
        self.adapters[APIProvider.ANTHROPIC] = AnthropicAdapter()
        self.adapters[APIProvider.GOOGLE] = SearchAPIAdapter()
        self.adapters[APIProvider.TOGETHER] = SearchAPIAdapter()

    def register_credentials(self, credentials: APICredentials):
        """
        인증 정보 등록.

        Args:
            credentials: API 인증 정보
        """
        self.credentials[credentials.provider] = credentials
        logger.info(f"Registered credentials for {credentials.provider.value}")

    def register_tool(self, tool: DirectTool):
        """
        도구 등록.

        Args:
            tool: 직접 API 도구
        """
        self.tools[tool.tool_id] = tool

        # 속도 제한기 생성
        if tool.endpoint.rate_limit:
            self.rate_limiters[tool.tool_id] = RateLimiter(tool.endpoint.rate_limit)

        logger.info(f"Registered direct tool: {tool.name} ({tool.provider.value})")

    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        도구 실행.

        Args:
            tool_id: 도구 ID
            parameters: 실행 파라미터

        Returns:
            실행 결과
        """
        if tool_id not in self.tools:
            raise ValueError(f"Tool {tool_id} not found")

        tool = self.tools[tool_id]

        if not tool.enabled:
            raise ValueError(f"Tool {tool_id} is disabled")

        # 사용량 기록
        tool.usage_count += 1
        tool.last_used = time.time()

        try:
            # 속도 제한 확인
            if tool_id in self.rate_limiters:
                await self.rate_limiters[tool_id].acquire()

            # 어댑터 확인
            if tool.provider not in self.adapters:
                raise ValueError(f"No adapter available for provider {tool.provider.value}")

            adapter = self.adapters[tool.provider]

            # 인증 정보 설정
            if tool.provider in self.credentials:
                creds = self.credentials[tool.provider]
                tool.endpoint.headers.update(self._get_auth_headers(creds))

            # 도구 실행
            start_time = time.time()
            result = await adapter.execute_tool(tool, parameters)
            execution_time = time.time() - start_time

            # 성공 기록
            tool.success_count += 1

            # 결과에 메타데이터 추가
            result.update({
                "tool_id": tool_id,
                "execution_time": execution_time,
                "provider": tool.provider.value,
                "direct_api": True
            })

            logger.info(f"Direct tool executed successfully: {tool.name} ({execution_time:.2f}s)")
            return result

        except Exception as e:
            logger.error(f"Direct tool execution failed: {tool.name}, error: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_id": tool_id,
                "provider": tool.provider.value,
                "direct_api": True
            }

    def _get_auth_headers(self, credentials: APICredentials) -> Dict[str, str]:
        """인증 헤더 생성."""
        headers = {}

        if credentials.auth_type == AuthenticationType.API_KEY:
            if credentials.provider == APIProvider.OPENAI:
                headers["Authorization"] = f"Bearer {credentials.api_key}"
            elif credentials.provider == APIProvider.ANTHROPIC:
                headers["x-api-key"] = credentials.api_key
                headers["anthropic-version"] = "2023-06-01"
            elif credentials.provider == APIProvider.GOOGLE:
                headers["Authorization"] = f"Bearer {credentials.api_key}"

        elif credentials.auth_type == AuthenticationType.BEARER_TOKEN:
            headers["Authorization"] = f"Bearer {credentials.access_token}"

        return headers

    def create_openai_tool(self, api_key: str, model: str = "gpt-3.5-turbo") -> DirectTool:
        """
        OpenAI 도구 생성.

        Args:
            api_key: API 키
            model: 모델 이름

        Returns:
            DirectTool 인스턴스
        """
        credentials = APICredentials(
            provider=APIProvider.OPENAI,
            auth_type=AuthenticationType.API_KEY,
            api_key=api_key
        )

        endpoint = APIEndpoint(
            url="https://api.openai.com/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            rate_limit=60  # requests per minute
        )

        return DirectTool(
            tool_id=f"openai_{model.replace('-', '_')}",
            name=f"OpenAI {model}",
            description=f"OpenAI {model} language model",
            provider=APIProvider.OPENAI,
            endpoint=endpoint,
            input_schema={
                "type": "object",
                "properties": {
                    "messages": {"type": "array"},
                    "temperature": {"type": "number"},
                    "max_tokens": {"type": "number"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                    "usage": {"type": "object"}
                }
            },
            credentials=credentials
        )

    def create_anthropic_tool(self, api_key: str, model: str = "claude-3-sonnet-20240229") -> DirectTool:
        """
        Anthropic 도구 생성.

        Args:
            api_key: API 키
            model: 모델 이름

        Returns:
            DirectTool 인스턴스
        """
        credentials = APICredentials(
            provider=APIProvider.ANTHROPIC,
            auth_type=AuthenticationType.API_KEY,
            api_key=api_key
        )

        endpoint = APIEndpoint(
            url="https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            rate_limit=50
        )

        return DirectTool(
            tool_id=f"anthropic_{model.replace('-', '_').replace('.', '_')}",
            name=f"Anthropic {model}",
            description=f"Anthropic {model} language model",
            provider=APIProvider.ANTHROPIC,
            endpoint=endpoint,
            input_schema={
                "type": "object",
                "properties": {
                    "messages": {"type": "array"},
                    "max_tokens": {"type": "number"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                    "usage": {"type": "object"}
                }
            },
            credentials=credentials
        )

    def create_google_search_tool(self, api_key: str, search_engine_id: str) -> DirectTool:
        """
        Google 검색 도구 생성.

        Args:
            api_key: Google API 키
            search_engine_id: 커스텀 검색 엔진 ID

        Returns:
            DirectTool 인스턴스
        """
        credentials = APICredentials(
            provider=APIProvider.GOOGLE,
            auth_type=AuthenticationType.API_KEY,
            api_key=api_key
        )

        endpoint = APIEndpoint(
            url="https://www.googleapis.com/customsearch/v1",
            method="GET",
            rate_limit=100
        )

        return DirectTool(
            tool_id="google_search",
            name="Google Search",
            description="Google Custom Search API",
            provider=APIProvider.GOOGLE,
            endpoint=endpoint,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer"},
                    "search_engine_id": {"type": "string"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "total_results": {"type": "integer"}
                }
            },
            credentials=credentials
        )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """사용 가능한 도구 목록."""
        return [
            {
                "tool_id": tool.tool_id,
                "name": tool.name,
                "description": tool.description,
                "provider": tool.provider.value,
                "enabled": tool.enabled,
                "success_rate": tool.success_rate,
                "usage_count": tool.usage_count
            }
            for tool in self.tools.values()
        ]

    def get_tool_stats(self) -> Dict[str, Any]:
        """도구 통계."""
        total_tools = len(self.tools)
        enabled_tools = sum(1 for t in self.tools.values() if t.enabled)
        total_usage = sum(t.usage_count for t in self.tools.values())
        avg_success_rate = sum(t.success_rate for t in self.tools.values()) / total_tools if total_tools > 0 else 0

        provider_stats = {}
        for tool in self.tools.values():
            provider = tool.provider.value
            if provider not in provider_stats:
                provider_stats[provider] = {"count": 0, "usage": 0}
            provider_stats[provider]["count"] += 1
            provider_stats[provider]["usage"] += tool.usage_count

        return {
            "total_tools": total_tools,
            "enabled_tools": enabled_tools,
            "total_usage": total_usage,
            "avg_success_rate": avg_success_rate,
            "provider_stats": provider_stats
        }


# 전역 직접 API 관리자 인스턴스
_direct_api_manager = None

def get_direct_api_manager() -> DirectAPIManager:
    """전역 직접 API 관리자 인스턴스 반환."""
    global _direct_api_manager
    if _direct_api_manager is None:
        _direct_api_manager = DirectAPIManager()
    return _direct_api_manager

def set_direct_api_manager(manager: DirectAPIManager):
    """전역 직접 API 관리자 설정."""
    global _direct_api_manager
    _direct_api_manager = manager


# 편의 함수들
def quick_setup_openai(api_key: str, model: str = "gpt-3.5-turbo") -> str:
    """
    OpenAI 도구 빠른 설정.

    Args:
        api_key: API 키
        model: 모델 이름

    Returns:
        도구 ID
    """
    manager = get_direct_api_manager()
    tool = manager.create_openai_tool(api_key, model)
    manager.register_tool(tool)
    return tool.tool_id

def quick_setup_anthropic(api_key: str, model: str = "claude-3-sonnet-20240229") -> str:
    """
    Anthropic 도구 빠른 설정.

    Args:
        api_key: API 키
        model: 모델 이름

    Returns:
        도구 ID
    """
    manager = get_direct_api_manager()
    tool = manager.create_anthropic_tool(api_key, model)
    manager.register_tool(tool)
    return tool.tool_id

def quick_setup_google_search(api_key: str, search_engine_id: str) -> str:
    """
    Google 검색 도구 빠른 설정.

    Args:
        api_key: Google API 키
        search_engine_id: 검색 엔진 ID

    Returns:
        도구 ID
    """
    manager = get_direct_api_manager()
    tool = manager.create_google_search_tool(api_key, search_engine_id)
    manager.register_tool(tool)
    return tool.tool_id
