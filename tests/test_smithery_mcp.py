#!/usr/bin/env python3
"""
Smithery MCP 서버 호출 및 연결 테스트 스크립트

이 스크립트는 Smithery MCP 서버를 Python에서 직접 호출하고 테스트하는 통합 도구입니다.

주요 기능:
- HTTP 기반 MCP 서버 호출 (streamablehttp_client)
- STDIO 기반 MCP 서버 호출 (stdio_client)
- 실제 도구 호출 및 결과 확인
- 병렬 서버 테스트
- LangChain 통합 예제

사용법:
    # 특정 서버 테스트
    python scripts/test_smithery_mcp.py --server semantic_scholar
    
    # STDIO 서버 테스트 (도구 호출 포함)
    python scripts/test_smithery_mcp.py --server fetch --test-tool
    
    # 모든 서버 테스트 (병렬)
    python scripts/test_smithery_mcp.py --all
    
    # LangChain 예제 보기
    python scripts/test_smithery_mcp.py --langchain-example
    
    # 결과를 JSON 파일로 저장
    python scripts/test_smithery_mcp.py --all --output results.json

환경 변수:
    SMITHERY_API_KEY: Smithery API 키 (필수)
    SMITHERY_PROFILE: Smithery 프로필 (선택사항)

참고:
    - MCP Authorization 명세 준수: Authorization 헤더 사용
    - STDIO 서버는 Node.js/npx 필요
    - HTTP 서버는 streamablehttp_client 사용
"""

import asyncio
import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import ListToolsResult, CallToolResult, TextContent
    from mcp.shared.exceptions import McpError
    MCP_AVAILABLE = True
except ImportError:
    print("❌ MCP 패키지가 설치되지 않았습니다. 'pip install mcp' 실행하세요.")
    sys.exit(1)

# HTTP client imports for error handling
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

# LangChain imports (선택적)
try:
    from langchain_core.tools import Tool
    from langchain.llms import OpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmitheryMCPTester:
    """
    Smithery MCP 서버 테스트 클라이언트
    
    HTTP 및 STDIO 기반 Smithery MCP 서버를 테스트하고 도구를 호출할 수 있습니다.
    """
    
    def __init__(self):
        self.api_key = os.getenv("SMITHERY_API_KEY", "")
        self.profile = os.getenv("SMITHERY_PROFILE", "")
        
        if not self.api_key:
            logger.warning("⚠️ SMITHERY_API_KEY 환경 변수가 설정되지 않았습니다")
        
        # HTTP 기반 Smithery 서버 목록
        self.http_servers = {
            "semantic_scholar": {
                "url": "https://server.smithery.ai/@hamid-vakilzadeh/mcpsemanticscholar/mcp",
                "description": "Semantic Scholar 학술 논문 검색",
                "tools": ["search_papers", "get_paper_details"],
                "params": {
                    "api_key": self.api_key,
                    "profile": self.profile
                }
            }
        }
        
        # STDIO 기반 Smithery 서버 목록
        self.stdio_servers = {
            "fetch": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@smithery-ai/fetch",
                    "--key",
                    self.api_key,
                    "--profile",
                    self.profile
                ],
                "description": "웹 페이지 페치 및 메타데이터 추출",
                "tools": ["fetch_url", "extract_metadata"]
            },
            "docfork": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@docfork/mcp",
                    "--key",
                    self.api_key,
                    "--profile",
                    self.profile
                ],
                "description": "문서 포크 및 처리",
                "tools": []
            },
            "context7-mcp": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@upstash/context7-mcp",
                    "--key",
                    self.api_key,
                    "--profile",
                    self.profile
                ],
                "description": "Context7 벡터 검색",
                "tools": []
            },
            "parallel-search": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@parallel/search",
                    "--key",
                    self.api_key,
                    "--profile",
                    self.profile
                ],
                "description": "병렬 웹 검색",
                "tools": ["parallel_search"]
            },
            "tavily-mcp": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@Jeetanshu18/tavily-mcp",
                    "--key",
                    self.api_key,
                    "--profile",
                    self.profile
                ],
                "description": "Tavily AI 검색",
                "tools": ["tavily_search"]
            },
            "WebSearch-MCP": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@mnhlt/WebSearch-MCP",
                    "--key",
                    self.api_key,
                    "--profile",
                    self.profile
                ],
                "description": "웹 검색",
                "tools": []
            }
        }
    
    def _resolve_env_vars(self, value: str) -> str:
        """환경 변수 치환 (${VAR} 형식)"""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, "")
        return value
    
    async def test_http_server(
        self, 
        server_name: str, 
        test_tool: bool = False,
        timeout: float = 15.0
    ) -> Dict[str, Any]:
        """
        HTTP 기반 Smithery MCP 서버 테스트
        
        Args:
            server_name: 서버 이름
            test_tool: 실제 도구 호출 테스트 여부
            timeout: 연결 타임아웃 (초)
            
        Returns:
            테스트 결과 딕셔너리
        """
        if server_name not in self.http_servers:
            return {
                "success": False,
                "error": f"HTTP 서버 '{server_name}'를 찾을 수 없습니다",
                "available_servers": list(self.http_servers.keys())
            }
        
        config = self.http_servers[server_name]
        url = config["url"]
        
        result = {
            "server_name": server_name,
            "type": "http",
            "url": url,
            "success": False,
            "tools": [],
            "tools_count": 0,
            "tool_results": {},
            "connection_time": None
        }
        
        start_time = datetime.now()
        
        try:
            logger.info(f"🔗 HTTP MCP 서버 연결 시도: {server_name}")
            logger.info(f"   URL: {url}")
            
            # MCP Authorization 명세 준수: Authorization 헤더 사용
            # URL 파라미터에 API 키 포함하지 않음
            headers = {}
            params = config.get("params", {})
            if params:
                api_key = params.get("api_key") or params.get("apiKey")
                if bool(api_key):
                    api_key = self._resolve_env_vars(api_key)
                    if bool(api_key):
                        # 큰따옴표(") 사용 - Python f-string 표준 (작은따옴표 아님)
                        # 헤더 값에 따옴표가 포함되지 않도록 f-string 사용
                        headers["Authorization"] = f"Bearer {api_key}"
                        logger.info(f"   Authorization 헤더 설정됨 (Bearer token)")
                        # 헤더 값 검증: 따옴표가 포함되어 있는지 확인
                        header_value = headers["Authorization"]
                        has_quotes = '"' in header_value or "'" in header_value
                        if has_quotes:
                            logger.warning(f"   ⚠️ 헤더 값에 따옴표가 포함되어 있습니다: {header_value[:30]}...")
                        # 헤더 값 미리보기 (보안상 일부만 표시)
                        if len(header_value) > 20:
                            logger.debug(f"   헤더 값 미리보기: {header_value[:15]}...{header_value[-5:]}")
                        else:
                            logger.debug(f"   헤더 값: {header_value[:10]}...")
                        logger.debug(f"   헤더 키 타입: {type('Authorization')}, 헤더 값 타입: {type(header_value)}")
            
            # streamable HTTP 클라이언트로 연결
            # unpacking 3 values (read, write, initialization_options) as per mcp library update
            # headers 파라미터는 dict 또는 None을 받음 (큰따옴표 사용 확인됨)
            async with streamablehttp_client(url, headers=headers if headers else None) as (read, write, _):
                async with ClientSession(read, write) as session:
                    # 초기화
                    await asyncio.wait_for(session.initialize(), timeout=timeout)
                    logger.info("✅ 서버 초기화 완료")
                    
                    # 도구 목록 가져오기
                    tools_result: ListToolsResult = await asyncio.wait_for(
                        session.list_tools(),
                        timeout=timeout
                    )
                    tools = tools_result.tools if hasattr(tools_result, 'tools') else []
                    
                    result["tools"] = [tool.name for tool in tools]
                    result["tools_count"] = len(tools)
                    logger.info(f"✅ 도구 목록 조회 완료: {len(tools)}개 도구 발견")
                    
                    # 도구 정보 출력
                    for tool in tools:
                        logger.info(f"   - {tool.name}: {tool.description[:80]}...")
                    
                    # 실제 도구 호출 테스트
                    if test_tool and tools:
                        logger.info(f"\n🔧 도구 호출 테스트 시작...")
                        test_tool_name = config.get("tools", [tools[0].name if tools else None])[0]
                        
                        if test_tool_name and any(t.name == test_tool_name for t in tools):
                            tool_result = await self._call_tool_example(
                                session, test_tool_name, tools, timeout
                            )
                            result["tool_results"][test_tool_name] = tool_result
                        else:
                            logger.warning(f"⚠️ 테스트 도구 '{test_tool_name}'를 찾을 수 없습니다")
                    
                    result["success"] = True
                    
        except asyncio.TimeoutError as e:
            result["success"] = False
            result["error"] = f"Connection timeout after {timeout}s"
            result["error_type"] = "timeout"
            logger.error(f"❌ HTTP 서버 연결 타임아웃: {server_name}")
        except McpError as e:
            result["success"] = False
            result["error"] = str(e)
            result["error_type"] = "mcp_error"
            error_code = getattr(e.error, 'code', None) if hasattr(e, 'error') else None
            if error_code:
                result["error"] += f" (code: {error_code})"
            logger.error(f"❌ HTTP 서버 MCP 오류: {e}")
        except ExceptionGroup as eg:
            # Unwrap ExceptionGroup to get the actual exception
            actual_error = None
            if HTTPX_AVAILABLE and httpx:
                for exc in (eg.exceptions if hasattr(eg, 'exceptions') else []):
                    if isinstance(exc, httpx.HTTPStatusError):
                        actual_error = exc
                        break
                    elif isinstance(exc, Exception):
                        actual_error = exc
            else:
                # If httpx not available, just use first exception
                if hasattr(eg, 'exceptions') and eg.exceptions:
                    actual_error = eg.exceptions[0]
            
            if actual_error:
                if HTTPX_AVAILABLE and httpx and isinstance(actual_error, httpx.HTTPStatusError):
                    result["success"] = False
                    result["error"] = f"HTTP {actual_error.response.status_code}: {actual_error.response.reason_phrase}"
                    result["error_type"] = "http_status_error"
                    result["status_code"] = actual_error.response.status_code
                    logger.error(f"❌ HTTP 서버 연결 실패: {result['error']}")
                else:
                    result["success"] = False
                    result["error"] = str(actual_error)
                    result["error_type"] = type(actual_error).__name__
                    logger.error(f"❌ HTTP 서버 연결 실패: {actual_error}")
            else:
                result["success"] = False
                result["error"] = str(eg)
                result["error_type"] = "exception_group"
                logger.error(f"❌ HTTP 서버 연결 실패 (ExceptionGroup): {eg}")
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
            logger.error(f"❌ HTTP 서버 연결 실패: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        connection_time = (datetime.now() - start_time).total_seconds()
        result["connection_time"] = connection_time
        
        return result
    
    async def test_stdio_server(
        self, 
        server_name: str, 
        test_tool: bool = False,
        timeout: float = 15.0
    ) -> Dict[str, Any]:
        """
        STDIO 기반 Smithery MCP 서버 테스트
        
        Args:
            server_name: 서버 이름
            test_tool: 실제 도구 호출 테스트 여부
            timeout: 연결 타임아웃 (초)
            
        Returns:
            테스트 결과 딕셔너리
        """
        if server_name not in self.stdio_servers:
            return {
                "success": False,
                "error": f"STDIO 서버 '{server_name}'를 찾을 수 없습니다",
                "available_servers": list(self.stdio_servers.keys())
            }
        
        config = self.stdio_servers[server_name]
        command = config["command"]
        args = config["args"]
        
        # 환경 변수 치환
        resolved_args = []
        for arg in args:
            resolved_args.append(self._resolve_env_vars(arg))
        
        # 빈 API 키 체크
        if "--key" in resolved_args:
            key_idx = resolved_args.index("--key")
            if key_idx + 1 < len(resolved_args) and not resolved_args[key_idx + 1]:
                return {
                    "success": False,
                    "error": "SMITHERY_API_KEY not set",
                    "error_type": "missing_api_key"
                }
        
        result = {
            "server_name": server_name,
            "type": "stdio",
            "command": f"{command} {' '.join(resolved_args)}",
            "success": False,
            "tools": [],
            "tools_count": 0,
            "tool_results": {},
            "connection_time": None
        }
        
        start_time = datetime.now()
        
        try:
            logger.info(f"🔗 STDIO MCP 서버 연결 시도: {server_name}")
            logger.info(f"   Command: {command}")
            logger.info(f"   Args: {' '.join(resolved_args[:5])}...")
            
            # STDIO 클라이언트로 연결
            server_params = StdioServerParameters(
                command=command,
                args=resolved_args
            )
            
            # unpacking 3 values (read, write, initialization_options) as per mcp library update
            async with stdio_client(server_params) as (read, write, _):
                async with ClientSession(read, write) as session:
                    # 초기화
                    await asyncio.wait_for(session.initialize(), timeout=timeout)
                    logger.info("✅ 서버 초기화 완료")
                    
                    # 도구 목록 가져오기
                    tools_result: ListToolsResult = await asyncio.wait_for(
                        session.list_tools(),
                        timeout=timeout
                    )
                    tools = tools_result.tools if hasattr(tools_result, 'tools') else []
                    
                    result["tools"] = [tool.name for tool in tools]
                    result["tools_count"] = len(tools)
                    logger.info(f"✅ 도구 목록 조회 완료: {len(tools)}개 도구 발견")
                    
                    # 도구 정보 출력
                    for tool in tools:
                        logger.info(f"   - {tool.name}: {tool.description[:80]}...")
                    
                    # 실제 도구 호출 테스트
                    if test_tool and tools:
                        logger.info(f"\n🔧 도구 호출 테스트 시작...")
                        test_tool_name = config.get("tools", [tools[0].name if tools else None])[0]
                        
                        if test_tool_name and any(t.name == test_tool_name for t in tools):
                            tool_result = await self._call_tool_example(
                                session, test_tool_name, tools, timeout
                            )
                            result["tool_results"][test_tool_name] = tool_result
                        else:
                            logger.warning(f"⚠️ 테스트 도구 '{test_tool_name}'를 찾을 수 없습니다")
                    
                    result["success"] = True
                    
        except asyncio.TimeoutError:
            result["success"] = False
            result["error"] = f"Connection timeout after {timeout}s"
            result["error_type"] = "timeout"
            logger.error(f"❌ STDIO 서버 연결 타임아웃: {server_name}")
        except McpError as e:
            result["success"] = False
            result["error"] = str(e)
            result["error_type"] = "mcp_error"
            error_code = getattr(e.error, 'code', None) if hasattr(e, 'error') else None
            if error_code:
                result["error"] += f" (code: {error_code})"
            logger.error(f"❌ STDIO 서버 MCP 오류: {e}")
        except ExceptionGroup as eg:
            # Unwrap ExceptionGroup to get the actual exception
            actual_error = None
            if hasattr(eg, 'exceptions') and eg.exceptions:
                actual_error = eg.exceptions[0]
            
            if actual_error:
                result["success"] = False
                result["error"] = str(actual_error)
                result["error_type"] = type(actual_error).__name__
                logger.error(f"❌ STDIO 서버 연결 실패: {actual_error}")
            else:
                result["success"] = False
                result["error"] = str(eg)
                result["error_type"] = "exception_group"
                logger.error(f"❌ STDIO 서버 연결 실패 (ExceptionGroup): {eg}")
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
            logger.error(f"❌ STDIO 서버 연결 실패: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        connection_time = (datetime.now() - start_time).total_seconds()
        result["connection_time"] = connection_time
        
        return result
    
    async def _call_tool_example(
        self, 
        session: ClientSession, 
        tool_name: str, 
        tools: List,
        timeout: float = 15.0
    ) -> Dict[str, Any]:
        """
        도구 호출 예제
        
        Args:
            session: MCP 클라이언트 세션
            tool_name: 호출할 도구 이름
            tools: 도구 목록
            timeout: 호출 타임아웃 (초)
            
        Returns:
            도구 호출 결과
        """
        result = {
            "tool_name": tool_name,
            "success": False,
            "result": None,
            "error": None
        }
        
        try:
            # 도구 찾기
            tool = next((t for t in tools if t.name == tool_name), None)
            if not tool:
                result["error"] = f"도구 '{tool_name}'를 찾을 수 없습니다"
                return result
            
            logger.info(f"   도구: {tool.name}")
            logger.info(f"   설명: {tool.description}")
            
            # 도구 파라미터 확인
            tool_params = {}
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                schema = tool.inputSchema
                if hasattr(schema, 'properties') and schema.properties:
                    logger.info(f"   파라미터: {list(schema.properties.keys())}")
                    
                    # 예제 파라미터 생성 (도구별)
                    if "search" in tool_name.lower() or "query" in tool_name.lower():
                        tool_params = {"query": "Python MCP tutorial"}
                    elif "url" in tool_name.lower() or "fetch" in tool_name.lower():
                        tool_params = {"url": "https://example.com"}
                    elif "paper" in tool_name.lower():
                        tool_params = {"query": "artificial intelligence"}
            
            logger.info(f"   호출 파라미터: {tool_params}")
            
            # 도구 호출
            tool_result: CallToolResult = await asyncio.wait_for(
                session.call_tool(tool_name, tool_params),
                timeout=timeout
            )
            
            # 결과 처리
            if hasattr(tool_result, 'content') and tool_result.content:
                content_text = ""
                for content in tool_result.content:
                    if isinstance(content, TextContent):
                        if hasattr(content, 'text'):
                            content_text += content.text
                        elif isinstance(content, str):
                            content_text += content
                    elif isinstance(content, dict) and 'text' in content:
                        content_text += content['text']
                
                result["result"] = content_text[:500]  # 처음 500자만
                logger.info(f"✅ 도구 호출 성공")
                logger.info(f"   결과 미리보기: {content_text[:200]}...")
            else:
                result["result"] = str(tool_result)
                logger.info(f"✅ 도구 호출 성공 (결과: {type(tool_result).__name__})")
            
            result["success"] = True
            
        except asyncio.TimeoutError:
            result["error"] = f"Tool call timeout after {timeout}s"
            result["error_type"] = "timeout"
            logger.error(f"❌ 도구 호출 타임아웃: {tool_name}")
        except Exception as e:
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
            logger.error(f"❌ 도구 호출 실패: {e}")
        
        return result
    
    async def test_all_servers(
        self, 
        test_tool: bool = False,
        timeout: float = 15.0,
        max_concurrency: int = 3
    ) -> Dict[str, Any]:
        """
        모든 Smithery 서버 테스트 (병렬 처리)
        
        Args:
            test_tool: 실제 도구 호출 테스트 여부
            timeout: 서버당 연결 타임아웃 (초)
            max_concurrency: 최대 동시 연결 수
            
        Returns:
            테스트 결과 딕셔너리
        """
        results = {
            "http_servers": {},
            "stdio_servers": {},
            "summary": {
                "total": 0,
                "success": 0,
                "failed": 0
            }
        }
        
        logger.info("=" * 80)
        logger.info("🚀 모든 Smithery MCP 서버 테스트 시작")
        logger.info(f"타임아웃: {timeout}초, 최대 동시 연결: {max_concurrency}")
        logger.info("=" * 80)
        
        # 병렬 처리용 semaphore
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def test_http_with_semaphore(server_name: str):
            async with semaphore:
                return await self.test_http_server(server_name, test_tool, timeout)
        
        async def test_stdio_with_semaphore(server_name: str):
            async with semaphore:
                return await self.test_stdio_server(server_name, test_tool, timeout)
        
        # HTTP 서버 테스트 (병렬)
        logger.info("\n📡 HTTP 기반 서버 테스트")
        http_tasks = [
            asyncio.create_task(test_http_with_semaphore(name))
            for name in self.http_servers.keys()
        ]
        http_results = await asyncio.gather(*http_tasks, return_exceptions=True)
        
        for i, result in enumerate(http_results):
            server_name = list(self.http_servers.keys())[i]
            if isinstance(result, Exception):
                results["http_servers"][server_name] = {
                    "server_name": server_name,
                    "success": False,
                    "error": str(result),
                    "error_type": type(result).__name__
                }
                results["summary"]["failed"] += 1
            else:
                results["http_servers"][server_name] = result
                if result["success"]:
                    results["summary"]["success"] += 1
                else:
                    results["summary"]["failed"] += 1
            results["summary"]["total"] += 1
        
        # STDIO 서버 테스트 (병렬)
        logger.info("\n\n💻 STDIO 기반 서버 테스트")
        stdio_tasks = [
            asyncio.create_task(test_stdio_with_semaphore(name))
            for name in self.stdio_servers.keys()
        ]
        stdio_results = await asyncio.gather(*stdio_tasks, return_exceptions=True)
        
        for i, result in enumerate(stdio_results):
            server_name = list(self.stdio_servers.keys())[i]
            if isinstance(result, Exception):
                results["stdio_servers"][server_name] = {
                    "server_name": server_name,
                    "success": False,
                    "error": str(result),
                    "error_type": type(result).__name__
                }
                results["summary"]["failed"] += 1
            else:
                results["stdio_servers"][server_name] = result
                if result["success"]:
                    results["summary"]["success"] += 1
                else:
                    results["summary"]["failed"] += 1
            results["summary"]["total"] += 1
        
        # 결과 요약
        logger.info("\n" + "=" * 80)
        logger.info("📊 테스트 결과 요약")
        logger.info("=" * 80)
        logger.info(f"총 서버 수: {results['summary']['total']}")
        logger.info(f"성공: {results['summary']['success']}")
        logger.info(f"실패: {results['summary']['failed']}")
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """테스트 결과 출력"""
        logger.info("\n" + "=" * 80)
        logger.info("📋 상세 결과")
        logger.info("=" * 80)
        
        # 성공한 서버
        successful = []
        failed = []
        
        for server_type in ["http_servers", "stdio_servers"]:
            for server_name, result in results.get(server_type, {}).items():
                if result.get("success"):
                    successful.append((server_name, result))
                else:
                    failed.append((server_name, result))
        
        if successful:
            logger.info("\n✅ 성공한 서버:")
            for server_name, result in successful:
                logger.info(f"  - {server_name}: {result.get('tools_count', 0)} tools "
                          f"({result.get('connection_time', 0):.2f}s)")
                if result.get('tools'):
                    logger.info(f"    도구: {', '.join(result['tools'][:5])}")
                    if len(result['tools']) > 5:
                        logger.info(f"    ... 외 {len(result['tools']) - 5}개")
        
        if failed:
            logger.info("\n❌ 실패한 서버:")
            for server_name, result in failed:
                logger.info(f"  - {server_name}: {result.get('error_type', 'unknown')}")
                logger.info(f"    에러: {result.get('error', 'Unknown error')[:100]}")


def print_langchain_example():
    """LangChain 통합 예제 출력"""
    if not LANGCHAIN_AVAILABLE:
        print("\n⚠️ LangChain이 설치되지 않았습니다. 'pip install langchain' 실행하세요.")
        return
    
    example_code = '''
# LangChain + Smithery MCP 통합 예제

from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain_core.tools import Tool
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import asyncio
import os

# Smithery MCP 서버 연결
async def create_mcp_tool(server_url: str, tool_name: str, api_key: str):
    """MCP 서버에서 도구를 LangChain Tool로 변환"""
    
    async def mcp_tool_func(**kwargs):
        async with streamablehttp_client(
            server_url,
            headers={"Authorization": f"Bearer {api_key}"}
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, kwargs)
                # 결과를 문자열로 변환
                if hasattr(result, 'content'):
                    return str(result.content)
                return str(result)
    
    return mcp_tool_func

# 사용 예제
async def main():
    api_key = os.getenv("SMITHERY_API_KEY")
    
    # Semantic Scholar 도구 생성
    search_tool_func = await create_mcp_tool(
        "https://server.smithery.ai/@hamid-vakilzadeh/mcpsemanticscholar/mcp",
        "search_papers",
        api_key
    )
    
    # LangChain Tool로 래핑
    tools = [
        Tool(
            name="semantic_scholar_search",
            func=lambda q: asyncio.run(search_tool_func(query=q)),
            description="Semantic Scholar에서 학술 논문 검색"
        )
    ]
    
    # LLM 및 에이전트 초기화
    llm = OpenAI()
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # 에이전트 실행
    result = agent.run("AI agent systems에 대한 최신 논문을 찾아줘")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    print("\n" + "=" * 80)
    print("📚 LangChain 통합 예제 코드")
    print("=" * 80)
    print(example_code)


async def main():
    parser = argparse.ArgumentParser(
        description="Smithery MCP 서버 호출 및 연결 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 특정 HTTP 서버 테스트
  python scripts/test_smithery_mcp.py --server semantic_scholar
  
  # STDIO 서버 테스트 (도구 호출 포함)
  python scripts/test_smithery_mcp.py --server fetch --test-tool
  
  # 모든 서버 테스트 (병렬)
  python scripts/test_smithery_mcp.py --all
  
  # LangChain 예제 보기
  python scripts/test_smithery_mcp.py --langchain-example
  
  # 결과를 JSON 파일로 저장
  python scripts/test_smithery_mcp.py --all --output results.json

환경 변수:
  SMITHERY_API_KEY: Smithery API 키 (필수)
  SMITHERY_PROFILE: Smithery 프로필 (선택사항)

주의사항:
  - MCP Authorization 명세 준수: Authorization 헤더 사용
  - STDIO 서버는 Node.js/npx 필요
  - HTTP 서버는 streamablehttp_client 사용
        """
    )
    
    parser.add_argument(
        "--server",
        type=str,
        help="테스트할 서버 이름 (http: semantic_scholar, stdio: fetch, parallel-search, tavily-mcp 등)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="모든 서버 테스트 (병렬 처리)"
    )
    parser.add_argument(
        "--test-tool",
        action="store_true",
        help="실제 도구 호출 테스트 포함"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="서버당 연결 타임아웃 (초, 기본값: 15)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="최대 동시 연결 수 (기본값: 3)"
    )
    parser.add_argument(
        "--langchain-example",
        action="store_true",
        help="LangChain 통합 예제 코드 출력"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="결과를 JSON 파일로 저장할 경로"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON 형식으로만 출력"
    )
    
    args = parser.parse_args()
    
    # LangChain 예제 출력
    if args.langchain_example:
        print_langchain_example()
        return
    
    # 클라이언트 생성
    client = SmitheryMCPTester()
    
    # API 키 확인 (필수는 아니지만 경고)
    if not client.api_key:
        logger.warning("⚠️ SMITHERY_API_KEY 환경 변수가 설정되지 않았습니다")
        logger.info("   일부 서버는 API 키 없이도 테스트할 수 있습니다")
    
    results = {}
    
    # 서버 테스트 실행
    if args.all:
        results = await client.test_all_servers(
            test_tool=args.test_tool,
            timeout=args.timeout,
            max_concurrency=args.concurrency
        )
        if not args.json:
            client.print_results(results)
    elif args.server:
        # 서버 타입 확인
        if args.server in client.http_servers:
            result = await client.test_http_server(
                args.server, 
                test_tool=args.test_tool,
                timeout=args.timeout
            )
            results = {"http_servers": {args.server: result}}
        elif args.server in client.stdio_servers:
            result = await client.test_stdio_server(
                args.server, 
                test_tool=args.test_tool,
                timeout=args.timeout
            )
            results = {"stdio_servers": {args.server: result}}
        else:
            logger.error(f"❌ 서버 '{args.server}'를 찾을 수 없습니다")
            logger.info(f"   HTTP 서버: {', '.join(client.http_servers.keys())}")
            logger.info(f"   STDIO 서버: {', '.join(client.stdio_servers.keys())}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
    
    # 결과 저장 또는 출력
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"\n💾 결과가 저장되었습니다: {output_path}")
    elif args.json or not args.all:
        # JSON 형식으로 출력
        print("\n" + "=" * 80)
        print("📋 테스트 결과 (JSON)")
        print("=" * 80)
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n테스트 중단됨")
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
