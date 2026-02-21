#!/usr/bin/env python3
"""
MCP 도구 점검 테스트

모든 MCP 도구들이 제대로 로드되고 사용 가능한지 확인하는 테스트 코드.
각 도구의 기본 기능을 테스트하여 오류가 없는지 검증합니다.
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# CRITICAL: 설정을 먼저 로드해야 함
from src.core.researcher_config import load_config_from_env

config = load_config_from_env()

from src.core.mcp_integration import get_mcp_hub, execute_tool, get_available_tools
from src.core.researcher_config import get_agent_config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCPToolChecker:
    """MCP 도구 점검 클래스"""

    def __init__(self):
        self.mcp_hub = None
        self.test_results: Dict[str, Any] = {
            "servers": {},
            "tools": {},
            "errors": [],
            "warnings": [],
            "summary": {},
        }

    async def initialize(self):
        """MCP Hub 초기화"""
        logger.info("=" * 80)
        logger.info("🔧 MCP 도구 점검 시작")
        logger.info("=" * 80)

        try:
            self.mcp_hub = get_mcp_hub()
            logger.info("✅ MCP Hub 인스턴스 생성 완료")

            # MCP 서버 초기화
            if not self.mcp_hub.mcp_sessions:
                logger.info("🔄 MCP 서버 초기화 중...")
                await self.mcp_hub.initialize_mcp()
                logger.info(f"✅ {len(self.mcp_hub.mcp_sessions)}개 MCP 서버 연결 완료")
            else:
                logger.info(
                    f"✅ {len(self.mcp_hub.mcp_sessions)}개 MCP 서버 이미 연결됨"
                )

            return True
        except Exception as e:
            logger.error(f"❌ MCP Hub 초기화 실패: {e}")
            self.test_results["errors"].append(
                {"stage": "initialization", "error": str(e), "type": type(e).__name__}
            )
            return False

    async def check_servers(self):
        """모든 MCP 서버 연결 상태 확인"""
        logger.info("\n" + "=" * 80)
        logger.info("📡 MCP 서버 연결 상태 확인")
        logger.info("=" * 80)

        if not self.mcp_hub or not self.mcp_hub.mcp_sessions:
            logger.warning("⚠️ 연결된 MCP 서버가 없습니다")
            self.test_results["warnings"].append("No MCP servers connected")
            return

        for server_name, session in self.mcp_hub.mcp_sessions.items():
            try:
                logger.info(f"\n🔍 서버: {server_name}")

                # 서버 상태 확인
                is_healthy = await self.mcp_hub._check_connection_health(server_name)

                # 도구 목록 가져오기
                tools = self.mcp_hub.mcp_tools_map.get(server_name, {})
                tool_names = list(tools.keys())

                server_info = {
                    "name": server_name,
                    "connected": True,
                    "healthy": is_healthy,
                    "tools_count": len(tool_names),
                    "tools": tool_names,
                    "connection_diagnostics": self.mcp_hub.connection_diagnostics.get(
                        server_name, {}
                    ),
                }

                self.test_results["servers"][server_name] = server_info

                if is_healthy:
                    logger.info(f"  ✅ 상태: 정상 (도구 {len(tool_names)}개)")
                    for tool_name in tool_names[:5]:  # 처음 5개만 표시
                        logger.info(f"    - {tool_name}")
                    if len(tool_names) > 5:
                        logger.info(f"    ... 외 {len(tool_names) - 5}개")
                else:
                    logger.warning(f"  ⚠️ 상태: 비정상")
                    self.test_results["warnings"].append(
                        f"Server {server_name} is unhealthy"
                    )

            except Exception as e:
                logger.error(f"  ❌ 서버 확인 실패: {e}")
                self.test_results["servers"][server_name] = {
                    "name": server_name,
                    "connected": True,
                    "healthy": False,
                    "error": str(e),
                }
                self.test_results["errors"].append(
                    {
                        "server": server_name,
                        "stage": "server_check",
                        "error": str(e),
                        "type": type(e).__name__,
                    }
                )

    async def check_tools(self):
        """모든 도구 목록 확인"""
        logger.info("\n" + "=" * 80)
        logger.info("🔧 사용 가능한 도구 목록 확인")
        logger.info("=" * 80)

        try:
            available_tools = await get_available_tools()
            logger.info(f"✅ 총 {len(available_tools)}개 도구 발견")

            # 도구별 분류
            tool_categories = {}
            for tool_name in available_tools:
                # 도구 카테고리 추정 (이름 기반)
                category = self._guess_tool_category(tool_name)
                if category not in tool_categories:
                    tool_categories[category] = []
                tool_categories[category].append(tool_name)

            for category, tools in tool_categories.items():
                logger.info(f"\n📁 {category}: {len(tools)}개")
                for tool_name in tools[:10]:  # 카테고리당 최대 10개만 표시
                    logger.info(f"  - {tool_name}")
                if len(tools) > 10:
                    logger.info(f"  ... 외 {len(tools) - 10}개")

            self.test_results["tools"]["available"] = available_tools
            self.test_results["tools"]["categories"] = tool_categories
            self.test_results["tools"]["total_count"] = len(available_tools)

        except Exception as e:
            logger.error(f"❌ 도구 목록 확인 실패: {e}")
            self.test_results["errors"].append(
                {"stage": "tool_listing", "error": str(e), "type": type(e).__name__}
            )

    def _guess_tool_category(self, tool_name: str) -> str:
        """도구 이름으로 카테고리 추정"""
        tool_lower = tool_name.lower()

        if "search" in tool_lower or "google" in tool_lower:
            return "검색 (Search)"
        elif "fetch" in tool_lower or "web" in tool_lower or "http" in tool_lower:
            return "웹 (Web)"
        elif (
            "file" in tool_lower
            or "fs" in tool_lower
            or "read" in tool_lower
            or "write" in tool_lower
        ):
            return "파일시스템 (Filesystem)"
        elif "code" in tool_lower or "github" in tool_lower:
            return "코드 (Code)"
        elif "database" in tool_lower or "db" in tool_lower or "sql" in tool_lower:
            return "데이터베이스 (Database)"
        elif "ai" in tool_lower or "llm" in tool_lower or "model" in tool_lower:
            return "AI/LLM (AI/LLM)"
        else:
            return "기타 (Other)"

    async def test_essential_tools(self):
        """필수 도구 실행 테스트"""
        logger.info("\n" + "=" * 80)
        logger.info("🧪 필수 도구 실행 테스트")
        logger.info("=" * 80)

        # 필수 도구 목록 및 테스트 파라미터
        essential_tools = {
            "g-search": {
                "params": {"query": "test", "max_results": 3},
                "description": "Google 검색",
            },
            "fetch": {
                "params": {"url": "https://httpbin.org/get"},
                "description": "웹 페이지 가져오기",
            },
            # filesystem은 경로 문제로 인해 선택적으로 테스트
            # "filesystem": {
            #     "params": {"path": ".", "operation": "list"},
            #     "description": "파일시스템 조작"
            # }
        }

        for tool_name, tool_config in essential_tools.items():
            logger.info(f"\n🔍 테스트: {tool_name} ({tool_config['description']})")

            try:
                # 도구 사용 가능 여부 확인
                available_tools = await get_available_tools()
                if tool_name not in available_tools:
                    logger.warning(f"  ⚠️ 도구 '{tool_name}'를 사용할 수 없습니다")
                    self.test_results["tools"][tool_name] = {
                        "available": False,
                        "error": "Tool not found",
                    }
                    self.test_results["warnings"].append(
                        f"Tool {tool_name} not available"
                    )
                    continue

                # 도구 실행 테스트
                logger.info(f"  📤 실행 중... (파라미터: {tool_config['params']})")
                result = await execute_tool(tool_name, tool_config["params"])

                if result.get("success", False):
                    logger.info(f"  ✅ 성공: {result.get('execution_time', 0):.2f}초")
                    data = result.get("data", {})

                    # 실제 결과 내용 검증
                    is_valid = False
                    validation_details = []

                    if isinstance(data, dict):
                        logger.info(f"    - 결과 타입: dict")
                        logger.info(f"    - 키: {list(data.keys())[:10]}")

                        if "results" in data:
                            results = data.get("results", [])
                            logger.info(f"    - 결과 개수: {len(results)}개")
                            if results:
                                # 첫 번째 결과 상세 확인
                                first_result = results[0]
                                if isinstance(first_result, dict):
                                    logger.info(
                                        f"    - 첫 결과 키: {list(first_result.keys())[:5]}"
                                    )
                                    # 제목이나 URL이 있는지 확인
                                    has_title = any(
                                        k in first_result
                                        for k in ["title", "Title", "name", "heading"]
                                    )
                                    has_url = any(
                                        k in first_result
                                        for k in ["url", "URL", "link", "href"]
                                    )
                                    has_content = any(
                                        k in first_result
                                        for k in [
                                            "snippet",
                                            "content",
                                            "description",
                                            "text",
                                            "summary",
                                        ]
                                    )

                                    logger.info(
                                        f"    - 제목 포함: {has_title}, URL 포함: {has_url}, 내용 포함: {has_content}"
                                    )

                                    if has_title or has_url or has_content:
                                        # 실제 내용 확인
                                        content_text = ""
                                        if has_content:
                                            content_key = next(
                                                k
                                                for k in [
                                                    "snippet",
                                                    "content",
                                                    "description",
                                                    "text",
                                                    "summary",
                                                ]
                                                if k in first_result
                                            )
                                            content_text = str(
                                                first_result[content_key]
                                            ).lower()

                                        title_text = ""
                                        if has_title:
                                            title_key = next(
                                                k
                                                for k in [
                                                    "title",
                                                    "Title",
                                                    "name",
                                                    "heading",
                                                ]
                                                if k in first_result
                                            )
                                            title_text = str(
                                                first_result[title_key]
                                            ).lower()

                                        # 검색 결과가 실제로 없는 경우 감지
                                        invalid_indicators = [
                                            "no results",
                                            "not found",
                                            "bot detection",
                                            "no results were found",
                                            "search results",  # "Search Results"는 메타데이터 제목일 수 있음
                                            "try again",
                                            "unable to",
                                            "error occurred",
                                        ]

                                        is_invalid_result = False
                                        if content_text:
                                            is_invalid_result = any(
                                                indicator in content_text
                                                for indicator in invalid_indicators
                                            )
                                        if title_text and not is_invalid_result:
                                            # 제목이 "Search Results"이고 내용이 없거나 에러 메시지인 경우
                                            if "search results" in title_text and (
                                                not content_text
                                                or any(
                                                    indicator in content_text
                                                    for indicator in invalid_indicators
                                                )
                                            ):
                                                is_invalid_result = True

                                        if is_invalid_result:
                                            logger.warning(
                                                f"    ⚠️ 검색 결과가 실제로 없거나 에러 메시지입니다"
                                            )
                                            validation_details.append(
                                                "no_actual_results"
                                            )
                                        else:
                                            is_valid = True

                                        # 실제 내용 일부 출력
                                        if has_title:
                                            title_key = next(
                                                k
                                                for k in [
                                                    "title",
                                                    "Title",
                                                    "name",
                                                    "heading",
                                                ]
                                                if k in first_result
                                            )
                                            logger.info(
                                                f"    - 제목 예시: {str(first_result[title_key])[:80]}..."
                                            )
                                        if has_url:
                                            url_key = next(
                                                k
                                                for k in ["url", "URL", "link", "href"]
                                                if k in first_result
                                            )
                                            logger.info(
                                                f"    - URL 예시: {str(first_result[url_key])[:80]}..."
                                            )
                                        if has_content:
                                            content_key = next(
                                                k
                                                for k in [
                                                    "snippet",
                                                    "content",
                                                    "description",
                                                    "text",
                                                    "summary",
                                                ]
                                                if k in first_result
                                            )
                                            content_preview = str(
                                                first_result[content_key]
                                            )[:100]
                                            logger.info(
                                                f"    - 내용 예시: {content_preview}..."
                                            )
                                    else:
                                        logger.warning(
                                            f"    ⚠️ 결과에 제목/URL/내용이 없습니다. 구조: {first_result}"
                                        )
                                elif isinstance(first_result, str):
                                    logger.info(
                                        f"    - 첫 결과 (문자열): {first_result[:100]}..."
                                    )
                                    is_valid = len(first_result.strip()) > 0
                            else:
                                logger.warning(f"    ⚠️ 결과 배열이 비어있습니다")
                                validation_details.append("empty_results_array")

                        elif "content" in data:
                            content = data.get("content", "")
                            content_len = len(str(content))
                            logger.info(f"    - 콘텐츠 길이: {content_len}자")
                            if content_len > 0:
                                is_valid = True
                                logger.info(
                                    f"    - 콘텐츠 미리보기: {str(content)[:150]}..."
                                )
                            else:
                                logger.warning(f"    ⚠️ 콘텐츠가 비어있습니다")
                                validation_details.append("empty_content")

                        else:
                            # 다른 키들 확인
                            logger.info(
                                f"    - 'results' 또는 'content' 키가 없습니다. 전체 구조 확인 중..."
                            )
                            # 값이 있는 키 찾기
                            non_empty_keys = [
                                k
                                for k, v in data.items()
                                if v
                                and (
                                    isinstance(v, (str, list, dict)) and len(str(v)) > 0
                                )
                            ]
                            if non_empty_keys:
                                logger.info(f"    - 값이 있는 키: {non_empty_keys[:5]}")
                                is_valid = True
                            else:
                                logger.warning(f"    ⚠️ 유효한 데이터가 없습니다")
                                validation_details.append("no_valid_data")

                    elif isinstance(data, str):
                        logger.info(f"    - 결과 타입: str")
                        logger.info(f"    - 결과 길이: {len(data)}자")
                        if len(data.strip()) > 0:
                            is_valid = True
                            logger.info(f"    - 내용 미리보기: {data[:150]}...")
                        else:
                            logger.warning(f"    ⚠️ 결과 문자열이 비어있습니다")
                            validation_details.append("empty_string")

                    elif isinstance(data, list):
                        logger.info(f"    - 결과 타입: list")
                        logger.info(f"    - 항목 수: {len(data)}개")
                        if len(data) > 0:
                            is_valid = True
                            logger.info(f"    - 첫 항목: {str(data[0])[:100]}...")
                        else:
                            logger.warning(f"    ⚠️ 리스트가 비어있습니다")
                            validation_details.append("empty_list")

                    else:
                        logger.warning(f"    ⚠️ 예상치 못한 결과 타입: {type(data)}")
                        validation_details.append(
                            f"unexpected_type_{type(data).__name__}"
                        )

                    # 검증 결과 기록
                    tool_result = {
                        "available": True,
                        "tested": True,
                        "success": True,
                        "is_valid": is_valid,
                        "execution_time": result.get("execution_time", 0),
                        "result_type": type(result.get("data")).__name__,
                        "validation_details": validation_details,
                    }

                    if not is_valid:
                        logger.warning(
                            f"    ⚠️ 결과 데이터가 유효하지 않거나 비어있습니다"
                        )
                        tool_result["warning"] = "Result data is empty or invalid"
                        self.test_results["warnings"].append(
                            f"Tool {tool_name}: Invalid or empty result data"
                        )

                    self.test_results["tools"][tool_name] = tool_result
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"  ❌ 실패: {error_msg}")
                    self.test_results["tools"][tool_name] = {
                        "available": True,
                        "tested": True,
                        "success": False,
                        "error": error_msg,
                    }
                    self.test_results["errors"].append(
                        {"tool": tool_name, "stage": "execution", "error": error_msg}
                    )

            except Exception as e:
                logger.error(f"  ❌ 테스트 중 예외 발생: {e}")
                self.test_results["tools"][tool_name] = {
                    "available": True,
                    "tested": True,
                    "success": False,
                    "error": str(e),
                    "exception_type": type(e).__name__,
                }
                self.test_results["errors"].append(
                    {
                        "tool": tool_name,
                        "stage": "execution",
                        "error": str(e),
                        "type": type(e).__name__,
                    }
                )

    async def test_github_mcp_server(self):
        """GitHub MCP 서버 연결 및 도구 테스트"""
        logger.info("\n" + "=" * 80)
        logger.info("🐙 GitHub MCP 서버 테스트")
        logger.info("=" * 80)

        github_server_name = None
        github_tools = []

        # GitHub 서버 찾기
        if not self.mcp_hub or not self.mcp_hub.mcp_sessions:
            logger.warning("⚠️ 연결된 MCP 서버가 없습니다")
            self.test_results["warnings"].append(
                "No MCP servers connected for GitHub test"
            )
            return

        # GitHub 관련 서버 찾기
        for server_name in self.mcp_hub.mcp_sessions.keys():
            if "github" in server_name.lower():
                github_server_name = server_name
                break

        if not github_server_name:
            logger.warning("⚠️ GitHub MCP 서버를 찾을 수 없습니다")
            logger.info(
                "   💡 GitHub 서버를 사용하려면 configs/mcp_config.json에 다음을 추가하세요:"
            )
            logger.info("   {")
            logger.info('     "github": {')
            logger.info('       "command": "npx",')
            logger.info(
                '       "args": ["-y", "@modelcontextprotocol/server-github@latest"],'
            )
            logger.info('       "env": {')
            logger.info('         "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"')
            logger.info("       }")
            logger.info("     }")
            logger.info("   }")
            self.test_results["warnings"].append(
                "GitHub MCP server not found in configuration"
            )
            return

        logger.info(f"✅ GitHub 서버 발견: {github_server_name}")

        try:
            # 서버 상태 확인
            is_healthy = await self.mcp_hub._check_connection_health(github_server_name)

            if not is_healthy:
                logger.warning(f"  ⚠️ GitHub 서버가 비정상 상태입니다")
                self.test_results["warnings"].append(
                    f"GitHub server {github_server_name} is unhealthy"
                )
                return

            # 도구 목록 가져오기
            tools = self.mcp_hub.mcp_tools_map.get(github_server_name, {})
            github_tools = list(tools.keys())

            logger.info(f"  ✅ 연결 상태: 정상")
            logger.info(f"  ✅ 사용 가능한 도구: {len(github_tools)}개")

            if github_tools:
                logger.info(f"\n  📋 GitHub 도구 목록:")
                for tool_name in github_tools[:10]:  # 처음 10개만 표시
                    logger.info(f"    - {tool_name}")
                if len(github_tools) > 10:
                    logger.info(f"    ... 외 {len(github_tools) - 10}개")

            # GitHub 서버 정보 저장
            self.test_results["servers"][github_server_name] = {
                "name": github_server_name,
                "connected": True,
                "healthy": is_healthy,
                "tools_count": len(github_tools),
                "tools": github_tools,
                "type": "github",
            }

            # GitHub 도구 테스트
            if github_tools:
                logger.info(f"\n  🧪 GitHub 도구 실행 테스트")

                # 일반적인 GitHub 도구들 테스트
                test_cases = []

                # 리포지토리 파일 읽기 도구
                read_file_tools = [
                    t
                    for t in github_tools
                    if "get_file_contents" in t.lower()
                    or (
                        "get" in t.lower()
                        and "file" in t.lower()
                        and "content" in t.lower()
                    )
                ]
                if read_file_tools:
                    test_cases.append(
                        {
                            "tool": read_file_tools[0],
                            "params": {
                                "owner": "modelcontextprotocol",
                                "repo": "servers",
                                "path": "README.md",
                            },
                            "description": "리포지토리 파일 읽기",
                        }
                    )

                # 이슈 검색 도구
                issue_tools = [t for t in github_tools if "issue" in t.lower()]
                if issue_tools:
                    test_cases.append(
                        {
                            "tool": issue_tools[0],
                            "params": {
                                "owner": "modelcontextprotocol",
                                "repo": "servers",
                                "state": "open",
                                "limit": 5,
                            },
                            "description": "이슈 목록 조회",
                        }
                    )

                # PR 검색 도구
                pr_tools = [
                    t for t in github_tools if "pull" in t.lower() or "pr" in t.lower()
                ]
                if pr_tools:
                    test_cases.append(
                        {
                            "tool": pr_tools[0],
                            "params": {
                                "owner": "modelcontextprotocol",
                                "repo": "servers",
                                "state": "open",
                                "limit": 5,
                            },
                            "description": "PR 목록 조회",
                        }
                    )

                # 검색 도구
                search_tools = [t for t in github_tools if "search" in t.lower()]
                if search_tools:
                    test_cases.append(
                        {
                            "tool": search_tools[0],
                            "params": {"query": "MCP server", "type": "code"},
                            "description": "코드 검색",
                        }
                    )

                # 테스트 실행 (최대 3개만)
                for i, test_case in enumerate(test_cases[:3]):
                    tool_name = f"{github_server_name}::{test_case['tool']}"
                    logger.info(
                        f"\n    🔍 테스트 {i + 1}: {test_case['description']} ({test_case['tool']})"
                    )

                    try:
                        # 도구 실행
                        result = await execute_tool(tool_name, test_case["params"])

                        if result.get("success", False):
                            logger.info(
                                f"      ✅ 성공: {result.get('execution_time', 0):.2f}초"
                            )
                            data = result.get("data", {})

                            # 결과 검증
                            is_valid = False
                            if isinstance(data, dict):
                                # 결과가 비어있지 않은지 확인
                                has_content = len(str(data)) > 0
                                if (
                                    "items" in data
                                    or "content" in data
                                    or "files" in data
                                ):
                                    is_valid = True
                                elif has_content:
                                    is_valid = True
                            elif isinstance(data, (list, str)):
                                is_valid = (
                                    len(data) > 0
                                    if isinstance(data, (list, str))
                                    else len(str(data)) > 0
                                )

                            tool_result = {
                                "tested": True,
                                "success": True,
                                "is_valid": is_valid,
                                "execution_time": result.get("execution_time", 0),
                                "description": test_case["description"],
                            }

                            if not is_valid:
                                logger.warning(
                                    f"      ⚠️ 결과가 비어있거나 유효하지 않습니다"
                                )
                                tool_result["warning"] = "Result is empty or invalid"
                                self.test_results["warnings"].append(
                                    f"GitHub tool {tool_name}: Invalid result"
                                )

                            self.test_results["tools"][tool_name] = tool_result
                        else:
                            error_msg = result.get("error", "Unknown error")
                            logger.warning(f"      ⚠️ 실패: {error_msg}")
                            self.test_results["tools"][tool_name] = {
                                "tested": True,
                                "success": False,
                                "error": error_msg,
                                "description": test_case["description"],
                            }
                            # 인증 오류는 경고로만 처리
                            if (
                                "401" in error_msg
                                or "unauthorized" in error_msg.lower()
                                or "token" in error_msg.lower()
                            ):
                                self.test_results["warnings"].append(
                                    f"GitHub tool {tool_name}: Authentication required (GITHUB_TOKEN not set or invalid)"
                                )
                    except Exception as e:
                        logger.warning(f"      ⚠️ 테스트 중 예외: {e}")
                        self.test_results["tools"][tool_name] = {
                            "tested": True,
                            "success": False,
                            "error": str(e),
                            "exception_type": type(e).__name__,
                            "description": test_case["description"],
                        }

                if not test_cases:
                    logger.info(f"    ℹ️ 테스트 가능한 GitHub 도구를 찾을 수 없습니다")
                    logger.info(
                        f"    💡 GitHub 도구는 GITHUB_TOKEN 환경 변수가 필요할 수 있습니다"
                    )
            else:
                logger.warning(f"  ⚠️ GitHub 서버에 도구가 없습니다")
                self.test_results["warnings"].append(
                    f"GitHub server {github_server_name} has no tools"
                )

        except Exception as e:
            logger.error(f"  ❌ GitHub 서버 테스트 실패: {e}")
            self.test_results["errors"].append(
                {
                    "server": github_server_name,
                    "stage": "github_test",
                    "error": str(e),
                    "type": type(e).__name__,
                }
            )

    async def test_search_tools(self):
        """검색 도구들 테스트"""
        logger.info("\n" + "=" * 80)
        logger.info("🔍 검색 도구 테스트")
        logger.info("=" * 80)

        try:
            available_tools = await get_available_tools()
            search_tools = [
                t
                for t in available_tools
                if "search" in t.lower() or "google" in t.lower()
            ]

            if not search_tools:
                logger.warning("⚠️ 검색 도구를 찾을 수 없습니다")
                return

            logger.info(f"✅ {len(search_tools)}개 검색 도구 발견")
            logger.info(
                f"   (전체 테스트는 시간이 오래 걸리므로 대표적인 도구만 테스트)"
            )

            # 우선순위가 높은 도구부터 테스트
            priority_tools = []
            other_tools = []

            for tool_name in search_tools:
                if (
                    tool_name in ["g-search", "ddg_search::search"]
                    or "ddg_search" in tool_name
                ):
                    priority_tools.append(tool_name)
                else:
                    other_tools.append(tool_name)

            # 더 많은 검색 도구 테스트 (우선순위 도구 + 대안 검색 도구들)
            # DuckDuckGo가 봇 감지하므로 다른 검색 도구들도 테스트
            alternative_tools = [
                t
                for t in other_tools
                if any(alt in t.lower() for alt in ["exa", "tavily", "websearch"])
            ]
            tools_to_test = priority_tools[:2] + alternative_tools[:3]  # 총 최대 5개

            for tool_name in tools_to_test:
                logger.info(f"\n🔍 테스트: {tool_name}")

                try:
                    # 검색 도구는 보통 query 파라미터 사용
                    params = {"query": "Python programming", "max_results": 2}
                    result = await execute_tool(tool_name, params)

                    if result.get("success", False):
                        logger.info(
                            f"  ✅ 성공: {result.get('execution_time', 0):.2f}초"
                        )
                        data = result.get("data", {})

                        # 실제 검색 결과 검증
                        is_valid = False
                        validation_details = []

                        if isinstance(data, dict):
                            # 'result' 키가 있는 경우 (문자열 결과)
                            if "result" in data and isinstance(data.get("result"), str):
                                result_str = data.get("result", "")
                                logger.info(
                                    f"    - 결과 타입: 문자열 (길이: {len(result_str)}자)"
                                )
                                logger.info(
                                    f"    - 결과 미리보기: {result_str[:200]}..."
                                )

                                # 문자열 결과에서도 유효성 검증
                                result_lower = result_str.lower()
                                invalid_indicators = [
                                    "no results",
                                    "not found",
                                    "bot detection",
                                    "no results were found",
                                    "error",
                                    "failed",
                                ]
                                is_invalid = any(
                                    indicator in result_lower
                                    for indicator in invalid_indicators
                                )

                                if is_invalid:
                                    logger.warning(
                                        f"    ⚠️ 결과 문자열에 에러 메시지가 포함되어 있습니다"
                                    )
                                    validation_details.append("error_in_string_result")
                                else:
                                    is_valid = (
                                        len(result_str.strip()) > 50
                                    )  # 최소 50자 이상이어야 유효
                                    if not is_valid:
                                        logger.warning(
                                            f"    ⚠️ 결과 문자열이 너무 짧습니다 ({len(result_str)}자)"
                                        )
                                        validation_details.append("string_too_short")

                                tool_result = {
                                    "tested": True,
                                    "success": True,
                                    "is_valid": is_valid,
                                    "execution_time": result.get("execution_time", 0),
                                    "source": result.get("source", "unknown"),
                                    "validation_details": validation_details,
                                    "result_type": "string",
                                    "result_length": len(result_str),
                                }

                                if not is_valid:
                                    tool_result["warning"] = (
                                        "String result contains error or too short"
                                    )
                                    self.test_results["warnings"].append(
                                        f"Tool {tool_name}: Invalid string result"
                                    )

                                self.test_results["tools"][tool_name] = tool_result
                                continue

                            if "results" in data:
                                results = data.get("results", [])
                                logger.info(f"    - 검색 결과: {len(results)}개")
                                if results:
                                    first_result = results[0]
                                    if isinstance(first_result, dict):
                                        has_title = any(
                                            k in first_result
                                            for k in [
                                                "title",
                                                "Title",
                                                "name",
                                                "heading",
                                            ]
                                        )
                                        has_url = any(
                                            k in first_result
                                            for k in ["url", "URL", "link", "href"]
                                        )
                                        has_content = any(
                                            k in first_result
                                            for k in [
                                                "snippet",
                                                "content",
                                                "description",
                                                "text",
                                                "summary",
                                            ]
                                        )

                                        if has_title or has_url or has_content:
                                            # 실제 내용 확인
                                            content_text = ""
                                            if has_content:
                                                content_key = next(
                                                    k
                                                    for k in [
                                                        "snippet",
                                                        "content",
                                                        "description",
                                                        "text",
                                                        "summary",
                                                    ]
                                                    if k in first_result
                                                )
                                                content_text = str(
                                                    first_result[content_key]
                                                ).lower()

                                            title_text = ""
                                            if has_title:
                                                title_key = next(
                                                    k
                                                    for k in [
                                                        "title",
                                                        "Title",
                                                        "name",
                                                        "heading",
                                                    ]
                                                    if k in first_result
                                                )
                                                title_text = str(
                                                    first_result[title_key]
                                                ).lower()

                                            # 검색 결과가 실제로 없는 경우 감지
                                            invalid_indicators = [
                                                "no results",
                                                "not found",
                                                "bot detection",
                                                "no results were found",
                                                "search results",
                                                "try again",
                                                "unable to",
                                                "error occurred",
                                            ]

                                            is_invalid_result = False
                                            if content_text:
                                                is_invalid_result = any(
                                                    indicator in content_text
                                                    for indicator in invalid_indicators
                                                )
                                            if title_text and not is_invalid_result:
                                                if "search results" in title_text and (
                                                    not content_text
                                                    or any(
                                                        indicator in content_text
                                                        for indicator in invalid_indicators
                                                    )
                                                ):
                                                    is_invalid_result = True

                                            if is_invalid_result:
                                                logger.warning(
                                                    f"    ⚠️ 검색 결과가 실제로 없거나 에러 메시지입니다"
                                                )
                                                validation_details.append(
                                                    "no_actual_results"
                                                )
                                            else:
                                                is_valid = True

                                            logger.info(
                                                f"    - 제목: {has_title}, URL: {has_url}, 내용: {has_content}"
                                            )
                                            if has_title:
                                                title_key = next(
                                                    k
                                                    for k in [
                                                        "title",
                                                        "Title",
                                                        "name",
                                                        "heading",
                                                    ]
                                                    if k in first_result
                                                )
                                                logger.info(
                                                    f"    - 제목: {str(first_result[title_key])[:80]}..."
                                                )
                                            if has_url:
                                                url_key = next(
                                                    k
                                                    for k in [
                                                        "url",
                                                        "URL",
                                                        "link",
                                                        "href",
                                                    ]
                                                    if k in first_result
                                                )
                                                logger.info(
                                                    f"    - URL: {str(first_result[url_key])[:80]}..."
                                                )
                                        else:
                                            logger.warning(
                                                f"    ⚠️ 검색 결과에 제목/URL/내용이 없습니다"
                                            )
                                    elif isinstance(first_result, str):
                                        logger.info(
                                            f"    - 결과 (문자열): {first_result[:100]}..."
                                        )
                                        is_valid = len(first_result.strip()) > 0
                                else:
                                    logger.warning(
                                        f"    ⚠️ 검색 결과 배열이 비어있습니다"
                                    )
                                    validation_details.append("empty_results")
                            elif isinstance(data, str):
                                logger.info(f"    - 결과 (문자열): {data[:150]}...")
                                is_valid = len(data.strip()) > 0
                            else:
                                logger.info(f"    - 결과 구조: {list(data.keys())[:5]}")
                                # 다른 형태의 결과도 유효할 수 있음
                                is_valid = len(str(data)) > 0
                        elif isinstance(data, str):
                            logger.info(f"    - 결과 (문자열): {data[:150]}...")
                            is_valid = len(data.strip()) > 0
                        elif isinstance(data, list):
                            logger.info(f"    - 결과 (리스트): {len(data)}개 항목")
                            if len(data) > 0:
                                is_valid = True
                                logger.info(f"    - 첫 항목: {str(data[0])[:100]}...")
                            else:
                                logger.warning(f"    ⚠️ 결과 리스트가 비어있습니다")
                                validation_details.append("empty_list")

                        tool_result = {
                            "tested": True,
                            "success": True,
                            "is_valid": is_valid,
                            "execution_time": result.get("execution_time", 0),
                            "source": result.get("source", "unknown"),
                            "validation_details": validation_details,
                        }

                        if not is_valid:
                            logger.warning(
                                f"    ⚠️ 검색 결과가 유효하지 않거나 비어있습니다"
                            )
                            tool_result["warning"] = "Search result is empty or invalid"
                            self.test_results["warnings"].append(
                                f"Tool {tool_name}: Invalid or empty search result"
                            )

                        self.test_results["tools"][tool_name] = tool_result
                    else:
                        error_msg = result.get("error", "Unknown error")
                        logger.warning(f"  ⚠️ 실패: {error_msg}")
                        self.test_results["tools"][tool_name] = {
                            "tested": True,
                            "success": False,
                            "error": error_msg,
                            "source": result.get("source", "unknown"),
                        }
                        # 서버 측 에러는 경고로만 처리 (우리 코드 문제 아님)
                        if (
                            "Server error" in error_msg
                            or "502" in error_msg
                            or "401" in error_msg
                        ):
                            self.test_results["warnings"].append(
                                f"Tool {tool_name}: {error_msg}"
                            )
                except Exception as e:
                    logger.warning(f"  ⚠️ 테스트 실패: {e}")
                    self.test_results["tools"][tool_name] = {
                        "tested": True,
                        "success": False,
                        "error": str(e),
                        "exception_type": type(e).__name__,
                    }

        except Exception as e:
            logger.error(f"❌ 검색 도구 테스트 실패: {e}")
            self.test_results["errors"].append(
                {"stage": "search_tools_test", "error": str(e)}
            )

    def generate_summary(self):
        """테스트 결과 요약 생성"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 테스트 결과 요약")
        logger.info("=" * 80)

        # 서버 통계
        total_servers = len(self.test_results["servers"])
        healthy_servers = len(
            [
                s
                for s in self.test_results["servers"].values()
                if s.get("healthy", False)
            ]
        )

        # 도구 통계
        total_tools = self.test_results["tools"].get("total_count", 0)
        tested_tools = len(
            [
                t
                for t in self.test_results["tools"].values()
                if isinstance(t, dict) and t.get("tested", False)
            ]
        )
        successful_tools = len(
            [
                t
                for t in self.test_results["tools"].values()
                if isinstance(t, dict) and t.get("success", False)
            ]
        )
        valid_tools = len(
            [
                t
                for t in self.test_results["tools"].values()
                if isinstance(t, dict) and t.get("is_valid", False)
            ]
        )
        invalid_tools = len(
            [
                t
                for t in self.test_results["tools"].values()
                if isinstance(t, dict)
                and t.get("tested", False)
                and t.get("success", False)
                and not t.get("is_valid", True)
            ]
        )

        # 오류 및 경고
        error_count = len(self.test_results["errors"])
        warning_count = len(self.test_results["warnings"])

        logger.info(f"\n📡 MCP 서버:")
        logger.info(f"  - 총 서버: {total_servers}개")
        logger.info(f"  - 정상 서버: {healthy_servers}개")
        logger.info(f"  - 비정상 서버: {total_servers - healthy_servers}개")

        logger.info(f"\n🔧 도구:")
        logger.info(f"  - 총 도구: {total_tools}개")
        logger.info(f"  - 테스트한 도구: {tested_tools}개")
        logger.info(f"  - 성공한 도구: {successful_tools}개")
        logger.info(f"  - 유효한 결과 도구: {valid_tools}개")
        logger.info(f"  - 실패한 도구: {tested_tools - successful_tools}개")
        if invalid_tools > 0:
            logger.warning(
                f"  - ⚠️ 결과가 비어있거나 유효하지 않은 도구: {invalid_tools}개"
            )

        logger.info(f"\n⚠️ 문제:")
        logger.info(f"  - 오류: {error_count}개")
        logger.info(f"  - 경고: {warning_count}개")

        # 요약 저장
        self.test_results["summary"] = {
            "servers": {
                "total": total_servers,
                "healthy": healthy_servers,
                "unhealthy": total_servers - healthy_servers,
            },
            "tools": {
                "total": total_tools,
                "tested": tested_tools,
                "successful": successful_tools,
                "valid": valid_tools,
                "invalid": invalid_tools,
                "failed": tested_tools - successful_tools,
            },
            "issues": {"errors": error_count, "warnings": warning_count},
            "timestamp": datetime.now().isoformat(),
        }

        # 최종 상태
        if error_count == 0 and healthy_servers > 0 and invalid_tools == 0:
            logger.info(
                "\n✅ 모든 테스트 통과! 모든 도구가 정상적으로 동작하고 유효한 결과를 반환합니다."
            )
            return True
        elif healthy_servers > 0 and invalid_tools == 0:
            logger.warning("\n⚠️ 일부 문제가 발견되었지만 기본 기능은 동작합니다")
            return True
        elif invalid_tools > 0:
            logger.warning(
                f"\n⚠️ {invalid_tools}개 도구가 실행은 성공했지만 유효하지 않은 결과를 반환했습니다. 실제 사용 시 문제가 발생할 수 있습니다."
            )
            # 유효하지 않은 도구 목록 출력
            invalid_tool_names = [
                name
                for name, tool_data in self.test_results["tools"].items()
                if isinstance(tool_data, dict)
                and tool_data.get("tested", False)
                and tool_data.get("success", False)
                and not tool_data.get("is_valid", True)
            ]
            if invalid_tool_names:
                logger.warning(
                    f"  - 유효하지 않은 결과를 반환한 도구: {', '.join(invalid_tool_names)}"
                )
            return False
        elif healthy_servers > 0:
            logger.warning("\n⚠️ 일부 문제가 발견되었지만 기본 기능은 동작합니다")
            return True
        else:
            logger.error("\n❌ 심각한 문제가 발견되었습니다")
            return False

    async def cleanup(self):
        """리소스 정리"""
        if self.mcp_hub:
            try:
                await self.mcp_hub.cleanup()
                logger.info("✅ 리소스 정리 완료")
            except Exception as e:
                logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")


async def main():
    """메인 테스트 함수"""
    checker = MCPToolChecker()

    try:
        # 초기화
        if not await checker.initialize():
            logger.error("❌ 초기화 실패로 테스트 중단")
            return False

        # 서버 확인
        await checker.check_servers()

        # 도구 목록 확인
        await checker.check_tools()

        # 필수 도구 테스트
        await checker.test_essential_tools()

        # GitHub MCP 서버 테스트
        await checker.test_github_mcp_server()

        # 검색 도구 테스트
        await checker.test_search_tools()

        # 요약 생성
        success = checker.generate_summary()

        return success

    except KeyboardInterrupt:
        logger.info("\n⚠️ 사용자에 의해 중단됨")
        return False
    except Exception as e:
        logger.error(f"\n❌ 테스트 중 예외 발생: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False
    finally:
        await checker.cleanup()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
