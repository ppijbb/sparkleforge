"""Tool registration/discovery mixin for UniversalMCPHub: builds LangChain tool wrappers, loads manual tool configs, merges auto-discovered tools."""
import asyncio
import json
import logging
from typing import Any, Dict, List

from src.core.mcp_auto_discovery import FastMCPMulti
from src.core.mcp_integration.executors.academic import _execute_academic_tool_sync
from src.core.mcp_integration.executors.code import _execute_code_tool_sync
from src.core.mcp_integration.executors.data import _execute_data_tool_sync
from src.core.mcp_integration.executors.search import _execute_search_tool_sync
from src.core.mcp_integration.mcp_runtime import (
    BaseTool,
    LANGCHAIN_AVAILABLE,
    StructuredTool,
    project_root,
)
from src.core.mcp_integration.parser import _structured_tool_description
from src.core.mcp_tool_loader import MCPToolLoader
from src.core.tools.registry import ToolCategory, ToolInfo

logger = logging.getLogger(__name__)

class ToolRegistrationMixin:
    def _load_tools_config(self):
        """tools_config.json에서 Tool 메타데이터 로드."""
        # configs 폴더에서 로드 시도 (우선)
        tools_config_file = project_root / "configs" / "tools_config.json"
        if not tools_config_file.exists():
            # 하위 호환성: 루트에서도 시도
            tools_config_file = project_root / "tools_config.json"

        if tools_config_file.exists():
            try:
                with open(tools_config_file, encoding="utf-8") as f:
                    self.tools_config = json.load(f)
                logger.info(f"✅ Loaded tools config from {tools_config_file}")
            except Exception as e:
                logger.warning(f"Failed to load tools config: {e}")
                self.tools_config = {}
        else:
            logger.warning(f"tools_config.json not found at {tools_config_file}")
            self.tools_config = {}
    def _create_langchain_tool_wrapper(
        self, tool_name: str, tool_config: Dict[str, Any]
    ) -> BaseTool | None:
        """tools_config.json의 설정을 기반으로 LangChain Tool 래퍼 생성.

        Args:
            tool_name: Tool 이름
            tool_config: tools_config.json에서 로드된 Tool 설정

        Returns:
            LangChain BaseTool 인스턴스 또는 None
        """
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, cannot create tool wrapper")
            return None

        try:
            # 카테고리 매핑
            category_map = {
                "search": ToolCategory.SEARCH,
                "data": ToolCategory.DATA,
                "code": ToolCategory.CODE,
                "academic": ToolCategory.ACADEMIC,
                "business": ToolCategory.BUSINESS,
                "utility": ToolCategory.UTILITY,
                "browser": ToolCategory.BROWSER,
                "document": ToolCategory.DOCUMENT,
                "file": ToolCategory.FILE,
            }

            category_str = tool_config.get("category", "utility")
            category_map.get(category_str, ToolCategory.UTILITY)
            description = _structured_tool_description(tool_config, tool_name)
            tool_config.get("parameters", {})

            # Pydantic 스키마 생성 - 최신 방식으로 단순화 (args_schema 없이도 동작)
            ToolSchema = None
            # LangChain StructuredTool은 args_schema 없이도 함수 시그니처에서 자동으로 파라미터를 추론함
            # 복잡한 동적 스키마 생성을 피하고 함수 파라미터로 처리

            # Tool 실행 함수 선택 (동기 래퍼 생성) - 함수 시그니처 명시
            def create_sync_func(tool_name_str, func_type):
                """동기 함수 래퍼 생성 - 명시적 함수 시그니처로 LangChain이 파라미터 추론."""
                if func_type == "search":

                    def search_wrapper(
                        query: str,
                        max_results: int = 10,
                        num_results: int = 10,
                        format: str = "detailed",
                    ) -> str:
                        params = {"query": query, "format": format}
                        if max_results:
                            params["max_results"] = max_results
                        elif num_results:
                            params["max_results"] = num_results
                        return _execute_search_tool_sync(tool_name_str, params)

                    return search_wrapper
                elif func_type == "academic":

                    def academic_wrapper(
                        query: str, max_results: int = 10, num_results: int = 10
                    ) -> str:
                        params = {"query": query}
                        if max_results:
                            params["max_results"] = max_results
                        elif num_results:
                            params["max_results"] = num_results
                        return _execute_academic_tool_sync(tool_name_str, params)

                    return academic_wrapper
                elif func_type == "data":
                    if tool_name_str == "fetch":

                        def fetch_wrapper(url: str, format: str = "detailed") -> str:
                            return _execute_data_tool_sync("fetch", {"url": url, "format": format})

                        return fetch_wrapper
                    elif tool_name_str == "filesystem":

                        def filesystem_wrapper(path: str, operation: str = "read") -> str:
                            return _execute_data_tool_sync(
                                "filesystem", {"path": path, "operation": operation}
                            )

                        return filesystem_wrapper
                    else:

                        def data_wrapper(**kwargs) -> str:
                            return _execute_data_tool_sync(tool_name_str, kwargs)

                        return data_wrapper
                elif func_type == "code":
                    if "interpreter" in tool_name_str.lower():

                        def code_wrapper(code: str, language: str = "python") -> str:
                            return _execute_code_tool_sync(
                                tool_name_str, {"code": code, "language": language}
                            )

                        return code_wrapper
                    else:

                        def code_wrapper(code: str) -> str:
                            return _execute_code_tool_sync(tool_name_str, {"code": code})

                        return code_wrapper
                else:
                    return None

            # Tool별 실행 함수 매핑
            func = None
            category_str = tool_config.get("category", "utility")

            if tool_name == "g-search":
                func = create_sync_func("g-search", "search")
            elif tool_name == "fetch":
                func = create_sync_func("fetch", "data")
            elif tool_name == "filesystem":
                func = create_sync_func("filesystem", "data")
            elif tool_name == "python_coder":
                func = create_sync_func("python_coder", "code")
            elif tool_name == "code_interpreter":
                func = create_sync_func("code_interpreter", "code")
            elif tool_name == "arxiv":
                func = create_sync_func("arxiv", "academic")
            elif tool_name == "scholar":
                func = create_sync_func("scholar", "academic")
            else:
                # 카테고리 기반으로 자동 선택 시도
                if category_str == "search":
                    func = create_sync_func(tool_name, "search")
                elif category_str == "data":
                    func = create_sync_func(tool_name, "data")
                elif category_str == "code":
                    func = create_sync_func(tool_name, "code")
                elif category_str == "academic":
                    func = create_sync_func(tool_name, "academic")
                elif category_str == "business":
                    # 전용 business executor가 없으면 일반 검색으로 graceful fallback
                    func = create_sync_func("g-search", "search")

            if func is None:
                logger.warning(
                    f"No execution function for tool: {tool_name}, category: {category_str}"
                )

                # 실행 함수가 없어도 기본 래퍼 함수 생성
                def generic_executor(**kwargs):
                    """Generic executor when specific function not available."""
                    raise RuntimeError(
                        f"Tool {tool_name} execution not implemented yet. Please configure execution function."
                    )

                func = generic_executor

            # StructuredTool 생성 - args_schema 없이도 생성 가능하도록
            try:
                if StructuredTool and ToolSchema:
                    langchain_tool = StructuredTool.from_function(
                        func=func,
                        name=tool_name,
                        description=description,
                        args_schema=ToolSchema,
                    )
                elif StructuredTool:
                    # args_schema 없이 생성 (파라미터는 함수 시그니처에서 자동 추론)
                    langchain_tool = StructuredTool.from_function(
                        func=func, name=tool_name, description=description
                    )
                else:
                    return None

                logger.info(f"✅ Created LangChain tool wrapper for {tool_name}")
                return langchain_tool
            except Exception as schema_error:
                # Schema 생성 실패 시 args_schema 없이 재시도
                logger.warning(
                    f"Failed to create tool with schema for {tool_name}: {schema_error}, trying without schema"
                )
                try:
                    if StructuredTool:
                        langchain_tool = StructuredTool.from_function(
                            func=func, name=tool_name, description=description
                        )
                        logger.info(
                            f"✅ Created LangChain tool wrapper for {tool_name} (without schema)"
                        )
                        return langchain_tool
                except Exception as e2:
                    logger.error(f"Failed to create tool without schema for {tool_name}: {e2}")
                    return None

        except Exception as e:
            logger.error(f"Failed to create LangChain tool wrapper for {tool_name}: {e}")
            return None
    def _initialize_tools(self):
        """도구 초기화 - tools_config.json 기반 + FastMCP 자동 발견."""
        # 1. 수동 등록 도구 초기화
        self._initialize_manual_tools()

        # 2. FastMCP 자동 발견 도구 초기화 (비동기)
        # 이미 실행 중인 이벤트 루프가 있으면 태스크로 실행, 없으면 새로 생성
        try:
            # 실행 중인 이벤트 루프 확인
            try:
                loop = asyncio.get_running_loop()
                # 이미 실행 중인 루프가 있으면 태스크로 실행 (asyncio.run() 사용 금지)
                # 태스크를 생성하지만 await하지 않음 (백그라운드 실행)
                loop.create_task(self._initialize_auto_discovered_tools())
                # 태스크가 완료될 때까지 기다리지 않음 (비동기 초기화)
                logger.debug("Auto-discovered MCP tools initialization started as background task")
            except RuntimeError:
                # 실행 중인 루프가 없으면 새 루프에서 실행
                asyncio.run(self._initialize_auto_discovered_tools())
        except Exception as e:
            logger.warning(f"Failed to initialize auto-discovered MCP tools: {e}")
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")
            # 자동 발견 실패 시에도 계속 진행

        # 3. 도구 통합 및 충돌 해결
        self._merge_tools()
    def _initialize_manual_tools(self):
        """수동 등록 도구 초기화 - tools_config.json 기반."""
        local_tools = self.tools_config.get("local_tools", {})

        for tool_name, tool_config in local_tools.items():
            # MCP 전용 Tool은 건너뛰기 (MCP 서버에서 동적 등록됨)
            if tool_config.get("implementation") == "mcp_only":
                continue

            # 카테고리 매핑
            category_map = {
                "search": ToolCategory.SEARCH,
                "data": ToolCategory.DATA,
                "code": ToolCategory.CODE,
                "academic": ToolCategory.ACADEMIC,
                "business": ToolCategory.BUSINESS,
                "utility": ToolCategory.UTILITY,
                "browser": ToolCategory.BROWSER,
                "document": ToolCategory.DOCUMENT,
                "file": ToolCategory.FILE,
            }

            category_str = tool_config.get("category", "utility")
            category = category_map.get(category_str, ToolCategory.UTILITY)
            description = tool_config.get("description", f"{tool_name} tool")

            # ToolInfo 생성
            tool_info = ToolInfo(
                name=tool_name,
                category=category,
                description=description,
                parameters=tool_config.get("parameters", {}),
                mcp_server=tool_config.get("mcp_server_name", ""),
            )

            # LangChain Tool 래퍼 생성
            langchain_tool = self._create_langchain_tool_wrapper(tool_name, tool_config)

            if langchain_tool:
                # Registry에 등록
                self.registry.register_local_tool(tool_info, langchain_tool)
                # 하위 호환성을 위해 self.tools에도 추가
                self.tools[tool_name] = tool_info
                logger.info(f"✅ Registered local tool: {tool_name}")
            else:
                # LangChain wrapper 생성 실패해도 기본 ToolInfo는 등록 (나중에 실행 시도 가능)
                logger.warning(
                    f"⚠️ Failed to create LangChain wrapper for {tool_name}, registering without wrapper"
                )
                self.registry.tools[tool_name] = tool_info
                self.tools[tool_name] = tool_info

        # ========================================================================
        # NATIVE TOOLS REGISTRATION (Overrides/Fallbacks)
        # ========================================================================
        try:
            from langchain_core.tools import Tool

            from src.core.tools.native_search import search_duckduckgo_json

            native_tool_name = "ddg_search"
            logger.info(f"🛠️ Registering Native Tool: {native_tool_name}")

            native_tool_info = ToolInfo(
                name=native_tool_name,
                category=ToolCategory.SEARCH,
                description="Robust native DuckDuckGo search (No MCP required)",
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results"},
                },
                mcp_server="",
            )

            def native_search_wrapper(query: str, max_results: int = 5):
                # Handler for both string input (query only) and structured input
                if isinstance(query, dict):
                    q = query.get("query", "")
                    m = query.get("max_results", 5)
                    return search_duckduckgo_json(q, m)
                return search_duckduckgo_json(query, max_results)

            native_langchain_tool = Tool(
                name=native_tool_name,
                func=native_search_wrapper,
                description="Search DuckDuckGo natively",
            )

            self.registry.register_local_tool(native_tool_info, native_langchain_tool)
            self.tools[native_tool_name] = native_tool_info

            # Also alias 'search' and 'g-search' to this if not present
            for alias in ["search", "g-search"]:
                if alias not in self.tools and alias not in self.registry.tools:
                    self.tools[alias] = native_tool_info
                    self.registry.register_local_tool(
                        native_tool_info, native_langchain_tool
                    )  # Re-registering with same object might verify alias support? No, straightforward.
                    logger.info(f"✅ Aliased '{alias}' to native {native_tool_name}")

        except Exception as e:
            logger.error(f"❌ Failed to register native tools: {e}")

        # ========================================================================
        # GIT TOOLS REGISTRATION
        # ========================================================================
        try:
            from langchain_core.tools import Tool

            git_tools = [
                {
                    "name": "git_status",
                    "description": "Check Git repository status (branch, staged/unstaged files)",
                    "parameters": {
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path (optional, defaults to current directory)",
                        }
                    },
                },
                {
                    "name": "git_commit",
                    "description": "Create a Git commit with automatic message generation",
                    "parameters": {
                        "message": {
                            "type": "string",
                            "description": "Commit message (optional, auto-generated if not provided)",
                        },
                        "auto_stage": {
                            "type": "boolean",
                            "description": "Automatically stage files (default: true)",
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path (optional)",
                        },
                    },
                },
                {
                    "name": "git_push",
                    "description": "Push Git branch to remote repository",
                    "parameters": {
                        "branch": {
                            "type": "string",
                            "description": "Branch to push (optional, defaults to current branch)",
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force push (default: false)",
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path (optional)",
                        },
                    },
                },
                {
                    "name": "git_create_pr",
                    "description": "Create a Pull Request using GitHub CLI",
                    "parameters": {
                        "title": {
                            "type": "string",
                            "description": "PR title (required)",
                        },
                        "body": {"type": "string", "description": "PR body (optional)"},
                        "base": {
                            "type": "string",
                            "description": "Base branch (default: main)",
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path (optional)",
                        },
                    },
                },
                {
                    "name": "git_commit_push_pr",
                    "description": "Complete workflow: commit, push, and create PR in one step",
                    "parameters": {
                        "commit_message": {
                            "type": "string",
                            "description": "Commit message (optional, auto-generated if not provided)",
                        },
                        "pr_title": {
                            "type": "string",
                            "description": "PR title (optional, uses commit message if not provided)",
                        },
                        "pr_body": {
                            "type": "string",
                            "description": "PR body (optional, auto-generated if not provided)",
                        },
                        "base": {
                            "type": "string",
                            "description": "Base branch (default: main)",
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path (optional)",
                        },
                    },
                },
            ]

            for git_tool_config in git_tools:
                tool_name = git_tool_config["name"]
                logger.info(f"🛠️ Registering Git Tool: {tool_name}")

                tool_info = ToolInfo(
                    name=tool_name,
                    category=ToolCategory.GIT,
                    description=git_tool_config["description"],
                    parameters=git_tool_config["parameters"],
                    mcp_server="",
                )

                # LangChain Tool 래퍼 생성
                def create_git_tool_wrapper(tool_name: str):
                    async def git_tool_wrapper(**kwargs):
                        from src.core.mcp_integration import (
                            _execute_git_tool,
                        )

                        result = await _execute_git_tool(tool_name, kwargs)
                        if result.success:
                            return (
                                result.data
                                if isinstance(result.data, dict)
                                else {"result": result.data}
                            )
                        else:
                            return {"error": result.error}

                    return git_tool_wrapper

                # 동기 래퍼 (LangChain Tool은 동기 함수를 기대)
                def sync_git_wrapper(tool_name: str):
                    def wrapper(**kwargs):
                        import asyncio

                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # 이미 실행 중인 루프가 있으면 새 태스크 생성
                                import concurrent.futures

                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(
                                        asyncio.run,
                                        create_git_tool_wrapper(tool_name)(**kwargs),
                                    )
                                    return future.result()
                            else:
                                return loop.run_until_complete(
                                    create_git_tool_wrapper(tool_name)(**kwargs)
                                )
                        except RuntimeError:
                            return asyncio.run(create_git_tool_wrapper(tool_name)(**kwargs))

                    return wrapper

                langchain_tool = Tool(
                    name=tool_name,
                    func=sync_git_wrapper(tool_name),
                    description=git_tool_config["description"],
                )

                self.registry.register_local_tool(tool_info, langchain_tool)
                self.tools[tool_name] = tool_info
                logger.info(f"✅ Registered Git tool: {tool_name}")

        except Exception as e:
            logger.error(f"❌ Failed to register Git tools: {e}", exc_info=True)

        # Registry의 tools를 self.tools와 동기화
        self.tools.update(self.registry.tools)

        logger.info(
            f"✅ Initialized {len(self.registry.tools)} tools in registry ({len(self.registry.langchain_tools)} with LangChain wrappers)"
        )
    async def _initialize_auto_discovered_tools(self):
        """FastMCP를 통한 자동 발견 도구 초기화."""
        # FastMCP 서버 설정 초기화
        self._initialize_fastmcp_servers()

        if not self.fastmcp_servers:
            logger.info("No FastMCP servers configured for auto-discovery")
            return

        # FastMCP 클라이언트 초기화
        self.fastmcp_multi = FastMCPMulti(self.fastmcp_servers)
        self.fastmcp_tool_loader = MCPToolLoader(self.fastmcp_multi)

        try:
            # 도구 자동 발견
            discovered_tools = await self.fastmcp_tool_loader.get_all_tools()
            tool_infos = await self.fastmcp_tool_loader.list_tool_info()

            # 발견된 도구 저장
            for tool, info in zip(discovered_tools, tool_infos):
                tool_name = info.name
                self.auto_discovered_tools[tool_name] = tool
                self.auto_discovered_tool_infos[tool_name] = info

            logger.info(
                f"Auto-discovered {len(discovered_tools)} tools from {len(self.fastmcp_servers)} FastMCP servers"
            )

        except Exception as e:
            logger.error(f"Failed to auto-discover MCP tools: {e}")
            raise
    def _initialize_fastmcp_servers(self):
        """환경 변수 및 구성에서 FastMCP 서버 설정 초기화."""
        # 환경 변수에서 서버 설정 로드 (예: FASTMCP_SERVERS)
        # 실제로는 config나 환경 변수에서 로드해야 함
        # 여기서는 예시로 빈 설정 유지
    def _merge_tools(self):
        """자동 발견 도구와 수동 등록 도구 통합 및 충돌 해결."""
        # 자동 발견된 도구들을 ToolRegistry에 통합
        for tool_name, tool in self.auto_discovered_tools.items():
            tool_info = self.auto_discovered_tool_infos[tool_name]

            # 카테고리 매핑 (MCP ToolInfo -> 기존 ToolCategory)
            category_map = {
                "search": ToolCategory.SEARCH,
                "data": ToolCategory.DATA,
                "code": ToolCategory.CODE,
                "academic": ToolCategory.ACADEMIC,
                "business": ToolCategory.BUSINESS,
                "utility": ToolCategory.UTILITY,
                "browser": ToolCategory.BROWSER,
                "document": ToolCategory.DOCUMENT,
                "file": ToolCategory.FILE,
            }

            # 도구 설명에서 카테고리 추론 (단순 키워드 기반)
            description_lower = tool_info.description.lower()
            category = ToolCategory.UTILITY  # 기본값
            for keyword, cat in category_map.items():
                if keyword in description_lower:
                    category = cat
                    break

            # 기존 ToolInfo 형식으로 변환
            legacy_tool_info = ToolInfo(
                name=tool_name,
                category=category,
                description=tool_info.description,
                parameters={},  # MCP 스키마는 별도 처리
                mcp_server=tool_info.server_guess,
            )

            # 이름 충돌 확인
            if tool_name in self.registry.tools:
                # 충돌 시 자동 발견 도구 우선 (mcp_auto_ 접두사로 구분)
                auto_tool_name = f"mcp_auto_{tool_name}"
                logger.warning(
                    f"Tool name conflict: '{tool_name}' already exists. Using '{auto_tool_name}' for auto-discovered tool."
                )
                legacy_tool_info.name = auto_tool_name

            # Registry에 등록
            self.registry.register_local_tool(legacy_tool_info, tool)

        # Registry와 self.tools 동기화
        self.tools.update(self.registry.tools)
        logger.info(f"✅ Merged tools: {len(self.registry.tools)} total tools in registry")
    def _initialize_clients(self):
        """클라이언트 초기화 - Gemini 직결 사용, OpenRouter 비활성화."""
        self.openrouter_client = None
        logger.info("✅ LLM routed via llm_manager (Gemini direct). OpenRouter disabled.")
    def get_all_langchain_tools(self) -> List[BaseTool]:
        """모든 LangChain Tool 리스트 반환."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available")
            return []
        return self.registry.get_all_langchain_tools()
