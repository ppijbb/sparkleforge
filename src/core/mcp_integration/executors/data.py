"""Data tool dispatch (ToolCategory.DATA): fetch/filesystem/browser/shell delegation."""
import asyncio
import logging
import time
from typing import Any, Dict

from src.core.mcp_integration.executors.browser import _execute_browser_tool
from src.core.mcp_integration.executors.file import _execute_file_tool
from src.core.mcp_integration.executors.shell import _execute_shell_tool
from src.core.tools.registry import ToolResult

logger = logging.getLogger(__name__)

def _execute_data_tool_sync(tool_name: str, parameters: Dict[str, Any]) -> str:
    """동기 버전 - LangChain Tool에서 호출."""
    try:
        import concurrent.futures

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _execute_data_tool(tool_name, parameters))
                    # timeout 설정으로 무한 대기 방지
                    result = future.result(timeout=300)  # 최대 5분
            else:
                result = loop.run_until_complete(_execute_data_tool(tool_name, parameters))
        except RuntimeError:
            result = asyncio.run(_execute_data_tool(tool_name, parameters))

        if result.success:
            import json

            return json.dumps(result.data, ensure_ascii=False, indent=2)
        else:
            raise RuntimeError(result.error or "Tool execution failed")
    except Exception as e:
        raise RuntimeError(f"Tool execution failed: {str(e)}")



async def _execute_data_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """실제 데이터 도구 실행."""
    start_time = time.time()

    try:
        if tool_name == "fetch":
            # src/utils에서 직접 사용
            try:
                from src.utils.web_utils import fetch_url

                url = parameters.get("url", "")
                max_length = parameters.get("max_length", 50000)
                timeout = parameters.get("timeout", 30)

                if not url:
                    raise ValueError("URL parameter is required for fetch tool")

                # src/utils의 fetch_url 직접 호출
                result = await fetch_url(url, max_length, timeout)

                if result.get("success"):
                    return ToolResult(
                        success=True,
                        data={
                            "url": url,
                            "content": result.get("content", ""),
                            "content_type": result.get("content_type", "unknown"),
                            "status_code": result.get("status_code", 200),
                            "character_count": result.get("character_count", 0),
                            "source": "embedded_fetch",
                        },
                        execution_time=time.time() - start_time,
                        confidence=0.9,
                    )
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=result.get("error", "Fetch failed"),
                        execution_time=time.time() - start_time,
                        confidence=0.0,
                    )
            except ImportError:
                logger.debug("src.utils.web_utils not available, using existing logic")
            except Exception as e:
                logger.warning(f"Embedded fetch failed: {e}, falling back to existing logic")

            # 기존 로직 (fallback)
            url = parameters.get("url", "")
            if not url:
                raise ValueError("URL parameter is required for fetch tool")

            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()

                return ToolResult(
                    success=True,
                    data={
                        "url": url,
                        "status": response.status_code,
                        "content": response.text[:10000],  # 처음 10000자만
                        "content_length": len(response.text),
                        "headers": dict(response.headers),
                    },
                    execution_time=time.time() - start_time,
                    confidence=0.9,
                )

        elif tool_name == "filesystem":
            # 파일시스템 접근 (실제 구현)
            return await _execute_file_tool(tool_name, parameters)

        elif tool_name == "browser":
            # 브라우저 자동화 (실제 구현)
            return await _execute_browser_tool(tool_name, parameters)

        elif tool_name == "shell":
            # 쉘 명령 실행 (실제 구현)
            return await _execute_shell_tool(tool_name, parameters)

        else:
            raise ValueError(f"Unknown data tool: {tool_name}")

    except Exception as e:
        logger.error(f"Data tool execution failed: {tool_name} - {e}")
        return ToolResult(
            success=False,
            data=None,
            error=f"Data tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0,
        )
