"""Code execution tool dispatch (ToolCategory.CODE): sandboxed code execution via remote backend or Docker/gVisor."""
import asyncio
import logging
import time
from typing import Any, Dict

from src.core.tools.registry import ToolResult

logger = logging.getLogger(__name__)


def _execute_code_tool_sync(tool_name: str, parameters: Dict[str, Any]) -> str:
    """동기 버전 - LangChain Tool에서 호출."""
    try:
        import concurrent.futures

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _execute_code_tool(tool_name, parameters))
                    # timeout 설정으로 무한 대기 방지
                    result = future.result(timeout=300)  # 최대 5분
            else:
                result = loop.run_until_complete(_execute_code_tool(tool_name, parameters))
        except RuntimeError:
            result = asyncio.run(_execute_code_tool(tool_name, parameters))

        if result.success:
            import json

            return json.dumps(result.data, ensure_ascii=False, indent=2)
        else:
            raise RuntimeError(result.error or "Tool execution failed")
    except Exception as e:
        raise RuntimeError(f"Tool execution failed: {str(e)}")



async def _execute_code_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """실제 코드 도구 실행 - Docker/gVisor 샌드박스 우선."""
    start_time = time.time()
    code = parameters.get("code", "")
    language = parameters.get("language", "python")
    sandbox_type = str(parameters.get("sandbox", "docker")).lower()

    # 1. 리소스 제한 체크
    try:
        from src.core.resource_limits import ResourceLimits

        code_bytes = code.encode("utf-8")
        if ResourceLimits.exceeds_code_limit(len(code_bytes)):
            error_msg = (
                f"Code size ({ResourceLimits.format_bytes(len(code_bytes))}) exceeds limit "
                f"({ResourceLimits.MAX_CODE_SIZE_HUMAN}). "
                f"Please reduce the code size or split into smaller chunks."
            )
            logger.error(error_msg)
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
                execution_time=time.time() - start_time,
                confidence=0.0,
            )
    except ImportError:
        # ResourceLimits 모듈이 없으면 경고만 하고 계속 진행
        logger.debug("ResourceLimits module not available, skipping size check")

    # 2. Optional remote sandbox (Runloop/Daytona/Modal) from env SANDBOX_BACKEND
    import os

    backend_name = (os.getenv("SANDBOX_BACKEND") or "").strip().lower()
    if backend_name in ("runloop", "daytona", "modal"):
        try:
            from src.core.sandbox.factory import get_sandbox_backend

            backend = get_sandbox_backend()
            if backend is not None:
                resp = await backend.execute_code(code, language)
                execution_time = time.time() - start_time
                return ToolResult(
                    success=resp.exit_code == 0,
                    data={
                        "code": code,
                        "language": language,
                        "output": resp.output,
                        "error": resp.error,
                        "exit_code": resp.exit_code,
                        "sandbox_type": backend.id,
                    },
                    error=resp.error,
                    execution_time=execution_time,
                    confidence=0.9 if resp.exit_code == 0 else 0.5,
                )
        except Exception as e:
            logger.debug("Remote sandbox (%s) failed, falling back: %s", backend_name, e)

    # 3. Docker/gVisor 샌드박스 사용 (기본값)
    if sandbox_type in ("docker", "gvisor", "runsc", "container"):
        # Docker 샌드박스 사용
        try:
            from src.core.sandbox.docker_sandbox import get_sandbox

            sandbox = get_sandbox()

            result = await sandbox.execute_code(code, language)
            execution_time = time.time() - start_time

            return ToolResult(
                success=result.success,
                data={
                    "code": code,
                    "language": language,
                    "output": result.output,
                    "error": result.error,
                    "exit_code": result.exit_code,
                    "sandbox_type": "docker",
                    "container_id": result.container_id,
                },
                error=result.error if not result.success else None,
                execution_time=execution_time,
                confidence=0.9 if result.success else 0.5,
            )

        except Exception as e:
            logger.error(f"Docker sandbox execution failed: {e}")
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                data=None,
                error=f"Docker sandbox failed: {str(e)}",
                execution_time=execution_time,
                confidence=0.0,
            )

    if sandbox_type not in ("docker", "gvisor", "runsc", "container"):
        return ToolResult(
            success=False,
            data=None,
            error=(f"Unsupported sandbox '{sandbox_type}'. " "Use 'docker' or 'runsc'."),
            execution_time=time.time() - start_time,
            confidence=0.0,
        )
