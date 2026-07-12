"""Shell tool dispatch (ToolCategory.SHELL): sandboxed command execution via ShellExecutor."""
import logging
import time
from typing import Any, Dict

from src.core.tools.registry import ToolResult

logger = logging.getLogger(__name__)


async def _execute_shell_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """Shell 명령 실행 도구 (완전 자동형 SparkleForge)."""
    start_time = time.time()

    try:
        from pathlib import Path

        from src.core.shell_executor import ShellExecutor

        # 작업 디렉토리 확인
        working_dir = parameters.get("working_dir")
        if working_dir:
            working_dir = Path(working_dir)
        else:
            working_dir = None

        # ShellExecutor 생성
        executor = ShellExecutor(
            require_confirmation=parameters.get("require_confirmation", False),
            max_execution_time=parameters.get("timeout", 300),
        )

        if tool_name == "run_shell_command":
            command = parameters.get("command")
            if not command:
                return ToolResult(
                    success=False,
                    data=None,
                    error="command parameter is required",
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                )

            confirm = parameters.get("confirm")
            timeout = parameters.get("timeout")
            result = await executor.run(
                command=command,
                working_dir=working_dir,
                confirm=confirm,
                timeout=timeout,
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0,
            )

        elif tool_name == "run_interactive_command":
            command = parameters.get("command")
            if not command:
                return ToolResult(
                    success=False,
                    data=None,
                    error="command parameter is required",
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                )

            input_data = parameters.get("input")
            result = await executor.run_interactive(
                command=command, working_dir=working_dir, input_data=input_data
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0,
            )

        elif tool_name == "run_background_command":
            command = parameters.get("command")
            if not command:
                return ToolResult(
                    success=False,
                    data=None,
                    error="command parameter is required",
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                )

            result = await executor.run_background(command=command, working_dir=working_dir)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0,
            )

        else:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown shell tool: {tool_name}",
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

    except Exception as e:
        logger.error(f"Shell tool execution failed: {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=str(e),
            execution_time=time.time() - start_time,
            confidence=0.0,
        )
