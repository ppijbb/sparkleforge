"""Git tool dispatch (ToolCategory.GIT): repository operations via GitWorkflow."""
import logging
import time
from typing import Any, Dict

from src.core.tools.registry import ToolResult

logger = logging.getLogger(__name__)


async def _execute_git_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """Git 워크플로우 도구 실행."""
    start_time = time.time()

    try:
        from pathlib import Path

        from src.core.git_workflow import GitWorkflow

        # 저장소 경로 확인
        repo_path = parameters.get("repo_path")
        if repo_path:
            repo_path = Path(repo_path)
        else:
            repo_path = None

        # GitWorkflow 생성
        git_workflow = GitWorkflow(repo_path=repo_path)

        if tool_name == "git_status":
            result = await git_workflow.git_status()
            return ToolResult(
                success=True,
                data=result,
                execution_time=time.time() - start_time,
                confidence=0.9,
            )

        elif tool_name == "git_commit":
            message = parameters.get("message")
            auto_stage = parameters.get("auto_stage", True)
            result = await git_workflow.git_commit(message=message, auto_stage=auto_stage)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0,
            )

        elif tool_name == "git_push":
            branch = parameters.get("branch")
            force = parameters.get("force", False)
            result = await git_workflow.git_push(branch=branch, force=force)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0,
            )

        elif tool_name == "git_create_pr":
            title = parameters.get("title")
            body = parameters.get("body")
            base = parameters.get("base", "main")

            if not title:
                return ToolResult(
                    success=False,
                    data=None,
                    error="title parameter is required",
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                )

            result = await git_workflow.git_create_pr(title=title, body=body, base=base)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0,
            )

        elif tool_name == "git_commit_push_pr":
            commit_message = parameters.get("commit_message")
            pr_title = parameters.get("pr_title")
            pr_body = parameters.get("pr_body")
            base = parameters.get("base", "main")

            result = await git_workflow.git_commit_push_pr(
                commit_message=commit_message,
                pr_title=pr_title,
                pr_body=pr_body,
                base=base,
            )
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
                error=f"Unknown git tool: {tool_name}",
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

    except Exception as e:
        logger.error(f"Git tool execution failed: {tool_name} - {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=f"Git tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0,
        )
