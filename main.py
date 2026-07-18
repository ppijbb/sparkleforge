#!/usr/bin/env python3
"""Autonomous Multi-Agent Research System - Main Entry Point
Implements 9 Core Innovations: Production-Grade Reliability, Universal MCP Hub, Streaming Pipeline

MCP agent 라이브러리 기반의 자율 리서처 시스템.
모든 하드코딩, fallback, mock 코드를 제거하고 실제 MCP agent를 사용.

현재 상태: Production Level 개발 진행 중 🚧

Usage:
    python main.py --request "연구 주제"                    # CLI 모드
    python main.py --web                                    # 웹 모드
    python main.py --mcp-server                            # MCP 서버 모드
    python main.py --streaming                             # 스트리밍 모드
"""

import argparse
import asyncio
import logging
import logging.handlers
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.researcher_config import load_config_from_env

# CRITICAL: Load configuration BEFORE importing any modules that depend on it
config = load_config_from_env()

# Configure logging for production-grade reliability
# Advanced logging setup: setup logger manually to ensure logs directory exists and avoid issues with logging.basicConfig (per best practices)
log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "researcher.log"

# Streamlit 경고 필터링 (CLI 모드에서 streamlit이 import될 때 발생하는 경고 무시)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)
logging.getLogger("streamlit").setLevel(logging.ERROR)


# HTTP 에러 메시지 필터링 클래스
class HTTPErrorFilter(logging.Filter):
    """HTML 에러 응답을 필터링하여 간단한 메시지만 출력"""

    def filter(self, record):
        message = record.getMessage()

        # HTML 에러 페이지 감지 및 필터링
        if "<!DOCTYPE html>" in message or "<html" in message.lower():
            # HTML에서 에러 메시지 추출 시도
            import re

            # HTTP 상태 코드 추출
            status_match = re.search(r"HTTP (\d{3})", message)
            status_code = status_match.group(1) if status_match else "Unknown"

            # 에러 제목 추출 시도
            title_match = re.search(r"<title>([^<]+)</title>", message, re.IGNORECASE)
            error_title = title_match.group(1).strip() if title_match else None

            # 간단한 에러 메시지 생성
            if error_title:
                record.msg = f"HTTP {status_code}: {error_title}"
            else:
                # 상태 코드에 따른 기본 메시지
                if status_code == "502":
                    record.msg = f"HTTP {status_code}: Bad Gateway - Server temporarily unavailable"
                elif status_code == "504":
                    record.msg = (
                        f"HTTP {status_code}: Gateway Timeout - Server response timeout"
                    )
                elif status_code == "503":
                    record.msg = f"HTTP {status_code}: Service Unavailable - Server temporarily unavailable"
                elif status_code == "401":
                    record.msg = (
                        f"HTTP {status_code}: Unauthorized - Authentication failed"
                    )
                elif status_code == "404":
                    record.msg = f"HTTP {status_code}: Not Found"
                elif status_code == "500":
                    record.msg = f"HTTP {status_code}: Internal Server Error"
                else:
                    record.msg = f"HTTP {status_code}: Server Error"

            record.args = ()  # args 초기화

        return True


from src.cli.cli_result import cli_result_succeeded, extract_cli_result_content
from src.core.autonomous_research_system import AutonomousResearchSystem
from src.cli.main_commands import (
    handle_actions_command,
    handle_approve_command,
    handle_cli_command,
    handle_deny_command,
    handle_docker_command,
    handle_health_command,
    handle_interactive_command,
    handle_mcp_command,
    handle_nightwelding_command,
    handle_run_command,
    handle_session_command,
    handle_setup_command,
    handle_tools_command,
    handle_web_command,
    handle_work_command,
    handle_report_command,
)


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicate logs on reloads
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# File handler (rotating: 10MB per file, 5 backups — matches src/utils/logger.py convention)
file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
console_handler.addFilter(HTTPErrorFilter())  # HTTP 에러 필터 추가
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# ============================================================================
# MCP / Runner / HTTP 관련 로거 억제 (과도한 로그 출력 방지)
# ============================================================================
# FastMCP Runner 로거 - WARNING 이상만 출력
runner_logger = logging.getLogger("Runner")
runner_logger.setLevel(logging.WARNING)
runner_logger.propagate = False  # 상위로 전파 차단

# MCP 관련 로거들
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("mcp.client").setLevel(logging.WARNING)
logging.getLogger("mcp.server").setLevel(logging.WARNING)

# HTTP 클라이언트 로거들
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# asyncio 관련 로거
logging.getLogger("asyncio").setLevel(logging.WARNING)


async def main():
    """Main function - 9가지 핵심 혁신 통합 실행 진입점 (Suna-style CLI)"""

    # Python 종료 시 발생하는 async generator 정리 오류 무시
    def ignore_async_gen_errors(loop, context):
        """Anyio cancel scope 및 async generator 종료 오류 무시"""
        exception = context.get("exception")
        if exception:
            error_msg = str(exception)
            # anyio cancel scope 오류는 무시
            if isinstance(exception, RuntimeError) and (
                "cancel scope" in error_msg.lower()
                or "different task" in error_msg.lower()
            ):
                return  # 무시
            # async generator 종료 오류는 무시
            if (
                isinstance(exception, GeneratorExit)
                or "async_generator" in error_msg.lower()
            ):
                return  # 무시
        # 기타 오류는 기본 handler로 전달
        loop.set_exception_handler(None)
        loop.call_exception_handler(context)
        loop.set_exception_handler(ignore_async_gen_errors)

    # asyncio exception handler 설정
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(ignore_async_gen_errors)

    # Suna-style 서브커맨드 구조로 개선
    parser = argparse.ArgumentParser(
        description="SparkleForge - Autonomous Multi-Agent Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SparkleForge: Where Ideas Sparkle and Get Forged ⚒️✨

EXAMPLES:
  # 연구 실행
  python main.py run "인공지능의 미래 전망"

  # 웹 대시보드 시작
  python main.py web

  # 시스템 헬스체크
  python main.py health

  # MCP 서버 상태 확인
  python main.py mcp status

  # 도구 목록 확인
  python main.py tools list

  # Docker 서비스 관리
  python main.py docker up
        """,
    )

    # 서브커맨드 추가
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    def add_research_command_options(command_parser):
        command_parser.add_argument("query", help="Research query")
        command_parser.add_argument("--output", "-o", help="Output file path")
        command_parser.add_argument(
            "--format",
            choices=["json", "markdown", "html"],
            default="markdown",
            help="Output format",
        )
        command_parser.add_argument(
            "--streaming", action="store_true", help="Enable streaming output"
        )
        command_parser.add_argument(
            "--max-tokens",
            type=int,
            default=None,
            help="Maximum output tokens requested by automation workflows",
        )
        command_parser.add_argument(
            "--model",
            default=None,
            help="Model override for this non-interactive research run",
        )
        command_parser.add_argument(
            "--task",
            default=None,
            help="Optional phase/task label prefixed onto the query, for automation traceability",
        )
        command_parser.add_argument(
            "--session",
            dest="session_id",
            default=None,
            metavar="SESSION_ID",
            help="Resume a specific prior session by ID instead of starting a new one",
        )
        command_parser.add_argument(
            "--continue",
            "-c",
            dest="continue_session",
            action="store_true",
            help="Resume the most recently active session",
        )
        command_parser.add_argument(
            "--mode",
            choices=["research", "work"],
            default="research",
            help="Execution mode: 'research' (default, one-shot research query) or "
            "'work' (coworker/tool-use goal execution, same path as the 'work' command)",
        )

    # run 커맨드
    run_parser = subparsers.add_parser("run", help="Execute research request")
    add_research_command_options(run_parser)

    # work 커맨드
    work_parser = subparsers.add_parser("work", help="Execute work goal as coworker")
    work_parser.add_argument("goal", nargs="+", help="Work goal")

    # session 커맨드 (REPL 밖에서도 세션 조회/재개 가능하도록)
    session_parser = subparsers.add_parser(
        "session", help="Inspect research sessions started via 'run --session'/'--continue'"
    )
    session_subparsers = session_parser.add_subparsers(
        dest="session_command", help="Session action"
    )
    session_list_parser = session_subparsers.add_parser("list", help="List sessions")
    session_list_parser.add_argument(
        "--limit", type=int, default=20, help="Maximum number of sessions to show"
    )
    session_show_parser = session_subparsers.add_parser("show", help="Show session details")
    session_show_parser.add_argument("session_id", help="Session ID")
    session_stats_parser = session_subparsers.add_parser(
        "stats", help="Show session statistics and concurrent-session quota usage"
    )
    session_quota_parser = session_subparsers.add_parser(
        "quota", help="Show or update a session's resource quota"
    )
    session_quota_parser.add_argument("session_id", help="Session ID")
    session_quota_parser.add_argument("--max-tokens", type=int, default=None, help="New token limit")
    session_quota_parser.add_argument("--budget", type=float, default=None, help="New cost budget")
    session_quota_parser.add_argument(
        "--timeout", type=int, default=None, help="New wall-clock timeout in seconds"
    )

    # actions 커맨드
    actions_parser = subparsers.add_parser("actions", help="List pending actions")

    # approve 커맨드
    approve_parser = subparsers.add_parser("approve", help="Approve action")
    approve_parser.add_argument("action_id", help="Action ID or 'all'")

    # deny 커맨드
    deny_parser = subparsers.add_parser("deny", help="Deny action")
    deny_parser.add_argument("action_id", help="Action ID")
    deny_parser.add_argument("reason", nargs="*", help="Reason for denial")

    # query 커맨드 (run과 동일 — sparkleforge query "..." 형태 지원)
    query_parser = subparsers.add_parser(
        "query", help="Send research query (alias for run)"
    )
    add_research_command_options(query_parser)

    # web 커맨드
    web_parser = subparsers.add_parser("web", help="Start web dashboard")
    web_parser.add_argument("--port", default="8501", help="Web server port")
    web_parser.add_argument("--host", default="0.0.0.0", help="Web server host")

    # mcp 커맨드
    mcp_parser = subparsers.add_parser("mcp", help="MCP server management")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", help="MCP commands")

    # mcp status
    mcp_status_parser = mcp_subparsers.add_parser(
        "status", help="Check MCP server status"
    )
    mcp_status_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    # mcp server
    mcp_server_parser = mcp_subparsers.add_parser("server", help="Start MCP server")

    # health 커맨드
    health_parser = subparsers.add_parser("health", help="System health check")
    health_parser.add_argument(
        "--detailed", action="store_true", help="Detailed health report"
    )

    # tools 커맨드
    tools_parser = subparsers.add_parser("tools", help="Tool management")
    tools_subparsers = tools_parser.add_subparsers(
        dest="tools_command", help="Tool commands"
    )

    # tools list
    tools_list_parser = tools_subparsers.add_parser("list", help="List available tools")
    tools_list_parser.add_argument("--category", help="Filter by category")

    # tools test
    tools_test_parser = tools_subparsers.add_parser(
        "test", help="Test tool functionality"
    )
    tools_test_parser.add_argument("tool_name", help="Tool name to test")

    # docker 커맨드
    docker_parser = subparsers.add_parser("docker", help="Docker service management")
    docker_subparsers = docker_parser.add_subparsers(
        dest="docker_command", help="Docker commands"
    )

    # docker up
    docker_up_parser = docker_subparsers.add_parser("up", help="Start Docker services")
    docker_up_parser.add_argument("--build", action="store_true", help="Rebuild images")
    docker_up_parser.add_argument(
        "--profile", action="append", help="Enable specific profiles (e.g., sandbox)"
    )

    # docker down
    docker_down_parser = docker_subparsers.add_parser(
        "down", help="Stop Docker services"
    )
    docker_down_parser.add_argument(
        "--volumes", action="store_true", help="Remove volumes"
    )
    docker_down_parser.add_argument(
        "--images", action="store_true", help="Remove images"
    )

    # docker logs
    docker_logs_parser = docker_subparsers.add_parser("logs", help="Show service logs")
    docker_logs_parser.add_argument("service", nargs="?", help="Specific service name")
    docker_logs_parser.add_argument(
        "--follow", "-f", action="store_true", help="Follow log output"
    )

    # docker status
    docker_status_parser = docker_subparsers.add_parser(
        "status", help="Show service status"
    )

    # docker build
    docker_build_parser = docker_subparsers.add_parser(
        "build", help="Build Docker images"
    )
    docker_build_parser.add_argument(
        "--no-cache", action="store_true", help="Build without cache"
    )

    # docker restart
    docker_restart_parser = docker_subparsers.add_parser(
        "restart", help="Restart Docker services"
    )
    docker_restart_parser.add_argument(
        "service", nargs="?", help="Specific service name"
    )

    # setup 커맨드
    setup_parser = subparsers.add_parser("setup", help="System setup and configuration")
    setup_parser.add_argument(
        "--force", action="store_true", help="Force reinstallation"
    )

    # cli 커맨드 (CLI 에이전트 관리)
    cli_parser = subparsers.add_parser("cli", help="CLI agent management")
    cli_subparsers = cli_parser.add_subparsers(
        dest="cli_command", help="CLI agent commands"
    )

    # cli list
    cli_list_parser = cli_subparsers.add_parser(
        "list", help="List available CLI agents"
    )

    # cli test
    cli_test_parser = cli_subparsers.add_parser("test", help="Test CLI agent")
    cli_test_parser.add_argument("agent_name", help="CLI agent name to test")

    # cli run
    cli_run_parser = cli_subparsers.add_parser("run", help="Run query with CLI agent")
    cli_run_parser.add_argument("agent_name", help="CLI agent name")
    cli_run_parser.add_argument("query", help="Query to execute")
    cli_run_parser.add_argument("--mode", help="Execution mode")
    cli_run_parser.add_argument("--files", nargs="*", help="Related files")

    # nightwelding 커맨드 (재현-우선 자율 이슈 수정 파이프라인)
    nightwelding_parser = subparsers.add_parser(
        "nightwelding", help="Reproduce-first autonomous issue fixer (writes a failing test, implements until green, opens a Draft PR)"
    )
    nightwelding_subparsers = nightwelding_parser.add_subparsers(
        dest="nightwelding_command", help="Nightwelding commands"
    )

    # nightwelding run
    nightwelding_run_parser = nightwelding_subparsers.add_parser(
        "run", help="Run Nightwelding once: a single issue, or a sweep of the backlog"
    )
    nightwelding_run_parser.add_argument("--issue", type=int, help="Specific GitHub issue number to process")
    nightwelding_run_parser.add_argument(
        "--backlog-label", default="auto-fix-failed",
        help="Label identifying the sweep backlog when --issue is not given (default: auto-fix-failed)",
    )
    nightwelding_run_parser.add_argument("--max-iterations", type=int, default=4, help="Max implementation repair attempts (1-6)")
    nightwelding_run_parser.add_argument("--max-per-run", type=int, default=3, help="Max issues to process per sweep")

    # nightwelding status
    nightwelding_status_parser = nightwelding_subparsers.add_parser("status", help="Show Nightwelding queue status")

    # nightwelding list
    nightwelding_list_parser = nightwelding_subparsers.add_parser("list", help="List Nightwelding queue history")

    # report parser
    report_parser = subparsers.add_parser(
        "report",
        help="Daily agent metric evaluation and critique"
    )
    report_subparsers = report_parser.add_subparsers(
        dest="report_command",
        help="Report command options"
    )
    report_subparsers.add_parser("generate", help="Generate the daily metric evaluation report")
    report_subparsers.add_parser("history", help="Show history of past agent evaluation scores")

    # 하위 호환성을 위한 기존 인자들 (deprecated)
    parser.add_argument(
        "--request",
        "--query",
        dest="legacy_request",
        help="Legacy: Use 'run' command instead",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        dest="legacy_web",
        help="Legacy: Use 'web' command instead",
    )
    parser.add_argument(
        "--mcp-server",
        action="store_true",
        dest="legacy_mcp_server",
        help="Legacy: Use 'mcp server' command instead",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        dest="legacy_health",
        help="Legacy: Use 'health' command instead",
    )
    parser.add_argument(
        "--check-mcp-servers",
        action="store_true",
        dest="legacy_mcp_status",
        help="Legacy: Use 'mcp status' command instead",
    )
    parser.add_argument(
        "--daemon", action="store_true", help="Start as long-running daemon (24/7 mode)"
    )
    parser.add_argument(
        "--prompt", help="Headless mode: research prompt (non-interactive)"
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    # Optional arguments
    parser.add_argument("--output", help="Output file path for research results")
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "stream-json"],
        default="text",
        help="Output format for headless mode",
    )
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument(
        "--checkpoint", help="Checkpoint file path to save/restore conversation"
    )
    parser.add_argument("--restore", help="Restore conversation from checkpoint file")

    # Long-running daemon options
    parser.add_argument(
        "--auto-save-interval",
        type=int,
        default=300,
        help="Auto-save interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-memory-mb",
        type=float,
        default=2048,
        help="Max memory usage in MB before restart (default: 2048)",
    )
    parser.add_argument(
        "--max-uptime-hours",
        type=float,
        default=0,
        help="Max uptime in hours before restart (0=unlimited, default: 0)",
    )
    parser.add_argument(
        "--no-auto-restart", action="store_true", help="Disable auto-restart on errors"
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml", "txt"],
        default="json",
        help="Output format for research results",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming pipeline for real-time results",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--skills",
        type=str,
        default=None,
        metavar="ID1,ID2",
        help="Comma-separated skill IDs to force-enable and inject into prompts (e.g. data_analyst,cursorrules)",
    )
    parser.add_argument(
        "--debug-bootstrap",
        action="store_true",
        help="Print BootstrapGraph stage timings and trust/runtime diagnostics",
    )

    args = parser.parse_args()

    # --skills: 강제 주입할 스킬 설정 (run/query 전에 적용)
    skills_arg = getattr(args, "skills", None)
    if skills_arg:
        from src.core.skills_manager import get_skill_manager
        get_skill_manager().set_forced_skills([s.strip() for s in skills_arg.split(",")])

    # 하위 호환성 처리
    if args.legacy_request:
        args.command = "run"
        args.query = args.legacy_request
        args.streaming = True
    elif args.legacy_web:
        args.command = "web"
    elif args.legacy_mcp_server:
        args.command = "mcp"
        args.mcp_command = "server"
    elif args.legacy_health:
        args.command = "health"
    elif args.legacy_mcp_status:
        args.command = "mcp"
        args.mcp_command = "status"

    # 기본값 설정: 인자 없으면 바로 REPL(sparkleforge>)로 진입 (Research query 루프 생략)
    if not args.command:
        if hasattr(args, "query") and args.query:
            args.command = "run"
        else:
            args.command = "repl"  # run 없이 바로 REPL로

    runtime_mode = "repl" if args.command in ("repl", "interactive") else str(
        args.command or "local"
    )

    from src.core.bootstrap_graph import BootstrapGraph

    bootstrap_result = await BootstrapGraph(
        project_root=project_root,
        runtime_mode=runtime_mode,
    ).run()
    if args.debug_bootstrap:
        for line in bootstrap_result.render_lines():
            print(line)
    if not bootstrap_result.ok:
        if not args.debug_bootstrap:
            print("Bootstrap failed:", file=sys.stderr)
            for line in bootstrap_result.render_lines():
                print(line, file=sys.stderr)
        return 1

    # Reassign global config using the pre-initialized config from BootstrapGraph
    global config
    config = bootstrap_result.values["config"]["config"]

    if args.debug_bootstrap and args.command == "repl" and not getattr(args, "prompt", None):
        return 0

    # 서브커맨드 처리 (반환 코드는 프로세스 종료까지 전달)
    cli_rc: int | None = None
    cmd = getattr(args, "command", None)
    if cmd == "run":
        cli_rc = await handle_run_command(args, config)
    elif cmd == "query":
        cli_rc = await handle_run_command(args, config)
    elif cmd == "work":
        cli_rc = await handle_work_command(args)
    elif cmd == "session":
        cli_rc = await handle_session_command(args)
    elif cmd == "actions":
        cli_rc = await handle_actions_command(args)
    elif cmd == "approve":
        cli_rc = await handle_approve_command(args)
    elif cmd == "deny":
        cli_rc = await handle_deny_command(args)
    elif cmd == "web":
        cli_rc = await handle_web_command(args)
    elif cmd == "mcp":
        cli_rc = await handle_mcp_command(args)
    elif cmd == "health":
        cli_rc = await handle_health_command(args)
    elif cmd == "tools":
        cli_rc = await handle_tools_command(args)
    elif cmd == "docker":
        cli_rc = await handle_docker_command(args)
    elif cmd == "setup":
        cli_rc = await handle_setup_command(args)
    elif cmd == "cli":
        cli_rc = await handle_cli_command(args)
    elif cmd == "nightwelding":
        cli_rc = await handle_nightwelding_command(args)
    elif cmd == "report":
        cli_rc = await handle_report_command(args)
    elif cmd == "interactive":
        cli_rc = await handle_interactive_command(args)
    elif cmd == "repl":
        # REPL로 바로 진입 (아무것도 안 하고 아래 is_repl_mode 블록으로 진행)
        pass
    else:
        parser.print_help()
        cli_rc = 0

    def _exit_code(rc: int | None) -> int:
        return 0 if rc is None else int(rc)

    # --query로 run/query 실행한 경우 여기서 종료 (이중 초기화·빈 워크플로우 요약 방지)
    if cmd in ("run", "query") and getattr(args, "query", None):
        return _exit_code(cli_rc)

    # 한 번만 실행하고 AutonomousResearchSystem 등 무거운 초기화로 넘어가면 안 되는 명령
    _STANDALONE_CLI = frozenset(
        {"health", "mcp", "tools", "docker", "setup", "cli", "web", "interactive", "work", "session", "actions", "approve", "deny", "report"}
    )
    if cmd in _STANDALONE_CLI:
        return _exit_code(cli_rc)

    # --query가 있으면 쿼리만 실행하고 종료. query 없을 때만 TLI(REPL) 진입
    is_repl_mode = getattr(args, "command", None) in ("interactive", "repl") or (
        getattr(args, "command", None) == "run"
        and not getattr(args, "prompt", None)
        and not getattr(args, "query", None)
    )

    # REPL 모드에서는 모든 로그를 완전히 억제 (ERROR만 표시)
    if is_repl_mode:
        import warnings

        # 모든 로거를 ERROR 레벨로 설정 (WARNING, INFO, DEBUG 모두 억제)
        logging.getLogger().setLevel(logging.ERROR)

        # 특정 모듈들의 로거도 ERROR로 설정
        for logger_name in [
            "__main__",
            "src",
            "src.core",
            "src.core.agent_orchestrator",
            "src.core.mcp_integration",
            "src.core.shared_memory",
            "src.core.skills_manager",
            "src.core.prompt_refiner_wrapper",
            "root",
            "streamlit",
            "streamlit.runtime",
            "local_researcher",
        ]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        # warnings도 완전히 억제
        warnings.filterwarnings("ignore")
    else:
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Initialize enhanced systems
    from src.core.error_handler import ErrorHandler, set_error_handler
    from src.core.progress_tracker import ProgressTracker, set_progress_tracker
    from src.utils.output_manager import (
        OutputFormat,
        OutputLevel,
        UserCenteredOutputManager,
        set_output_manager,
    )

    # 출력 매니저 초기화
    output_manager = UserCenteredOutputManager(
        output_level=OutputLevel.USER,
        output_format=OutputFormat.TEXT,
        enable_colors=True,
        stream_output=True,
        show_progress=True,
    )
    set_output_manager(output_manager)

    # 에러 핸들러 초기화
    error_handler = ErrorHandler(log_errors=True, enable_recovery=True)
    set_error_handler(error_handler)

    # 진행 상황 추적기 초기화 (세션별로 생성)
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    progress_tracker = ProgressTracker(
        session_id=session_id, enable_real_time_updates=True, update_interval=1.0
    )
    set_progress_tracker(progress_tracker)

    # 진행 상황 추적기 콜백 설정 (출력 매니저와 연동)
    last_stage = [None]  # 클로저를 위한 리스트 (nonlocal 대신)
    last_progress = [0]

    async def progress_callback(workflow_progress):
        """진행 상황 업데이트 시 출력 매니저에 표시."""
        try:
            progress_pct = int(workflow_progress.overall_progress * 100)
            stage_name = workflow_progress.current_stage.value

            # 단계가 변경되었을 때만 start_progress 호출
            if last_stage[0] != stage_name:
                last_stage[0] = stage_name
                eta_str = ""
                if workflow_progress.estimated_completion:
                    eta_seconds = max(
                        0, int(workflow_progress.estimated_completion - time.time())
                    )
                    eta_str = f" (예상 {eta_seconds}초 남음)"

                # 진행률 바 스타일로 표시 (단계 변경 시에만)
                await output_manager.start_progress(
                    stage_name,
                    100,
                    f"{progress_pct}% 완료",
                    workflow_progress.estimated_completion,
                )

            # 진행률이 실제로 변경되었을 때만 업데이트 (1% 이상 차이)
            if abs(progress_pct - last_progress[0]) >= 1 or progress_pct == 100:
                last_progress[0] = progress_pct
                await output_manager.update_progress(progress_pct)

        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")

    progress_tracker.add_progress_callback(progress_callback)

    # 시스템 초기화 (REPL 모드가 아닐 때만 전체 초기화)
    system = None
    if not is_repl_mode:
        # Initialize system
        system = AutonomousResearchSystem(bootstrap_result)

    try:
        # 체크포인트 복원 (있는 경우)
        if args.restore:
            from src.core.checkpoint_manager import CheckpointManager

            checkpoint_manager = CheckpointManager()
            restored_state = await checkpoint_manager.restore_checkpoint(args.restore)
            if restored_state:
                logger.info(f"✅ Restored checkpoint: {args.restore}")
                # 복원된 상태를 사용하여 계속 진행
            else:
                logger.error(f"❌ Failed to restore checkpoint: {args.restore}")
                return

        # Headless 모드 (비대화형)
        if args.prompt:
            from src.core.autonomous_orchestrator import AutonomousOrchestrator

            orchestrator = AutonomousOrchestrator()

            result = await orchestrator.run_research(args.prompt)

            # 출력 형식에 따라 결과 출력
            if args.output_format == "json":
                import json

                output = json.dumps(result, indent=2, ensure_ascii=False)
                print(output)
            elif args.output_format == "stream-json":
                # 스트리밍 JSON (newline-delimited)
                import json

                if isinstance(result, dict):
                    for key, value in result.items():
                        print(json.dumps({key: value}, ensure_ascii=False))
                else:
                    print(json.dumps(result, ensure_ascii=False))
            else:
                # 기본 텍스트 출력
                if isinstance(result, dict):
                    if "content" in result:
                        print(result["content"])
                    elif "final_synthesis" in result:
                        print(result["final_synthesis"].get("content", ""))
                    else:
                        print(str(result))
                else:
                    print(str(result))

            # 체크포인트 저장 (요청된 경우)
            if args.checkpoint:
                from src.core.checkpoint_manager import CheckpointManager

                checkpoint_manager = CheckpointManager()
                checkpoint_id = await checkpoint_manager.save_checkpoint(
                    state={"result": result, "prompt": args.prompt},
                    metadata={"mode": "headless", "output_format": args.output_format},
                )
                logger.info(f"✅ Checkpoint saved: {checkpoint_id}")

            return

        # 기본 동작: REPL 모드 (아무 옵션도 없으면)
        # 또는 --cli 옵션이 있으면
        if is_repl_mode:
            scheduler = None
            try:
                from src.cli.repl_cli import REPLCLI
                from src.core.scheduler import (
                    configure_scheduler_execution,
                    get_scheduler,
                )

                scheduler = configure_scheduler_execution(get_scheduler())
                await scheduler.start()
                cli = REPLCLI()
                await cli.run()
            except (EOFError, KeyboardInterrupt, SystemExit):
                # 정상 종료
                pass
            finally:
                if scheduler is not None:
                    try:
                        await scheduler.stop()
                    except Exception:
                        pass
            return

        # Interactive 모드 (기존)
        if args.interactive:
            from src.cli.repl_cli import REPLCLI
            from src.core.scheduler import (
                configure_scheduler_execution,
                get_scheduler,
            )

            # 스케줄러 초기화 및 시작
            scheduler = configure_scheduler_execution(get_scheduler())
            await scheduler.start()

            cli = REPLCLI()
            try:
                await cli.run()
            finally:
                await scheduler.stop()
            return

        request_arg = getattr(args, "request", None)
        if request_arg:
            # CLI Research Mode with 8 innovations
            logger.info("🚀 Starting Local Researcher with enhanced systems...")

            # 진행 상황 추적 시작
            await progress_tracker.start_tracking()

            # 초기화 단계 명시적으로 설정
            from src.core.progress_tracker import WorkflowStage

            progress_tracker.set_workflow_stage(
                WorkflowStage.INITIALIZING, {"message": "시스템 초기화 중..."}
            )

            # 워크플로우 시작 알림
            await output_manager.output(
                f"🔬 연구 주제: {request_arg}", level=OutputLevel.USER
            )
            await output_manager.output(
                "실시간 진행 상황 추적 및 향상된 에러 처리가 활성화되었습니다.",
                level=OutputLevel.SERVICE,
            )

            # 연구 실행
            await system.run_research(
                request_arg,
                args.output,
                streaming=args.streaming,
                output_format=args.format,
            )

        elif getattr(args, "web", False):
            # Web Application Mode with Streaming Pipeline
            system.run_web_app()

        elif getattr(args, "mcp_server", False):
            # MCP Server Mode with Universal MCP Hub - 실제 연결 수행
            logger.info("Initializing MCP servers...")
            try:
                await system.mcp_hub.initialize_mcp()
                logger.info("✅ MCP servers initialized")

                # 연결된 서버 상태 출력
                if system.mcp_hub.mcp_sessions:
                    print("\n" + "=" * 80)
                    print("✅ MCP 서버 연결 완료")
                    print("=" * 80)
                    for server_name in system.mcp_hub.mcp_sessions.keys():
                        tools_count = len(
                            system.mcp_hub.mcp_tools_map.get(server_name, {})
                        )
                        print(f"✅ {server_name}: {tools_count} tools available")
                    print("=" * 80)
                    print("\nMCP Hub is running. Press Ctrl+C to stop.")

                    # 계속 실행 대기
                    try:
                        await asyncio.sleep(3600)  # 1시간 대기 (또는 Ctrl+C로 종료)
                    except KeyboardInterrupt:
                        logger.info("Shutting down MCP Hub...")
                else:
                    logger.warning("⚠️ No MCP servers connected")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to initialize MCP servers: {e}")
                sys.exit(1)

        elif getattr(args, "mcp_client", False):
            # MCP Client Mode with Smart Tool Selection
            success = await system.run_mcp_client()
            if not success:
                sys.exit(1)

        elif getattr(args, "health_check", False):
            # Health Check Mode
            await system.run_health_check()

        elif getattr(args, "check_mcp_servers", False):
            # MCP 서버 확인 모드
            await system.check_mcp_servers()

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user (KeyboardInterrupt)")
        if system is not None:
            system._shutdown_requested = True
            try:
                await system._graceful_shutdown()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        # sys.exit(0) 제거 - asyncio.run()이 자동으로 처리
    except asyncio.CancelledError:
        # 취소된 경우 정리 후 종료
        logger.info("Operation cancelled")
        if system is not None:
            system._shutdown_requested = True
            try:
                await system._graceful_shutdown()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        # asyncio.CancelledError는 다시 raise하여 정상적인 취소 흐름 유지
        raise
    except Exception as e:
        # 향상된 에러 처리

        from src.core.error_handler import ErrorCategory, ErrorContext, ErrorSeverity

        error_context = ErrorContext(
            component="main",
            operation="run_research"
            if (getattr(args, "request", None) or getattr(args, "query", None))
            else "system_operation",
            session_id=session_id,
        )

        await error_handler.handle_error(
            e,
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.HIGH,
            context=error_context,
            custom_message=f"시스템 실행 중 치명적 오류 발생: {str(e)}",
        )

        if system is not None:
            system._shutdown_requested = True
            try:
                await system._graceful_shutdown()
            except Exception as e2:
                logger.error(f"Error during shutdown: {e2}")
        # 에러 발생 시 종료 코드 1로 종료
        sys.exit(1)
    finally:
        # 진행 상황 추적 중지 및 요약 출력
        try:
            await progress_tracker.stop_tracking()

            # run/query는 args.query, 레거시는 args.request (Namespace에 없을 수 있음)
            had_research_request = getattr(args, "request", None) or getattr(
                args, "query", None
            )
            if had_research_request:
                # 워크플로우 완료 요약
                await output_manager.complete_progress(success=True)
                await output_manager.output_workflow_summary()

                # 진행 상황 통계 출력
                stats = progress_tracker.get_statistics()
                await output_manager.output(
                    f"📈 세션 통계: {stats['total_agents_created']}개 에이전트 생성, "
                    f"{stats['agents_completed']}개 완료, {stats['agents_failed']}개 실패",
                    level=OutputLevel.SERVICE,
                )

        except Exception as e:
            logger.warning(f"Failed to finalize progress tracking: {e}")

        # 최종 정리 보장 (system이 초기화된 경우에만)
        if (
            system is not None
            and hasattr(system, "mcp_hub")
            and system.mcp_hub
            and hasattr(system.mcp_hub, "mcp_sessions")
        ):
            try:
                if system.mcp_hub.mcp_sessions:
                    logger.info("Final cleanup of MCP connections...")
                    await system.mcp_hub.cleanup()

                # MCP 백그라운드 헬스체크 서비스 종료 (선택적)
                if hasattr(system, "mcp_health_service") and system.mcp_health_service:
                    try:
                        await system.mcp_health_service.stop()
                        logger.info("✅ MCP Health Background Service stopped")
                    except Exception as e:
                        logger.debug(f"Error stopping MCP health service: {e}")
            except Exception as e:
                logger.debug(f"Error in final cleanup: {e}")

    return 0


# 서브커맨드 핸들러 함수들


def main_entry():
    """Entry point for sparkleforge / sparkle CLI (called from src.cli.entry or __main__)."""
    exit_code = 0
    try:
        result = asyncio.run(main())
        exit_code = int(result) if isinstance(result, int) else 0
    finally:
        # Flush Langfuse traces for short-lived CLI (optional; no-op if disabled)
        try:
            from src.core.observability import get_langfuse_client

            client = get_langfuse_client()
            if client is not None:
                client.flush()
        except Exception:
            # Do not log exception message: may contain URLs or internal details
            logger.warning("Langfuse trace flush failed (traces may be incomplete).")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main_entry()
