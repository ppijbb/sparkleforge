"""CLI subcommand handlers for main.py's argparse dispatch.

Extracted from main.py (Anvil Phase Sigma, issue #507 -- main.py was
3,331 lines with these ~20 handle_*_command functions inlined, one of
the two files #507's Sigma-1 checklist item claimed to have already
split).

`handle_run_command` takes `config` as an explicit parameter rather than
reading a module global: main.py reassigns its module-level `config` via
`global config` after BootstrapGraph runs, and a bare `from main import
config` here would only ever see the pre-bootstrap value. Threading it
through as a parameter (main.py passes its current `config` at the call
site) keeps the exact runtime value main() has calculated, without a
package import depending on main.py's module state.
"""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from src.cli.cli_result import cli_result_succeeded, extract_cli_result_content
from src.core.autonomous_research_system import (
    WebAppManager,
    _load_autonomous_orchestrator,
    project_root,
)
from src.monitoring.system_monitor import HealthMonitor

logger = logging.getLogger(__name__)


def _ensure_database_driver_for_cli() -> None:
    """Initialize SQLite driver for lightweight CLI commands if needed."""
    from src.core.db.database_driver import get_database_driver, set_database_driver
    from src.core.db.sqlite_driver import SQLiteDriver

    if get_database_driver() is None:
        sqlite_db_path = project_root / "data" / "sparkleforge.db"
        set_database_driver(SQLiteDriver(str(sqlite_db_path)))
        logger.info("✅ SQLite database driver initialized: %s", sqlite_db_path)


async def _resolve_run_session(args) -> tuple[str, str | None]:
    """`--session`/`--continue`를 처리해 세션 ID를 결정하고, 이어가는 경우 이전 컨텍스트를 query에 반영.

    Returns:
        (session_id, error_message). error_message가 있으면 caller가 즉시 중단해야 함.
    """
    resume_id = getattr(args, "session_id", None)
    should_continue = getattr(args, "continue_session", False)

    from src.core.session_control import get_session_control

    if not resume_id and not should_continue:
        new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            get_session_control().register_active_session(new_session_id, getattr(args, "query", ""))
        except RuntimeError as e:
            return "", f"❌ {e}"
        return new_session_id, None

    session_control = get_session_control()

    target_id = resume_id
    if not target_id:
        recent = await session_control.search_sessions(limit=1)
        if not recent:
            return "", "❌ No previous session found to continue (--continue)."
        target_id = recent[0].session_id

    restored_state = await session_control.restore_session(target_id)
    if restored_state is None:
        return "", f"❌ Session not found: {target_id}"

    prior_query = restored_state.get("user_query")
    if prior_query:
        args.query = (
            f"[Continuing session {target_id}]\nPrevious request: {prior_query}\n\n"
            f"New request: {args.query}"
        )
    logger.info(f"↩️  Resumed session: {target_id}")
    return target_id, None


def _persist_run_session(session_id: str, query: str, output_text: str | None) -> None:
    """연구 실행 완료 후 세션을 저장해 이후 --session/--continue로 이어갈 수 있게 함."""
    try:
        from src.core.session_manager import get_session_manager

        get_session_manager().save_session(
            session_id,
            agent_state={
                "user_query": query,
                "last_output_preview": (output_text or "")[:2000],
            },
            metadata={"tags": ["cli-run"]},
        )
    except Exception:
        logger.debug("Session persistence skipped for %s", session_id, exc_info=True)


async def handle_run_command(args, config):
    """연구 실행 커맨드 처리"""
    # --mode research를 명시한 경우만 이 아래의 AutonomousOrchestrator 리서치
    # 파이프라인으로 직행한다. mode 미지정(기본값) 또는 --mode work는 동일하게
    # 에이전트의 classify/planner 노드(_execute_coworker_goal -> AgentHarness)로
    # 보내 research/work 여부를 LLM이 쿼리별로 직접 판단하게 한다 -- CLI가
    # 정적 플래그로 미리 결정해버리지 않는다.
    if getattr(args, "mode", None) != "research":
        from src.cli.commands.run import run_command as _run_command
        return await _run_command(args, config)

    def _apply_runtime_overrides() -> None:
        model_override = getattr(args, "model", None)
        if model_override:
            os.environ["OPEN_CODE_MODEL_PATH"] = model_override
            if config.llm.provider == "opencode":
                config.llm.open_code_model_path = model_override
            else:
                os.environ["LLM_MODEL"] = model_override
                for key in (
                    "PLANNING_MODEL",
                    "REASONING_MODEL",
                    "VERIFICATION_MODEL",
                    "GENERATION_MODEL",
                    "COMPRESSION_MODEL",
                ):
                    os.environ[key] = model_override
                config.llm.primary_model = model_override
                config.llm.planning_model = model_override
                config.llm.reasoning_model = model_override
                config.llm.verification_model = model_override
                config.llm.generation_model = model_override
                config.llm.compression_model = model_override

        max_tokens = getattr(args, "max_tokens", None)
        if max_tokens is not None:
            os.environ["LLM_MAX_TOKENS"] = str(max_tokens)
            config.llm.max_tokens = max_tokens

    def _sanitize_embedded_cli_flags(query: str) -> tuple[str, bool]:
        """Query 문자열에 잘못 포함된 CLI 플래그를 제거.

        예: "질문 ... --format markdown --output out.md"
        """
        if not query:
            return query, False
        markers = (
            " --format ",
            " --output ",
            " --streaming",
            " --max-tokens ",
            " --model ",
            " -o ",
        )
        cut_positions = [query.find(m) for m in markers if query.find(m) != -1]
        if not cut_positions:
            return query, False
        cut_at = min(cut_positions)
        cleaned = query[:cut_at].strip()
        # CLI 플래그가 제거된 결과가 빈 문자열이더라도, "플래그 제거가 감지됨"은 맞으므로 True 유지.
        # 대신 caller에서 빈 쿼리를 실행하지 않도록 처리한다.
        return cleaned, True

    original_query = getattr(args, "query", "")
    sanitized_query, was_sanitized = _sanitize_embedded_cli_flags(original_query)
    if was_sanitized:
        if not sanitized_query.strip():
            logger.warning(
                "Detected embedded CLI flags inside query text, but removing them left an empty query. "
                "Aborting run to avoid executing a misleading/empty request."
            )
            return 1
        logger.warning(
            "Detected embedded CLI flags inside query text; sanitized query for research execution."
        )
        args.query = sanitized_query

    task_label = getattr(args, "task", None)
    if task_label:
        args.query = f"[{task_label}] {args.query}"

    session_id, session_error = await _resolve_run_session(args)
    if session_error:
        logger.error(session_error)
        return 1

    from src.core.observe.system_collector import (
        check_disk_space_safety,
        check_network_connectivity,
    )
    from src.core.session_control import get_session_control

    # Everything from here on must go through the finally below so the
    # session slot is always released -- including the disk check and
    # _apply_runtime_overrides(), either of which can raise or return early
    # before the old try block (which used to start after both) ever ran.
    try:
        disk_ok, disk_message = check_disk_space_safety()
        if not disk_ok:
            logger.error(disk_message)
            return 1

        network_ok, network_message = check_network_connectivity()
        if not network_ok:
            logger.warning(network_message)

        _apply_runtime_overrides()
        logger.info(f"🔬 Starting research: {args.query}")

        _ensure_database_driver_for_cli()
        # Autonomous Orchestrator 초기화
        AutonomousOrchestrator = _load_autonomous_orchestrator()
        orchestrator = AutonomousOrchestrator()

        # CLI에서 추가 단서를 즉시 받아 재시도할지 여부
        # - 인터랙티브(실제 콘솔 입력 가능)일 때만 사용
        # - stdin이 TTY가 아니면 자동으로 기존 비대화형 동작으로 폴백
        should_interact = (
            hasattr(sys, "stdin")
            and hasattr(sys, "stdout")
            and sys.stdin is not None
            and sys.stdout is not None
            and sys.stdin.isatty()
            and sys.stdout.isatty()
            and os.getenv("SPARKLEFORGE_CLI_INTERACTIVE", "true").lower() == "true"
        )

        def _needs_clarification(content: str) -> bool:
            if not content:
                return False
            # "리포트에 질문을 넣는" 기존 패턴을 감지
            keywords = (
                "추가로 사용자가 제공해야 할 단서",
                "후보 확정을 위한 추가 정보 요청",
                "추가 질문",
                "추가 단서",
                "식별을 위해",
            )
            return any(k in content for k in keywords) and (
                "확정" in content or "단정" in content or "미확정" in content
            )

        def _extract_clarification_block(content: str) -> str:
            # 콘솔 표시용: 특정 섹션 헤딩 주변만 보여준다.
            markers = (
                "추가로 사용자가 제공해야 할 단서",
                "후보 확정을 위한 추가 정보 요청",
            )
            for m in markers:
                idx = content.find(m)
                if idx != -1:
                    # 다음 '---'까지 대략 잘라냄
                    end = content.find("\n---", idx)
                    if end == -1:
                        end = min(len(content), idx + 1400)
                    return content[idx : end].strip()
            return ""

        # 연구 실행 (run_research 사용)
        base_query = args.query
        user_addendum = ""
        max_rounds = 3 if should_interact else 1

        result: dict[str, Any] | None = None
        for round_idx in range(max_rounds):
            active_query = base_query if not user_addendum else f"{base_query}\n\n{user_addendum}"
            result = await orchestrator.run_research(
                user_request=active_query,
                context={},
            )

            # 추가 단서가 필요하면 콘솔에서 받아서 이어가기
            content = extract_cli_result_content(result)
            if should_interact and _needs_clarification(content) and round_idx < max_rounds - 1:
                clarification_block = _extract_clarification_block(content)
                if clarification_block:
                    print("\n[추가 단서 요청 감지]\n")
                    print(clarification_block)

                prompt = (
                    "\n위 요청에 맞는 추가 단서(자유 입력)를 넣고 Enter를 누르면 "
                    "SparkleForge가 이어서 재시도합니다. (원하면 빈 입력으로 건너뜁니다): "
                )
                try:
                    user_input = input(prompt).strip()
                except EOFError:
                    user_input = ""

                if not user_input:
                    break

                # 다음 라운드에서 프롬프트에 사용자 단서 반영
                user_addendum = f"[사용자 추가 단서]\n{user_input}"
                continue

            break

        # 결과 출력/저장 (output 미지정 시 output/ 아래 기본 파일 생성)
        output_path = args.output
        if not output_path:
            output_dir = project_root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = ".json" if args.format == "json" else ".md"
            output_path = str(output_dir / f"query_{ts}{ext}")

        text = extract_cli_result_content(result)
        succeeded = cli_result_succeeded(result, text)
        _persist_run_session(session_id, base_query, text)

        with open(output_path, "w", encoding="utf-8") as f:
            if args.format == "json":
                json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                f.write(text)
        logger.info(f"✅ Results saved to {output_path}")

        if text:
            print(text)
        else:
            print(result)

        # 실행 성공/실패를 결과 페이로드 기준으로 일관되게 반환
        if not succeeded:
            logger.error("❌ Research completed with failure state")
            if not text:
                logger.error("❌ Research produced no deliverable content")
            return 1

    except Exception as e:
        logger.error(f"❌ Research failed: {e}")
        return 1
    finally:
        get_session_control().release_active_session(session_id)
    return 0


def parse_heat_duration(duration: str) -> float:
    """Parse a Heat time-budget string ('30m', '1h', '90s', or a bare number of seconds) to seconds.

    Raises ValueError with a clear message on an unparseable input.
    """
    text = duration.strip().lower()
    units = {"s": 1.0, "m": 60.0, "h": 3600.0}
    if text and text[-1] in units:
        value_part, unit = text[:-1], text[-1]
    else:
        value_part, unit = text, "s"
    try:
        value = float(value_part)
    except ValueError:
        raise ValueError(
            f"Invalid --heat duration '{duration}': expected a number optionally "
            "suffixed with s/m/h, e.g. '30m', '1h', '90s'."
        )
    if value <= 0:
        raise ValueError(f"Invalid --heat duration '{duration}': must be greater than 0.")
    return value * units[unit]


async def _execute_coworker_goal(
    goal: str, heat_seconds: float | None = None, force_coworker: bool = True
) -> int:
    """목표를 실행하는 공통 경로.

    force_coworker=True(기본, `work` 커맨드/명시적 `--mode work`): identity를
    "coder"로 고정해 곧장 autonomous Hermes 루프로 보낸다.
    force_coworker=False(`run` 기본 경로, --mode 미지정): identity/mode를
    강제하지 않고 AgentHarness의 classify/TaskRouter(LLM) 노드가 이 목표가
    research인지 실제 작업인지 요청마다 직접 판단하게 한다.
    """
    from src.core.observe.system_collector import (
        check_disk_space_safety,
        check_network_connectivity,
    )

    disk_ok, disk_message = check_disk_space_safety()
    if not disk_ok:
        logger.error(disk_message)
        return 1

    network_ok, network_message = check_network_connectivity()
    if not network_ok:
        logger.warning(network_message)

    if force_coworker:
        logger.info(f"🤝 Starting coworker session for: {goal}")
    else:
        logger.info(f"🧭 Letting the agent classify and route: {goal}")
    from rich import get_console

    from src.cli.ui.spinner import stage_status
    from src.core.agent_orchestrator import get_orchestrator

    custom_state = {"current_goal": goal}
    if force_coworker:
        custom_state["mode"] = "coworker"

    # issue #1508: without an explicit session_id, AgentOrchestrator.execute()
    # defaults every `work` invocation to the literal "default_session", so
    # concurrent runs collide on the same session state. Generate a unique id
    # per invocation, mirroring _resolve_run_session's pattern for `run`.
    session_id = f"work_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    with stage_status(get_console(), "Working...", ("src.core.agent_harness", "src.core.agent_loop")):
        orchestrator = get_orchestrator()
        result = await orchestrator.execute(
            goal,
            session_id=session_id,
            custom_state=custom_state,
            heat_seconds=heat_seconds,
        )

    # issue #1506: the harness catches its own failures internally and
    # returns {"success": False, "error": ...} instead of raising, so this
    # was falling through to `return 0` unconditionally -- a rate-limited or
    # otherwise failed run exited clean with no visible error at all.
    if not result.get("success", True):
        error_message = result.get("error") or "coworker session failed for an unknown reason"
        logger.error(f"Coworker session failed: {error_message}")
        print(f"❌ Coworker session failed: {error_message}")
        return 1

    print(result.get("content", ""))

    heat_report = result.get("metadata", {}).get("heat_report")
    if heat_report:
        print("\n--- Heat wrap-up report ---")
        print(f"Elapsed: {heat_report['elapsed_seconds']:.0f}s / {heat_report['heat_budget_seconds']:.0f}s budget")
        print(f"Completed: {len(heat_report['completed'])} step(s)")
        for item in heat_report["completed"]:
            print(f"  ✅ {item['tool']}: {item['summary']}")
        if heat_report["failed"]:
            print(f"Failed: {len(heat_report['failed'])} step(s)")
            for item in heat_report["failed"]:
                print(f"  ❌ {item['tool']}: {item['error']}")
        print(f"Next recommended action: {heat_report['next_recommended_action']}")

    return 0


async def handle_work_command(args):
    """협업 세션 실행 커맨드 처리"""
    heat_seconds = None
    heat_arg = getattr(args, "heat", None)
    if heat_arg:
        try:
            heat_seconds = parse_heat_duration(heat_arg)
        except ValueError as e:
            logger.error(str(e))
            print(f"❌ {e}")
            return 1
    return await _execute_coworker_goal(" ".join(args.goal), heat_seconds=heat_seconds)


async def handle_work_command_from_query(args):
    """`run --mode work`에서 진입하는 coworker 실행 경로 (query를 goal로 사용)."""
    return await _execute_coworker_goal(getattr(args, "query", "") or "")


async def handle_session_command(args):
    """REPL 밖에서 세션을 조회/관리하는 커맨드 처리 (session list / show / stats / quota)."""
    from rich.console import Console

    from src.cli.commands.session import (
        session_list_command,
        session_quota_command,
        session_show_command,
        session_stats_command,
    )
    from src.core.session_control import SessionControl

    class _SessionCliShim:
        def __init__(self):
            self.session_control = SessionControl()
            self.console = Console(force_terminal=True)

    shim = _SessionCliShim()
    sub = getattr(args, "session_command", None)
    if sub == "show":
        await session_show_command(shim, [args.session_id])
    elif sub == "stats":
        await session_stats_command(shim, [])
    elif sub == "quota":
        quota_args = [args.session_id]
        if args.max_tokens is not None:
            quota_args += ["--max-tokens", str(args.max_tokens)]
        if args.budget is not None:
            quota_args += ["--budget", str(args.budget)]
        if args.timeout is not None:
            quota_args += ["--timeout", str(args.timeout)]
        await session_quota_command(shim, quota_args)
    else:
        await session_list_command(shim, [str(getattr(args, "limit", 20))])
    return 0


async def _resolve_actions_session(args) -> tuple[str | None, bool]:
    """--session이 없으면 가장 최근 활성 세션으로 대체 (handle_run_command의 --continue와 동일한 패턴).

    Returns (session_id, was_inferred). KNOWN LIMITATION: AgentOrchestrator.execute()
    (the coworker/`work` path) never persists its session state via
    SessionManager.save_session -- only `run`/`query` does (see
    _persist_run_session). So when session_id isn't given explicitly, "most
    recently active session" may resolve to an unrelated research session
    that happens to be the newest thing in that store, not the coworker
    session that actually produced the pending actions. `was_inferred` lets
    callers surface that assumption instead of silently acting on it.
    """
    session_id = getattr(args, "session_id", None)
    if session_id:
        return session_id, False

    from src.core.session_control import get_session_control

    recent = await get_session_control().search_sessions(limit=1)
    return (recent[0].session_id if recent else None), True


class _ActionsCliShim:
    """actions/approve/deny 커맨드용 REPL 호환 shim.

    src.cli.commands.work의 actions_command/approve_command/deny_command는
    REPL(`cli.console`, `cli.session_control.current_session_id`)을 기대하므로,
    REPL 밖(main.py의 서브커맨드)에서도 동일한 함수를 그대로 재사용하기 위한
    최소 어댑터. handle_session_command/handle_report_command가 이미 쓰는
    패턴과 동일.
    """

    def __init__(self, session_id: str | None):
        from rich.console import Console

        from src.core.session_control import SessionControl

        self.session_control = SessionControl()
        self.session_control.current_session_id = session_id
        self.console = Console(force_terminal=True)


def _warn_if_inferred_session(shim, session_id: str, was_inferred: bool) -> None:
    if was_inferred:
        shim.console.print(
            f"[dim]Using most recently active session ({session_id}) -- "
            f"pass --session to target a specific one.[/dim]"
        )


async def handle_actions_command(args):
    from src.cli.commands.work import actions_command

    session_id, was_inferred = await _resolve_actions_session(args)
    shim = _ActionsCliShim(session_id)
    if not session_id:
        shim.console.print("[yellow]No active or recent session found.[/yellow]")
        return 0
    _warn_if_inferred_session(shim, session_id, was_inferred)
    await actions_command(shim, [])
    return 0


async def handle_approve_command(args):
    from src.cli.commands.work import approve_command

    session_id, was_inferred = await _resolve_actions_session(args)
    shim = _ActionsCliShim(session_id)
    if not session_id:
        shim.console.print("[yellow]No active or recent session found.[/yellow]")
        return 0
    _warn_if_inferred_session(shim, session_id, was_inferred)
    await approve_command(shim, [args.action_id])
    return 0


async def handle_deny_command(args):
    from src.cli.commands.work import deny_command

    session_id, was_inferred = await _resolve_actions_session(args)
    shim = _ActionsCliShim(session_id)
    if not session_id:
        shim.console.print("[yellow]No active or recent session found.[/yellow]")
        return 0
    _warn_if_inferred_session(shim, session_id, was_inferred)
    await deny_command(shim, [args.action_id] + list(getattr(args, "reason", None) or []))
    return 0

async def handle_web_command(args):
    """웹 대시보드 시작 커맨드 처리"""
    logger.info("🌐 Starting web dashboard...")

    web_manager = WebAppManager()
    os.environ["STREAMLIT_PORT"] = args.port
    os.environ["STREAMLIT_ADDRESS"] = args.host

    try:
        web_manager.start_web_app()
    except KeyboardInterrupt:
        logger.info("🛑 Web dashboard stopped")
    except Exception as e:
        logger.error(f"❌ Failed to start web dashboard: {e}")
        return 1
    return 0


async def handle_mcp_command(args):
    """MCP 관리 커맨드 처리"""
    if args.mcp_command == "status":
        logger.info("🔍 Checking MCP server status...")
        mcp_hub = None

        try:
            # MCP Hub 초기화 및 상태 확인
            from src.core.mcp_integration import get_mcp_hub

            mcp_hub = get_mcp_hub()

            if args.verbose:
                await mcp_hub.initialize_mcp()

            server_status = await mcp_hub.check_mcp_servers()
            mcp_hub.print_server_status(server_status, verbose=args.verbose)

        except Exception as e:
            logger.error(f"❌ MCP status check failed: {e}")
            return 1
        finally:
            if mcp_hub is not None:
                await mcp_hub.cleanup()

    elif args.mcp_command == "server":
        logger.info("🚀 Starting MCP server...")

        try:
            from src.core.mcp_integration import get_mcp_hub

            mcp_hub = get_mcp_hub()
            await mcp_hub.initialize_mcp()
            logger.info("✅ MCP Hub running. Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            logger.info("🛑 MCP server stopped")
        except Exception as e:
            logger.error(f"❌ Failed to start MCP server: {e}")
            return 1

    return 0


async def handle_health_command(args):
    """시스템 헬스체크 커맨드 처리"""
    logger.info("🏥 Running system health check...")

    try:
        health_monitor = HealthMonitor()

        # 능동 검증: Docker 응답성, 샌드박스 실제 실행, OpenRouter API 연결성
        active_checks = await health_monitor.run_active_subsystem_checks()
        for name, check in active_checks.items():
            if check["ok"] is True:
                logger.info(f"✅ {name}: ok")
            elif check["ok"] is None:
                logger.info(f"⏭️  {name}: {check['detail']}")
            else:
                logger.warning(f"⚠️  {name}: {check['detail']}")

        if args.detailed:
            # 상세 헬스체크
            health_report = await health_monitor.run_comprehensive_health_check()
            health_report["active_checks"] = active_checks
            health_monitor.print_detailed_health_report(health_report)
        else:
            # 간단한 헬스체크
            is_healthy = await health_monitor.quick_health_check()
            if is_healthy:
                logger.info("✅ System is healthy")
            else:
                logger.error("❌ System has issues")
                return 1

        # 샌드박스가 기본적인 명령조차 실행하지 못하면 백엔드가 실질적으로 고장난 것
        if active_checks["sandbox_write"]["ok"] is False:
            logger.error("❌ Sandbox cannot execute commands")
            return 1

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return 1
    return 0


async def handle_tools_command(args):
    """도구 관리 커맨드 처리"""

    def _default_tool_test_parameters(tool_name: str) -> Dict[str, Any]:
        """Return minimal non-destructive parameters for CLI tool smoke tests."""
        name = tool_name.lower()
        if "fetch" in name:
            return {"url": "https://example.com"}
        if "arxiv" in name or "scholar" in name:
            return {"query": "machine learning", "max_results": 1}
        if "search" in name or name in {"ddg_search", "g-search", "exa", "tavily"}:
            return {"query": "SparkleForge MCP", "num_results": 1, "max_results": 1}
        return {}

    if args.tools_command == "list":
        logger.info("🔧 Listing available tools...")
        mcp_hub = None

        try:
            from src.core.mcp_integration import get_mcp_hub

            mcp_hub = get_mcp_hub()
            try:
                # 전체 초기화가 지연될 수 있으므로 타임아웃 후 부분 결과로 진행
                await asyncio.wait_for(mcp_hub.initialize_mcp(), timeout=25.0)
            except TimeoutError:
                logger.warning(
                    "⚠️ MCP initialization timed out; showing currently discovered tools only"
                )

            # 도구 목록 출력 (ToolInfo dataclass 또는 dict 모두 허용)
            tools_by_category: Dict[str, List[str]] = {}
            for tool_name, tool_info in mcp_hub.tools.items():
                if isinstance(tool_info, dict):
                    raw_cat = tool_info.get("category", "unknown")
                    category = (
                        raw_cat.value
                        if hasattr(raw_cat, "value")
                        else str(raw_cat)
                    )
                else:
                    cat = getattr(tool_info, "category", None)
                    category = cat.value if hasattr(cat, "value") else str(
                        cat or "unknown"
                    )
                if category not in tools_by_category:
                    tools_by_category[category] = []
                tools_by_category[category].append(tool_name)

            for category, tools in tools_by_category.items():
                if not args.category or args.category == category:
                    print(f"\n📂 {category.upper()}:")
                    for tool in sorted(tools):
                        print(f"  - {tool}")

            if not tools_by_category:
                logger.error(
                    "❌ No tools discovered. Check MCP server connectivity with `sparkleforge mcp status`."
                )
                return 1

        except Exception as e:
            logger.error(f"❌ Failed to list tools: {e}")
            return 1
        finally:
            if mcp_hub is not None:
                await mcp_hub.cleanup()

    elif args.tools_command == "test":
        logger.info(f"🧪 Testing tool: {args.tool_name}")
        mcp_hub = None

        try:
            from src.core.mcp_integration import get_mcp_hub

            mcp_hub = get_mcp_hub()
            await asyncio.wait_for(mcp_hub.initialize_mcp(), timeout=25.0)

            # 도구 테스트
            result = await mcp_hub.execute_tool(
                args.tool_name,
                _default_tool_test_parameters(args.tool_name),
            )
            if result.get("success"):
                print(f"✅ Tool {args.tool_name} is working")
            else:
                print(f"❌ Tool {args.tool_name} failed: {result.get('error')}")
                return 1

        except Exception as e:
            logger.error(f"❌ Tool test failed: {e}")
            return 1
        finally:
            if mcp_hub is not None:
                await mcp_hub.cleanup()

    return 0


async def handle_docker_command(args):
    """Docker 서비스 관리 커맨드 처리 (Suna-style)"""
    import os
    import subprocess

    # Docker Compose 명령어 자동 감지
    def get_docker_compose_cmd():
        """Docker Compose 명령어 자동 감지"""
        # Docker Compose v2 (docker compose)
        try:
            subprocess.run(
                ["docker", "compose", "version"], capture_output=True, check=True
            )
            return ["docker", "compose"]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Docker Compose v1 (docker-compose)
        try:
            subprocess.run(
                ["docker-compose", "version"], capture_output=True, check=True
            )
            return ["docker-compose"]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return None

    # Docker 설치 확인
    def check_docker():
        """Docker 설치 상태 확인"""
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    # Docker Compose 파일 존재 확인
    def check_compose_file():
        """docker-compose.yaml 파일 존재 확인"""
        compose_files = ["docker-compose.yaml", "docker-compose.yml"]
        for filename in compose_files:
            if (project_root / filename).exists():
                return filename
        return None

    # Docker 환경 확인
    if not check_docker():
        logger.error("❌ Docker is not installed or not running")
        logger.info("Please install Docker: https://docs.docker.com/get-docker/")
        return 1

    compose_cmd = get_docker_compose_cmd()
    if not compose_cmd:
        logger.error("❌ Docker Compose is not installed")
        logger.info(
            "Please install Docker Compose: https://docs.docker.com/compose/install/"
        )
        return 1

    compose_file = check_compose_file()
    if not compose_file:
        logger.error("❌ docker-compose.yaml file not found")
        logger.info("Please ensure docker-compose.yaml exists in the project root")
        return 1

    if args.docker_command == "up":
        logger.info("🐳 Starting Docker services...")
        logger.info(f"Using Docker Compose: {' '.join(compose_cmd)}")
        logger.info(f"Compose file: {compose_file}")

        try:
            cmd = compose_cmd + ["-f", compose_file, "up", "-d"]
            if args.build:
                cmd.append("--build")
                logger.info("🔨 Building images...")

            # 프로필 지원 (예: sandbox)
            if hasattr(args, "profile") and args.profile:
                for profile in args.profile:
                    cmd.extend(["--profile", profile])
                    logger.info(f"🔧 Enabling profile: {profile}")

            # 환경 변수 로드 (.env 파일)
            env = os.environ.copy()
            env_file = project_root / ".env"
            if env_file.exists():
                logger.info("📄 Loading environment from .env file")
                # .env 파일에서 환경 변수 로드 (간단한 구현)
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            env[key] = value

            result = subprocess.run(cmd, cwd=str(project_root), env=env)
            if result.returncode == 0:
                logger.info("✅ Docker services started successfully")
                logger.info("🌐 Services:")
                logger.info("   - Backend API: http://localhost:8000")
                logger.info("   - Frontend: http://localhost:8501")
                logger.info("   - Redis: localhost:6379")
                logger.info("📊 View logs: python main.py docker logs")
                logger.info("📊 Check status: python main.py docker status")
            else:
                logger.error("❌ Failed to start Docker services")
                return 1

        except Exception as e:
            logger.error(f"❌ Docker command failed: {e}")
            return 1

    elif args.docker_command == "down":
        logger.info("🐳 Stopping Docker services...")

        try:
            cmd = compose_cmd + ["-f", compose_file, "down"]
            if hasattr(args, "volumes") and args.volumes:
                cmd.append("--volumes")
                logger.info("🗑️ Removing volumes...")
            if hasattr(args, "images") and args.images:
                cmd.append("--rmi")
                cmd.append("all")
                logger.info("🖼️ Removing images...")

            result = subprocess.run(cmd, cwd=str(project_root))
            if result.returncode == 0:
                logger.info("✅ Docker services stopped successfully")
            else:
                logger.error("❌ Failed to stop Docker services")
                return 1

        except Exception as e:
            logger.error(f"❌ Docker command failed: {e}")
            return 1

    elif args.docker_command == "logs":
        service_name = getattr(args, "service", None)
        logger.info(
            f"📊 Showing Docker service logs{f' for {service_name}' if service_name else ''}..."
        )

        try:
            cmd = compose_cmd + ["-f", compose_file, "logs"]
            if service_name:
                cmd.append(service_name)
            if hasattr(args, "follow") and args.follow:
                cmd.append("-f")

            if hasattr(args, "follow") and args.follow:
                subprocess.run(cmd, cwd=str(project_root))
            else:
                result = subprocess.run(
                    cmd, cwd=str(project_root), capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(result.stdout)
                else:
                    logger.error("❌ Failed to get logs")
                    return 1
        except KeyboardInterrupt:
            logger.info("🛑 Stopped log monitoring")
        except Exception as e:
            logger.error(f"❌ Failed to show logs: {e}")
            return 1

    elif args.docker_command == "status":
        logger.info("📊 Checking Docker service status...")

        try:
            cmd = compose_cmd + ["-f", compose_file, "ps"]
            result = subprocess.run(
                cmd, cwd=str(project_root), capture_output=True, text=True
            )
            if result.returncode == 0:
                print("🐳 Docker Services Status:")
                print("=" * 50)
                print(result.stdout)
            else:
                logger.error("❌ Failed to get service status")
                return 1
        except Exception as e:
            logger.error(f"❌ Failed to check status: {e}")
            return 1

    elif args.docker_command == "build":
        logger.info("🔨 Building Docker images...")

        try:
            cmd = compose_cmd + ["-f", compose_file, "build"]
            if hasattr(args, "no_cache") and args.no_cache:
                cmd.append("--no-cache")
                logger.info("🧹 Building without cache...")

            result = subprocess.run(cmd, cwd=str(project_root))
            if result.returncode == 0:
                logger.info("✅ Docker images built successfully")
            else:
                logger.error("❌ Failed to build Docker images")
                return 1
        except Exception as e:
            logger.error(f"❌ Build failed: {e}")
            return 1

    elif args.docker_command == "restart":
        service_name = getattr(args, "service", None)
        logger.info(
            f"🔄 Restarting Docker services{f' ({service_name})' if service_name else ''}..."
        )

        try:
            cmd = compose_cmd + ["-f", compose_file, "restart"]
            if service_name:
                cmd.append(service_name)

            result = subprocess.run(cmd, cwd=str(project_root))
            if result.returncode == 0:
                logger.info("✅ Docker services restarted successfully")
            else:
                logger.error("❌ Failed to restart Docker services")
                return 1
        except Exception as e:
            logger.error(f"❌ Restart failed: {e}")
            return 1

    else:
        logger.error(f"❌ Unknown Docker command: {args.docker_command}")
        logger.info("Available commands: up, down, logs, status, build, restart")
        return 1

    return 0


async def handle_setup_command(args):
    """시스템 설정 커맨드 처리"""
    logger.info("⚙️ Running system setup...")

    try:
        # 간단한 설정 확인
        required_files = [
            "pyproject.toml",
            "requirements.txt",
            "src/core/configs/researcher_config.yaml",
        ]

        missing_files = []
        for file_path in required_files:
            if not (project_root / file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return 1

        # 환경 변수 확인
        required_env_vars = ["OPENROUTER_API_KEY"]
        missing_env_vars = []
        for env_var in required_env_vars:
            if not os.getenv(env_var):
                missing_env_vars.append(env_var)

        if missing_env_vars:
            logger.warning(f"⚠️ Missing environment variables: {missing_env_vars}")
            logger.info("Please set these in your .env file or environment")

        logger.info("✅ System setup completed")

    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1
    return 0


async def handle_dwell_command(args):
    """Environmental Self-Dwell & Auto-Install Engine (issue #1108).

    Detects the target OS/architecture/runtime, auto-installs missing
    dependencies/MCP servers, installs itself as a background resident
    service, and verifies cross-platform readiness so a single
    `python main.py dwell` is sufficient on a clean Linux/macOS host.
    """
    import platform
    import shutil
    import subprocess

    from src.core.env_configurator import EnvironmentConfigurator

    logger.info("🏠 SparkleForge Self-Dwell: detecting environment...")

    env_configurator = EnvironmentConfigurator()
    env_summary = env_configurator.detect_environment()
    logger.info(
        "🖥️  OS=%s arch=%s python=%s uv=%s docker=%s",
        env_summary.get("os"),
        env_summary.get("architecture"),
        env_summary.get("python_version"),
        "yes" if env_summary.get("uv_available") else "no",
        "yes" if env_summary.get("docker_available") else "no",
    )

    missing = env_summary.get("missing_dependencies", [])
    if missing:
        logger.info("📦 Auto-installing missing dependencies: %s", ", ".join(missing))
        install_ok, install_message = env_configurator.auto_install(missing)
        if not install_ok:
            logger.error("❌ Auto-install failed: %s", install_message)
            return 1
        logger.info("✅ Auto-install complete: %s", install_message)
    else:
        logger.info("✅ All required dependencies already present")

    missing_mcp = env_summary.get("missing_mcp_servers", [])
    if missing_mcp:
        logger.info("🔌 Provisioning missing MCP servers: %s", ", ".join(missing_mcp))
        mcp_ok, mcp_message = env_configurator.provision_mcp_servers(missing_mcp)
        if not mcp_ok:
            logger.warning("⚠️  MCP provisioning incomplete: %s", mcp_message)
        else:
            logger.info("✅ MCP servers provisioned: %s", mcp_message)

    install_service = getattr(args, "install_service", True)
    if install_service:
        logger.info("🛠️  Installing background resident service...")
        service_ok, service_message = env_configurator.install_resident_service()
        if service_ok:
            logger.info("✅ Resident service installed: %s", service_message)
        else:
            logger.warning("⚠️  Resident service install skipped: %s", service_message)

    verify = getattr(args, "verify", True)
    if verify:
        logger.info("🔍 Verifying cross-platform readiness...")
        verify_script = project_root / "scripts" / "verify_environment.py"
        if not verify_script.exists():
            logger.warning("⚠️  scripts/verify_environment.py not found; skipping verification")
        else:
            python_bin = sys.executable or "python3"
            try:
                result = subprocess.run(
                    [python_bin, str(verify_script)],
                    cwd=str(project_root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                print(result.stdout)
                if result.returncode != 0:
                    logger.error("❌ Environment verification failed (exit %s)", result.returncode)
                    if result.stderr:
                        logger.error(result.stderr)
                    return 1
                logger.info("✅ Environment verification passed")
            except Exception as e:
                logger.error("❌ Environment verification raised: %s", e)
                return 1

    logger.info("🏠 SparkleForge is now dwelling in this environment 🎉")
    return 0


async def handle_nightwelding_command(args):
    """Nightwelding(재현-우선 자율 이슈 수정 파이프라인) 커맨드 처리."""
    from src.core.nightwelding.models import NightweldingQueue

    if args.nightwelding_command == "run":
        from src.core.nightwelding.runner import (
            run_nightwelding_issue,
            run_nightwelding_sweep,
        )

        try:
            target_issue = getattr(args, "file", None) or getattr(args, "issue", None)
            provider = getattr(args, "provider", None)

            if target_issue:
                logger.info(f"🌙 Nightwelding: running issue {target_issue} (provider={provider or 'auto'})")
                item = await run_nightwelding_issue(
                    target_issue,
                    max_iterations=args.max_iterations,
                    provider=provider,
                )
                items = [item]
            else:
                logger.info(f"🌙 Nightwelding: sweeping backlog label '{args.backlog_label}' (provider={provider or 'auto'})")
                items = await run_nightwelding_sweep(
                    backlog_label=args.backlog_label,
                    max_per_run=args.max_per_run,
                    max_iterations=args.max_iterations,
                    provider=provider,
                )

            if not items:
                logger.info("Nightwelding: no eligible issues found.")
                return 0

            failed = 0
            for item in items:
                if item.status.value == "draft_opened":
                    logger.info(f"✅ Issue #{item.issue_number}: Published -> {item.pr_url}")
                else:
                    failed += 1
                    logger.error(f"❌ Issue #{item.issue_number}: {item.status.value} — {item.failure_reason}")
            return 1 if failed and failed == len(items) else 0

        except Exception as e:
            logger.error(f"❌ Nightwelding run failed: {e}")
            return 1

    elif args.nightwelding_command == "status":
        try:
            queue = NightweldingQueue()
            items = queue.list()
            if not items:
                logger.info("Nightwelding: queue is empty.")
                return 0
            for item in items[:20]:
                status_str = f"#{item.issue_number}: {item.status.value}"
                if item.status.value == "failed" and item.failure_reason:
                    status_str += f" — {item.failure_reason.splitlines()[0]}"
                logger.info(f"{status_str} (updated {item.updated_at})")
        except Exception as e:
            logger.error(f"❌ Failed to read Nightwelding status: {e}")
            return 1

    elif args.nightwelding_command == "list":
        try:
            queue = NightweldingQueue()
            verbose = getattr(args, "verbose", False)
            for item in queue.list():
                line = f"#{item.issue_number}: {item.status.value} pr={item.pr_url or '-'}"
                if item.status.value == "failed" and item.failure_reason:
                    line += f" | reason: {item.failure_reason.splitlines()[0]}"
                log_val = getattr(item, "log", None)
                if verbose and log_val:
                    line += f"\n  log: {log_val}"
                logger.info(line)
        except Exception as e:
            logger.error(f"❌ Failed to list Nightwelding queue: {e}")
            return 1

    else:
        logger.error(f"❌ Unknown nightwelding command: {args.nightwelding_command}")
        logger.info("Available commands: run, status, list")
        return 1

    return 0


async def handle_ci_command(args):
    """CI 게이트 커맨드 처리 (code-review / issue-triage / merge-decision / fix-issue)."""
    from pathlib import Path

    if args.ci_command == "code-review":
        from src.core.ci.code_review import code_review

        return await code_review(Path(args.diff_file))
    elif args.ci_command == "issue-triage":
        from src.core.ci.issue_triage import issue_triage

        cerebras_file = Path(args.cerebras_file) if args.cerebras_file else None
        open_issues_file = Path(args.open_issues_file) if args.open_issues_file else None
        return await issue_triage(Path(args.review_file), cerebras_file, open_issues_file)
    elif args.ci_command == "merge-decision":
        from src.core.ci.merge_decision import merge_decision

        cerebras_file = Path(args.cerebras_file) if args.cerebras_file else None
        return await merge_decision(Path(args.pr_meta_file), Path(args.review_file), cerebras_file)
    elif args.ci_command == "fix-issue":
        from src.core.ci.fix_issue import fix_issue

        extra_context = Path(args.extra_context) if args.extra_context else None
        return await fix_issue(Path(args.issue_context), extra_context)
    elif args.ci_command == "publish":
        import json

        from src.core.ci.publish import commit_push_and_open_pr

        commit_body = Path(args.commit_body_file).read_text(encoding="utf-8") if args.commit_body_file else None
        pr_body = Path(args.pr_body_file).read_text(encoding="utf-8")
        try:
            result = commit_push_and_open_pr(
                repo=args.repo,
                repo_root=Path.cwd(),
                branch=args.branch,
                base_branch=args.base,
                commit_title=args.commit_title,
                commit_body=commit_body,
                paths=args.paths,
                pr_title=args.pr_title,
                pr_body=pr_body,
                labels=args.labels,
            )
        except RuntimeError as e:
            logger.error(f"❌ ci publish failed: {e}")
            return 1
        Path("publish_result.json").write_text(
            json.dumps(result.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if result.skipped_reason:
            print(f"ci publish: {result.skipped_reason}")
        else:
            print(f"ci publish: opened/reused {result.pr_url}")
        return 0
    elif args.ci_command == "select-issue":
        import json

        from src.core.ci.issue_selection import select_fixable_issue

        issues = json.loads(Path(args.issues_file).read_text(encoding="utf-8") or "[]")
        open_prs = json.loads(Path(args.open_prs_file).read_text(encoding="utf-8") or "[]")
        selected = select_fixable_issue(issues, open_prs)
        # Write to a file rather than printing the number for `$(...)`
        # capture -- a background init path unrelated to this command logs a
        # stray line to stdout after the command "returns" (see the
        # daily-roadmap-prompt handler above for the same issue), which would
        # corrupt a direct stdout capture.
        Path("selected_issue.txt").write_text(str(selected) if selected is not None else "", encoding="utf-8")
        return 0
    elif args.ci_command == "classify-commit":
        import json

        from src.core.ci.commit_classify import classify_conventional_commit

        commit = classify_conventional_commit(args.title)
        Path("commit_classification.json").write_text(
            json.dumps({"type": commit.type, "subject": commit.subject}), encoding="utf-8"
        )
        return 0
    elif args.ci_command == "assess-substantiality":
        import json

        from src.core.ci.fix_substantiality import (
            SubstantialityVerdict,
            assess_fix_substantiality,
            count_unchecked,
            gather_diff_stats,
        )

        issue_text = Path(args.issue_file).read_text(encoding="utf-8")
        try:
            diff_text, changed_files, changed_lines = gather_diff_stats(args.range)
            verdict = assess_fix_substantiality(
                issue_text=issue_text,
                diff_text=diff_text,
                changed_files=changed_files,
                changed_lines=changed_lines,
            )
        except Exception as e:
            logger.error(f"❌ assess-substantiality: diff gathering failed: {e}")
            verdict = SubstantialityVerdict(
                substantial=False,
                reason=" and the scope-overlap check itself failed to run, so it could not be verified",
                unchecked=count_unchecked(issue_text),
            )
        Path("substantiality_result.json").write_text(
            json.dumps(
                {"substantial": verdict.substantial, "reason": verdict.reason, "unchecked": verdict.unchecked}
            ),
            encoding="utf-8",
        )
        return 0
    elif args.ci_command == "classify-scenario-outcome":
        import json

        from src.core.ci.scenario_classify import classify_scenario_outcome

        report = json.loads(Path(args.report_file).read_text(encoding="utf-8"))
        outcome = classify_scenario_outcome(report)
        Path("scenario_outcome.json").write_text(
            json.dumps(
                {
                    "overall_score": outcome.overall_score,
                    "infra_failed": outcome.infra_failed,
                    "total": outcome.total,
                    "infra_ratio": outcome.infra_ratio,
                    "is_infra_outage": outcome.is_infra_outage,
                }
            ),
            encoding="utf-8",
        )
        return 0
    elif args.ci_command == "stagnation-issue":
        import json
        import os

        from src.core.ci.stagnation_issue import build_stagnation_issue, create_github_issue, load_history

        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        history = load_history(Path(args.history))
        issue = build_stagnation_issue(report, history)
        if issue is None:
            return 0

        repo = args.repo or os.getenv("GITHUB_REPOSITORY", "")
        if not repo:
            logger.error("GITHUB_REPOSITORY not set; cannot create stagnation issue.")
            return 0
        create_github_issue(repo, issue)
        return 0
    elif args.ci_command == "collect-todos":
        import json

        from src.core.ci.todo_inventory import collect_todos, generate_inventory, generate_json_inventory

        todos = collect_todos(project_root)
        docs_dir = project_root / "docs"
        docs_dir.mkdir(exist_ok=True)
        (docs_dir / "todo_inventory.md").write_text(generate_inventory(todos), encoding="utf-8")
        (docs_dir / "todo_inventory.json").write_text(
            json.dumps(generate_json_inventory(todos), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Collected {len(todos)} TODO/FIXME item(s).")
        return 0
    elif args.ci_command == "plan-todo-issues":
        import json

        from src.core.ci.todo_issue_plan import known_anchors, plan_todo_issues

        inventory = json.loads(Path(args.inventory_file).read_text(encoding="utf-8"))
        existing_issues = json.loads(Path(args.existing_issues_file).read_text(encoding="utf-8"))
        plan = plan_todo_issues(inventory, existing_issues)
        Path(args.plan_out).write_text(
            json.dumps([{"anchor": p.anchor, "title": p.title, "body": p.body} for p in plan]),
            encoding="utf-8",
        )
        print(f"Planned {len(plan)} new todo-debt issue(s); {len(known_anchors(existing_issues))} already tracked.")
        return 0

    logger.error(f"❌ Unknown ci command: {args.ci_command}")
    return 2


async def handle_autofix_command(args):
    """OpenCode repair-loop 커맨드 처리 (opencode-auto-fix.yml의 bash 재시도 루프를 내재화)."""
    from pathlib import Path

    if args.autofix_command == "run":
        from src.core.autofix.runner import run_autofix_repair_loop

        result = run_autofix_repair_loop(
            issue_context_path=Path(args.issue_context),
            repo_root=Path.cwd(),
            commit_title=args.commit_title,
            max_iterations=args.max_iterations,
            verify_command=args.verify_command,
            self_verify_command=args.self_verify_command or None,
        )
        if result.success:
            logger.info(f"✅ Autofix repair loop succeeded after {result.attempts} attempt(s).")
            return 0
        logger.error(f"❌ Autofix repair loop failed: {result.reason}")
        return 1

    logger.error(f"❌ Unknown autofix command: {args.autofix_command}")
    return 2


async def handle_interactive_command(args):
    """인터랙티브 모드 처리"""
    logger.info("💬 Starting interactive mode...")
    scheduler = None

    async def _shutdown():
        # REPL work/coworker commands start MCP server subprocesses via the
        # get_mcp_hub() singleton (stdio transport, inherits this process's
        # stdin/stdout). Every other command path calls mcp_hub.cleanup() in
        # a finally block -- this one didn't, so those subprocesses outlived
        # the REPL and held the terminal open after "exit" (issue: exit
        # appeared to hang).
        if scheduler is not None:
            try:
                await scheduler.stop()
            except Exception:
                logger.debug("Scheduler stop during shutdown raised", exc_info=True)
        try:
            from src.core.mcp_integration import get_mcp_hub

            await get_mcp_hub().cleanup()
        except Exception:
            logger.debug("MCP Hub cleanup during shutdown raised", exc_info=True)

    try:
        from src.cli.repl_cli import REPLCLI
        from src.core.scheduler import (
            configure_scheduler_execution,
            get_scheduler,
        )

        scheduler = configure_scheduler_execution(get_scheduler())
        await scheduler.start()

        try:
            cli = REPLCLI()
            try:
                await cli.run()
            finally:
                await _shutdown()
        except asyncio.CancelledError:
            logger.info("👋 Interactive mode cancelled; shutting down")
            raise
        return 0

    except (EOFError, KeyboardInterrupt, SystemExit):
        logger.info("👋 Goodbye!")
        await _shutdown()
        return 0
    except Exception as e:
        logger.error(f"❌ Interactive mode failed: {e}")
        await _shutdown()
        return 1
    return 0


async def handle_cli_command(args):
    """CLI 에이전트 관리 커맨드 처리"""
    from src.core.cli_agents.cli_agent_manager import get_cli_agent_manager
    from src.core.researcher_config import initialize_cli_agents

    # CLI 에이전트 초기화
    if not initialize_cli_agents():
        logger.warning("⚠️ CLI agents not enabled or failed to initialize")

    cli_manager = get_cli_agent_manager()

    if args.cli_command == "list":
        logger.info("🤖 Available CLI Agents:")

        try:
            available_agents = cli_manager.get_available_agents()
            if not available_agents:
                logger.info("  No CLI agents configured")
                return 0

            for agent_name in available_agents:
                agent_info = cli_manager.get_agent_info(agent_name)
                if agent_info:
                    status = (
                        "✅ Available" if agent_info.get("instance") else "⚠️ Configured"
                    )
                    logger.info(f"  - {agent_name}: {status}")
                    if agent_info.get("type"):
                        logger.info(f"    Type: {agent_info['type']}")
                    if agent_info.get("command"):
                        logger.info(f"    Command: {agent_info['command']}")
                else:
                    logger.info(f"  - {agent_name}: ❌ Not configured")

        except Exception as e:
            logger.error(f"❌ Failed to list CLI agents: {e}")
            return 1

    elif args.cli_command == "test":
        agent_name = args.agent_name
        logger.info(f"🧪 Testing CLI agent: {agent_name}")

        try:
            # 헬스체크
            agent = cli_manager.create_agent(agent_name)
            if not agent:
                logger.error(f"❌ CLI agent not available: {agent_name}")
                return 1

            is_healthy = await agent.health_check()
            if is_healthy:
                logger.info(f"✅ CLI agent {agent_name} is healthy")
                # 추가 정보 표시
                info = agent.get_info()
                logger.info(f"   Name: {info.get('name')}")
                logger.info(f"   Command: {info.get('command')}")
                logger.info(f"   Timeout: {info.get('timeout')}s")
            else:
                logger.error(f"❌ CLI agent {agent_name} is not healthy")
                return 1

        except Exception as e:
            logger.error(f"❌ CLI agent test failed: {e}")
            return 1

    elif args.cli_command == "run":
        agent_name = args.agent_name
        query = args.query
        logger.info(f"🚀 Running query with CLI agent: {agent_name}")
        logger.info(f"   Query: {query}")

        try:
            # 실행 옵션 준비
            kwargs = {}
            if hasattr(args, "mode") and args.mode:
                kwargs["mode"] = args.mode
            if hasattr(args, "files") and args.files:
                kwargs["files"] = args.files

            # CLI 에이전트로 쿼리 실행
            result = await cli_manager.execute_with_agent(agent_name, query, **kwargs)

            if result.get("success"):
                logger.info("✅ CLI agent execution successful")
                logger.info("📄 Response:")
                print(result.get("response", ""))

                # 메타데이터 표시
                metadata = result.get("metadata", {})
                if metadata:
                    logger.info("📊 Metadata:")
                    for key, value in metadata.items():
                        if key != "execution_time":  # 실행 시간은 별도로 표시
                            logger.info(f"   {key}: {value}")

                execution_time = metadata.get("execution_time", 0)
                if execution_time:
                    logger.info(f"⏱️ Execution time: {execution_time:.2f}s")

                confidence = result.get("confidence", 0)
                logger.info(f"🎯 Confidence: {confidence:.2f}")

            else:
                logger.error("❌ CLI agent execution failed")
                error_msg = result.get("error", "Unknown error")
                logger.error(f"   Error: {error_msg}")
                return 1

        except Exception as e:
            logger.error(f"❌ CLI agent execution failed: {e}")
            return 1

    else:
        logger.error(f"❌ Unknown CLI command: {args.cli_command}")
        logger.info("Available commands: list, test, run")
        return 1

    return 0


async def handle_report_command(args):
    """보고서 및 에이전트 평가 명령어 처리."""
    if getattr(args, "report_command", None) == "daily-roadmap-prompt":
        # Piped straight into a file by the daily-roadmap workflow -- must be plain
        # text, so this bypasses the rich Console(force_terminal=True) shim below
        # (that would emit ANSI escape codes into the redirected file).
        import datetime
        import os
        import sys
        from zoneinfo import ZoneInfo

        from src.core.daily_roadmap import build_daily_roadmap_mission_brief

        today = getattr(args, "today", None) or datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
        sys.stdout.write(build_daily_roadmap_mission_brief(today) + "\n")
        sys.stdout.flush()
        # Some background init path (unrelated to this subcommand -- reproduces even on
        # `sparkleforge health`) logs a stray line to stdout after the command "returns".
        # This command's whole job is printing static text for a workflow to redirect to
        # a file, so exit immediately rather than risk that noise corrupting the output.
        os._exit(0)

    report_command_name = getattr(args, "report_command", None)
    if report_command_name in (
        "roadmap-target",
        "roadmap-fallback-issue",
        "roadmap-issue-body",
        "sync-anvil-doc",
    ):
        # Same stray-stdout-log concern as daily-roadmap-prompt above: these are
        # all piped straight into a file (or captured via `$(...)`) by the
        # workflow, so bypass the rich Console shim and exit immediately.
        import json as json_module
        import os
        import sys
        from pathlib import Path

        if report_command_name == "roadmap-target":
            from src.core.roadmap.target_selection import (
                render_planning_context,
                select_anvil_target,
                target_file_contents,
            )

            milestone_file = Path(args.milestone_file)
            raw = milestone_file.read_text(encoding="utf-8").strip() if milestone_file.exists() else ""
            milestone = json_module.loads(raw) if raw else None
            status_file = Path(args.subissue_status_file)
            sub_status = json_module.loads(status_file.read_text(encoding="utf-8") or "[]") if status_file.exists() else []

            sys.stdout.write(render_planning_context(milestone, sub_status) + "\n")
            sys.stdout.flush()
            target = select_anvil_target(milestone, sub_status)
            Path(args.target_out).write_text(target_file_contents(target), encoding="utf-8")
            os._exit(0)

        if report_command_name == "roadmap-fallback-issue":
            from src.core.roadmap.planning import build_fallback_roadmap

            context_file = Path(args.context_file)
            context_md = context_file.read_text(encoding="utf-8") if context_file.exists() else ""
            target_file = Path(args.anvil_target_file)
            anvil_target = target_file.read_text(encoding="utf-8").strip() if target_file.exists() else ""

            sys.stdout.write(
                build_fallback_roadmap(
                    context_md=context_md,
                    anvil_target=anvil_target,
                    rc=args.rc,
                    invalid_reason=args.invalid_reason,
                    output_bytes=args.output_bytes,
                    console_bytes=args.console_bytes,
                    error_bytes=args.error_bytes,
                )
            )
            sys.stdout.flush()
            os._exit(0)

        if report_command_name == "roadmap-issue-body":
            from src.core.roadmap.planning import build_issue_body

            roadmap_file = Path(args.roadmap_file)
            roadmap_text = roadmap_file.read_text(encoding="utf-8") if roadmap_file.exists() else ""
            previous_file = Path(args.previous_body_file)
            previous_body = previous_file.read_text(encoding="utf-8") if previous_file.exists() else ""

            sys.stdout.write(
                build_issue_body(
                    today=args.today,
                    status=args.status,
                    roadmap_text=roadmap_text,
                    previous_body=previous_body,
                )
            )
            sys.stdout.flush()
            os._exit(0)

        if report_command_name == "sync-anvil-doc":
            from src.core.roadmap.anvil_doc_sync import sync_anvil_doc

            milestone_file = Path(args.milestone_file)
            raw = milestone_file.read_text(encoding="utf-8").strip() if milestone_file.exists() else ""
            milestone = json_module.loads(raw) if raw else None
            status_file = Path(args.subissue_status_file)
            sub_status = json_module.loads(status_file.read_text(encoding="utf-8") or "[]") if status_file.exists() else []

            if not milestone:
                sys.stdout.write("nochange\n")
                sys.stdout.flush()
                os._exit(0)

            total = len(sub_status)
            closed = sum(1 for item in sub_status if item.get("state") == "CLOSED")
            changed = sync_anvil_doc(Path(args.plan_file), milestone["number"], closed, total)
            sys.stdout.write(("changed" if changed else "nochange") + "\n")
            sys.stdout.flush()
            os._exit(0)

    from src.cli.commands.report import report_command
    from rich.console import Console

    class CliShim:
        def __init__(self):
            self.console = Console(force_terminal=True)

    cli = CliShim()
    report_args = []
    if getattr(args, "report_command", None):
        report_args.append(args.report_command)
    await report_command(cli, report_args)
    return 0



