import logging
"""SparkleForge REPL CLI (완전 CLI 환경)

prompt_toolkit 기반의 완전한 REPL 환경 제공.
- 히스토리 관리 (파일 기반)
- 자동완성 (명령어, 파일 경로, 세션 ID 등)
- 역검색 (Ctrl+R)
- 컬러 프롬프트 및 출력
"""

import locale
import logging
import shlex
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich import get_console
from rich.panel import Panel
from rich.text import Text

from src.cli.completion import SparkleForgeCompleter
from src.cli.history import SparkleForgeHistory
from src.core.execution_registry import ExecutionRegistry, RegisteredCommand
from src.core.prompt_router import PromptRouter, RouteTargetType
from src.core.trust_gate import get_current_trust_context
from src.utils.sparkleforge_history import (
    end_history_session,
    log_history_event,
    start_history_session,
)

# REPL 모드에서는 기본 로깅 레벨을 WARNING으로 설정하고
# src.cli 네임스페이스만 INFO 레벨을 허용하여 로그 누출 방지
logging.getLogger().setLevel(logging.WARNING)
for name in logging.root.manager.loggerDict:
    if not name.startswith("src.cli"):
        logging.getLogger(name).setLevel(logging.WARNING)

import sys
from datetime import datetime

logger = logging.getLogger(__name__)


class REPLCLI:
    """SparkleForge REPL CLI."""

    def __init__(self, suppress_logging: bool = True):
        """초기화."""
        import warnings

        if suppress_logging:
            # 잡음성 리프 로거만 개별로 낮춘다. root/"src"/"src.core"를 통째로
            # ERROR로 내리면 main.py가 REPL 진입 시 이미 설정해둔 INFO/allowlist를
            # 이 생성자가 곧바로 덮어써서, agent_loop/agent_harness/llm_manager
            # 등의 실제 진행(iteration, tool 호출, 재시도) 로그가 콘솔은 물론
            # 로그 파일에도 전혀 남지 않게 된다 (issue #1255).
            from src.cli.ui.logging_policy import apply_repl_quiet_mode

            apply_repl_quiet_mode()

            # warnings도 완전히 억제
            warnings.filterwarnings("ignore")

        # rich's process-wide singleton, not a private Console() -- anything
        # else in this process (output_manager's tool-call trace, LLM-call
        # progress ticks, ...) that also uses get_console() shares this same
        # Live/Status region instead of fighting it for the terminal. Two
        # independent Console instances both trying to own the terminal at
        # once corrupts/hides whichever one isn't "active" (this is why tool
        # calls never showed up during a turn -- output_manager was writing
        # through a completely separate Console the whole time).
        self.console = get_console()

        # Supabase 작업 히스토리 세션 -- run()이 실제 세션을 시작하기 전까지의
        # 기본값. run() 없이 handle_command()가 직접 호출되는 경로(테스트 등)를
        # 위해 getattr(..., None)로 방어적으로 읽는다.
        self._history_session_id = None

        # 히스토리 초기화
        self.history_manager = SparkleForgeHistory()
        self.history = self.history_manager.get_file_history()

        # PromptSession 초기화
        self.session = PromptSession(
            history=self.history,
            completer=SparkleForgeCompleter(self),
            enable_history_search=True,
            complete_while_typing=False,
        )

        # 명령어 핸들러
        self.command_handlers = {}
        self.prompt_router = PromptRouter()
        self._register_handlers()

        # 컨텍스트 및 체크포인트 매니저
        self.context_loader = None
        self.checkpoint_manager = None
        self.session_control = None

        try:
            from src.core.checkpoint_manager import CheckpointManager
            from src.core.context_loader import ContextLoader
            from src.core.session_control import get_session_control

            self.context_loader = ContextLoader()
            self.checkpoint_manager = CheckpointManager()
            self.session_control = get_session_control()
        except Exception as e:
            logger.warning(f"Failed to initialize context/checkpoint/session: {e}")

    def _register_handlers(self):
        """명령어 핸들러 등록."""
        from src.cli.commands.checkpoint import (
            checkpoint_delete_command,
            checkpoint_list_command,
            checkpoint_restore_command,
            checkpoint_save_command,
        )
        from src.cli.commands.config import (
            config_get_command,
            config_set_command,
            config_show_command,
        )
        from src.cli.commands.context import (
            context_reload_command,
            context_show_command,
        )
        from src.cli.commands.forge_master import (
            forge_master_batch_command,
            forge_master_command,
        )
        from src.cli.commands.help import help_command
        from src.cli.commands.mcp import (
            mcp_attach_command,
            mcp_detach_command,
            mcp_list_command,
        )
        from src.cli.commands.nightwelding import (
            nightwelding_list_command,
            nightwelding_run_command,
            nightwelding_status_command,
        )
        from src.cli.commands.research import research_command
        from src.cli.commands.schedule import (
            schedule_add_command,
            schedule_create_command,
            schedule_delete_command,
            schedule_disable_command,
            schedule_enable_command,
            schedule_history_command,
            schedule_list_command,
            schedule_pause_command,
            schedule_remove_command,
            schedule_resume_command,
            schedule_run_command,
            schedule_show_command,
            schedule_stats_command,
        )
        from src.cli.commands.session import (
            session_cancel_command,
            session_delete_command,
            session_list_command,
            session_pause_command,
            session_quota_command,
            session_resume_command,
            session_search_command,
            session_show_command,
            session_stats_command,
            session_tasks_command,
        )
        from src.cli.commands.work import (
            actions_command,
            approve_command,
            deny_command,
            work_command,
        )

        self.command_handlers = {
            "research": research_command,
            "forge-master": forge_master_command,
            "forge-batch": forge_master_batch_command,
            "work": work_command,
            "actions": actions_command,
            "approve": approve_command,
            "deny": deny_command,
            "session": {
                "list": session_list_command,
                "show": session_show_command,
                "pause": session_pause_command,
                "resume": session_resume_command,
                "cancel": session_cancel_command,
                "delete": session_delete_command,
                "search": session_search_command,
                "stats": session_stats_command,
                "tasks": session_tasks_command,
                "quota": session_quota_command,
            },
            "context": {
                "show": context_show_command,
                "reload": context_reload_command,
            },
            "checkpoint": {
                "save": checkpoint_save_command,
                "list": checkpoint_list_command,
                "restore": checkpoint_restore_command,
                "delete": checkpoint_delete_command,
            },
            "schedule": {
                "list": schedule_list_command,
                "create": schedule_create_command,
                "show": schedule_show_command,
                "pause": schedule_pause_command,
                "resume": schedule_resume_command,
                "delete": schedule_delete_command,
                "history": schedule_history_command,
                "stats": schedule_stats_command,
                "run": schedule_run_command,
                "add": schedule_add_command,
                "remove": schedule_remove_command,
                "enable": schedule_enable_command,
                "disable": schedule_disable_command,
            },
            "config": {
                "show": config_show_command,
                "set": config_set_command,
                "get": config_get_command,
            },
            "nightwelding": {
                "run": nightwelding_run_command,
                "status": nightwelding_status_command,
                "list": nightwelding_list_command,
            },
            "mcp": {
                "attach": mcp_attach_command,
                "detach": mcp_detach_command,
                "list": mcp_list_command,
            },
            "help": help_command,
            "exit": self._handle_exit,
            "quit": self._handle_exit,
            "clear": self._handle_clear,
        }

    async def run(self):
        """REPL 루프 실행."""
        self._history_session_id = start_history_session("repl")

        # 시작 배너
        await self._show_banner()

        # 컨텍스트 로드
        if self.context_loader:
            try:
                context = await self.context_loader.load_context()
                if context:
                    self.console.print(
                        "[dim]📄 Project context loaded from SPARKLEFORGE.md[/dim]\n"
                    )
            except Exception as e:
                logger.debug(f"Failed to load context: {e}")

        # REPL 루프
        self.console.print(
            "[bold cyan]💬 Ready to chat![/bold cyan] "
            "[dim]Type your prompt or command below:[/dim]\n"
        )
        # Ensure terminal cursor is visible (unhide cursor ANSI code \033[?25h).
        # Piped/non-tty stdout has no cursor to unhide, and the raw escape
        # code would otherwise render as literal "[?25h" text in the output.
        if sys.stdout.isatty():
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

        while True:
            try:
                prompt_text = HTML(
                    "<b><cyan>sparkleforge</cyan> <brightcyan>❯</brightcyan> </b>"
                )
                text = await self.session.prompt_async(prompt_text)

                if not text.strip():
                    continue

                await self.handle_command(text)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                continue
            except EOFError:
                # exit 명령어 또는 Ctrl+D로 종료
                # _handle_exit에서 이미 "Goodbye!" 메시지를 출력했으므로 여기서는 중복 출력하지 않음
                # 단, Ctrl+D로 직접 종료한 경우를 위해 확인
                break
            except Exception as e:
                logger.error(f"Error in REPL CLI: {e}", exc_info=True)
                self.console.print(f"[red]❌ Error: {e}[/red]")

        end_history_session(self._history_session_id, "succeeded")

        # 루프 종료 후 정리 작업
        try:
            # PromptSession 정리
            if hasattr(self, "session") and self.session:
                # prompt_toolkit 세션은 자동으로 정리됨
                pass
        except Exception as e:
            logger.debug(f"Final cleanup error (ignored): {e}")

    async def _get_greeting_message(self) -> str:
        """현재 시간과 지역에 맞는 인사 메시지 반환 (즉시 반환)."""
        try:
            hour = datetime.now().hour
            is_korean = False
            try:
                lang = (locale.getlocale()[0] or "").lower()
                is_korean = lang.startswith("ko")
            except Exception:
                pass

            if is_korean:
                if 5 <= hour < 12:
                    return "좋은 아침입니다"
                elif 12 <= hour < 18:
                    return "좋은 오후입니다"
                elif 18 <= hour < 22:
                    return "좋은 저녁입니다"
                else:
                    return "편안한 밤 되세요"
            else:
                if 5 <= hour < 12:
                    return "Good morning"
                elif 12 <= hour < 18:
                    return "Good afternoon"
                elif 18 <= hour < 22:
                    return "Good evening"
                else:
                    return "Good night"
        except Exception:
            return "Welcome"

    async def _show_banner(self):
        """시작 배너 표시."""
        # 인사 메시지 가져오기
        greeting = await self._get_greeting_message()

        # 배너 내용 구성
        banner_content = Text()
        banner_content.append("⚒️  ", style="bold yellow")
        banner_content.append(greeting, style="bold cyan")

        # Chat & Available Commands 구성
        commands_text = Text()
        commands_text.append("💬 Chat & Interactive Agent Mode:\n", style="bold green")
        commands_text.append("  • ", style="dim")
        commands_text.append(
            "Type ANY prompt or request to chat with SparkleForge AI Agent\n",
            style="bold white",
        )
        commands_text.append(
            '    (e.g., "이 프로젝트 구조 설명해줘", "버그 수정해줘", "연구 수행해줘")\n\n',
            style="dim",
        )

        commands_text.append("⚡ Available System Commands:\n", style="bold cyan")
        commands_text.append("  ", style="dim")
        commands_text.append("research <query>", style="cyan")
        commands_text.append("  - Start an autonomous research task\n", style="dim")
        commands_text.append("  ", style="dim")
        commands_text.append("session list", style="cyan")
        commands_text.append("      - List all active sessions\n", style="dim")
        commands_text.append("  ", style="dim")
        commands_text.append("context show", style="cyan")
        commands_text.append("     - Show project context\n", style="dim")
        commands_text.append("  ", style="dim")
        commands_text.append("schedule list", style="cyan")
        commands_text.append("     - Manage cron-style schedules\n", style="dim")
        commands_text.append("  ", style="dim")
        commands_text.append("nightwelding status", style="cyan")
        commands_text.append(" - Autonomous issue-fixer queue\n", style="dim")
        commands_text.append("  ", style="dim")
        commands_text.append("help", style="cyan")
        commands_text.append("             - Show detailed help message\n", style="dim")
        commands_text.append("  ", style="dim")
        commands_text.append("exit", style="cyan")
        commands_text.append("             - Exit REPL\n\n", style="dim")
        commands_text.append("Type 'help' for detailed command information", style="dim")

        # 전체 내용 합치기
        full_content = Text()
        full_content.append(banner_content)
        full_content.append("\n\n", style="dim")
        full_content.append(commands_text)

        banner = Panel(
            full_content,
            border_style="cyan",
            padding=(1, 2),
            title="[bold cyan]💬 SparkleForge Interactive Agent & Research System[/bold cyan]",
            subtitle="[dim]Version 1.0.0[/dim]",
        )
        self.console.print(banner)
        self.console.print()

    async def handle_command(self, text: str):
        """명령어 처리."""
        history_session_id = getattr(self, "_history_session_id", None)
        if history_session_id:
            log_history_event(history_session_id, "message", text, role="user")
        try:
            # shlex로 파싱 (따옴표 처리)
            parts = shlex.split(text)
            if not parts:
                return

            command = parts[0].lower()

            if command in ["exit", "quit"]:
                await self._handle_exit()
                return

            if command == "clear":
                await self._handle_clear()
                return

            if command == "help":
                await self.command_handlers["help"](self.console)
                return

            # 명령어 라우팅
            if command in self.command_handlers:
                handler = self.command_handlers[command]

                if isinstance(handler, dict):
                    # 서브 명령어
                    if len(parts) < 2:
                        self.console.print(f"[red]Usage: {command} <subcommand>[/red]")
                        self.console.print(f"[dim]Type '{command} help' for subcommands[/dim]")
                        return

                    subcommand = parts[1].lower()
                    if subcommand in handler:
                        await handler[subcommand](self, parts[2:])
                    else:
                        self.console.print(f"[red]Unknown subcommand: {subcommand}[/red]")
                        self.console.print(f"[dim]Available: {', '.join(handler.keys())}[/dim]")
                else:
                    # 직접 명령어
                    await handler(self, parts[1:])
            else:
                routed = await self._try_route_command(text)
                if routed:
                    return
                # 등록된 커맨드에 매칭되지 않으면 work 경로(AgentHarness)로 보내
                # classify/planner 노드가 research vs 실제 작업 여부를 직접 판단하게 한다.
                # research_command는 도구가 전혀 붙어있지 않은 별도 파이프라인이라
                # 여기서 고정 호출하면 AI가 작업 여부를 판단할 기회 자체가 없다.
                await self.command_handlers["work"](self, [text])

        except EOFError:
            # exit 명령어에서 발생한 EOFError는 다시 raise하여 run()에서 처리
            raise
        except Exception as e:
            logger.error(f"Error handling command: {e}", exc_info=True)
            self.console.print(f"[red]❌ Error: {e}[/red]")

    async def _handle_exit(self):
        """종료 처리."""
        self.console.print("[bold]Goodbye! 👋[/bold]")
        # EOFError를 raise하여 run() 메서드의 루프를 종료
        raise EOFError()

    async def _handle_clear(self):
        """화면 지우기."""
        self.console.clear()

    async def _build_execution_registry(self) -> ExecutionRegistry:
        """Build a lightweight registry for prompt routing."""
        from src.core.mcp_integration import get_mcp_hub
        from src.core.scheduler import get_scheduler
        from src.core.skills_manager import get_skill_manager

        return await ExecutionRegistry.build(
            mcp_hub=get_mcp_hub(),
            skill_manager=get_skill_manager(),
            scheduler=get_scheduler(),
            trust=get_current_trust_context(),
        )

    def _is_safe_command_route(
        self,
        text: str,
        command: RegisteredCommand,
    ) -> bool:
        """Only auto-dispatch commands when the prompt clearly expresses them."""
        prompt_tokens = set(self.prompt_router._tokenize(text))
        if set(command.dispatch).issubset(prompt_tokens) and not command.requires_args:
            return True

        for alias in command.aliases:
            alias_tokens = set(self.prompt_router._tokenize(alias))
            if alias_tokens and alias_tokens.issubset(prompt_tokens) and not command.requires_args:
                return True

        return False

    async def _try_route_command(self, text: str) -> bool:
        """Attempt safe command routing before falling back to research."""
        registry = await self._build_execution_registry()
        routes = await self.prompt_router.route(
            text,
            registry,
            get_current_trust_context(),
        )
        if not routes:
            return False

        top = routes[0]
        if top.target_type != RouteTargetType.COMMAND:
            return False

        command = registry.lookup(top.target)
        if not isinstance(command, RegisteredCommand):
            return False

        if not self._is_safe_command_route(text, command):
            return False

        canonical = " ".join(command.dispatch)
        await self.handle_command(canonical)
        return True
