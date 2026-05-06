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
from datetime import datetime

import pytz
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.cli.completion import SparkleForgeCompleter
from src.cli.history import SparkleForgeHistory
from src.core.execution_registry import ExecutionRegistry, RegisteredCommand
from src.core.prompt_router import PromptRouter, RouteTargetType
from src.core.trust_gate import get_current_trust_context

logger = logging.getLogger(__name__)


class REPLCLI:
    """SparkleForge REPL CLI."""

    def __init__(self):
        """초기화."""
        import logging
        import warnings

        # REPL 모드에서는 모든 로그를 완전히 억제 (ERROR만 표시)
        logging.getLogger().setLevel(logging.ERROR)

        # 모든 주요 모듈의 로거를 ERROR로 설정
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

        self.console = Console()

        # 히스토리 초기화
        self.history_manager = SparkleForgeHistory()
        self.history = self.history_manager.get_file_history()

        # PromptSession 초기화
        self.session = PromptSession(
            history=self.history,
            completer=SparkleForgeCompleter(self),
            enable_history_search=True,
            complete_while_typing=True,
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
        from src.cli.commands.help import help_command
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
            session_resume_command,
            session_search_command,
            session_show_command,
            session_stats_command,
            session_tasks_command,
        )
        from src.cli.commands.work import (
            work_command,
            actions_command,
            approve_command,
            deny_command,
        )

        self.command_handlers = {
            "research": research_command,
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
            "help": help_command,
            "exit": self._handle_exit,
            "quit": self._handle_exit,
            "clear": self._handle_clear,
        }

    async def run(self):
        """REPL 루프 실행."""
        # 시작 배너
        await self._show_banner()

        # 로딩 표시와 함께 컨텍스트 로드
        with self.console.status(
            "[bold cyan]Initializing SparkleForge...", spinner="dots"
        ):
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
        while True:
            try:
                # ANSI 색상 코드를 사용하여 프롬프트 색상 적용
                prompt_text = ANSI("\033[1;36msparkleforge\033[0m> ")
                text = await self.session.prompt_async(prompt_text)

                if not text.strip():
                    continue

                await self.handle_command(text)

            except KeyboardInterrupt:
                self.console.print(
                    "\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]"
                )
                continue
            except EOFError:
                # exit 명령어 또는 Ctrl+D로 종료
                # _handle_exit에서 이미 "Goodbye!" 메시지를 출력했으므로 여기서는 중복 출력하지 않음
                # 단, Ctrl+D로 직접 종료한 경우를 위해 확인
                break
            except Exception as e:
                logger.error(f"Error in REPL CLI: {e}", exc_info=True)
                self.console.print(f"[red]❌ Error: {e}[/red]")

        # 루프 종료 후 정리 작업
        try:
            # PromptSession 정리
            if hasattr(self, "session") and self.session:
                # prompt_toolkit 세션은 자동으로 정리됨
                pass
        except Exception as e:
            logger.debug(f"Final cleanup error (ignored): {e}")

    async def _get_greeting_message(self) -> str:
        """현재 시간과 지역에 맞는 인사 메시지를 LLM으로 생성."""
        try:
            # 시간대 감지
            try:
                local_tz = pytz.timezone("Asia/Seoul")  # 기본값
                # 시스템 시간대 가져오기
                import time

                local_tz_name = time.tzname[0] if time.tzname else "UTC"
                # 주요 시간대 매핑
                tz_mapping = {
                    "KST": "Asia/Seoul",
                    "JST": "Asia/Tokyo",
                    "CST": "Asia/Shanghai",
                    "PST": "America/Los_Angeles",
                    "EST": "America/New_York",
                    "GMT": "Europe/London",
                    "CET": "Europe/Paris",
                }
                for tz_abbr, tz_name in tz_mapping.items():
                    if tz_abbr in local_tz_name:
                        local_tz = pytz.timezone(tz_name)
                        break
            except:
                local_tz = pytz.UTC

            # 현재 시간
            now = datetime.now(local_tz)
            hour = now.hour
            date_str = now.strftime("%Y-%m-%d %H:%M")

            # 언어 감지
            try:
                lang_code = locale.getlocale()[0] or "en_US"
                if lang_code.startswith("ko"):
                    language = "Korean"
                elif lang_code.startswith("ja"):
                    language = "Japanese"
                elif lang_code.startswith("zh"):
                    language = "Chinese"
                elif lang_code.startswith("es"):
                    language = "Spanish"
                elif lang_code.startswith("fr"):
                    language = "French"
                elif lang_code.startswith("de"):
                    language = "German"
                else:
                    language = "English"
            except:
                language = "English"

            # 시간대 이름
            tz_name = str(local_tz)

            # LLM 호출
            from src.core.llm_manager import TaskType, execute_llm_task

            prompt = f"""Generate a brief, friendly greeting message for SparkleForge (an autonomous multi-agent research system).

Current time: {date_str} ({tz_name})
Time of day: {"Morning" if 5 <= hour < 12 else "Afternoon" if 12 <= hour < 18 else "Evening" if 18 <= hour < 22 else "Night"}
Language: {language}

Requirements:
- Keep it very brief (maximum 10 words)
- Use the appropriate language ({language})
- Match the time of day (morning/afternoon/evening/night)
- Be professional but friendly
- Do NOT include "REPL Mode" or "SparkleForge" in the message
- Return ONLY the greeting message, nothing else

Example outputs:
- Morning in Korean: "좋은 아침입니다"
- Afternoon in English: "Good afternoon"
- Evening in Japanese: "こんばんは"

Generate the greeting:"""

            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.CREATIVE,
                system_message="You are a helpful assistant that generates brief, culturally appropriate greetings.",
            )

            greeting = result.content.strip()
            # 따옴표 제거
            greeting = greeting.strip("\"'")
            # 첫 줄만 사용
            if "\n" in greeting:
                greeting = greeting.split("\n")[0].strip()

            return greeting if greeting else "Welcome"

        except Exception as e:
            logger.debug(f"Failed to generate greeting: {e}")
            # 기본 인사 메시지
            hour = datetime.now().hour
            if 5 <= hour < 12:
                return "Good morning"
            elif 12 <= hour < 18:
                return "Good afternoon"
            elif 18 <= hour < 22:
                return "Good evening"
            else:
                return "Good night"

    async def _show_banner(self):
        """시작 배너 표시."""
        # 인사 메시지 가져오기
        greeting = await self._get_greeting_message()

        # 배너 내용 구성
        banner_content = Text()
        banner_content.append("⚒️  ", style="bold yellow")
        banner_content.append(greeting, style="bold cyan")

        # Available Commands를 박스 안에 포함
        commands_text = Text()
        commands_text.append("Available Commands:\n", style="bold")
        commands_text.append("  ", style="dim")
        commands_text.append("research <query>", style="cyan")
        commands_text.append("  - Start a research task\n", style="dim")
        commands_text.append("  ", style="dim")
        commands_text.append("session list", style="cyan")
        commands_text.append("      - List all sessions\n", style="dim")
        commands_text.append("  ", style="dim")
        commands_text.append("context show", style="cyan")
        commands_text.append("     - Show project context\n", style="dim")
        commands_text.append("  ", style="dim")
        commands_text.append("help", style="cyan")
        commands_text.append("             - Show help message\n", style="dim")
        commands_text.append("  ", style="dim")
        commands_text.append("exit", style="cyan")
        commands_text.append("             - Exit REPL\n", style="dim")
        commands_text.append("\n", style="dim")
        commands_text.append(
            "Type 'help' for detailed command information", style="dim"
        )

        # 전체 내용 합치기
        full_content = Text()
        full_content.append(banner_content)
        full_content.append("\n\n", style="dim")
        full_content.append(commands_text)

        banner = Panel(
            full_content,
            border_style="cyan",
            padding=(1, 2),
            title="[bold cyan]Autonomous Multi-Agent Research System[/bold cyan]",
            subtitle="[dim]Version 1.0.0[/dim]",
        )
        self.console.print(banner)
        self.console.print()

    async def handle_command(self, text: str):
        """명령어 처리."""
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
                        self.console.print(
                            f"[dim]Type '{command} help' for subcommands[/dim]"
                        )
                        return

                    subcommand = parts[1].lower()
                    if subcommand in handler:
                        await handler[subcommand](self, parts[2:])
                    else:
                        self.console.print(
                            f"[red]Unknown subcommand: {subcommand}[/red]"
                        )
                        self.console.print(
                            f"[dim]Available: {', '.join(handler.keys())}[/dim]"
                        )
                else:
                    # 직접 명령어
                    await handler(self, parts[1:])
            else:
                routed = await self._try_route_command(text)
                if routed:
                    return
                # 연구 요청으로 처리 (명령어가 없으면)
                # 중복 출력 방지: research_command에서 이미 출력하므로 여기서는 호출만
                await self.command_handlers["research"](self, [text])

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
