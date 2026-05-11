"""Interactive CLI (완전 자동형 SparkleForge)

대화형 터미널 인터페이스, 명령 히스토리, 자동 완성, 컬러 출력, 진행 상황 표시 기능 제공.
gemini-cli의 CLI 패턴을 참고하여 구현.
"""

import asyncio
import logging
import shlex
import uuid
from datetime import datetime
from typing import List

from rich.console import Console

from src.core.execution_registry import ExecutionRegistry, RegisteredCommand
from src.core.prompt_router import PromptRouter, RouteTargetType
from src.core.trust_gate import get_current_trust_context

logger = logging.getLogger(__name__)


class InteractiveCLI:
    """대화형 CLI 인터페이스."""

    def __init__(self):
        """초기화."""
        self.history: List[str] = []
        self.console = Console()
        self.prompt_router = PromptRouter()
        self.context_loader = None
        self.checkpoint_manager = None

        try:
            from src.core.checkpoint_manager import CheckpointManager
            from src.core.context_loader import ContextLoader

            self.context_loader = ContextLoader()
            self.checkpoint_manager = CheckpointManager()
        except Exception as e:
            logger.warning(f"Failed to initialize context/checkpoint: {e}")

    async def run(self):
        """대화형 CLI 실행."""
        print("\n" + "=" * 80)
        print("⚒️  SparkleForge - Interactive Mode")
        print("=" * 80)
        print("Type 'help' for commands, 'exit' to quit\n")

        # 컨텍스트 로드
        if self.context_loader:
            try:
                context = await self.context_loader.load_context()
                if context:
                    print("📄 Project context loaded from SPARKLEFORGE.md\n")
            except Exception as e:
                logger.debug(f"Failed to load context: {e}")

        while True:
            try:
                # 프롬프트 표시
                prompt = "sparkleforge> "
                user_input = input(prompt).strip()

                if not user_input:
                    continue

                # 히스토리에 추가
                self.history.append(user_input)

                # 명령 처리
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Goodbye! 👋")
                    break
                elif user_input.lower() == "help":
                    self._show_help()
                elif user_input.lower().startswith("checkpoint"):
                    await self._handle_checkpoint(user_input)
                elif user_input.lower().startswith("context"):
                    await self._handle_context(user_input)
                elif user_input.lower().startswith("session"):
                    await self._handle_session(user_input)
                elif user_input.lower().startswith("task"):
                    await self._handle_task(user_input)
                elif user_input.lower().startswith("schedule"):
                    await self._handle_schedule(user_input)
                else:
                    routed = await self._try_route_command(user_input)
                    if routed:
                        continue
                    # 연구 요청으로 처리
                    await self._handle_research_request(user_input)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except EOFError:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error in interactive CLI: {e}", exc_info=True)
                print(f"❌ Error: {e}")

    def _show_help(self):
        """도움말 표시."""
        help_text = """
Available commands:
  help                    - Show this help message
  exit / quit / q         - Exit interactive mode
  
  # Checkpoint commands
  checkpoint list         - List all checkpoints
  checkpoint save [name]  - Save current conversation as checkpoint
  checkpoint load <id>    - Load checkpoint by ID
  checkpoint delete <id>  - Delete checkpoint by ID
  
  # Context commands
  context show            - Show loaded project context
  context reload          - Reload project context
  
  # Session management (Backoffice)
  session list            - List all sessions
  session search <query>  - Search sessions
  session show <id>       - Show session details
  session pause <id>      - Pause session
  session resume <id>     - Resume paused session
  session cancel <id>     - Cancel session
  session delete <id>     - Delete session
  session restore <id>    - Restore session (continuity)
  session tasks <id>      - List tasks for session
  session stats           - Show session statistics
  
  # Task control
  task pause <session_id> <task_id>   - Pause task
  task resume <session_id> <task_id>  - Resume task
  task cancel <session_id> <task_id>  - Cancel task
  task show <session_id> <task_id>    - Show task details
  
  # Schedule management (Cron-based)
  schedule list                        - List all schedules
  schedule create <name> <cron> <query> - Create schedule
  schedule show <id>                  - Show schedule details
  schedule pause <id>                 - Pause schedule
  schedule resume <id>                - Resume schedule
  schedule delete <id>                - Delete schedule
  schedule history [id]               - Show execution history
  schedule stats                      - Show schedule statistics
  
  <research query>        - Execute research request
  Example: "Latest AI trends in 2025"
        """
        print(help_text)

    async def _handle_checkpoint(self, command: str):
        """체크포인트 명령 처리."""
        if not self.checkpoint_manager:
            print("❌ Checkpoint manager not available")
            return

        parts = command.split()
        if len(parts) < 2:
            print("Usage: checkpoint <list|save|load|delete> [args]")
            return

        action = parts[1].lower()

        if action == "list":
            checkpoints = await self.checkpoint_manager.list_checkpoints(limit=10)
            if checkpoints:
                print("\n📋 Recent checkpoints:")
                for cp in checkpoints:
                    print(f"  {cp['checkpoint_id']} - {cp['timestamp']}")
            else:
                print("No checkpoints found")

        elif action == "save":
            name = parts[2] if len(parts) > 2 else None
            # 현재 상태 저장 (간단한 구현)
            checkpoint_id = await self.checkpoint_manager.save_checkpoint(
                state={"history": self.history[-10:]},  # 최근 10개 명령
                metadata={"name": name, "timestamp": datetime.now().isoformat()},
            )
            print(f"✅ Checkpoint saved: {checkpoint_id}")

        elif action == "load":
            if len(parts) < 3:
                print("Usage: checkpoint load <checkpoint_id>")
                return
            checkpoint_id = parts[2]
            restored = await self.checkpoint_manager.restore_checkpoint(checkpoint_id)
            if restored:
                print(f"✅ Checkpoint restored: {checkpoint_id}")
                if "history" in restored:
                    print(f"   History: {len(restored['history'])} commands")
            else:
                print(f"❌ Failed to restore checkpoint: {checkpoint_id}")

        elif action == "delete":
            if len(parts) < 3:
                print("Usage: checkpoint delete <checkpoint_id>")
                return
            checkpoint_id = parts[2]
            if await self.checkpoint_manager.delete_checkpoint(checkpoint_id):
                print(f"✅ Checkpoint deleted: {checkpoint_id}")
            else:
                print(f"❌ Failed to delete checkpoint: {checkpoint_id}")

        else:
            print(f"Unknown checkpoint action: {action}")

    async def _handle_context(self, command: str):
        """컨텍스트 명령 처리."""
        if not self.context_loader:
            print("❌ Context loader not available")
            return

        parts = command.split()
        if len(parts) < 2:
            print("Usage: context <show|reload>")
            return

        action = parts[1].lower()

        if action == "show":
            context = await self.context_loader.load_context()
            if context:
                print("\n📄 Project Context:")
                print("=" * 80)
                print(context[:1000])  # 처음 1000자만
                if len(context) > 1000:
                    print(f"\n... ({len(context) - 1000} more characters)")
            else:
                print("No context file found (SPARKLEFORGE.md)")

        elif action == "reload":
            self.context_loader.clear_cache()
            context = await self.context_loader.load_context()
            if context:
                print("✅ Context reloaded")
            else:
                print("No context file found")

        else:
            print(f"Unknown context action: {action}")

    async def _handle_session(self, command: str):
        """세션 관리 명령 처리."""
        from src.core.session_control import SessionStatus, get_session_control

        session_control = get_session_control()

        parts = command.split()
        if len(parts) < 2:
            print(
                "Usage: session <list|search|show|pause|resume|cancel|delete|restore|tasks|stats> [args]"
            )
            return

        action = parts[1].lower()

        if action == "list":
            sessions = await session_control.search_sessions(limit=20)
            if sessions:
                print("\n📋 Sessions:")
                print("=" * 100)
                for s in sessions:
                    status_icon = {
                        SessionStatus.ACTIVE: "🟢",
                        SessionStatus.PAUSED: "🟡",
                        SessionStatus.COMPLETED: "✅",
                        SessionStatus.FAILED: "❌",
                        SessionStatus.CANCELLED: "🚫",
                        SessionStatus.WAITING: "⏳",
                    }.get(s.status, "⚪")
                    print(
                        f"{status_icon} {s.session_id[:20]}... | {s.status.value:10} | {s.progress_percentage:5.1f}% | {s.last_activity.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    if s.user_query:
                        print(f"   Query: {s.user_query[:60]}...")
            else:
                print("No sessions found")

        elif action == "search":
            query = " ".join(parts[2:]) if len(parts) > 2 else None
            sessions = await session_control.search_sessions(query=query, limit=20)
            if sessions:
                print(f"\n🔍 Found {len(sessions)} sessions:")
                for s in sessions:
                    print(f"  {s.session_id} | {s.status.value} | {s.user_query or 'N/A'}")
            else:
                print("No sessions found")

        elif action == "show":
            if len(parts) < 3:
                print("Usage: session show <session_id>")
                return
            session_id = parts[2]
            session_info = await session_control.get_session(session_id)
            if session_info:
                print(f"\n📊 Session: {session_id}")
                print("=" * 80)
                print(f"Status: {session_info.status.value}")
                print(f"Created: {session_info.created_at}")
                print(f"Last Activity: {session_info.last_activity}")
                print(f"Progress: {session_info.progress_percentage:.1f}%")
                if session_info.user_query:
                    print(f"Query: {session_info.user_query}")
                if session_info.current_task:
                    print(f"Current Task: {session_info.current_task}")
                print(f"Errors: {session_info.error_count}, Warnings: {session_info.warning_count}")
            else:
                print(f"Session not found: {session_id}")

        elif action == "pause":
            if len(parts) < 3:
                print("Usage: session pause <session_id>")
                return
            session_id = parts[2]
            if await session_control.pause_session(session_id):
                print(f"✅ Session paused: {session_id}")
            else:
                print(f"❌ Failed to pause session: {session_id}")

        elif action == "resume":
            if len(parts) < 3:
                print("Usage: session resume <session_id>")
                return
            session_id = parts[2]
            if await session_control.resume_session(session_id):
                print(f"✅ Session resumed: {session_id}")
            else:
                print(f"❌ Failed to resume session: {session_id}")

        elif action == "cancel":
            if len(parts) < 3:
                print("Usage: session cancel <session_id>")
                return
            session_id = parts[2]
            if await session_control.cancel_session(session_id):
                print(f"✅ Session cancelled: {session_id}")
            else:
                print(f"❌ Failed to cancel session: {session_id}")

        elif action == "delete":
            if len(parts) < 3:
                print("Usage: session delete <session_id>")
                return
            session_id = parts[2]
            confirm = input(f"Delete session {session_id}? (yes/no): ").strip().lower()
            if confirm == "yes":
                if await session_control.delete_session(session_id):
                    print(f"✅ Session deleted: {session_id}")
                else:
                    print(f"❌ Failed to delete session: {session_id}")
            else:
                print("Cancelled")

        elif action == "restore":
            if len(parts) < 3:
                print("Usage: session restore <session_id>")
                return
            session_id = parts[2]
            restored = await session_control.restore_session(session_id)
            if restored:
                print(f"✅ Session restored: {session_id}")
            else:
                print(f"❌ Failed to restore session: {session_id}")

        elif action == "tasks":
            if len(parts) < 3:
                print("Usage: session tasks <session_id>")
                return
            session_id = parts[2]
            tasks = session_control.get_session_tasks(session_id)
            if tasks:
                print(f"\n📋 Tasks for {session_id}:")
                for task in tasks:
                    print(
                        f"  {task.task_id[:20]}... | {task.status.value:10} | {task.task_type:15} | {task.description[:40]}"
                    )
            else:
                print(f"No tasks found for session: {session_id}")

        elif action == "stats":
            stats = session_control.get_session_statistics()
            print("\n📊 Session Statistics:")
            print("=" * 80)
            print(f"Active Sessions: {stats['active_sessions']}")
            print(f"Total Tasks: {stats['total_tasks']}")
            print(f"Status Distribution: {stats['status_distribution']}")

        else:
            print(f"Unknown session action: {action}")

    async def _handle_task(self, command: str):
        """작업 제어 명령 처리."""
        from src.core.session_control import get_session_control

        session_control = get_session_control()

        parts = command.split()
        if len(parts) < 2:
            print("Usage: task <pause|resume|cancel|show> <session_id> <task_id>")
            return

        action = parts[1].lower()

        if len(parts) < 4:
            print(f"Usage: task {action} <session_id> <task_id>")
            return

        session_id = parts[2]
        task_id = parts[3]

        if action == "pause":
            if await session_control.pause_task(session_id, task_id):
                print(f"✅ Task paused: {task_id}")
            else:
                print(f"❌ Failed to pause task: {task_id}")

        elif action == "resume":
            if await session_control.resume_task(session_id, task_id):
                print(f"✅ Task resumed: {task_id}")
            else:
                print(f"❌ Failed to resume task: {task_id}")

        elif action == "cancel":
            if await session_control.cancel_task(session_id, task_id):
                print(f"✅ Task cancelled: {task_id}")
            else:
                print(f"❌ Failed to cancel task: {task_id}")

        elif action == "show":
            task_info = session_control.get_task(session_id, task_id)
            if task_info:
                print(f"\n📋 Task: {task_id}")
                print("=" * 80)
                print(f"Status: {task_info.status.value}")
                print(f"Type: {task_info.task_type}")
                print(f"Description: {task_info.description}")
                print(f"Progress: {task_info.progress:.1f}%")
                print(f"Created: {task_info.created_at}")
                if task_info.started_at:
                    print(f"Started: {task_info.started_at}")
                if task_info.completed_at:
                    print(f"Completed: {task_info.completed_at}")
                if task_info.error:
                    print(f"Error: {task_info.error}")
            else:
                print(f"Task not found: {task_id}")

        else:
            print(f"Unknown task action: {action}")

    async def _handle_schedule(self, command: str):
        """스케줄 관리 명령 처리."""
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

        try:
            parts = shlex.split(command)
        except ValueError as e:
            print(f"❌ Invalid schedule command: {e}")
            return

        if len(parts) < 2:
            print("Usage: schedule <create|list|show|pause|resume|delete|history|stats|run>")
            return

        handlers = {
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
        }

        action = parts[1].lower()
        handler = handlers.get(action)
        if not handler:
            print(f"Unknown schedule action: {action}")
            print(f"Available: {', '.join(handlers.keys())}")
            return

        await handler(self, parts[2:])

    async def _handle_research_request(self, request: str):
        """연구 요청 처리."""
        from src.core.session_control import SessionStatus, get_session_control

        session_control = get_session_control()

        # 세션 ID 생성
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # 활성 세션 등록
        session_control.register_active_session(session_id, request)

        print(f"\n🔬 Processing: {request}")
        print(f"📋 Session ID: {session_id}\n")

        try:
            from src.core.autonomous_orchestrator import AutonomousOrchestrator

            orchestrator = AutonomousOrchestrator()

            # 세션 제어 체크를 위한 래퍼
            async def check_and_wait():
                while not session_control.check_session_control(session_id):
                    await session_control.wait_for_resume(session_id, timeout=1.0)

            # 주기적으로 세션 제어 체크
            async def monitor_session():
                while True:
                    await check_and_wait()
                    await asyncio.sleep(0.5)

            # 모니터링 태스크 시작
            monitor_task = asyncio.create_task(monitor_session())

            # 스트리밍 출력을 위한 간단한 콜백
            async def progress_callback(message: str):
                session_control.update_session_progress(session_id, current_task=message)
                print(f"  {message}")

            result = await orchestrator.execute_full_research_workflow(request)

            # 모니터링 태스크 중지
            monitor_task.cancel()

            # 세션 상태 업데이트
            session_control.update_session_progress(session_id, progress=100.0)
            if session_id in session_control.active_sessions:
                session_control.active_sessions[session_id]["status"] = SessionStatus.COMPLETED

            print("\n" + "=" * 80)
            print("✅ Research Complete")
            print("=" * 80)

            if isinstance(result, dict):
                if "content" in result:
                    print(result["content"][:500])
                    if len(result.get("content", "")) > 500:
                        print("\n... (truncated)")
                elif "final_synthesis" in result:
                    synthesis = result["final_synthesis"]
                    if isinstance(synthesis, dict) and "content" in synthesis:
                        print(synthesis["content"][:500])
                else:
                    print("Result available (use --output to save)")
            else:
                print(str(result)[:500])

            print()

        except Exception as e:
            logger.error(f"Research request failed: {e}", exc_info=True)
            if session_id in session_control.active_sessions:
                session_control.active_sessions[session_id]["status"] = SessionStatus.FAILED
                session_control.active_sessions[session_id]["error_count"] += 1
            print(f"❌ Error: {e}")

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
        command: RegisteredCommand,
        user_input: str,
    ) -> bool:
        prompt_tokens = set(self.prompt_router._tokenize(user_input))
        if set(command.dispatch).issubset(prompt_tokens) and not command.requires_args:
            return True

        for alias in command.aliases:
            alias_tokens = set(self.prompt_router._tokenize(alias))
            if alias_tokens and alias_tokens.issubset(prompt_tokens) and not command.requires_args:
                return True

        return False

    async def _try_route_command(self, user_input: str) -> bool:
        """Attempt safe command routing before defaulting to research."""
        registry = await self._build_execution_registry()
        routes = await self.prompt_router.route(
            user_input,
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

        if not self._is_safe_command_route(command, user_input):
            return False

        await self._dispatch_routed_command(command.dispatch)
        return True

    async def _dispatch_routed_command(self, dispatch: tuple[str, ...]):
        """Dispatch a routed command back into the interactive handlers."""
        command = " ".join(dispatch)
        if command == "help":
            self._show_help()
            return
        if command.startswith("context"):
            await self._handle_context(command)
            return
        if command.startswith("session"):
            await self._handle_session(command)
            return
        if command.startswith("checkpoint"):
            await self._handle_checkpoint(command)
            return
        if command.startswith("schedule"):
            await self._handle_schedule(command)
