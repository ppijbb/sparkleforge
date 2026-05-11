"""연구 요청 명령어"""

import logging
from typing import List

from rich.panel import Panel

logger = logging.getLogger(__name__)


async def research_command(cli, args: List[str]):
    """연구 요청 실행."""
    if not args:
        cli.console.print("[red]Usage: research <query>[/red]")
        cli.console.print("[dim]Or just type your query directly[/dim]")
        return

    query = " ".join(args)

    # 중복 출력 제거: Research Request는 한 번만 출력
    cli.console.print(f"\n[bold cyan]🔬 Research Request:[/bold cyan] {query}\n")

    try:
        from src.core.autonomous_orchestrator import AutonomousOrchestrator

        orchestrator = AutonomousOrchestrator()

        # 로그 핸들러를 통한 진행 상황 표시
        import logging

        class REPLProgressHandler(logging.Handler):
            """REPL용 진행 상황 핸들러."""

            def __init__(self, console):
                super().__init__()
                self.console = console
                self.last_message = None
                self.stage_patterns = {
                    "🔍": "analysis",
                    "📋": "planning",
                    "⚙️": "execution",
                    "🗜️": "compression",
                    "✅": "verification",
                    "📊": "evaluation",
                    "📝": "synthesis",
                }

            def emit(self, record):
                """로그 레코드 처리."""
                try:
                    msg = self.format(record)

                    # 중복 메시지 방지
                    if msg == self.last_message:
                        return
                    self.last_message = msg

                    # 특정 패턴만 표시 (진행 상황 관련)
                    if any(
                        keyword in msg
                        for keyword in [
                            "Thinking",
                            "Analyzing",
                            "Planning",
                            "Executing",
                            "Compressing",
                            "Verifying",
                            "Evaluating",
                            "Synthesizing",
                            "Starting",
                            "Completed",
                            "Searching",
                            "Researching",
                            "Gathering",
                            "Processing",
                            "Reviewing",
                            "Checking",
                        ]
                    ):
                        # 아이콘 추출 및 색상 적용
                        icon = None
                        color = "white"

                        for ic, stage in self.stage_patterns.items():
                            if ic in msg:
                                icon = ic
                                break

                        if not icon:
                            # 메시지에서 단계 추론
                            if "Analyzing" in msg or "analysis" in msg.lower():
                                icon = "🔍"
                                color = "cyan"
                            elif "Planning" in msg or "plan" in msg.lower():
                                icon = "📋"
                                color = "blue"
                            elif (
                                "Executing" in msg
                                or "execution" in msg.lower()
                                or "Searching" in msg
                                or "Researching" in msg
                            ):
                                icon = "⚙️"
                                color = "yellow"
                            elif "Compressing" in msg:
                                icon = "🗜️"
                                color = "magenta"
                            elif "Verifying" in msg or "verification" in msg.lower():
                                icon = "✅"
                                color = "green"
                            elif "Evaluating" in msg:
                                icon = "📊"
                                color = "blue"
                            elif "Synthesizing" in msg or "synthesis" in msg.lower():
                                icon = "📝"
                                color = "cyan"
                            elif "Completed" in msg or "complete" in msg.lower():
                                icon = "✨"
                                color = "green"

                        # 메시지 정리 (불필요한 부분 제거)
                        clean_msg = msg
                        # 로그 레벨 제거
                        clean_msg = (
                            clean_msg.split(" - ", 1)[-1] if " - " in clean_msg else clean_msg
                        )
                        # Research Request 중복 제거
                        if "Research Request:" in clean_msg:
                            return

                        # 출력
                        if icon:
                            self.console.print(f"[{color}]{icon} {clean_msg}[/{color}]")
                        else:
                            self.console.print(f"[dim]{clean_msg}[/dim]")
                except Exception:
                    pass

        # 핸들러 추가
        progress_handler = REPLProgressHandler(cli.console)
        progress_handler.setLevel(logging.INFO)

        # 특정 로거에만 핸들러 추가
        orchestrator_logger = logging.getLogger("src.core.autonomous_orchestrator")
        orchestrator_logger.addHandler(progress_handler)
        orchestrator_logger.setLevel(logging.INFO)

        try:
            # run_research 메서드 사용
            result = await orchestrator.run_research(query)
        finally:
            # 핸들러 제거
            orchestrator_logger.removeHandler(progress_handler)

        # 결과 출력
        if isinstance(result, dict):
            if "final_synthesis" in result:
                content = result["final_synthesis"].get("content", "")
                if content:
                    cli.console.print(Panel(content, title="Research Result", border_style="green"))
            elif "content" in result:
                cli.console.print(
                    Panel(result["content"], title="Research Result", border_style="green")
                )
            elif "deliverable" in result:
                deliverable = result.get("deliverable", {})
                content = (
                    deliverable.get("content", "")
                    if isinstance(deliverable, dict)
                    else str(deliverable)
                )
                if content:
                    cli.console.print(Panel(content, title="Research Result", border_style="green"))
                else:
                    cli.console.print("[green]✅ Research completed[/green]")
            else:
                cli.console.print("[green]✅ Research completed[/green]")
                if result:
                    cli.console.print(f"[dim]Result keys: {list(result.keys())[:5]}...[/dim]")
        else:
            cli.console.print("[green]✅ Research completed[/green]")
            if result:
                cli.console.print(str(result))

    except AttributeError as e:
        if "execute_full_research_workflow" in str(e) or "run_research" in str(e):
            cli.console.print(
                "[red]❌ Research method not available. Please check the orchestrator implementation.[/red]"
            )
            logger.debug(f"Research method error: {e}", exc_info=True)
        else:
            logger.error(f"Research failed: {e}", exc_info=True)
            cli.console.print(f"[red]❌ Research failed: {e}[/red]")
    except Exception as e:
        logger.error(f"Research failed: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Research failed: {e}[/red]")
