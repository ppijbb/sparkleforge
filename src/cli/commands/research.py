"""연구 요청 명령어"""

import logging
from typing import List

from src.cli.ui.markdown_stream import render_markdown_result
from src.cli.ui.spinner import stage_status

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

        with stage_status(cli.console, "🔬 Researching...", ["src.core.autonomous_orchestrator"]):
            result = await orchestrator.run_research(query)

        # 결과 출력
        if isinstance(result, dict):
            if "final_synthesis" in result:
                content = result["final_synthesis"].get("content", "")
                if content:
                    render_markdown_result(cli.console, content, title="Research Result")
            elif "content" in result:
                render_markdown_result(cli.console, result["content"], title="Research Result")
            elif "deliverable" in result:
                deliverable = result.get("deliverable", {})
                content = (
                    deliverable.get("content", "")
                    if isinstance(deliverable, dict)
                    else str(deliverable)
                )
                if content:
                    render_markdown_result(cli.console, content, title="Research Result")
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
