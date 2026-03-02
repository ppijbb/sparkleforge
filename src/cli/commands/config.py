"""설정 관리 명령어"""

import logging
from typing import Any, List

from rich.panel import Panel

logger = logging.getLogger(__name__)

# Keys that must not be printed (API keys, tokens, passwords)
_SECRET_CONFIG_KEYS = frozenset(
    {
        "api_key",
        "openrouter_api_key",
        "claude_code_api_key",
        "gemini_cli_api_key",
        "password",
        "secret",
        "token",
    }
)


def _redact_secret(key: str, value: Any) -> Any:
    """Return redacted value for known secret keys."""
    key_lower = key.lower().replace(".", "_")
    if any(secret in key_lower for secret in _SECRET_CONFIG_KEYS):
        return "***" if value else value
    if isinstance(value, dict):
        return {k: _redact_secret(k, v) for k, v in value.items()}
    return value


async def config_show_command(cli, args: List[str]):
    """설정 표시."""
    try:
        from src.core.researcher_config import get_research_config

        config = get_research_config()

        config_text = f"""
[bold]LLM Provider:[/bold] {getattr(config, "llm_provider", "N/A")}
[bold]LLM Model:[/bold] {getattr(config, "llm_model", "N/A")}
[bold]Max Tokens:[/bold] {getattr(config, "max_tokens", "N/A")}
[bold]Temperature:[/bold] {getattr(config, "temperature", "N/A")}
"""

        cli.console.print(
            Panel(config_text.strip(), title="Configuration", border_style="cyan")
        )

    except Exception as e:
        logger.error(f"Failed to show config: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to show config: {e}[/red]")


async def config_set_command(cli, args: List[str]):
    """설정 변경."""
    if len(args) < 2:
        cli.console.print("[red]Usage: config set <key> <value>[/red]")
        return

    key = args[0]
    value = args[1]

    cli.console.print("[yellow]⚠️  Config setting is not yet implemented[/yellow]")
    cli.console.print(f"[dim]Key: {key}, Value: {value}[/dim]")


async def config_get_command(cli, args: List[str]):
    """설정 값 가져오기."""
    if not args:
        cli.console.print("[red]Usage: config get <key>[/red]")
        return

    key = args[0]

    try:
        from src.core.researcher_config import get_research_config

        config = get_research_config()
        value = getattr(config, key, None)

        if value is not None:
            value = _redact_secret(key, value)
            cli.console.print(f"[green]{key}: {value}[/green]")
        else:
            cli.console.print(f"[yellow]Config key not found: {key}[/yellow]")

    except Exception as e:
        logger.error(f"Failed to get config: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to get config: {e}[/red]")
