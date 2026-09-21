"""설정 관리 명령어"""

import logging
import os
import re
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

        cli.console.print(Panel(config_text.strip(), title="Configuration", border_style="cyan"))

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

    try:
        from src.core.researcher_config import get_research_config
        config = get_research_config()

        # Handle nested keys and setting
        parts = key.split(".")
        curr = config
        for part in parts[:-1]:
            if isinstance(curr, dict):
                if part not in curr:
                    curr[part] = {}
                curr = curr[part]
            else:
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    setattr(curr, part, {})
                    curr = getattr(curr, part)

        last_part = parts[-1]
        if isinstance(curr, dict):
            curr[last_part] = value
        else:
            setattr(curr, last_part, value)

        # Handle depth alias if applicable
        if key == "research_depth" or key == "depth":
            if hasattr(config, 'research') and hasattr(config.research, 'research_depth'):
                config.research.research_depth.default_preset = value

        # Model cascade logic with proper env check
        if key in ("llm_model", "model"):
            for env_k, old_val in [
                ("PLANNING_MODEL", getattr(config, "planning_model", value)),
                ("REASONING_MODEL", getattr(config, "reasoning_model", value)),
                ("VERIFICATION_MODEL", getattr(config, "verification_model", value)),
                ("GENERATION_MODEL", getattr(config, "generation_model", value)),
                ("COMPRESSION_MODEL", getattr(config, "compression_model", value)),
            ]:
                if env_k not in os.environ or os.environ.get(env_k) == old_val:
                    os.environ[env_k] = value

        cli.console.print(f"[green]Successfully set {key} to {value}[/green]")
    except Exception as e:
        logger.error(f"Failed to set config: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to set config: {e}[/red]")


def _parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() not in {"0", "false", "no", "off", "f", "n"}
    return bool(val)


def _sanitize_embedded_cli_flags(prompt: str) -> str:
    """Sanitize embedded CLI flags like --depth from prompt strings."""
    # Matches --depth followed by space or equals and a word/value, plus optional leading/trailing spaces
    pattern = r'(?:\s+|^)--depth(?:\s+|\=)\w+'
    return re.sub(pattern, '', prompt).strip()


async def handle_run_command(cli, args: List[str], autopilot_override: Any = None):
    """Handle run command execution with autopilot and flag sanitization."""
    try:
        is_autopilot = _parse_bool(autopilot_override) if autopilot_override is not None else False
        cli.console.print(f"[cyan]Running command with autopilot: {is_autopilot}[/cyan]")
    except Exception as e:
        logger.error(f"Failed to handle run command: {e}", exc_info=True)


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

        parts = key.split(".")
        curr = config
        found = True
        for part in parts:
            if isinstance(curr, dict):
                if part in curr:
                    curr = curr[part]
                else:
                    found = False
                    break
            elif hasattr(curr, part):
                curr = getattr(curr, part)
            else:
                found = False
                break
        value = curr if found else None

        if value is not None:
            value = _redact_secret(key, value)
            cli.console.print(f"[green]{key}: {value}[/green]")
        else:
            cli.console.print(f"[yellow]Config key not found: {key}[/yellow]")

    except Exception as e:
        logger.error(f"Failed to get config: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to get config: {e}[/red]")
