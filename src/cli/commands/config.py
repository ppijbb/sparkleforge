"""설정 관리 명령어"""

import logging
import os
import re
from typing import Any, List, Optional

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
        from src.utils.config_manager import get_config_manager

        cfg_mgr = get_config_manager()
        if cfg_mgr and hasattr(cfg_mgr, "show_config"):
            await cfg_mgr.show_config(cli, args)
            return
    except Exception:
        pass

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
    if len(args) < 1:
        cli.console.print("[red]Usage: config set <key> [value] [--depth <preset>] [--model <model>] [--provider <provider>][/red]")
        return

    try:
        from src.core.researcher_config import get_research_config, save_config
        cfg = get_research_config()
    except Exception as e:
        logger.error(f"Failed to load config for setting: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to load config: {e}[/red]")
        return

    key = args[0]
    value = args[1] if len(args) > 1 else None

    # Handle depth alias / preset shortcut or standard key-value
    if key == "depth" and value:
        if hasattr(cfg, "research") and cfg.research is not None and hasattr(cfg.research, "research_depth"):
            cfg.research.research_depth.default_preset = value
            cli.console.print(f"[green]✓ Set default research depth preset to: {value}[/green]")
        else:
            cli.console.print("[yellow]⚠️ research.research_depth configuration structure not found[/yellow]")
        try:
            save_config(cfg)
        except Exception as ex:
            logger.warning(f"Could not persist config save: {ex}")
        return

    if value is None:
        cli.console.print(f"[red]Usage: config set {key} <value>[/red]")
        return

    # Support dotted path setting or direct attribute
    parts = key.split(".")
    curr = cfg
    for part in parts[:-1]:
        if isinstance(curr, dict):
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
        else:
            if not hasattr(curr, part):
                setattr(curr, part, {})
            curr = getattr(curr, part)

    leaf = parts[-1]
    target_val = value

    # Coerce boolean/numeric if appropriate
    if isinstance(target_val, str):
        lower_val = target_val.lower()
        if lower_val in ("true", "yes", "1", "on"):
            target_val = True
        elif lower_val in ("false", "no", "0", "off"):
            target_val = False
        else:
            try:
                if "." in target_val:
                    target_val = float(target_val)
                else:
                    target_val = int(target_val)
            except ValueError:
                pass

    if isinstance(curr, dict):
        curr[leaf] = target_val
    else:
        setattr(curr, leaf, target_val)

    # Handle model cascade logic safely (checking if env_k is genuinely set or not)
    if key in ("llm_model", "model", "planning_model", "reasoning_model"):
        role_env_map = {
            "llm_model": ["LLM_MODEL", "PLANNING_MODEL", "REASONING_MODEL", "GENERATION_MODEL"],
            "model": ["LLM_MODEL", "PLANNING_MODEL", "REASONING_MODEL", "GENERATION_MODEL"],
            "planning_model": ["PLANNING_MODEL"],
            "reasoning_model": ["REASONING_MODEL"],
        }
        env_keys = role_env_map.get(key, ["LLM_MODEL"])
        new_model = str(target_val)
        for env_k in env_keys:
            old_val = os.getenv(env_k)
            if env_k not in os.environ or (old_val is not None and os.environ.get(env_k) == old_val):
                os.environ[env_k] = new_model

    try:
        save_config(cfg)
        cli.console.print(f"[green]✓ Config updated: {key} = {value}[/green]")
    except Exception as e:
        logger.error(f"Failed to save config: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to save config: {e}[/red]")


def _parse_bool(val: Any) -> bool:
    """Parse boolean values consistently across CLI commands."""
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).lower() not in {"0", "false", "no", "off", "f", "n"}


def _sanitize_embedded_cli_flags(query_str: str) -> tuple[str, dict[str, Any]]:
    """Extract embedded flags like --depth=quick or --depth quick from query string."""
    overrides: dict[str, Any] = {}
    
    # Match patterns like --depth=value or --depth value
    # Regex to match --flag=value or --flag value
    depth_match = re.search(r'(?i)\s+--depth(?:\s+|\=)([^\s]+)', query_str)
    if depth_match:
        overrides["depth"] = depth_match.group(1)
        query_str = query_str.replace(depth_match.group(0), "")
    elif query_str.lstrip().startswith("--depth"):
        parts = query_str.lstrip().split(maxsplit=2)
        if len(parts) > 1 and not parts[1].startswith("-"):
            overrides["depth"] = parts[1]
            query_str = query_str.replace(parts[0] + " " + parts[1], "", 1)
        else:
            query_str = query_str.replace("--depth", "", 1)

    model_match = re.search(r'(?i)\s+--(?:model|llm-model)(?:\s+|\=)([^\s]+)', query_str)
    if model_match:
        overrides["model"] = model_match.group(1)
        query_str = query_str.replace(model_match.group(0), "")

    autopilot_match = re.search(r'(?i)\s+--autopilot(?:\s+|\=)([^\s]+)', query_str)
    if autopilot_match:
        overrides["autopilot"] = autopilot_match.group(1)
        query_str = query_str.replace(autopilot_match.group(0), "")

    return query_str.strip(), overrides


async def handle_run_command(cli, args: List[str]):
    """Run command handler supporting robust flags and execution."""
    if not args:
        cli.console.print("[red]Usage: run <query> [--depth <preset>] [--model <model>] [--autopilot <bool>][/red]")
        return

    raw_query = " ".join(args)
    query, embedded_flags = _sanitize_embedded_cli_flags(raw_query)

    autopilot_override = embedded_flags.get("autopilot")
    is_autopilot = _parse_bool(autopilot_override) if autopilot_override is not None else getattr(cli, "autopilot", False)

    depth_override = embedded_flags.get("depth")
    model_override = embedded_flags.get("model")

    try:
        from src.core.agent_harness import AgentHarness
        from src.core.researcher_config import get_research_config

        config = get_research_config()
        if depth_override:
            if hasattr(config, "research") and config.research is not None and hasattr(config.research, "research_depth"):
                config.research.research_depth.default_preset = depth_override
        if model_override:
            setattr(config, "llm_model", model_override)

        harness = AgentHarness(config=config)
        cli.console.print(f"[cyan]🚀 Running query: {query}[/cyan]")
        
        if hasattr(harness, "run_async"):
            result = await harness.run_async(query, autopilot=is_autopilot)
        elif hasattr(harness, "run"):
            result = harness.run(query, autopilot=is_autopilot)
        else:
            from src.cli.commands.run import run_command
            result = await run_command([query], config)

        cli.console.print(f"[green]✓ Result: {result}[/green]")
    except Exception as e:
        logger.error(f"Failed to execute run command: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Execution failed: {e}[/red]")


async def config_get_command(cli, args: List[str]):
    """설정 값 가져오기."""
    if not args:
        cli.console.print("[red]Usage: config get <key>[/red]")
        return

    key = args[0]

    try:
        from src.core.researcher_config import get_research_config

        config = get_research_config()
        
        # Dotted-path traversal supporting dicts first to avoid built-in dict methods like keys/values/items
        parts = key.split(".")
        curr = config
        value = None
        found = True

        for part in parts:
            if isinstance(curr, dict):
                if part in curr:
                    curr = curr[part]
                else:
                    found = False
                    break
            elif hasattr(curr, "__dict__") or hasattr(curr, "__slots__"):
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    found = False
                    break
            else:
                found = False
                break

        if found:
            value = curr
        else:
            value = None

        if value is not None:
            value = _redact_secret(key, value)
            cli.console.print(f"[green]{key}: {value}[/green]")
        else:
            cli.console.print(f"[yellow]Config key not found: {key}[/yellow]")

    except Exception as e:
        logger.error(f"Failed to get config: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to get config: {e}[/red]")


async def config_command(cli, args: List[str]):
    """설정 관리 진입점."""
    if not args:
        await config_show_command(cli, args)
        return

    subcommand = args[0].lower()
    sub_args = args[1:]

    if subcommand == "show":
        await config_show_command(cli, sub_args)
    elif subcommand == "set":
        await config_set_command(cli, sub_args)
    elif subcommand == "get":
        await config_get_command(cli, sub_args)
    else:
        cli.console.print(f"[red]Unknown config subcommand: {subcommand}[/red]")
        cli.console.print("[yellow]Usage: config [show|get|set] ...[/yellow]")
