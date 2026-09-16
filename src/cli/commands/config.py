"""설정 관리 명령어 (Anvil Phase K / REPL 런타임 설정)."""

import logging
import os
from typing import Any, Dict, List, Optional

from rich.panel import Panel

logger = logging.getLogger(__name__)

# Keys that must not be printed or set directly (API keys, tokens, passwords)
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

# Friendly aliases mapping user-facing option names to canonical config paths
_CONFIG_ALIASES: Dict[str, str] = {
    "model": "llm.primary_model",
    "primary_model": "llm.primary_model",
    "temperature": "llm.temperature",
    "max_tokens": "llm.max_tokens",
    "budget_limit": "llm.budget_limit",
    "budget": "llm.budget_limit",
    "depth": "research.research_depth.default_preset",
    "research_depth": "research.research_depth.default_preset",
    "autopilot": "autopilot_mode",
    "autopilot_mode": "autopilot_mode",
    "approval_policy": "approval_policy",
}


def _is_secret_key(key: str) -> bool:
    """Check if key or any sub-part is considered a sensitive secret."""
    key_lower = key.lower().replace(".", "_")
    return any(secret in key_lower for secret in _SECRET_CONFIG_KEYS)


def _redact_secret(key: str, value: Any) -> Any:
    """Return redacted value for known secret keys."""
    if _is_secret_key(key):
        return "***" if value else value
    if isinstance(value, dict):
        return {k: _redact_secret(k, v) for k, v in value.items()}
    return value


def _get_root_config():
    """Ensure and return the global ResearcherSystemConfig instance."""
    from src.core import researcher_config

    if researcher_config.config is None:
        researcher_config.load_config_from_env()
    return researcher_config.config


def _parse_bool(value: str) -> Optional[bool]:
    """Parse string representation of boolean."""
    v = value.strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    return None


async def config_show_command(cli, args: List[str]):
    """설정 표시."""
    try:
        cfg = _get_root_config()
        from src.core.autonomous_orchestrator import _autopilot_mode_enabled

        llm_provider = getattr(cfg.llm, "provider", "N/A") if hasattr(cfg, "llm") else "N/A"
        llm_model = getattr(cfg.llm, "primary_model", "N/A") if hasattr(cfg, "llm") else "N/A"
        max_tokens = getattr(cfg.llm, "max_tokens", "N/A") if hasattr(cfg, "llm") else "N/A"
        temperature = getattr(cfg.llm, "temperature", "N/A") if hasattr(cfg, "llm") else "N/A"

        depth_preset = "auto"
        if hasattr(cfg, "research") and hasattr(cfg.research, "research_depth"):
            depth_preset = getattr(cfg.research.research_depth, "default_preset", "auto")

        autopilot_val = _autopilot_mode_enabled()
        budget_limit = (
            getattr(cfg.llm, "budget_limit", "unlimited")
            if hasattr(cfg, "llm")
            else "N/A"
        )
        approval_policy = os.getenv("APPROVAL_POLICY", "ask")

        config_text = f"""
[bold]LLM Provider:[/bold] {llm_provider}
[bold]LLM Model:[/bold] {llm_model}
[bold]Max Tokens:[/bold] {max_tokens}
[bold]Temperature:[/bold] {temperature}
[bold]Research Depth:[/bold] {depth_preset}
[bold]Autopilot Mode:[/bold] {autopilot_val}
[bold]Budget Limit:[/bold] {budget_limit}
[bold]Approval Policy:[/bold] {approval_policy}
"""
        cli.console.print(Panel(config_text.strip(), title="Configuration", border_style="cyan"))

    except Exception as e:
        logger.error(f"Failed to show config: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to show config: {e}[/red]")


async def config_set_command(cli, args: List[str]):
    """설정 변경 (현재 세션 및 런타임 환경에 적용)."""
    if len(args) < 2:
        cli.console.print("[red]Usage: config set <key> <value>[/red]")
        return

    raw_key = args[0]
    value_str = " ".join(args[1:])

    if _is_secret_key(raw_key):
        cli.console.print(
            f"[red]❌ Modifying API keys or secrets ('{raw_key}') via 'config set' is not allowed[/red]"
        )
        return

    canonical_key = _CONFIG_ALIASES.get(raw_key.lower(), raw_key)

    try:
        cfg = _get_root_config()

        # Handle special key: autopilot_mode
        if canonical_key == "autopilot_mode":
            parsed_bool = _parse_bool(value_str)
            if parsed_bool is None:
                cli.console.print(
                    f"[red]❌ Invalid boolean for autopilot_mode: '{value_str}'[/red]"
                )
                return
            from src.core.autonomous_orchestrator import _autopilot_mode_enabled
            old_val = _autopilot_mode_enabled()
            os.environ["SPARKLEFORGE_AUTOPILOT_MODE"] = "true" if parsed_bool else "false"
            cli.console.print(f"[green]✓ {raw_key}: {old_val} -> {parsed_bool}[/green]")
            return

        # Handle special key: approval_policy
        if canonical_key == "approval_policy":
            val_clean = value_str.lower().strip()
            if val_clean not in ("ask", "allowlist", "autopilot"):
                cli.console.print(
                    f"[red]❌ Invalid approval policy: '{value_str}'. "
                    "Allowed: ask, allowlist, autopilot[/red]"
                )
                return
            old_val = os.getenv("APPROVAL_POLICY", "ask")
            os.environ["APPROVAL_POLICY"] = val_clean
            cli.console.print(f"[green]✓ {raw_key}: {old_val} -> {val_clean}[/green]")
            return

        # Handle special key: depth preset
        if canonical_key in ("research.research_depth.default_preset", "research_depth", "depth"):
            val_clean = value_str.lower().strip()
            valid_presets = {"quick", "medium", "deep", "auto"}
            if val_clean not in valid_presets:
                allowed_str = ", ".join(sorted(valid_presets))
                cli.console.print(
                    f"[red]❌ Invalid research depth: '{value_str}'. Allowed: {allowed_str}[/red]"
                )
                return
            old_val = getattr(cfg.research.research_depth, "default_preset", "auto")
            cfg.research.research_depth.default_preset = val_clean
            os.environ["RESEARCH_DEPTH_PRESET"] = val_clean
            cli.console.print(f"[green]✓ {raw_key}: {old_val} -> {val_clean}[/green]")
            return

        # Handle model selection alias
        if canonical_key in ("llm.primary_model", "primary_model", "model"):
            old_val = cfg.llm.primary_model
            new_model = value_str.strip()
            if not new_model:
                cli.console.print("[red]❌ Primary model must not be empty[/red]")
                return
            cfg.llm.primary_model = new_model
            for role in (
                "planning_model",
                "reasoning_model",
                "verification_model",
                "generation_model",
                "compression_model",
            ):
                if hasattr(cfg.llm, role):
                    setattr(cfg.llm, role, new_model)
            os.environ["LLM_MODEL"] = new_model
            for env_k in (
                "PLANNING_MODEL",
                "REASONING_MODEL",
                "VERIFICATION_MODEL",
                "GENERATION_MODEL",
                "COMPRESSION_MODEL",
            ):
                os.environ[env_k] = new_model
            cli.console.print(f"[green]✓ {raw_key}: {old_val} -> {new_model}[/green]")
            return

        # Dotted path traversal on ResearcherSystemConfig
        parts = canonical_key.split(".")
        target = cfg
        for part in parts[:-1]:
            if not hasattr(target, part):
                cli.console.print(f"[red]❌ Unknown config key: '{raw_key}'[/red]")
                return
            target = getattr(target, part)

        last_part = parts[-1]
        if not hasattr(target, last_part):
            cli.console.print(f"[red]❌ Unknown config key: '{raw_key}'[/red]")
            return

        old_val = getattr(target, last_part)
        old_type = type(old_val) if old_val is not None else str

        # Convert value_str to target type
        if old_type is bool:
            converted: Any = _parse_bool(value_str)
            if converted is None:
                cli.console.print(f"[red]❌ Invalid boolean: '{value_str}'[/red]")
                return
        elif old_type is int:
            try:
                converted = int(value_str)
            except ValueError:
                cli.console.print(f"[red]❌ Invalid integer: '{value_str}'[/red]")
                return
        elif old_type is float:
            try:
                converted = float(value_str)
            except ValueError:
                cli.console.print(f"[red]❌ Invalid float: '{value_str}'[/red]")
                return
        else:
            converted = value_str.strip()

        setattr(target, last_part, converted)
        cli.console.print(f"[green]✓ {raw_key}: {old_val} -> {converted}[/green]")

    except Exception as e:
        logger.error(f"Failed to set config: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to set config: {e}[/red]")


async def config_get_command(cli, args: List[str]):
    """설정 값 가져오기."""
    if not args:
        cli.console.print("[red]Usage: config get <key>[/red]")
        return

    raw_key = args[0]
    canonical_key = _CONFIG_ALIASES.get(raw_key.lower(), raw_key)

    try:
        cfg = _get_root_config()

        # Handle special key: autopilot_mode
        if canonical_key == "autopilot_mode":
            from src.core.autonomous_orchestrator import _autopilot_mode_enabled
            val = _autopilot_mode_enabled()
            cli.console.print(f"[green]{raw_key}: {val}[/green]")
            return

        # Handle special key: approval_policy
        if canonical_key == "approval_policy":
            val = os.getenv("APPROVAL_POLICY", "ask")
            cli.console.print(f"[green]{raw_key}: {val}[/green]")
            return

        # Direct attribute or dotted path traversal
        parts = canonical_key.split(".")
        curr = cfg
        for part in parts:
            if hasattr(curr, part):
                curr = getattr(curr, part)
            elif isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                cli.console.print(f"[yellow]Config key not found: {raw_key}[/yellow]")
                return

        redacted_val = _redact_secret(raw_key, curr)
        cli.console.print(f"[green]{raw_key}: {redacted_val}[/green]")

    except Exception as e:
        logger.error(f"Failed to get config: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to get config: {e}[/red]")
