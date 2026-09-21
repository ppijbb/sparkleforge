"""설정 관리 명령어 (Anvil Phase K / REPL 런타임 설정)."""

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional

from rich.panel import Panel

from src.core import researcher_config
from src.core.autonomous_orchestrator import _autopilot_mode_enabled

logger = logging.getLogger(__name__)

# Guards the read-modify-write os.environ/config mutations below against
# concurrent `config set` invocations -- an asyncio.Lock (not threading.Lock)
# because every caller (REPL command, handle_run_command's runtime overrides)
# is a coroutine on the same event loop; this serializes those coroutines
# without ever blocking the loop itself.
CONFIG_MUTATION_LOCK = asyncio.Lock()

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
# Word-boundary match against the whole secret token (e.g. "api_key"), not a
# bare substring search -- "_" is a \w char, so \b never falls inside a run
# of letters/underscores, and "my_api_key_backup" correctly does not match.
_SECRET_KEY_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in _SECRET_CONFIG_KEYS) + r")\b"
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
    """Check if key contains a known secret token as a whole word."""
    return bool(_SECRET_KEY_RE.search(key.lower()))


def _redact_secret(key: str, value: Any) -> Any:
    """Return redacted value for known secret keys."""
    if _is_secret_key(key):
        return "***" if value else value
    if isinstance(value, dict):
        return {k: _redact_secret(k, v) for k, v in value.items()}
    return value


def _get_root_config():
    """Ensure and return the global ResearcherSystemConfig instance."""
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
        # Shares the mutation lock as a reader, but only while snapshotting
        # values -- the lock is released before cli.console.print (I/O) runs,
        # so a slow render can't hold up a concurrent config_set_command.
        async with CONFIG_MUTATION_LOCK:
            cfg = _get_root_config()

            llm_provider = getattr(cfg.llm, "provider", "N/A") if hasattr(cfg, "llm") else "N/A"
            llm_model = getattr(cfg.llm, "primary_model", "N/A") if hasattr(cfg, "llm") else "N/A"
            max_tokens = getattr(cfg.llm, "max_tokens", "N/A") if hasattr(cfg, "llm") else "N/A"
            temperature = getattr(cfg.llm, "temperature", "N/A") if hasattr(cfg, "llm") else "N/A"

            depth_preset = "auto"
            if hasattr(cfg, "research") and hasattr(cfg.research, "research_depth"):
                depth_preset = getattr(cfg.research.research_depth, "default_preset", "auto")

            autopilot_val = (
                cfg.autopilot_mode if hasattr(cfg, "autopilot_mode") else _autopilot_mode_enabled()
            )
            budget_limit = (
                getattr(cfg.llm, "budget_limit", "unlimited")
                if hasattr(cfg, "llm")
                else "N/A"
            )
            approval_policy = getattr(cfg, "approval_policy", "ask")

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
        cli.console.print("[red]❌ Failed to show config (see log for details)[/red]")


async def config_set_command(cli, args: List[str]):
    """설정 변경 (프로세스 전역 config 객체와 os.environ을 직접 수정 -- 세션 범위 아님)."""
    if len(args) < 2:
        cli.console.print("[red]Usage: config set <key> <value>[/red]")
        return

    raw_key = args[0]
    value_str = " ".join(args[1:])
    canonical_key = _CONFIG_ALIASES.get(raw_key.lower(), raw_key)

    # Check both the raw key and its resolved canonical path -- an alias
    # that maps a plain-looking name to a secret-shaped canonical key (none
    # do today, but nothing stops one being added later) must not bypass
    # this by only ever checking the pre-alias-resolution string.
    if _is_secret_key(raw_key) or _is_secret_key(canonical_key):
        cli.console.print(
            f"[red]❌ Modifying API keys or secrets ('{raw_key}') via 'config set' is not allowed[/red]"
        )
        return

    def _do_set(cfg) -> str:
        """All validation and mutation, returning the message to print. Takes
        no lock itself and does no I/O -- the caller holds CONFIG_MUTATION_LOCK
        only around this call, then prints the result after releasing it."""
        # Handle special key: autopilot_mode
        if canonical_key == "autopilot_mode":
            parsed_bool = _parse_bool(value_str)
            if parsed_bool is None:
                return (
                    f"[red]❌ Invalid boolean for autopilot_mode: '{value_str}'. "
                    "Expected true/false/yes/no/1/0/on/off[/red]"
                )
            old_val = (cfg.autopilot_mode if hasattr(cfg, "autopilot_mode") else _autopilot_mode_enabled())
            cfg.autopilot_mode = parsed_bool
            os.environ["SPARKLEFORGE_AUTOPILOT_MODE"] = "true" if parsed_bool else "false"
            return f"[green]✓ {raw_key}: {old_val} -> {parsed_bool}[/green]"

        # Handle special key: approval_policy
        if canonical_key == "approval_policy":
            val_clean = value_str.lower().strip()
            if val_clean not in ("ask", "allowlist", "autopilot"):
                return (
                    f"[red]❌ Invalid approval policy: '{value_str}'. "
                    "Allowed: ask, allowlist, autopilot[/red]"
                )
            old_val = getattr(cfg, "approval_policy", "ask")
            cfg.approval_policy = val_clean
            os.environ["APPROVAL_POLICY"] = val_clean
            return f"[green]✓ {raw_key}: {old_val} -> {val_clean}[/green]"

        # Handle special key: depth preset
        if canonical_key in ("research.research_depth.default_preset", "research_depth", "depth"):
            val_clean = value_str.lower().strip()
            valid_presets = {"quick", "medium", "deep", "auto"}
            if val_clean not in valid_presets:
                allowed_str = ", ".join(sorted(valid_presets))
                return f"[red]❌ Invalid research depth: '{value_str}'. Allowed: {allowed_str}[/red]"
            if not hasattr(cfg, "research") or not hasattr(cfg.research, "research_depth"):
                return "[red]❌ Research depth setting is unavailable right now[/red]"
            old_val = getattr(cfg.research.research_depth, "default_preset", "auto")
            cfg.research.research_depth.default_preset = val_clean
            os.environ["RESEARCH_DEPTH_PRESET"] = val_clean
            return f"[green]✓ {raw_key}: {old_val} -> {val_clean}[/green]"

        # Handle model selection alias
        if canonical_key in ("llm.primary_model", "primary_model", "model"):
            old_val = cfg.llm.primary_model
            new_model = value_str.strip()
            if not new_model:
                return "[red]❌ Primary model must not be empty[/red]"
            cfg.llm.primary_model = new_model
            os.environ["LLM_MODEL"] = new_model
            # Only cascade a role when BOTH its config attribute and its
            # env var still mirror the old primary -- checking only the
            # config side let a role whose env var was pinned by hand
            # (export REASONING_MODEL=...) get clobbered anyway, since
            # the cfg attribute alone still matched.
            for role, env_k in (
                ("planning_model", "PLANNING_MODEL"),
                ("reasoning_model", "REASONING_MODEL"),
                ("verification_model", "VERIFICATION_MODEL"),
                ("generation_model", "GENERATION_MODEL"),
                ("compression_model", "COMPRESSION_MODEL"),
            ):
                cfg_matches = hasattr(cfg.llm, role) and getattr(cfg.llm, role) == old_val
                env_matches = os.getenv(env_k, old_val) == old_val
                if cfg_matches and env_matches:
                    setattr(cfg.llm, role, new_model)
                    os.environ[env_k] = new_model
            return f"[green]✓ {raw_key}: {old_val} -> {new_model}[/green]"

        # Dotted path traversal on ResearcherSystemConfig -- mirrors
        # config_get_command's dict-before-hasattr precedence so a dict
        # branch in the config tree can be traversed the same way by
        # both `config get` and `config set`.
        parts = canonical_key.split(".")
        target = cfg
        for part in parts[:-1]:
            if isinstance(target, dict):
                if part not in target:
                    return f"[red]❌ Unknown config key: '{raw_key}'[/red]"
                target = target[part]
            elif hasattr(target, part):
                target = getattr(target, part)
            else:
                return f"[red]❌ Unknown config key: '{raw_key}'[/red]"

        last_part = parts[-1]
        target_is_dict = isinstance(target, dict)
        if target_is_dict:
            if last_part not in target:
                return f"[red]❌ Unknown config key: '{raw_key}'[/red]"
            old_val = target[last_part]
        else:
            if not hasattr(target, last_part):
                return f"[red]❌ Unknown config key: '{raw_key}'[/red]"
            old_val = getattr(target, last_part)

        # Convert value_str to target type
        if old_val is None:
            # No prior value to infer a type from -- best-effort sniff the
            # input itself (int, then float, then bool) instead of forcing
            # str, so a None-defaulted numeric/bool field doesn't get stuck
            # as a string on its first set. int/float are tried before
            # bool because "0"/"1" are valid bool aliases too, and a bare
            # digit should become a number, not True/False.
            converted: Any
            try:
                converted = int(value_str)
            except ValueError:
                try:
                    converted = float(value_str)
                except ValueError:
                    parsed_bool = _parse_bool(value_str)
                    converted = value_str.strip() if parsed_bool is None else parsed_bool
        elif isinstance(old_val, bool):
            converted = _parse_bool(value_str)
            if converted is None:
                return f"[red]❌ Invalid boolean: '{value_str}'[/red]"
        elif isinstance(old_val, int):
            try:
                converted = int(value_str)
            except ValueError:
                return f"[red]❌ Invalid integer: '{value_str}'[/red]"
        elif isinstance(old_val, float):
            try:
                converted = float(value_str)
            except ValueError:
                return f"[red]❌ Invalid float: '{value_str}'[/red]"
        else:
            converted = value_str.strip()

        if target_is_dict:
            target[last_part] = converted
        else:
            setattr(target, last_part, converted)
        return f"[green]✓ {raw_key}: {old_val} -> {converted}[/green]"

    try:
        async with CONFIG_MUTATION_LOCK:
            message = _do_set(_get_root_config())
        cli.console.print(message)

    except Exception as e:
        logger.error(f"Failed to set config '{raw_key}': {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to set config '{raw_key}' (see log for details)[/red]")


async def config_get_command(cli, args: List[str]):
    """설정 값 가져오기."""
    if not args:
        cli.console.print("[red]Usage: config get <key>[/red]")
        return

    raw_key = args[0]
    canonical_key = _CONFIG_ALIASES.get(raw_key.lower(), raw_key)

    try:
        # Gather the result (or a not-found marker) while holding the lock,
        # then print outside it -- I/O has no business extending how long a
        # concurrent config_set_command has to wait.
        async with CONFIG_MUTATION_LOCK:
            cfg = _get_root_config()

            if canonical_key == "autopilot_mode":
                found = True
                result = (
                    cfg.autopilot_mode if hasattr(cfg, "autopilot_mode") else _autopilot_mode_enabled()
                )
            elif canonical_key == "approval_policy":
                found = True
                result = getattr(cfg, "approval_policy", "ask")
            else:
                # Direct attribute or dotted path traversal
                parts = canonical_key.split(".")
                curr = cfg
                found = True
                for part in parts:
                    # dict membership takes priority over hasattr: for a plain
                    # dict, hasattr(curr, "keys") is also true (it's a dict
                    # method), so a dict key literally named "keys"/"items"/etc
                    # would otherwise resolve to the bound method instead of
                    # the stored value.
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
                result = _redact_secret(raw_key, curr) if found else None

        if not found:
            cli.console.print(f"[yellow]Config key not found: {raw_key}[/yellow]")
        else:
            cli.console.print(f"[green]{raw_key}: {result}[/green]")

    except Exception as e:
        logger.error(f"Failed to get config: {e}", exc_info=True)
        cli.console.print("[red]❌ Failed to get config (see log for details)[/red]")
