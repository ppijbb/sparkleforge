"""Tests for REPL runtime config commands and CLI one-shot overrides (Issue #1650)."""

import argparse
import os
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from rich.console import Console

from src.cli.commands import config as config_module
from src.cli.commands.config import (
    _is_secret_key,
    config_get_command,
    config_set_command,
    config_show_command,
)
from src.cli.main_commands import handle_run_command
from src.core import researcher_config
from src.core.autonomous_orchestrator import _autopilot_mode_enabled


class MockCLI:
    """Mock CLI container with captured rich console output."""

    def __init__(self):
        self.output_messages = []
        self.console = MagicMock(spec=Console)
        self.console.print.side_effect = lambda *msg, **kwargs: self.output_messages.append(
            str(msg[0]) if msg else ""
        )


@pytest.fixture(autouse=True)
def setup_config_env(monkeypatch):
    old_env = dict(os.environ)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setenv("LLM_MODEL", "google/gemini-2.5-flash-lite-preview")
    researcher_config.config = None
    researcher_config.load_config_from_env()
    yield
    os.environ.clear()
    os.environ.update(old_env)
    researcher_config.config = None


@pytest.mark.asyncio
async def test_config_get_aliases():
    cli = MockCLI()
    cfg = researcher_config.config
    expected_model = cfg.llm.primary_model
    await config_get_command(cli, ["model"])
    assert any(expected_model in m for m in cli.output_messages)

    cli.output_messages.clear()
    await config_get_command(cli, ["depth"])
    assert any("auto" in m for m in cli.output_messages)

    cli.output_messages.clear()
    await config_get_command(cli, ["autopilot"])
    assert any("True" in m for m in cli.output_messages)


@pytest.mark.asyncio
async def test_config_get_dotted_path():
    cli = MockCLI()
    cfg = researcher_config.config
    expected_model = cfg.llm.primary_model
    await config_get_command(cli, ["llm.primary_model"])
    assert any(expected_model in m for m in cli.output_messages)


@pytest.mark.asyncio
async def test_config_get_secret_redaction():
    cli = MockCLI()
    await config_get_command(cli, ["llm.openrouter_api_key"])
    assert any("***" in m for m in cli.output_messages)
    assert not any("sk-or-test-key" in m for m in cli.output_messages)


@pytest.mark.asyncio
async def test_config_get_not_found():
    cli = MockCLI()
    await config_get_command(cli, ["non_existent_attribute"])
    assert any("Config key not found" in m for m in cli.output_messages)


@pytest.mark.asyncio
async def test_config_set_model_alias():
    cli = MockCLI()
    new_model = "anthropic/claude-3.5-sonnet"
    await config_set_command(cli, ["model", new_model])

    assert any(new_model in m for m in cli.output_messages)
    cfg = researcher_config.config
    assert cfg.llm.primary_model == new_model
    assert cfg.llm.planning_model == new_model
    assert os.getenv("LLM_MODEL") == new_model


@pytest.mark.asyncio
async def test_config_set_depth_alias():
    cli = MockCLI()
    await config_set_command(cli, ["depth", "deep"])

    assert any("deep" in m for m in cli.output_messages)
    cfg = researcher_config.config
    assert cfg.research.research_depth.default_preset == "deep"
    assert os.getenv("RESEARCH_DEPTH_PRESET") == "deep"

    # Test invalid depth preset
    cli.output_messages.clear()
    await config_set_command(cli, ["depth", "ultra-mega-deep"])
    assert any("Invalid research depth" in m for m in cli.output_messages)


@pytest.mark.asyncio
async def test_config_set_autopilot_alias():
    cli = MockCLI()
    await config_set_command(cli, ["autopilot", "false"])

    assert any("False" in m for m in cli.output_messages)
    assert _autopilot_mode_enabled() is False

    cli.output_messages.clear()
    await config_set_command(cli, ["autopilot", "true"])
    assert any("True" in m for m in cli.output_messages)
    assert _autopilot_mode_enabled() is True


@pytest.mark.asyncio
async def test_config_autopilot_roundtrips_through_config_object():
    """Issue: autopilot_mode previously lived only in os.environ, with no
    config-object source of truth (same class of gap as approval_policy)."""
    cli = MockCLI()
    await config_set_command(cli, ["autopilot", "false"])

    cfg = researcher_config.config
    assert cfg.autopilot_mode is False
    assert os.getenv("SPARKLEFORGE_AUTOPILOT_MODE") == "false"

    cli.output_messages.clear()
    await config_get_command(cli, ["autopilot"])
    assert any("False" in m for m in cli.output_messages)


def test_is_secret_key_exact_match_not_substring():
    """Issue #1659: substring matching false-flagged e.g. my_api_key_backup."""
    assert _is_secret_key("api_key")
    assert _is_secret_key("llm.openrouter_api_key")
    assert _is_secret_key("token")
    assert not _is_secret_key("my_api_key_backup")
    assert not _is_secret_key("tokenizer_config")


@pytest.mark.asyncio
async def test_config_set_model_alias_preserves_pinned_role():
    cli = MockCLI()
    cfg = researcher_config.config
    pinned = "anthropic/claude-3.5-opus"
    cfg.llm.planning_model = pinned  # simulate a role the user pinned earlier

    new_model = "anthropic/claude-3.5-sonnet"
    await config_set_command(cli, ["model", new_model])

    cfg = researcher_config.config
    assert cfg.llm.primary_model == new_model
    assert cfg.llm.reasoning_model == new_model  # mirrored old primary -> cascades
    assert cfg.llm.planning_model == pinned  # explicitly pinned -> untouched


@pytest.mark.asyncio
async def test_config_set_model_alias_preserves_env_pinned_role(monkeypatch):
    """A role whose config attribute still mirrors the old primary but whose
    env var was hand-pinned (export REASONING_MODEL=...) must not cascade
    either -- checking only the config side let this slip through before."""
    cli = MockCLI()
    cfg = researcher_config.config
    old_primary = cfg.llm.primary_model
    assert cfg.llm.reasoning_model == old_primary  # not pinned on the cfg side
    monkeypatch.setenv("REASONING_MODEL", "hand-pinned/model")

    new_model = "anthropic/claude-3.5-sonnet"
    await config_set_command(cli, ["model", new_model])

    cfg = researcher_config.config
    assert cfg.llm.primary_model == new_model
    assert cfg.llm.reasoning_model == old_primary  # env-pinned -> untouched
    assert os.getenv("REASONING_MODEL") == "hand-pinned/model"


@pytest.mark.asyncio
async def test_config_set_none_default_infers_type(monkeypatch):
    cli = MockCLI()
    fake_cfg = SimpleNamespace(section=SimpleNamespace(value=None))
    monkeypatch.setattr(config_module, "_get_root_config", lambda: fake_cfg)

    await config_set_command(cli, ["section.value", "42"])
    assert fake_cfg.section.value == 42  # int, not "42"


@pytest.mark.asyncio
async def test_config_set_none_default_zero_stays_int(monkeypatch):
    """"0" is a bool alias too -- int/float must be tried before bool, or a
    None-defaulted numeric field set to "0" becomes False instead of 0."""
    cli = MockCLI()
    fake_cfg = SimpleNamespace(section=SimpleNamespace(value=None))
    monkeypatch.setattr(config_module, "_get_root_config", lambda: fake_cfg)

    await config_set_command(cli, ["section.value", "0"])
    assert fake_cfg.section.value == 0
    assert fake_cfg.section.value is not False


@pytest.mark.asyncio
async def test_config_set_none_default_bool_word_still_works(monkeypatch):
    cli = MockCLI()
    fake_cfg = SimpleNamespace(section=SimpleNamespace(value=None))
    monkeypatch.setattr(config_module, "_get_root_config", lambda: fake_cfg)

    await config_set_command(cli, ["section.value", "true"])
    assert fake_cfg.section.value is True


@pytest.mark.asyncio
async def test_config_approval_policy_roundtrips_through_config_object():
    """Issue: APPROVAL_POLICY previously lived only in os.environ, with no
    config-object source of truth. It must now live on the config object."""
    cli = MockCLI()
    await config_set_command(cli, ["approval_policy", "allowlist"])
    assert any("allowlist" in m for m in cli.output_messages)

    cfg = researcher_config.config
    assert cfg.approval_policy == "allowlist"
    assert os.getenv("APPROVAL_POLICY") == "allowlist"

    cli.output_messages.clear()
    await config_get_command(cli, ["approval_policy"])
    assert any("allowlist" in m for m in cli.output_messages)


@pytest.mark.asyncio
async def test_config_get_dict_key_shadowing_dict_method(monkeypatch):
    """A dict key named like a dict method (e.g. "keys") must win over hasattr."""
    cli = MockCLI()
    fake_cfg = SimpleNamespace(section={"keys": "custom-value"})
    monkeypatch.setattr(config_module, "_get_root_config", lambda: fake_cfg)

    await config_get_command(cli, ["section.keys"])
    assert any("custom-value" in m for m in cli.output_messages)


@pytest.mark.asyncio
async def test_config_set_supports_dict_traversal(monkeypatch):
    """config_set_command previously had no dict branch at all, so it could
    not update a key living inside a plain dict -- only config_get_command
    could read one. Both must support the same shape."""
    cli = MockCLI()
    fake_cfg = SimpleNamespace(section={"nested": 1})
    monkeypatch.setattr(config_module, "_get_root_config", lambda: fake_cfg)

    await config_set_command(cli, ["section.nested", "5"])
    assert fake_cfg.section["nested"] == 5


@pytest.mark.asyncio
async def test_config_set_depth_alias_missing_path_guarded(monkeypatch):
    cli = MockCLI()
    fake_cfg = SimpleNamespace()  # no `research` attribute at all
    monkeypatch.setattr(config_module, "_get_root_config", lambda: fake_cfg)

    await config_set_command(cli, ["depth", "deep"])
    # Distinct from "Unknown config key" -- this is a missing schema
    # structure, not a typo'd key name.
    assert any("internal config error" in m for m in cli.output_messages)


@pytest.mark.asyncio
async def test_config_set_secret_forbidden():
    cli = MockCLI()
    await config_set_command(cli, ["openrouter_api_key", "sk-or-hacked"])
    assert any("Modifying API keys or secrets" in m for m in cli.output_messages)
    assert os.getenv("OPENROUTER_API_KEY") == "sk-or-test-key"


@pytest.mark.asyncio
async def test_config_set_unknown_key_rejected():
    cli = MockCLI()
    await config_set_command(cli, ["llm.unknown_setting", "123"])
    assert any("Unknown config key" in m for m in cli.output_messages)


@pytest.mark.asyncio
async def test_config_show_includes_extended_fields():
    cli = MockCLI()
    await config_show_command(cli, [])

    panel_calls = [call for call in cli.console.print.call_args_list if call[0]]
    assert len(panel_calls) > 0
    panel_arg = panel_calls[0][0][0]
    renderable_text = getattr(panel_arg, "renderable", str(panel_arg))

    assert "Research Depth:" in renderable_text
    assert "Autopilot Mode:" in renderable_text
    assert "Budget Limit:" in renderable_text
    assert "Approval Policy:" in renderable_text


@pytest.mark.asyncio
async def test_run_command_runtime_overrides(monkeypatch):
    cfg = researcher_config.config

    args = argparse.Namespace(
        command="run",
        query="test query",
        model="openai/gpt-4o-mini",
        max_tokens=1500,
        depth="quick",
        autopilot="false",
        task=None,
        session_id=None,
        continue_session=False,
    )

    fake_executed = False

    async def fake_run_command(run_args, run_cfg):
        nonlocal fake_executed
        fake_executed = True
        return 0

    monkeypatch.setattr("src.cli.commands.run.run_command", fake_run_command)
    monkeypatch.setattr(
        "src.core.observe.system_collector.check_disk_space_safety",
        lambda: (True, "Disk OK"),
    )
    monkeypatch.setattr(
        "src.core.observe.system_collector.check_network_connectivity",
        lambda: (True, "Network OK"),
    )

    rc = await handle_run_command(args, cfg)
    assert rc == 0
    assert fake_executed is True

    assert cfg.llm.primary_model == "openai/gpt-4o-mini"
    assert cfg.llm.max_tokens == 1500
    assert cfg.research.research_depth.default_preset == "quick"
    assert os.getenv("RESEARCH_DEPTH_PRESET") == "quick"
    assert os.getenv("SPARKLEFORGE_AUTOPILOT_MODE") == "false"
    assert _autopilot_mode_enabled() is False
    # --autopilot went through handle_run_command's own override path, not
    # config_set_command -- it must still land on the config object, or this
    # entry point re-creates the same split-brain the config-object fix closed.
    assert cfg.autopilot_mode is False


@pytest.mark.asyncio
async def test_run_command_sanitizes_embedded_equals_flag(monkeypatch):
    """Issue: sanitizer only matched "--depth " (space-separated), missing --depth=quick."""
    cfg = researcher_config.config
    captured_query = None

    args = argparse.Namespace(
        command="run",
        query="do the research --depth=quick",
        model=None,
        max_tokens=None,
        depth=None,
        autopilot=None,
        task=None,
        session_id=None,
        continue_session=False,
    )

    async def fake_run_command(run_args, run_cfg):
        nonlocal captured_query
        captured_query = run_args.query
        return 0

    monkeypatch.setattr("src.cli.commands.run.run_command", fake_run_command)
    monkeypatch.setattr(
        "src.core.observe.system_collector.check_disk_space_safety",
        lambda: (True, "Disk OK"),
    )
    monkeypatch.setattr(
        "src.core.observe.system_collector.check_network_connectivity",
        lambda: (True, "Network OK"),
    )

    rc = await handle_run_command(args, cfg)
    assert rc == 0
    assert captured_query == "do the research"


@pytest.mark.asyncio
async def test_run_command_invalid_autopilot_value_ignored(monkeypatch):
    """An unparseable --autopilot value must not silently coerce to True."""
    cfg = researcher_config.config
    monkeypatch.delenv("SPARKLEFORGE_AUTOPILOT_MODE", raising=False)

    args = argparse.Namespace(
        command="run",
        query="test query",
        model=None,
        max_tokens=None,
        depth=None,
        autopilot="maybe",
        task=None,
        session_id=None,
        continue_session=False,
    )

    async def fake_run_command(run_args, run_cfg):
        return 0

    monkeypatch.setattr("src.cli.commands.run.run_command", fake_run_command)
    monkeypatch.setattr(
        "src.core.observe.system_collector.check_disk_space_safety",
        lambda: (True, "Disk OK"),
    )
    monkeypatch.setattr(
        "src.core.observe.system_collector.check_network_connectivity",
        lambda: (True, "Network OK"),
    )

    rc = await handle_run_command(args, cfg)
    assert rc == 0
    assert os.getenv("SPARKLEFORGE_AUTOPILOT_MODE") is None
