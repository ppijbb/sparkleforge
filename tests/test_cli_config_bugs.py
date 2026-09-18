import os
import pytest
from unittest.mock import MagicMock
from src.cli.commands.config import (
    config_get_command,
    config_set_command,
    handle_run_command,
    _sanitize_embedded_cli_flags,
    _parse_bool,
)

class MockConsole:
    def __init__(self):
        self.prints = []
    def print(self, msg, *args, **kwargs):
        self.prints.append(str(msg))

class MockCLI:
    def __init__(self):
        self.console = MockConsole()

@pytest.mark.asyncio
async def test_config_get_dict_keys_method_conflict(monkeypatch):
    """Test 1: config_get dotted-path traversal doesn't mistakenly call dict.keys method."""
    mock_cfg = MagicMock()
    mock_cfg.some_dict = {"keys": "secret_key_value_123"}
    
    monkeypatch.setattr("src.core.researcher_config.get_research_config", lambda: mock_cfg)
    
    cli = MockCLI()
    await config_get_command(cli, ["some_dict.keys"])
    
    output = "".join(cli.console.prints)
    assert "secret_key_value_123" in output
    assert "<method 'keys'" not in output

def test_parse_bool_and_autopilot():
    """Test 2: handle_run_command autopilot parsing uses robust boolean rules."""
    assert _parse_bool("true") is True
    assert _parse_bool("1") is True
    assert _parse_bool("maybe") is True
    assert _parse_bool("hello") is True
    assert _parse_bool("false") is False
    assert _parse_bool("0") is False
    assert _parse_bool("no") is False
    assert _parse_bool("off") is False

@pytest.mark.asyncio
async def test_config_set_depth_alias_missing_research(monkeypatch):
    """Test 3: config_set depth alias handles missing cfg.research gracefully."""
    mock_cfg = MagicMock(spec=[]) # No research attribute
    monkeypatch.setattr("src.core.researcher_config.get_research_config", lambda: mock_cfg)
    monkeypatch.setattr("src.core.researcher
