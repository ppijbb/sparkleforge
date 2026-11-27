"""Tests for multi-host plugin manifests and portable MCP launcher (#2032)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_json(rel_path: str) -> dict:
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"Missing manifest file: {rel_path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_claude_marketplace_and_plugin_manifests_match() -> None:
    marketplace = _read_json(".claude-plugin/marketplace.json")
    plugin = _read_json(".claude-plugin/plugin.json")

    assert marketplace["$schema"] == "https://anthropic.com/claude-code/marketplace.schema.json"
    assert marketplace["name"] == "sparkleforge"
    assert isinstance(marketplace.get("owner", {}).get("name"), str)
    assert len(marketplace.get("plugins", [])) == 1

    entry = marketplace["plugins"][0]
    assert entry["name"] == plugin["name"] == "sparkleforge"
    assert entry["source"] == "./"
    assert isinstance(plugin.get("version"), str) and plugin["version"]

    mcp_cfg = plugin["mcpServers"]["sparkleforge"]
    assert mcp_cfg["command"] == "bash"
    assert "${CLAUDE_PLUGIN_ROOT}/scripts/run_mcp_server.sh" in mcp_cfg["args"]


def test_antigravity_codex_and_gemini_manifests_are_valid() -> None:
    agy_plugin = _read_json("plugin.json")
    agy_mcp = _read_json("mcp_config.json")
    codex_plugin = _read_json(".codex-plugin/plugin.json")
    gemini_ext = _read_json("gemini-extension.json")

    allowed_agy_keys = {
        "name",
        "displayName",
        "description",
        "version",
        "logo",
        "suggestedPrompts",
        "author",
        "homepage",
        "repository",
        "disabled",
    }
    assert set(agy_plugin.keys()).issubset(allowed_agy_keys)
    assert agy_plugin["name"] == "sparkleforge"
    assert (REPO_ROOT / agy_plugin["logo"]).is_file()
    assert len(agy_plugin["suggestedPrompts"]) <= 3

    assert "sparkleforge" in agy_mcp["mcpServers"]
    assert "${PLUGIN_ROOT}/scripts/run_mcp_server.sh" in agy_mcp["mcpServers"]["sparkleforge"]["args"]

    assert codex_plugin["name"] == "sparkleforge"
    assert (REPO_ROOT / "skills" / "sparkleforge" / "SKILL.md").is_file()

    assert gemini_ext["name"] == "sparkleforge"
    assert "sparkleforge" in gemini_ext["mcpServers"]


def test_launcher_script_syntax_and_permissions() -> None:
    launcher = REPO_ROOT / "scripts" / "run_mcp_server.sh"
    assert launcher.is_file()
    assert os.access(launcher, os.X_OK)
    res = subprocess.run(["bash", "-n", str(launcher)], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr


def test_launcher_stdio_handshake_from_external_cwd(tmp_path: Path) -> None:
    launcher = REPO_ROOT / "scripts" / "run_mcp_server.sh"
    # Verify whether any local interpreter has fastmcp before running stdio handshake
    probe = subprocess.run(
        [
            "bash",
            "-c",
            "for p in \"$HOME/miniconda3/bin/python\" \"$(command -v python3)\"; do "
            "[ -x \"$p\" ] && \"$p\" -c 'import fastmcp' 2>/dev/null && exit 0; done; exit 1",
        ],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("No Python interpreter with fastmcp available in test environment")

    proc = subprocess.Popen(
        ["bash", str(launcher)],
        cwd=str(tmp_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdin is not None and proc.stdout is not None
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0.0"},
            },
        }
        proc.stdin.write(json.dumps(init_req) + "\n")
        proc.stdin.flush()
        first_line = proc.stdout.readline()
        assert first_line, f"Server exited without stdout; stderr={proc.stderr.read() if proc.stderr else ''}"
        init_resp = json.loads(first_line)
        assert init_resp["result"]["serverInfo"]["name"] == "sparkleforge"

        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n")
        proc.stdin.flush()
        tools_resp = json.loads(proc.stdout.readline())
        tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
        assert {
            "run_task",
            "start_research",
            "get_report",
            "nightwelding_status",
            "search_skills",
            "get_skill",
        }.issubset(tool_names)
    finally:
        proc.terminate()
        proc.wait(timeout=5)
