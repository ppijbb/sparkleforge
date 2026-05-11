"""MCP stdio 서버용 Python 실행 파일 결정.

`mcp_config.json`에서 흔히 `command: "python"`으로 두는데, 이때 PATH의 python은
앱을 실행한 venv와 다를 수 있어 `fastmcp` 등이 없고 서버 모듈이 실패합니다.
부모(오케스트레이터)와 동일한 인터프리터를 쓰도록 `sys.executable`로 정규화합니다.

환경 변수:
- MCP_PYTHON_PATH: 명시적으로 사용할 python 경로(최우선).
"""

from __future__ import annotations

import os
import sys


def resolve_mcp_python_executable(configured: str | None) -> str:
    """설정된 command 문자열을 실제 실행 가능한 Python 경로로 해석."""
    explicit = (os.getenv("MCP_PYTHON_PATH") or "").strip()
    if explicit:
        return explicit
    if configured is None or (isinstance(configured, str) and not configured.strip()):
        return sys.executable
    c = configured.strip()
    # 일반적인 플레이스홀더만 동일 인터프리터로 치환 (사용자 지정 절대경로는 유지)
    if c in ("python", "python3"):
        return sys.executable
    return configured  # type: ignore[return-value]


def normalize_mcp_servers_python_commands(mcp_servers: dict) -> dict:
    """McpServers dict의 각 항목에 대해 command를 정규화."""
    out: dict = {}
    for name, cfg in mcp_servers.items():
        if not isinstance(cfg, dict):
            out[name] = cfg
            continue
        merged = dict(cfg)
        cmd = merged.get("command")
        if isinstance(cmd, str):
            merged["command"] = resolve_mcp_python_executable(cmd)
        out[name] = merged
    return out
