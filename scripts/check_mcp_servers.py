#!/usr/bin/env python3
"""
MCP 서버 연결 상태 확인 스크립트

사용법:
    python scripts/check_mcp_servers.py
    또는
    python main.py --check-mcp-servers
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.mcp_integration import check_mcp_servers

if __name__ == "__main__":
    asyncio.run(check_mcp_servers())

