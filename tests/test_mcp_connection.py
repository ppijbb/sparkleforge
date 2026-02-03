#!/usr/bin/env python3
"""
MCP 서버 연결 테스트 스크립트

사용법:
    python scripts/test_mcp_connection.py [server_name]
    또는
    python main.py --mcp-server  # 모든 서버 연결
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.researcher_config import load_config_from_env
from src.core.mcp_integration import UniversalMCPHub

# anyio 오류 무시 설정
def ignore_async_gen_errors(loop, context):
    """anyio cancel scope 및 async generator 종료 오류 무시"""
    exception = context.get('exception')
    if exception:
        error_msg = str(exception)
        # anyio cancel scope 오류는 무시
        if isinstance(exception, RuntimeError) and ("cancel scope" in error_msg.lower() or "different task" in error_msg.lower()):
            return  # 무시
        # async generator 종료 오류는 무시
        if isinstance(exception, GeneratorExit) or "async_generator" in error_msg.lower():
            return  # 무시
    # 기타 오류는 기본 handler로 전달
    loop.set_exception_handler(None)
    loop.call_exception_handler(context)
    loop.set_exception_handler(ignore_async_gen_errors)

async def test_mcp_connections():
    """MCP 서버 연결 테스트."""
    # 환경 변수 로드
    config = load_config_from_env()
    
    # MCP Hub 초기화
    hub = UniversalMCPHub()
    
    print("\n" + "=" * 80)
    print("🔌 MCP 서버 연결 테스트")
    print("=" * 80)
    print(f"설정된 서버 수: {len(hub.mcp_server_configs)}")
    print()
    
    if not hub.mcp_server_configs:
        print("⚠️ No MCP servers configured in mcp_config.json")
        return
    
    # 연결 시도
    print("Connecting to MCP servers...")
    print()
    
    connected_count = 0
    failed_count = 0
    
    for server_name, server_config in hub.mcp_server_configs.items():
        print(f"Testing {server_name}...", end=" ", flush=True)
        try:
            # 서버별 설정 가져오기 (더 긴 타임아웃 적용)
            server_settings = hub._get_server_specific_settings(server_name, server_config)
            server_timeout = server_settings["timeout"]
            success = await hub._connect_to_mcp_server(server_name, server_config, timeout=server_timeout)
            if success:
                tools_count = len(hub.mcp_tools_map.get(server_name, {}))
                print(f"✅ Connected ({tools_count} tools)")
                connected_count += 1
            else:
                print("❌ Failed")
                failed_count += 1
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            failed_count += 1
    
    print()
    print("=" * 80)
    print(f"✅ Connected: {connected_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📊 Total: {len(hub.mcp_server_configs)}")
    print("=" * 80)
    
    # 연결된 서버의 Tool 목록 출력
    if connected_count > 0:
        print("\n📋 Available Tools:")
        for server_name in hub.mcp_sessions.keys():
            if server_name in hub.mcp_tools_map:
                tools = hub.mcp_tools_map[server_name]
                print(f"\n  {server_name}:")
                for tool_name in tools.keys():
                    print(f"    - {server_name}::{tool_name}")
    
    # 정리
    await hub.cleanup()

if __name__ == "__main__":
    # anyio 오류 무시 설정
    async def main():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(ignore_async_gen_errors)
        await test_mcp_connections()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n테스트 중단됨")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

