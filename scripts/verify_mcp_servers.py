#!/usr/bin/env python3
"""
Verify MCP servers can be started and list their tools.
"""

import asyncio
import subprocess
import sys
import time
import json
from pathlib import Path

# Add project root to path - use absolute path
project_root = Path("/home/user/workspace/mcp_agent/sparkleforge")
sys.path.insert(0, str(project_root))


async def test_server_startup(server_name: str, timeout: float = 10.0) -> dict:
    """Test if an MCP server can start and respond to list_tools."""
    # embedded_mcp_servers는 더 이상 MCP 서버로 실행하지 않음
    # src/utils에서 직접 함수 호출로 사용
    return {"success": False, "error": "Embedded servers are now used directly from src/utils, not as MCP servers"}
    
    cmd = [config["command"]] + config["args"]
    env = {**config.get("env", {}), **dict(__import__("os").environ)}
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        # Wait for the server to start
        await asyncio.sleep(2.0)
        
        # Check if process is still running
        if proc.returncode is not None:
            stdout, stderr = await proc.communicate()
            return {
                "success": False,
                "error": f"Process exited with code {proc.returncode}",
                "stderr": stderr.decode()[:500]
            }
        
        # Terminate the process
        proc.terminate()
        await proc.wait()
        
        return {
            "success": True,
            "message": "Server started successfully (basic check)"
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


async def test_server_functions(server_name: str) -> dict:
    """Test server by importing and running its functions directly."""
    try:
        if server_name == "web-fetch":
            from src.utils.web_utils import fetch_url
            result = await fetch_url("https://example.com", max_length=500)
            return {"success": result.get("success", False), "function": "fetch_url"}
        
        elif server_name == "search":
            from src.utils.search_utils import search_duckduckgo
            result = await search_duckduckgo("test", num_results=3)
            return {"success": result.get("success", False), "function": "search_duckduckgo"}
        
        elif server_name == "arxiv":
            from src.utils.academic_utils import search_arxiv
            result = await search_arxiv("machine learning", max_results=3)
            return {"success": result.get("success", False), "function": "search_arxiv"}
        
        elif server_name == "memory":
            # Memory는 아직 src/utils로 이동하지 않음 (필요시 추가)
            return {"success": False, "error": "Memory utilities not yet migrated to src/utils"}
        
        elif server_name == "filesystem":
            # Filesystem은 아직 src/utils로 이동하지 않음 (필요시 추가)
            return {"success": False, "error": "Filesystem utilities not yet migrated to src/utils"}
        
        elif server_name == "github":
            # GitHub는 아직 src/utils로 이동하지 않음 (필요시 추가)
            return {"success": False, "error": "GitHub utilities not yet migrated to src/utils"}
        
        else:
            return {"success": False, "error": "Unknown server"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}


async def main():
    print("=" * 60)
    print("🔍 MCP Server Verification Tests")
    print("=" * 60)
    
    servers = ["web-fetch", "search", "arxiv", "memory", "filesystem", "github"]
    results = []
    
    for server in servers:
        print(f"\nTesting {server}...")
        
        # Test function directly
        result = await test_server_functions(server)
        results.append((server, result))
        
        status = "✅" if result.get("success") else "❌"
        print(f"  {status} Function test: {result.get('function', 'N/A')} - {result.get('success', False)}")
        
        if not result.get("success"):
            print(f"     Error: {result.get('error', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print("📊 Results Summary")
    print("=" * 60)
    
    passed = 0
    for server, result in results:
        status = "✅ PASS" if result.get("success") else "❌ FAIL"
        print(f"  {server}: {status}")
        if result.get("success"):
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} servers verified")
    
    if passed == len(results):
        print("\n🎉 All MCP servers are working correctly!")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} servers need attention")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
