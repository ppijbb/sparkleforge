#!/usr/bin/env python3
"""
간단한 테스트 스크립트 - 프로젝트 기본 동작 확인
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.researcher_config import load_config_from_env
from src.core.mcp_integration import execute_tool

async def test_basic():
    """기본 기능 테스트"""
    print("🔧 설정 로드 테스트...")
    try:
        config = load_config_from_env()
        print("✅ 설정 로드 성공")
    except Exception as e:
        print(f"❌ 설정 로드 실패: {e}")
        return False
    
    print("\n🔍 MCP 도구 테스트...")
    try:
        # 간단한 검색 테스트
        result = await execute_tool("g-search", {
            "query": "test",
            "max_results": 1
        })
        
        print(f"✅ 도구 테스트 결과: success={result.get('success')}")
        if result.get('success'):
            print(f"   데이터: {result.get('data')}")
        else:
            print(f"   오류: {result.get('error')}")
            
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ 도구 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_basic())
    sys.exit(0 if result else 1)


