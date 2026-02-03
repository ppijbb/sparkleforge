"""
SparkleForge Entry Point for A2A Integration

StandardAgentRunner가 SparkleForge를 호출할 수 있도록 하는 entry point
"""

import asyncio
import logging
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


async def run_sparkleforge_agent(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    SparkleForge Agent 실행 함수

    StandardAgentRunner에서 호출되는 entry point

    Args:
        input_data: 입력 데이터 (A2A 메시지 payload)

    Returns:
        실행 결과
    """
    try:
        # SparkleForge A2A wrapper import
        from sparkleforge.common.sparkleforge_a2a_wrapper import SparkleForgeA2AWrapper

        logger.info("SparkleForge Agent 실행 시작")

        # A2A wrapper 생성
        wrapper = SparkleForgeA2AWrapper()

        # 요청 실행
        result = await wrapper.execute_request(input_data)

        logger.info("SparkleForge Agent 실행 완료")

        return result

    except Exception as e:
        logger.error(f"SparkleForge Agent 실행 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())

        return {
            'success': False,
            'error': str(e),
            'agent': 'sparkleforge'
        }


# 모듈 형태로도 사용할 수 있게 클래스 형태 제공
class SparkleForgeAgent:
    """SparkleForge Agent 클래스"""

    def __init__(self):
        self.wrapper = None

    async def initialize(self):
        """초기화"""
        from sparkleforge.common.sparkleforge_a2a_wrapper import SparkleForgeA2AWrapper
        self.wrapper = SparkleForgeA2AWrapper()
        logger.info("SparkleForge Agent 초기화 완료")

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """실행"""
        if self.wrapper is None:
            await self.initialize()

        return await self.wrapper.execute_request(input_data)

    async def cleanup(self):
        """정리"""
        if self.wrapper:
            await self.wrapper.stop_listener()
        logger.info("SparkleForge Agent 정리 완료")


# 전역 인스턴스 (필요한 경우)
_sparkleforge_agent_instance = None

async def get_sparkleforge_agent() -> SparkleForgeAgent:
    """SparkleForge Agent 싱글톤 인스턴스 가져오기"""
    global _sparkleforge_agent_instance

    if _sparkleforge_agent_instance is None:
        _sparkleforge_agent_instance = SparkleForgeAgent()
        await _sparkleforge_agent_instance.initialize()

    return _sparkleforge_agent_instance


if __name__ == "__main__":
    # 직접 테스트
    async def test():
        print("🔍 SparkleForge Entry Point 테스트...")

        test_input = {
            'request': '블록체인 기술의 최신 동향 분석',
            'streaming': False
        }

        try:
            result = await run_sparkleforge_agent(test_input)

            print("✅ Entry Point 테스트 성공:")
            print(f"성공 여부: {result.get('success', False)}")
            print(f"요약: {result.get('summary', 'N/A')}")

        except Exception as e:
            print(f"❌ Entry Point 테스트 실패: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test())
