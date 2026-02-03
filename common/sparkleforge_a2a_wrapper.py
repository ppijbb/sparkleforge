"""
SparkleForge A2A Wrapper

sparkleforge 프로젝트를 A2A 통신이 가능한 형태로 감싸는 wrapper
"""

import asyncio
import logging
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
import json

# sparkleforge 프로젝트 경로 추가
# __file__이 sparkleforge/common/sparkleforge_a2a_wrapper.py이므로
# parent.parent가 sparkleforge 디렉토리
sparkleforge_path = Path(__file__).parent.parent
sys.path.insert(0, str(sparkleforge_path))

from srcs.common.a2a_integration import (
    A2AAdapter,
    A2AMessage,
    MessagePriority,
    get_global_broker,
    get_global_registry,
)
from srcs.common.agent_interface import AgentMetadata, AgentType

logger = logging.getLogger(__name__)


class SparkleForgeA2AWrapper(A2AAdapter):
    """SparkleForge를 A2A 통신이 가능하게 하는 wrapper"""

    def __init__(
        self,
        agent_id: str = "sparkleforge_agent",
        agent_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        SparkleForge A2A Wrapper 초기화

        Args:
            agent_id: Agent ID
            agent_metadata: Agent 메타데이터 (선택)
        """
        # 기본 메타데이터 설정
        if agent_metadata is None:
            agent_metadata = {
                'agent_id': agent_id,
                'agent_name': 'SparkleForge Multi-Agent Research System',
                'entry_point': 'sparkleforge.common.sparkleforge_a2a_wrapper',
                'agent_type': AgentType.MCP_AGENT,  # MCP_AGENT로 취급
                'capabilities': [
                    'research',
                    'multi_agent_collaboration',
                    'source_validation',
                    'creative_synthesis',
                    'domain_exploration'
                ],
                'description': '혁신적인 다중 에이전트 연구 시스템. 아이디어가 반짝이고 단련되는 곳'
            }

        super().__init__(agent_id, agent_metadata)

        # sparkleforge 환경 설정
        self._setup_sparkleforge_env()

        # sparkleforge orchestrator 초기화 (lazy loading)
        self._orchestrator = None

    def _setup_sparkleforge_env(self):
        """sparkleforge 환경 설정"""
        try:
            # sparkleforge 프로젝트 디렉토리로 이동
            os.chdir(sparkleforge_path)

            # 환경 변수 설정 (필요한 경우)
            if 'OPENROUTER_API_KEY' not in os.environ:
                logger.warning("OPENROUTER_API_KEY 환경 변수가 설정되지 않았습니다.")

            logger.info("SparkleForge 환경 설정 완료")

        except Exception as e:
            logger.error(f"SparkleForge 환경 설정 실패: {e}")
            raise

    async def _get_orchestrator(self):
        """AgentOrchestrator 인스턴스 가져오기 (lazy loading)"""
        if self._orchestrator is None:
            try:
                # sparkleforge 설정 로드
                from src.core.researcher_config import load_config_from_env
                config = load_config_from_env()

                # AgentOrchestrator 초기화
                from src.core.agent_orchestrator import AgentOrchestrator
                self._orchestrator = AgentOrchestrator()

                logger.info("SparkleForge AgentOrchestrator 초기화 완료")

            except Exception as e:
                logger.error(f"AgentOrchestrator 초기화 실패: {e}")
                raise

        return self._orchestrator

    async def _execute_sparkleforge_request(
        self,
        request: str,
        output_path: Optional[str] = None,
        streaming: bool = False
    ) -> Dict[str, Any]:
        """
        sparkleforge 연구 요청 실행

        Args:
            request: 연구 요청
            output_path: 결과 출력 파일 경로
            streaming: 스트리밍 모드

        Returns:
            실행 결과
        """
        # 현재 작업 디렉토리 저장
        original_cwd = os.getcwd()

        try:
            # sparkleforge 디렉토리로 이동
            os.chdir(sparkleforge_path)

            # orchestrator 가져오기
            orchestrator = await self._get_orchestrator()

            # 연구 실행
            logger.info(f"SparkleForge 연구 시작: {request}")

            if streaming:
                # 스트리밍 모드로 실행
                result = await self._execute_streaming_request(orchestrator, request)
            else:
                # 일반 모드로 실행
                result = await orchestrator.execute(request)

            # 결과 포맷팅
            formatted_result = self._format_sparkleforge_result(result)

            # 출력 파일 저장 (요청된 경우)
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(formatted_result, f, ensure_ascii=False, indent=2)

            logger.info("SparkleForge 연구 완료")
            return formatted_result

        except Exception as e:
            logger.error(f"SparkleForge 실행 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'SparkleForge 실행 실패: {str(e)}',
                'agent': 'sparkleforge'
            }
        finally:
            # 원래 작업 디렉토리 복원
            os.chdir(original_cwd)

    def _format_sparkleforge_result(self, raw_result: Any) -> Dict[str, Any]:
        """sparkleforge 결과를 표준 포맷으로 변환"""
        try:
            if isinstance(raw_result, dict):
                # 이미 dict 형태인 경우
                formatted = {
                    'success': True,
                    'agent': 'sparkleforge',
                    'result': raw_result,
                    'summary': raw_result.get('summary', '연구 완료'),
                    'timestamp': raw_result.get('timestamp', str(asyncio.get_event_loop().time()))
                }
            elif isinstance(raw_result, str):
                # 문자열 결과인 경우
                formatted = {
                    'success': True,
                    'agent': 'sparkleforge',
                    'result': {'content': raw_result},
                    'summary': raw_result[:200] + '...' if len(raw_result) > 200 else raw_result,
                    'timestamp': str(asyncio.get_event_loop().time())
                }
            else:
                # 기타 형태
                formatted = {
                    'success': True,
                    'agent': 'sparkleforge',
                    'result': {'data': str(raw_result)},
                    'summary': 'SparkleForge 연구 결과',
                    'timestamp': str(asyncio.get_event_loop().time())
                }

            return formatted

        except Exception as e:
            logger.error(f"결과 포맷팅 실패: {e}")
            return {
                'success': False,
                'error': f'결과 포맷팅 실패: {str(e)}',
                'agent': 'sparkleforge'
            }

    async def execute_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        A2A를 통해 들어온 요청 처리

        Args:
            input_data: 입력 데이터 (A2A 메시지 payload)

        Returns:
            실행 결과
        """
        try:
            # 요청 데이터 추출
            request = input_data.get('request', input_data.get('query', ''))
            if not request:
                return {
                    'success': False,
                    'error': '요청이 비어있습니다. request 또는 query 필드를 제공해주세요.',
                    'agent': 'sparkleforge'
                }

            # 옵션 파라미터들
            output_path = input_data.get('output_path')
            streaming = input_data.get('streaming', False)

            logger.info(f"SparkleForge A2A 요청 처리: {request[:100]}...")

            # sparkleforge 실행
            result = await self._execute_sparkleforge_request(
                request=request,
                output_path=output_path,
                streaming=streaming
            )

            return result

        except Exception as e:
            logger.error(f"A2A 요청 처리 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())

            return {
                'success': False,
                'error': str(e),
                'agent': 'sparkleforge'
            }

    # A2A 필수 메서드들 (부모 클래스에서 상속)

    async def send_message(
        self,
        target_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.MEDIUM
    ) -> bool:
        """메시지 전송"""
        return await super().send_message(target_agent, message_type, payload, priority)

    async def start_listener(self) -> None:
        """리스너 시작"""
        await super().start_listener()

    async def stop_listener(self) -> None:
        """리스너 중지"""
        await super().stop_listener()

    async def register_capabilities(self, capabilities: List[str]) -> None:
        """능력 등록"""
        await super().register_capabilities(capabilities)


# 편의를 위한 함수들
async def create_sparkleforge_agent(
    agent_id: str = "sparkleforge_agent"
) -> SparkleForgeA2AWrapper:
    """SparkleForge A2A agent 생성"""
    return SparkleForgeA2AWrapper(agent_id)


async def execute_sparkleforge_via_a2a(
    request: str,
    output_path: Optional[str] = None,
    streaming: bool = False
) -> Dict[str, Any]:
    """
    A2A를 통해 SparkleForge 실행

    Args:
        request: 연구 요청
        output_path: 출력 파일 경로
        streaming: 스트리밍 모드

    Returns:
        실행 결과
    """
    wrapper = await create_sparkleforge_agent()
    input_data = {
        'request': request,
        'output_path': output_path,
        'streaming': streaming
    }

    return await wrapper.execute_request(input_data)


if __name__ == "__main__":
    # 테스트 실행
    async def test():
        print("🔍 SparkleForge A2A Wrapper 테스트...")

        try:
            result = await execute_sparkleforge_via_a2a(
                request="인공지능의 미래 전망 분석",
                streaming=False
            )

            print("✅ 테스트 성공:")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test())
