"""
Open Code CLI Agent

OpenCode CLI 도구를 위한 어댑터
"""

import json
import re
from typing import Dict, Any, Optional
from .base_cli_agent import BaseCLIAgent, CLIAgentConfig, CLIExecutionResult


class OpenCodeAgent(BaseCLIAgent):
    """
    OpenCode CLI 에이전트

    특징:
    - 오픈소스 기반 코드 생성/분석
    - 로컬 실행으로 프라이버시 보장
    - 다양한 프로그래밍 언어 지원
    """

    def __init__(self, model_path: Optional[str] = None):
        config = CLIAgentConfig(
            name="open_code",
            command="opencode",
            args=["--json"],
            env={"OPEN_CODE_MODEL": model_path} if model_path else {},
            timeout=300,
            output_format="json"
        )
        super().__init__(config)

    async def execute_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        OpenCode로 쿼리 실행

        Args:
            query: 코드 생성/분석 요청
            **kwargs: 추가 옵션들
                - language: 프로그래밍 언어
                - context: 코드 컨텍스트
                - mode: "generate", "analyze", "complete"

        Returns:
            표준화된 결과
        """
        language = kwargs.get('language', 'python')
        context = kwargs.get('context', '')
        mode = kwargs.get('mode', 'generate')

        # OpenCode 명령 구성
        args = [mode]

        if language:
            args.extend(["--language", language])

        if context:
            args.extend(["--context", context])

        args.append(query)

        # 설정 업데이트
        original_args = self.config.args.copy()
        self.config.args = args

        try:
            # 명령 실행
            result = await self._execute_command(self.config.command)

            # 결과 파싱
            parsed_result = self.parse_output(result)

            return {
                'success': result.success and parsed_result.get('success', True),
                'response': parsed_result.get('code', ''),
                'confidence': parsed_result.get('confidence', 0.8),
                'metadata': {
                    'agent': 'open_code',
                    'language': language,
                    'mode': mode,
                    'execution_time': result.execution_time
                },
                'usage': parsed_result.get('usage', {})
            }

        finally:
            # 설정 복원
            self.config.args = original_args

    def parse_output(self, result: CLIExecutionResult) -> Dict[str, Any]:
        """
        OpenCode 출력을 파싱

        OpenCode의 출력 형식:
        {
            "code": "생성된 코드",
            "confidence": 0.85,
            "usage": {"tokens": 150},
            "language": "python"
        }
        """
        if not result.success:
            return {
                'success': False,
                'error': result.error,
                'code': '',
                'confidence': 0.0
            }

        try:
            # JSON 출력 파싱
            if self.config.output_format == "json":
                data = json.loads(result.output.strip())
                return {
                    'success': True,
                    'code': data.get('code', ''),
                    'confidence': data.get('confidence', 0.8),
                    'usage': data.get('usage', {}),
                    'language': data.get('language', 'python')
                }

            # 텍스트 출력 파싱
            else:
                code = result.output.strip()

                # 신뢰도 추출
                confidence_match = re.search(r'confidence:?\s*([0-9.]+)', code, re.IGNORECASE)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.8

                return {
                    'success': True,
                    'code': code,
                    'confidence': min(confidence, 1.0),
                    'usage': {}
                }

        except json.JSONDecodeError:
            return {
                'success': True,
                'code': result.output.strip(),
                'confidence': 0.7,
                'usage': {}
            }

        except Exception as e:
            self.logger.error(f"Failed to parse OpenCode output: {e}")
            return {
                'success': False,
                'error': f"Parsing failed: {e}",
                'code': result.output,
                'confidence': 0.0
            }