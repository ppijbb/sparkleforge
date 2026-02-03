"""
Claude Code CLI Agent

Anthropic의 Claude Code CLI 도구를 위한 어댑터
"""

import json
import re
from typing import Dict, Any, Optional
from .base_cli_agent import BaseCLIAgent, CLIAgentConfig, CLIExecutionResult


class ClaudeCodeAgent(BaseCLIAgent):
    """
    Claude Code CLI 에이전트

    특징:
    - Anthropic Claude 기반
    - 코드 생성/수정 전문
    - 터미널 기반 인터랙션
    """

    def __init__(self, api_key: Optional[str] = None):
        config = CLIAgentConfig(
            name="claude_code",
            command="claude",
            args=["--output-format", "json"],
            env={"ANTHROPIC_API_KEY": api_key} if api_key else {},
            timeout=600,  # Claude Code는 복잡한 작업에 시간이 걸릴 수 있음
            output_format="json"
        )
        super().__init__(config)

    async def execute_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Claude Code로 쿼리 실행

        Args:
            query: 실행할 쿼리 (코드 생성/수정 요청)
            **kwargs: 추가 옵션들
                - files: 관련 파일 목록
                - context: 추가 컨텍스트
                - mode: "code" 또는 "chat"

        Returns:
            표준화된 결과
        """
        mode = kwargs.get('mode', 'code')
        files = kwargs.get('files', [])
        context = kwargs.get('context', '')

        # Claude Code 명령 구성
        args = []

        if mode == "code":
            args.extend(["code", "--query", query])
        else:
            args.extend(["chat", query])

        # 파일 지정
        if files:
            for file in files:
                args.extend(["--file", file])

        # 컨텍스트 추가
        if context:
            args.extend(["--context", context])

        # 설정 업데이트
        original_args = self.config.args.copy()
        self.config.args.extend(args)

        try:
            # 명령 실행
            result = await self._execute_command(self.config.command)

            # 결과 파싱
            parsed_result = self.parse_output(result)

            return {
                'success': result.success and parsed_result.get('success', True),
                'response': parsed_result.get('response', ''),
                'confidence': parsed_result.get('confidence', 0.8),
                'metadata': {
                    'agent': 'claude_code',
                    'mode': mode,
                    'files': files,
                    'execution_time': result.execution_time
                },
                'usage': parsed_result.get('usage', {})
            }

        finally:
            # 설정 복원
            self.config.args = original_args

    def parse_output(self, result: CLIExecutionResult) -> Dict[str, Any]:
        """
        Claude Code 출력을 파싱

        Claude Code의 출력 형식:
        {
            "response": "생성된 코드 또는 응답",
            "confidence": 0.85,
            "usage": {"input_tokens": 150, "output_tokens": 200},
            "files_modified": ["file1.py", "file2.py"]
        }
        """
        if not result.success:
            return {
                'success': False,
                'error': result.error,
                'response': '',
                'confidence': 0.0
            }

        try:
            # JSON 출력 파싱 시도
            if self.config.output_format == "json":
                data = json.loads(result.output.strip())
                return {
                    'success': True,
                    'response': data.get('response', ''),
                    'confidence': data.get('confidence', 0.8),
                    'usage': data.get('usage', {}),
                    'files_modified': data.get('files_modified', [])
                }

            # 텍스트 출력 파싱
            else:
                # Claude Code의 텍스트 출력을 파싱
                response = result.output.strip()

                # 신뢰도 추출 (선택적)
                confidence_match = re.search(r'confidence:?\s*([0-9.]+)', response, re.IGNORECASE)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.8

                return {
                    'success': True,
                    'response': response,
                    'confidence': min(confidence, 1.0),
                    'usage': {}
                }

        except json.JSONDecodeError:
            # JSON 파싱 실패 시 텍스트로 처리
            return {
                'success': True,
                'response': result.output.strip(),
                'confidence': 0.7,
                'usage': {}
            }

        except Exception as e:
            self.logger.error(f"Failed to parse Claude Code output: {e}")
            return {
                'success': False,
                'error': f"Parsing failed: {e}",
                'response': result.output,
                'confidence': 0.0
            }