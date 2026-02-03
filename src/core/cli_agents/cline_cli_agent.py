"""
Cline CLI Agent

Cline CLI 도구를 위한 어댑터
"""

import json
import re
from typing import Dict, Any, Optional
from .base_cli_agent import BaseCLIAgent, CLIAgentConfig, CLIExecutionResult


class ClineCLIAgent(BaseCLIAgent):
    """
    Cline CLI 에이전트

    특징:
    - 코드 에디팅 및 리팩토링 전문
    - 자동화된 코드 수정
    - 다중 파일 동시 수정 지원
    """

    def __init__(self, config_path: Optional[str] = None):
        config = CLIAgentConfig(
            name="cline_cli",
            command="cline",
            args=["--json-output"],
            env={"CLINE_CONFIG": config_path} if config_path else {},
            timeout=600,  # 코드 수정 작업은 시간이 오래 걸릴 수 있음
            output_format="json"
        )
        super().__init__(config)

    async def execute_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Cline CLI로 쿼리 실행

        Args:
            query: 코드 수정/리팩토링 요청
            **kwargs: 추가 옵션들
                - files: 대상 파일 목록
                - operation: "edit", "refactor", "generate", "analyze"
                - context: 추가 컨텍스트
                - dry_run: 변경 미리보기만

        Returns:
            표준화된 결과
        """
        files = kwargs.get('files', [])
        operation = kwargs.get('operation', 'edit')
        context = kwargs.get('context', '')
        dry_run = kwargs.get('dry_run', False)

        # Cline 명령 구성
        args = [operation]

        # 파일 지정
        if files:
            for file in files:
                args.extend(["--file", file])

        # 컨텍스트 추가
        if context:
            args.extend(["--context", context])

        # dry run 모드
        if dry_run:
            args.append("--dry-run")

        # 쿼리 추가
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
                'response': parsed_result.get('changes', ''),
                'confidence': parsed_result.get('confidence', 0.8),
                'metadata': {
                    'agent': 'cline_cli',
                    'operation': operation,
                    'files': files,
                    'dry_run': dry_run,
                    'execution_time': result.execution_time,
                    'changes_applied': parsed_result.get('changes_applied', [])
                },
                'usage': parsed_result.get('usage', {})
            }

        finally:
            # 설정 복원
            self.config.args = original_args

    def parse_output(self, result: CLIExecutionResult) -> Dict[str, Any]:
        """
        Cline CLI 출력을 파싱

        Cline의 출력 형식:
        {
            "changes": "적용된 변경사항 요약",
            "confidence": 0.85,
            "usage": {"tokens": 200},
            "changes_applied": [
                {"file": "file1.py", "line": 10, "change": "added import"},
                {"file": "file2.py", "line": 25, "change": "modified function"}
            ],
            "dry_run": false
        }
        """
        if not result.success:
            return {
                'success': False,
                'error': result.error,
                'changes': '',
                'confidence': 0.0
            }

        try:
            # JSON 출력 파싱
            if self.config.output_format == "json":
                data = json.loads(result.output.strip())
                return {
                    'success': True,
                    'changes': data.get('changes', ''),
                    'confidence': data.get('confidence', 0.8),
                    'usage': data.get('usage', {}),
                    'changes_applied': data.get('changes_applied', [])
                }

            # 텍스트 출력 파싱
            else:
                changes = result.output.strip()

                # 신뢰도 추출
                confidence_match = re.search(r'confidence:?\s*([0-9.]+)', changes, re.IGNORECASE)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.8

                # 변경사항 파일 추출
                changes_applied = []
                file_matches = re.findall(r'(\w+\.\w+):(\d+):\s*(.+)', changes)
                for file_match, line, change_desc in file_matches:
                    changes_applied.append({
                        'file': file_match,
                        'line': int(line),
                        'change': change_desc.strip()
                    })

                return {
                    'success': True,
                    'changes': changes,
                    'confidence': min(confidence, 1.0),
                    'usage': {},
                    'changes_applied': changes_applied
                }

        except json.JSONDecodeError:
            # 텍스트로 처리
            changes = result.output.strip()
            return {
                'success': True,
                'changes': changes,
                'confidence': 0.7,
                'usage': {},
                'changes_applied': []
            }

        except Exception as e:
            self.logger.error(f"Failed to parse Cline CLI output: {e}")
            return {
                'success': False,
                'error': f"Parsing failed: {e}",
                'changes': result.output,
                'confidence': 0.0
            }