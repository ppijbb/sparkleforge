"""Hermes CLI Agent - Hermes AI Agent CLI adapter

Hermes 자율 에이전트 CLI 도구를 위한 어댑터
"""

import json
import re
from typing import Any, Dict

from .base_cli_agent import BaseCLIAgent, CLIAgentConfig, CLIExecutionResult


class HermesCLIAgent(BaseCLIAgent):
    """Hermes CLI 에이전트

    특징:
    - Hermes 커스텀 에이전트 및 태스크 워크플로우 전용
    - 커스텀 툴 콜 및 도메인 전용 시나리오 수행
    - 터미널 CLI / 템플릿 기반 구동
    """

    def __init__(self, api_key: str | None = None, command: str = "hermes"):
        config = CLIAgentConfig(
            name="hermes",
            command=command,
            args=["--format", "json"],
            env={"HERMES_API_KEY": api_key} if api_key else {},
            timeout=300,
            output_format="json",
        )
        super().__init__(config)

    async def execute_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Hermes CLI에 쿼리를 실행

        Args:
            query: 실행할 요청
            **kwargs:
                - task_type: 작업 유형 (workflow, reasoning, agentic)
                - context: 추가 컨텍스트
                - tools: 사용 허용 도구 리스트

        Returns:
            표준화된 결과
        """
        task_type = kwargs.get("task_type", "agentic")
        context = kwargs.get("context", "")
        tools = kwargs.get("tools", [])

        args = ["run", "--query", query, "--type", task_type]

        if context:
            args.extend(["--context", context])
        if tools:
            args.extend(["--tools", ",".join(tools)])

        original_args = self.config.args.copy()
        self.config.args.extend(args)

        try:
            result = await self._execute_command(self.config.command)
            parsed_result = self.parse_output(result)

            return {
                "success": result.success and parsed_result.get("success", True),
                "response": parsed_result.get("response", ""),
                "confidence": parsed_result.get("confidence", 0.80),
                "metadata": {
                    "agent": "hermes",
                    "task_type": task_type,
                    "execution_time": result.execution_time,
                },
                "usage": parsed_result.get("usage", {}),
            }

        finally:
            self.config.args = original_args

    def parse_output(self, result: CLIExecutionResult) -> Dict[str, Any]:
        """Hermes CLI 출력 파싱"""
        if not result.success:
            return {
                "success": False,
                "error": result.error or "Hermes execution failed",
                "response": "",
                "confidence": 0.0,
            }

        try:
            if self.config.output_format == "json":
                data = json.loads(result.output.strip())
                return {
                    "success": True,
                    "response": data.get("response") or data.get("output") or result.output.strip(),
                    "confidence": float(data.get("confidence", 0.80)),
                    "usage": data.get("usage", {}),
                    "artifacts": data.get("artifacts", []),
                }
            else:
                response = result.output.strip()
                confidence_match = re.search(r"confidence:?\s*([0-9.]+)", response, re.IGNORECASE)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.80
                return {
                    "success": True,
                    "response": response,
                    "confidence": min(confidence, 1.0),
                    "usage": {},
                }

        except json.JSONDecodeError:
            return {
                "success": True,
                "response": result.output.strip(),
                "confidence": 0.70,
                "usage": {},
            }
        except Exception as e:
            self.logger.error(f"Failed to parse Hermes CLI output: {e}")
            return {
                "success": False,
                "error": f"Parsing failed: {e}",
                "response": result.output,
                "confidence": 0.0,
            }
