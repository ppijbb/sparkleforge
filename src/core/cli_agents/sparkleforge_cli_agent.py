"""SparkleForge CLI Agent

SparkleForge CLI 도구를 위한 어댑터
"""

import json
import re
from typing import Any, Dict

from .base_cli_agent import BaseCLIAgent, CLIAgentConfig, CLIExecutionResult


class SparkleForgeCLIAgent(BaseCLIAgent):
    """SparkleForge CLI 에이전트

    특징:
    - SparkleForge 멀티 에이전트 연구 및 엔지니어링 CLI 기반
    - 명령줄 인터페이스로 접근
    - 다양한 작업 유형 지원
    """

    def __init__(self, api_key: str | None = None, model: str = "google/gemini-3.5-flash-lite"):
        env: Dict[str, str] = {"SPARKLE_LLM_MODEL_PATH": model}
        if api_key:
            env["SPARKLEFORGE_CLI_API_KEY"] = api_key

        config = CLIAgentConfig(
            name="sparkleforge_cli",
            command="sparkleforge",
            args=["--autopilot", "true"],
            env=env,
            timeout=180,
            output_format="text",
        )
        super().__init__(config)

    async def execute_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """SparkleForge CLI로 쿼리 실행

        Args:
            query: 실행할 쿼리
            **kwargs: 추가 옵션들
                - model: 사용할 모델

        Returns:
            표준화된 결과
        """
        model = kwargs.get("model", self.config.env.get("SPARKLE_LLM_MODEL_PATH", "google/gemini-3.5-flash-lite"))

        args = ["run"]
        if model:
            args.extend(["--model", model])
        args.append(query)

        # self.config.args (`["--autopilot", "true"]`)는 _execute_command가 뒤에 덧붙인다.
        result = await self._execute_command([self.config.command] + args)

        parsed_result = self.parse_output(result)

        return {
            "success": result.success and parsed_result.get("success", True),
            "response": parsed_result.get("response", ""),
            "confidence": parsed_result.get("confidence", 0.8),
            "metadata": {
                "agent": "sparkleforge_cli",
                "model": model,
                "execution_time": result.execution_time,
            },
            "usage": parsed_result.get("usage", {}),
        }

    def parse_output(self, result: CLIExecutionResult) -> Dict[str, Any]:
        """SparkleForge CLI 출력을 파싱"""
        if not result.success:
            return {
                "success": False,
                "error": result.error,
                "response": "",
                "confidence": 0.0,
            }

        try:
            stripped = result.output.strip()
            if self.config.output_format == "json" or stripped.startswith("{"):
                data = json.loads(stripped)
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "confidence": data.get("confidence", 0.8),
                    "usage": data.get("usage", {}),
                    "model": data.get("model", "google/gemini-3.5-flash-lite"),
                }

            confidence_match = re.search(r"confidence:?\s*([0-9.]+)", stripped, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.8

            return {
                "success": True,
                "response": stripped,
                "confidence": min(confidence, 1.0),
                "usage": {},
            }

        except json.JSONDecodeError:
            return {
                "success": True,
                "response": result.output.strip(),
                "confidence": 0.7,
                "usage": {},
            }

        except Exception as e:
            self.logger.error(f"Failed to parse SparkleForge CLI output: {e}")
            return {
                "success": False,
                "error": f"Parsing failed: {e}",
                "response": result.output,
                "confidence": 0.0,
            }
