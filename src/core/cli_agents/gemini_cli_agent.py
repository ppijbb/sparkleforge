"""Gemini CLI Agent

Google Gemini CLI 도구를 위한 어댑터
"""

import json
import re
from typing import Any, Dict

from .base_cli_agent import BaseCLIAgent, CLIAgentConfig, CLIExecutionResult


class GeminiCLIAgent(BaseCLIAgent):
    """Gemini CLI 에이전트

    특징:
    - Google Gemini 모델 기반
    - 명령줄 인터페이스로 접근
    - 다양한 작업 유형 지원
    """

    def __init__(self, api_key: str | None = None, model: str = "gemini-pro"):
        config = CLIAgentConfig(
            name="gemini_cli",
            command="gemini",
            args=["--output-format", "json"],
            env=(
                {"GEMINI_API_KEY": api_key, "GEMINI_MODEL": model}
                if api_key
                else {"GEMINI_MODEL": model}
            ),
            timeout=180,
            output_format="json",
        )
        super().__init__(config)

    async def execute_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Gemini CLI로 쿼리 실행

        Args:
            query: 실행할 쿼리
            **kwargs: 추가 옵션들
                - model: 사용할 모델

        Returns:
            표준화된 결과
        """
        model = kwargs.get("model", "gemini-pro")

        # Gemini CLI는 서브커맨드 없이 프롬프트를 위치 인자로만 받음
        args = []
        if model != "gemini-pro":
            args.extend(["--model", model])
        args.append(query)

        # 명령 실행: self.config.args는 _execute_command가 자동으로 덧붙이므로
        # 여기서 다시 합치면 중복 인자가 생겨 CLI가 거부한다 (공유 상태는 그대로 둠)
        result = await self._execute_command([self.config.command] + args)

        # 결과 파싱
        parsed_result = self.parse_output(result)

        return {
            "success": result.success and parsed_result.get("success", True),
            "response": parsed_result.get("response", ""),
            "confidence": parsed_result.get("confidence", 0.8),
            "metadata": {
                "agent": "gemini_cli",
                "model": model,
                "execution_time": result.execution_time,
            },
            "usage": parsed_result.get("usage", {}),
        }

    def parse_output(self, result: CLIExecutionResult) -> Dict[str, Any]:
        """Gemini CLI 출력을 파싱

        Gemini CLI의 출력 형식:
        {
            "response": "Gemini 응답",
            "confidence": 0.85,
            "usage": {"input_tokens": 10, "output_tokens": 50},
            "model": "gemini-pro"
        }
        """
        if not result.success:
            return {
                "success": False,
                "error": result.error,
                "response": "",
                "confidence": 0.0,
            }

        try:
            # JSON 출력 파싱
            if self.config.output_format == "json":
                data = json.loads(result.output.strip())
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "confidence": data.get("confidence", 0.8),
                    "usage": data.get("usage", {}),
                    "model": data.get("model", "gemini-pro"),
                }

            # 텍스트 출력 파싱
            else:
                response = result.output.strip()

                # 신뢰도 추출
                confidence_match = re.search(r"confidence:?\s*([0-9.]+)", response, re.IGNORECASE)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.8

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
                "confidence": 0.7,
                "usage": {},
            }

        except Exception as e:
            self.logger.error(f"Failed to parse Gemini CLI output: {e}")
            return {
                "success": False,
                "error": f"Parsing failed: {e}",
                "response": result.output,
                "confidence": 0.0,
            }
