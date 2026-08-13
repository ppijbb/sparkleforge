"""Codex CLI Agent - OpenAI Codex / CLI tool adapter

OpenAI Codex 기반 코드 생성, 구문 보정 및 리팩토링 CLI 에이전트
"""

import logging
import json
import re
from typing import Any, Dict

from .base_cli_agent import BaseCLIAgent, CLIAgentConfig, CLIExecutionResult


class CodexCLIAgent(BaseCLIAgent):
    """Codex CLI 에이전트

    특징:
    - OpenAI Codex 기반 코드 생성 및 구문 수정
    - 빠른 코드 조각 생성 및 독립 함수 합성
    - 터미널 CLI 기반 상호작용
    """

    def __init__(self, api_key: str | None = None, command: str = "codex"):
        config = CLIAgentConfig(
            name="codex",
            command=command,
            # --json is an `exec`-subcommand option, not a top-level codex flag;
            # it is appended after "exec" in execute_query, not here.
            args=[],
            env={"OPENAI_API_KEY": api_key} if api_key else {},
            timeout=300,
            output_format="json",
        )
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    async def execute_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Codex CLI에 쿼리를 실행

        Args:
            query: 실행할 요청 (코드 생성/수정/구문검사)
            **kwargs:
                - language: 프로그래밍 언어 (python, typescript 등)
                - context: 추가 문맥

        Returns:
            표준화된 결과
        """
        language = kwargs.get("language", "python")
        context = kwargs.get("context", "")

        # `codex exec`는 프롬프트를 위치 인자로만 받음 (--prompt/--lang/--context 없음)
        prompt_parts = [query]
        if language and language != "python":
            prompt_parts.append(f"[Language]: {language}")
        if context:
            prompt_parts.append(f"[Context]\n{context}")
        full_prompt = "\n\n".join(prompt_parts)

        args = ["exec", "--json", full_prompt]

        # 명령 실행 (공유 상태 self.config.args를 수정하지 않음)
        # _execute_command가 self.config.args를 자동으로 덧붙이므로 extra args만 전달
        result = await self._execute_command([self.config.command] + args)
        parsed_result = self.parse_output(result)

        return {
            "success": result.success and parsed_result.get("success", True),
            "response": parsed_result.get("response", ""),
            "confidence": parsed_result.get("confidence", 0.85),
            "metadata": {
                "agent": "codex",
                "language": language,
                "execution_time": result.execution_time,
            },
            "usage": parsed_result.get("usage", {}),
        }

    def parse_output(self, result: CLIExecutionResult) -> Dict[str, Any]:
        """Codex CLI 출력 파싱"""
        if not result.success:
            return {
                "success": False,
                "error": result.error or "Codex execution failed",
                "response": "",
                "confidence": 0.0,
            }

        try:
            if self.config.output_format == "json":
                data = json.loads(result.output.strip())
                return {
                    "success": True,
                    "response": data.get("response") or data.get("code") or result.output.strip(),
                    "confidence": float(data.get("confidence", 0.85)),
                    "usage": data.get("usage", {}),
                }
            else:
                response = result.output.strip()
                confidence_match = re.search(r"confidence:?\s*([0-9.]+)", response, re.IGNORECASE)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.85
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
                "confidence": 0.75,
                "usage": {},
            }
        except Exception as e:
            self.logger.error(f"Failed to parse Codex CLI output: {e}")
            return {
                "success": False,
                "error": f"Parsing failed: {e}",
                "response": result.output,
                "confidence": 0.0,
            }
