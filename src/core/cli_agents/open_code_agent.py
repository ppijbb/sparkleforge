"""Open Code Agent — OpenRouter API direct call (Kimi K 2.5 default)

opencode CLI run 명령이 non-interactive 환경에서 hang되므로,
OpenRouter REST API를 직접 호출하여 동일한 역할을 수행한다.
"""

import logging
import os
import time
from typing import Any, Dict

import aiohttp

from .base_cli_agent import BaseCLIAgent, CLIAgentConfig, CLIExecutionResult

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GOOGLE_GENAI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "moonshotai/kimi-k2.5"
OPENROUTER_FALLBACKS = [
    "moonshotai/kimi-k2",
    "qwen/qwen3-32b",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-r1-0528",
]
GOOGLE_FALLBACK_MODEL = "gemini-2.5-flash-lite"


class OpenCodeAgent(BaseCLIAgent):
    """LLM agent: OpenRouter primary (Kimi K 2.5) → OpenRouter fallbacks → Google Gemini API."""

    def __init__(self, model_path: str | None = None):
        raw = model_path or os.getenv("OPEN_CODE_MODEL_PATH") or DEFAULT_MODEL
        if "/" not in raw:
            raw = f"moonshotai/{raw}"
        self._model = raw
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")
        self._google_key = os.getenv("GOOGLE_API_KEY", "")
        config = CLIAgentConfig(
            name="open_code",
            command="opencode",
            args=[],
            env={},
            timeout=120,
            output_format="text",
        )
        super().__init__(config)

    async def execute_query(self, query: str, **kwargs) -> Dict[str, Any]:
        system_msg = kwargs.get("system_message") or "You are a helpful research assistant."
        start = time.time()
        try:
            text = await self._call_openrouter(query, system_msg)
            elapsed = time.time() - start
            return {
                "success": bool(text),
                "response": text,
                "confidence": 0.85 if text else 0.0,
                "metadata": {
                    "agent": "open_code",
                    "model": self._model,
                    "execution_time": elapsed,
                },
                "usage": {},
            }
        except Exception as e:
            elapsed = time.time() - start
            logger.error("OpenCodeAgent API call failed: %s", e)
            return {
                "success": False,
                "response": f"[ERROR] {e}",
                "confidence": 0.0,
                "metadata": {
                    "agent": "open_code",
                    "model": self._model,
                    "execution_time": elapsed,
                    "error": str(e),
                },
                "usage": {},
            }

    async def _call_openrouter(self, user_msg: str, system_msg: str) -> str:
        models_to_try = [self._model] + [m for m in OPENROUTER_FALLBACKS if m != self._model]
        last_err = None
        for model in models_to_try:
            try:
                return await self._call_openrouter_single(model, user_msg, system_msg)
            except RuntimeError as e:
                last_err = e
                err_str = str(e)
                if any(code in err_str for code in ("402", "403", "404", "429")) or "limit" in err_str.lower():
                    logger.warning("Model %s unavailable (%s), trying fallback...", model, err_str[:80])
                    continue
                raise
        # OpenRouter 전부 실패 → Google Gemini API fallback
        if self._google_key:
            logger.info("All OpenRouter models exhausted, falling back to Google Gemini API")
            try:
                return await self._call_google_genai(user_msg, system_msg)
            except Exception as ge:
                logger.warning("Google Gemini fallback also failed: %s", ge)
        raise last_err or RuntimeError("All models failed")

    async def _call_openrouter_single(self, model: str, user_msg: str, system_msg: str) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://sparkleforge.local",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 4096,
            "temperature": 0.2,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=90)
            ) as resp:
                body = await resp.json()
                if resp.status != 200:
                    err = body.get("error", {}).get("message", str(body))
                    raise RuntimeError(f"OpenRouter {resp.status}: {err}")
                choices = body.get("choices", [])
                if not choices:
                    raise RuntimeError("OpenRouter returned no choices")
                return choices[0].get("message", {}).get("content", "")

    async def _call_google_genai(self, user_msg: str, system_msg: str) -> str:
        url = GOOGLE_GENAI_URL.format(model=GOOGLE_FALLBACK_MODEL)
        payload = {
            "contents": [{"parts": [{"text": user_msg}]}],
            "systemInstruction": {"parts": [{"text": system_msg}]},
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                params={"key": self._google_key},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                body = await resp.json()
                if resp.status != 200:
                    err = body.get("error", {}).get("message", str(body))
                    raise RuntimeError(f"Google Gemini {resp.status}: {err}")
                candidates = body.get("candidates", [])
                if not candidates:
                    raise RuntimeError("Google Gemini returned no candidates")
                parts = candidates[0].get("content", {}).get("parts", [])
                return parts[0].get("text", "") if parts else ""

    def parse_output(self, result: CLIExecutionResult) -> Dict[str, Any]:
        return {"success": True, "text": result.output or "", "raw": result.output or ""}
