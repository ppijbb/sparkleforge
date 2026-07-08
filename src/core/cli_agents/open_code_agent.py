"""Open Code Agent — provider 우선 호출 (Google Gemini / OpenRouter)

OPENCODE_PRIMARY=google 이면 Gemini 우선 호출로 OpenRouter 일일 한도 소진 방지.
opencode CLI run 명령이 non-interactive 환경에서 hang되므로, REST API 직접 호출.
"""

import json
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
    "google/gemini-3.1-flash-lite-preview",
    "moonshotai/kimi-k2",
    "qwen/qwen3-32b",
    "deepseek/deepseek-r1-0528",
]
GOOGLE_FALLBACK_MODEL = "gemini-3.1-flash-lite-preview"


# OPENCODE_PRIMARY: "google" = Gemini 우선 (한도 절약), "openrouter" = OpenRouter 우선, "nvidia" = NVIDIA NIM
def _primary_provider() -> str:
    raw = (os.getenv("OPENCODE_PRIMARY") or "").strip().lower()
    if raw in ("google", "openrouter", "nvidia"):
        return raw
    if os.getenv("NVIDIA_API_KEY"):
        return "nvidia"
    return "google" if os.getenv("GOOGLE_API_KEY") else "openrouter"


class OpenCodeAgent(BaseCLIAgent):
    """LLM agent: OPENCODE_PRIMARY에 따라 Google, NVIDIA NIM, OpenRouter 우선 호출 및 fallback."""

    def __init__(self, model_path: str | None = None):
        raw = model_path or os.getenv("OPEN_CODE_MODEL_PATH") or DEFAULT_MODEL
        if "/" not in raw:
            raw = f"moonshotai/{raw}"
        self._model = raw
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")
        self._google_key = os.getenv("GOOGLE_API_KEY", "")
        self._nvidia_key = os.getenv("NVIDIA_API_KEY", "")
        self._primary = _primary_provider()
        self._max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
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
        max_tokens = int(kwargs.get("max_tokens") or self._max_tokens)
        start = time.time()
        try:
            text = await self._call_llm(query, system_msg, max_tokens=max_tokens)
            elapsed = time.time() - start
            return {
                "success": bool(text),
                "response": text,
                "confidence": 0.85 if text else 0.0,
                "metadata": {
                    "agent": "open_code",
                    "model": self._model,
                    "max_tokens": max_tokens,
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
                    "max_tokens": max_tokens,
                    "execution_time": elapsed,
                    "error": str(e),
                },
                "usage": {},
            }

    def _google_model(self) -> str:
        if self._model.startswith("google/"):
            stripped = self._model.split("/", 1)[1]
            if stripped.startswith("models/"):
                stripped = stripped.split("/", 1)[1]
            return stripped
        if self._model.startswith("models/"):
            return self._model.split("/", 1)[1]
        if self._model.startswith("gemini-"):
            return self._model
        return GOOGLE_FALLBACK_MODEL

    async def _call_llm(self, user_msg: str, system_msg: str, max_tokens: int) -> str:
        """OPENCODE_PRIMARY에 따라 Google, NVIDIA, OpenRouter 호출."""
        if self._primary == "nvidia" and self._nvidia_key:
            try:
                return await self._call_nvidia_nim(user_msg, system_msg, max_tokens)
            except Exception as e:
                logger.warning(
                    "NVIDIA NIM primary failed (%s), trying Google or OpenRouter...", str(e)[:60]
                )

        if self._primary == "google" and self._google_key:
            try:
                return await self._call_google_genai(user_msg, system_msg, max_tokens)
            except Exception as e:
                logger.warning(
                    "Google Gemini primary failed (%s), trying OpenRouter...", str(e)[:60]
                )
        if self._api_key:
            try:
                return await self._call_openrouter_chain(user_msg, system_msg, max_tokens)
            except RuntimeError as e:
                if self._nvidia_key:
                    logger.info("OpenRouter failed, falling back to NVIDIA NIM")
                    try:
                        return await self._call_nvidia_nim(user_msg, system_msg, max_tokens)
                    except Exception as ne:
                        logger.warning("NVIDIA NIM fallback also failed: %s", ne)
                if self._google_key:
                    logger.info("OpenRouter chain failed, falling back to Google Gemini")
                    try:
                        return await self._call_google_genai(user_msg, system_msg, max_tokens)
                    except Exception as ge:
                        logger.warning("Google Gemini fallback also failed: %s", ge)
                raise e
        if self._nvidia_key:
            return await self._call_nvidia_nim(user_msg, system_msg, max_tokens)
        if self._google_key:
            return await self._call_google_genai(user_msg, system_msg, max_tokens)
        raise RuntimeError(
            "No API key available. Set GOOGLE_API_KEY, NVIDIA_API_KEY, and/or OPENROUTER_API_KEY."
        )

    async def _call_openrouter_chain(self, user_msg: str, system_msg: str, max_tokens: int) -> str:
        """OpenRouter 모델 순서대로 시도 후 실패 시 예외."""
        models_to_try = [self._model] + [m for m in OPENROUTER_FALLBACKS if m != self._model]
        last_err = None
        for model in models_to_try:
            try:
                return await self._call_openrouter_single(model, user_msg, system_msg, max_tokens)
            except RuntimeError as e:
                last_err = e
                err_str = str(e)
                if (
                    any(c in err_str for c in ("402", "403", "404", "429"))
                    or "limit" in err_str.lower()
                ):
                    logger.warning(
                        "Model %s unavailable (%s), trying fallback...", model, err_str[:80]
                    )
                    continue
                raise
        raise last_err or RuntimeError("All OpenRouter models failed")

    async def _call_openrouter_single(
        self, model: str, user_msg: str, system_msg: str, max_tokens: int
    ) -> str:
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
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                raw = await resp.text()
                try:
                    body = json.loads(raw) if raw else {}
                except Exception:
                    raise RuntimeError(
                        f"OpenRouter {resp.status}: {raw[:200] if raw else 'empty response'}"
                    )
                if resp.status != 200:
                    err = body.get("error", {}).get("message", str(body))
                    raise RuntimeError(f"OpenRouter {resp.status}: {err}")
                choices = body.get("choices", [])
                if not choices:
                    raise RuntimeError("OpenRouter returned no choices")
                return choices[0].get("message", {}).get("content", "")

    async def _call_google_genai(
        self, user_msg: str, system_msg: str, max_tokens: int
    ) -> str:
        url = GOOGLE_GENAI_URL.format(model=self._google_model())
        payload = {
            "contents": [{"parts": [{"text": user_msg}]}],
            "systemInstruction": {"parts": [{"text": system_msg}]},
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                params={"key": self._google_key},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                raw = await resp.text()
                try:
                    body = json.loads(raw) if raw else {}
                except Exception:
                    raise RuntimeError(
                        f"Google Gemini {resp.status}: {raw[:200] if raw else 'empty response'}"
                    )
                if resp.status != 200:
                    err = body.get("error", {}).get("message", str(body))
                    raise RuntimeError(f"Google Gemini {resp.status}: {err}")
                candidates = body.get("candidates", [])
                if not candidates:
                    raise RuntimeError("Google Gemini returned no candidates")
                parts = candidates[0].get("content", {}).get("parts", [])
                return parts[0].get("text", "") if parts else ""

    async def _call_nvidia_nim(
        self, user_msg: str, system_msg: str, max_tokens: int
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self._nvidia_key}",
            "Content-Type": "application/json",
        }
        model = self._model
        if not model.startswith("z-ai/") and model != "glm-5.2":
            model = "z-ai/glm-5.2"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                raw = await resp.text()
                try:
                    body = json.loads(raw) if raw else {}
                except Exception:
                    raise RuntimeError(
                        f"NVIDIA NIM {resp.status}: {raw[:200] if raw else 'empty response'}"
                    )
                if resp.status != 200:
                    err = body.get("error", {}).get("message", str(body))
                    raise RuntimeError(f"NVIDIA NIM {resp.status}: {err}")
                choices = body.get("choices", [])
                if not choices:
                    raise RuntimeError("NVIDIA NIM returned no choices")
                return choices[0].get("message", {}).get("content", "")

    def parse_output(self, result: CLIExecutionResult) -> Dict[str, Any]:
        return {"success": True, "text": result.output or "", "raw": result.output or ""}
