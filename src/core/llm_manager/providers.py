"""Per-provider execution adapters (Gemini/OpenRouter/Groq/Cerebras/OpenAI/NVIDIA/LangChain).

Split out of the former monolithic llm_manager.py (issue #582).
"""

import asyncio
import json
import logging
import os
import warnings
from typing import Any, Dict

import requests

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        import google.generativeai as genai
    except ImportError:
        genai = None  # type: ignore[assignment]
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    HumanMessage = None  # type: ignore[assignment]
    SystemMessage = None  # type: ignore[assignment]

from src.core.llm_manager.model_registry import (
    SAFETY_SETTINGS_BLOCK_NONE,
    _parse_openrouter_json_response,
)
from src.core.observability import get_langfuse_run_config

logger = logging.getLogger(__name__)


class ProviderAdaptersMixin:
    """Per-provider model execution (Gemini, OpenRouter, Groq, Cerebras, OpenAI, NVIDIA, LangChain)."""

    def _build_gemini_prompt_ordered(self, system_message: str | None, prompt: str) -> str:
        """Build prompt with context ordering for caching: static (system) first, then dynamic."""
        if not system_message:
            return prompt
        return f"{system_message}\n\n{prompt}"


    async def _execute_gemini_with_cached_content(
        self,
        model_name: str,
        model_config: Any,
        full_prompt: str,
        system_message: str,
        prompt: str,
    ) -> Dict[str, Any]:
        """Use Gemini explicit prompt caching when google-genai SDK is available."""
        from google import genai as genai_v2
        from google.genai import types

        api_key = self.llm_config.api_key
        model_id = getattr(model_config, "model_id", None) or model_name
        if not model_id.startswith("models/"):
            model_id = (
                f"models/{model_id}" if not model_id.startswith("gemini") else f"models/{model_id}"
            )

        client = genai_v2.Client(api_key=api_key)
        http_options = types.HttpOptions(timeout=60000)  # ms; bounds transport, not just the awaiting coroutine
        config = types.CreateCachedContentConfig(
            display_name=f"sparkleforge_{model_name}",
            system_instruction=system_message,
            ttl="3600s",
            http_options=http_options,
        )
        cache = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None, lambda: client.caches.create(model=model_id, config=config)
            ),
            timeout=60.0,
        )
        response = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        cached_content=cache.name,
                        temperature=getattr(model_config, "temperature", 0.1),
                        max_output_tokens=getattr(model_config, "max_tokens", 4000),
                        http_options=http_options,
                    ),
                ),
            ),
            timeout=60.0,
        )
        text = ""
        if response and getattr(response, "candidates", None):
            cand = response.candidates[0]
            if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                for part in cand.content.parts:
                    if getattr(part, "text", None):
                        text += part.text
        return {
            "content": text,
            "confidence": 0.9,
            "metadata": {"cached_content": True},
        }


    # JSON Schema keywords the harness's tool definitions carry that Gemini's
    # (much smaller) OpenAPI-subset Schema proto rejects outright ("Unknown
    # field for Schema: ...") rather than ignoring.
    _GEMINI_UNSUPPORTED_SCHEMA_KEYS = {"default", "additionalProperties", "$schema", "examples", "title"}

    def _clean_gemini_schema(self, schema: Any) -> Any:
        """Recursively drop JSON Schema keywords Gemini's Schema proto doesn't accept."""
        if isinstance(schema, dict):
            cleaned = {
                k: self._clean_gemini_schema(v)
                for k, v in schema.items()
                if k not in self._GEMINI_UNSUPPORTED_SCHEMA_KEYS
            }
            if "properties" in cleaned and isinstance(cleaned["properties"], dict):
                cleaned["properties"] = {
                    k: self._clean_gemini_schema(v) for k, v in cleaned["properties"].items()
                }
            return cleaned
        if isinstance(schema, list):
            return [self._clean_gemini_schema(item) for item in schema]
        return schema

    def _build_gemini_tools(self, tools: Any) -> list | None:
        """Convert OpenAI-style tool schemas to Gemini's function_declarations shape."""
        if not tools:
            return None
        declarations = []
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else None
            if not fn or not fn.get("name"):
                continue
            parameters = fn.get("parameters") or {"type": "object", "properties": {}}
            declarations.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": self._clean_gemini_schema(parameters),
                }
            )
        return [{"function_declarations": declarations}] if declarations else None

    def _extract_gemini_tool_calls(self, response: Any) -> list[Dict[str, Any]]:
        """Pull function_call parts out of a Gemini response into OpenAI-like tool_calls."""
        tool_calls = []
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                function_call = getattr(part, "function_call", None)
                if not function_call or not getattr(function_call, "name", None):
                    continue
                tool_calls.append(
                    {
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": function_call.name,
                            "arguments": json.dumps(dict(function_call.args or {})),
                        },
                    }
                )
        return tool_calls

    async def _execute_gemini_model(
        self, model_name: str, prompt: str, system_message: str = None, **kwargs
    ) -> Dict[str, Any]:
        """Gemini 모델 실행 (rate limit 재시도 포함). Prompt caching: static prefix first."""
        client = self.model_clients[model_name]
        model_config = self.models[model_name]

        # Context ordering for caching: static (system) first, then dynamic (prompt)
        full_prompt = self._build_gemini_prompt_ordered(system_message, prompt)

        # Convert OpenAI-style tool schemas (what the harness/other providers use)
        # into Gemini's function_declarations shape. Without this, gemini-flash was
        # never given any tools at all, so it could only ever answer in prose --
        # unrelated to the model's actual function-calling capability.
        gemini_tools = self._build_gemini_tools(kwargs.get("tools"))

        # Optional: explicit prompt caching when google-genai and env are set
        if (
            os.getenv("ENABLE_GEMINI_PROMPT_CACHING", "").lower() in ("1", "true", "yes")
            and system_message
            and len(system_message) >= 1024
        ):
            try:
                return await self._execute_gemini_with_cached_content(
                    model_name, model_config, full_prompt, system_message, prompt
                )
            except Exception as cache_e:
                logger.debug("Gemini prompt cache path skipped: %s", cache_e)

        # Rate limit 재시도 로직
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 실행
                response = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: client.generate_content(
                            full_prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=model_config.temperature,
                                max_output_tokens=model_config.max_tokens,
                            ),
                            tools=gemini_tools,
                            request_options={"timeout": 60},
                        ),
                    ),
                    timeout=60.0,
                )
                break  # 성공 시 루프 종료
            except Exception as e:
                error_str = str(e).lower()
                # Rate limit 에러 감지
                if (
                    "429" in error_str
                    or "rate limit" in error_str
                    or "quota exceeded" in error_str
                    or "resource_exhausted" in error_str
                ) and attempt < max_retries - 1:
                    wait_time = 5 * (2**attempt)  # 지수 백오프: 5초, 10초, 20초
                    logger.warning(
                        f"Gemini API rate limit (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Rate limit이 아니거나 최대 재시도 횟수 초과
                    raise

        # Function-call turn: response.text raises when the only parts are
        # function_call (no text part), so handle this before any text extraction.
        tool_calls = self._extract_gemini_tool_calls(response)
        if tool_calls:
            return {
                "content": "",
                "confidence": 0.8,
                "quality_score": 0.8,
                "metadata": {
                    "model": model_name,
                    "temperature": model_config.temperature,
                    "max_tokens": model_config.max_tokens,
                    "tool_calls": tool_calls,
                },
            }

        # finish_reason 체크 및 안전한 응답 처리
        finish_reason = None
        finish_reason_int = None
        has_valid_part = False

        # 먼저 candidates와 parts 확인
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason if hasattr(candidate, "finish_reason") else None

            # parts 확인
            if hasattr(candidate, "content") and candidate.content:
                if hasattr(candidate.content, "parts") and candidate.content.parts:
                    # text 속성을 가진 part가 있는지 확인
                    for part in candidate.content.parts:
                        if hasattr(part, "text"):
                            has_valid_part = True
                            break

            # finish_reason이 enum인 경우 숫자로 변환
            if finish_reason is not None:
                try:
                    # FinishReason enum인 경우
                    if hasattr(finish_reason, "value"):
                        finish_reason_int = finish_reason.value
                    elif hasattr(finish_reason, "name"):
                        # SAFETY = 2
                        if "SAFETY" in finish_reason.name or "SAFETY" in str(finish_reason):
                            finish_reason_int = 2
                    # 숫자인 경우
                    elif isinstance(finish_reason, int):
                        finish_reason_int = finish_reason
                    # 문자열인 경우
                    elif isinstance(finish_reason, str):
                        if "SAFETY" in finish_reason.upper() or finish_reason == "2":
                            finish_reason_int = 2
                except Exception:
                    # 변환 실패 시 그대로 사용
                    finish_reason_int = finish_reason if isinstance(finish_reason, int) else None

        # finish_reason이 2 (SAFETY)이거나 유효한 Part가 없는 경우
        if (
            finish_reason_int == 2
            or (finish_reason is not None and ("SAFETY" in str(finish_reason).upper()))
            or not has_valid_part
        ):
            if finish_reason_int == 2 or (
                finish_reason is not None and ("SAFETY" in str(finish_reason).upper())
            ):
                logger.warning(
                    f"Gemini API safety filter triggered (finish_reason={finish_reason}). Returning empty content."
                )
            else:
                logger.warning(
                    f"Gemini API response has no valid Part (finish_reason={finish_reason}). Returning empty content."
                )
            return {
                "content": "[Content blocked by safety filters. Please try rephrasing the request.]",
                "confidence": 0.0,
                "quality_score": 0.0,
                "metadata": {
                    "model": model_name,
                    "temperature": model_config.temperature,
                    "max_tokens": model_config.max_tokens,
                    "finish_reason": finish_reason,
                    "safety_filter_triggered": True,
                },
            }

        # 안전한 텍스트 추출 (has_valid_part가 True이면 안전하게 접근 가능)
        try:
            content = response.text
        except ValueError as e:
            # 예외 발생 시 직접 추출 시도
            logger.warning(f"Gemini API response.text failed: {e}. Trying direct extraction.")
            content = ""
            if hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and candidate.content:
                        if hasattr(candidate.content, "parts"):
                            for part in candidate.content.parts:
                                if hasattr(part, "text"):
                                    content += part.text

            if not content:
                content = "[Unable to extract content from response. This may be due to safety filters or other restrictions.]"

        return {
            "content": content,
            "confidence": 0.8,  # 기본 신뢰도
            "quality_score": 0.8,
            "metadata": {
                "model": model_name,
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_tokens,
                "finish_reason": finish_reason,
            },
        }


    async def _execute_openrouter_model(
        self, model_name: str, prompt: str, system_message: str = None, **kwargs
    ) -> Dict[str, Any]:
        """OpenRouter 모델 실행."""
        model_config = self.models[model_name]

        # OpenRouter에 실제 존재하는 모델 ID 확인 및 변환
        model_id = self._get_valid_openrouter_model_id(model_config.model_id, model_name)

        # 메시지 구성
        history = kwargs.pop("history_messages", [])
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        if history:
            messages.extend(history)

        # If the last message in history is already the user prompt, don't duplicate
        if not history or history[-1].get("content") != prompt:
            messages.append({"role": "user", "content": prompt})

        # OpenRouter API 직접 호출
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mcp-agent.local",
            "X-Title": "MCP Agent Hub",
        }

        payload = {
            "model": model_id,  # 실제 OpenRouter 모델 ID 사용
            "messages": messages,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            **kwargs,
        }

        # 재시도 로직: 502, 500, 503, 429 등 서버 에러와 rate limit은 재시도, 401/403/404는 재시도 안 함
        max_retries = 3
        retryable_status_codes = [
            429,
            500,
            502,
            503,
            504,
        ]  # Rate limit과 서버 에러 재시도
        response = None

        for attempt in range(max_retries):
            try:
                response = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30,
                    ),
                )

                if response.status_code == 200:
                    # 성공
                    break

                # HTML 에러 페이지 필터링
                error_text = response.text
                if "<!DOCTYPE html>" in error_text or "<html" in error_text.lower():
                    # HTML에서 간단한 에러 메시지 추출
                    import re

                    status_code = response.status_code
                    title_match = re.search(r"<title>([^<]+)</title>", error_text, re.IGNORECASE)
                    if title_match:
                        error_msg = f"HTTP {status_code}: {title_match.group(1).strip()}"
                    elif status_code == 502:
                        error_msg = (
                            f"HTTP {status_code}: Bad Gateway - Server temporarily unavailable"
                        )
                    elif status_code == 500:
                        error_msg = f"HTTP {status_code}: Internal Server Error"
                    else:
                        error_msg = f"HTTP {status_code}: Server Error"
                else:
                    error_msg = f"HTTP {response.status_code}: {error_text[:200]}"

                # Rate limit (429) 에러 처리: Provider를 rate-limited로 표시하고 다음 Provider로 전환
                if response.status_code == 429:
                    # OpenRouter Provider를 rate-limited로 표시
                    self._mark_provider_rate_limited("openrouter")
                    logger.warning(
                        "OpenRouter rate-limited (429), will use next provider in rotation"
                    )
                    # Rate limit is not retried; immediately switch to the next provider.
                    raise RuntimeError(f"OpenRouter API rate-limited (429): {error_msg}")

                # 재시도 가능한 에러인지 확인 (429는 이미 처리했으므로 제외)
                if response.status_code in retryable_status_codes and attempt < max_retries - 1:
                    wait_time = 2**attempt  # 지수 백오프: 1초, 2초, 4초
                    logger.warning(
                        f"OpenRouter API error (attempt {attempt + 1}/{max_retries}): {error_msg}, retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue  # 재시도
                else:
                    # 재시도 불가능한 에러 (401, 403 등) 또는 최대 재시도 횟수 초과
                    # 429 (Rate limit), 400 (Invalid model ID), 404 (model doesn't support the
                    # request, e.g. tool use) 에러 발생 시 fallback 모델 시도
                    if response.status_code in (429, 400, 404):
                        logger.warning(
                            f"Model {model_id} returned HTTP {response.status_code} in OpenRouter, trying fallback models..."
                        )
                        fallback_models = self.get_openrouter_fallback_models()

                        for fallback_model in fallback_models:
                            if fallback_model == model_id:
                                continue
                            try:
                                logger.info(f"Trying fallback model: {fallback_model}")
                                payload["model"] = fallback_model
                                fallback_response = await asyncio.get_event_loop().run_in_executor(
                                    None,
                                    lambda: requests.post(
                                        "https://openrouter.ai/api/v1/chat/completions",
                                        headers=headers,
                                        json=payload,
                                        timeout=30,
                                    ),
                                )

                                if fallback_response.status_code == 200:
                                    response = fallback_response
                                    model_id = fallback_model  # 실제 사용된 모델 ID 업데이트
                                    logger.info(
                                        f"✅ Successfully used fallback model: {fallback_model}"
                                    )
                                    break
                            except Exception as fallback_error:
                                logger.debug(
                                    f"Fallback model {fallback_model} failed: {fallback_error}"
                                )
                                continue

                        if response.status_code != 200:
                            logger.error(f"OpenRouter API error: {error_msg}")
                            raise RuntimeError(f"OpenRouter API error: {error_msg}")
                    else:
                        logger.error(f"OpenRouter API error: {error_msg}")
                        raise RuntimeError(f"OpenRouter API error: {error_msg}")

            except requests.exceptions.RequestException as e:
                # 네트워크 에러도 재시도
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"OpenRouter API request failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        "OpenRouter API request failed after %d attempts: %s", max_retries, e
                    )
                    raise

        # 마지막 시도 결과 확인
        if not response or response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API error after {max_retries} attempts: HTTP {response.status_code if response else 'No response'}"
            )

        data = _parse_openrouter_json_response(response, "chat completion")
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls", [])

        return {
            "content": content,
            "confidence": 0.8,
            "quality_score": 0.8,
            "metadata": {
                "model": model_name,
                "provider": "openrouter",
                "model_id": model_config.model_id,
                "tokens_used": len(content.split()) if content else 0,
                "usage": data.get("usage", {}),
                "tool_calls": tool_calls,
            },
        }


    async def _execute_groq_model(
        self, model_name: str, prompt: str, system_message: str = None, **kwargs
    ) -> Dict[str, Any]:
        """Groq 모델 실행."""
        if model_name not in self.model_clients:
            raise ValueError(f"Groq client not initialized for {model_name}")

        client = self.model_clients[model_name]
        model_config = self.models[model_name]

        # Groq에 실제 존재하는 모델 ID 사용
        model_id = self._get_valid_groq_model_id(model_config.model_id)

        # 메시지 구성
        history = kwargs.pop("history_messages", [])
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if history:
            messages.extend(history)
        if not history or (history and history[-1].get("content") != prompt):
            messages.append({"role": "user", "content": prompt})

        max_tokens = kwargs.pop("max_tokens", model_config.max_tokens)

        try:
            # Groq API 호출
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model_id,  # 실제 Groq 모델 ID 사용
                    messages=messages,
                    temperature=model_config.temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ),
            )

            content = response.choices[0].message.content
            tool_calls = getattr(response.choices[0].message, "tool_calls", [])

            return {
                "content": content,
                "confidence": 0.8,
                "quality_score": 0.8,
                "metadata": {
                    "model": model_name,
                    "provider": "groq",
                    "model_id": model_id,  # 실제 사용된 모델 ID (변환된 것일 수 있음)
                    "original_model_id": model_config.model_id,  # 원래 요청한 모델 ID
                    "tokens_used": (
                        response.usage.total_tokens
                        if hasattr(response, "usage")
                        else len(str(content).split())
                    ),
                    "tool_calls": tool_calls,
                },
            }
        except Exception as e:
            error_str = str(e).lower()

            # 모델 존재하지 않음 (404) 또는 Decommissioned 모델 감지 및 자동 대체
            if (
                "does not exist" in error_str
                or "model_not_found" in error_str
                or "decommissioned" in error_str
                or "model_decommissioned" in error_str
            ):
                logger.warning(
                    f"Groq model {model_name} ({model_config.model_id}) is not available"
                )

                # 실제 존재하는 Groq 모델로 대체 시도
                replacement_models = [
                    "openai/gpt-oss-20b",
                    "openai/gpt-oss-120b",
                ]

                for replacement_model in replacement_models:
                    logger.info(f"Attempting to use replacement model: {replacement_model}")
                    try:
                        # 대체 모델로 재시도
                        replacement_response = await asyncio.get_running_loop().run_in_executor(
                            None,
                            lambda rm=replacement_model: (
                                client.chat.completions.create(
                                    model=rm,
                                    messages=messages,
                                    temperature=model_config.temperature,
                                    max_tokens=max_tokens,
                                    **kwargs,
                                )
                            ),
                        )
                        content = replacement_response.choices[0].message.content
                        logger.info(f"✅ Successfully used replacement model: {replacement_model}")

                        # 모델 설정 업데이트 (다음 요청을 위해)
                        self.models[model_name].model_id = replacement_model

                        return {
                            "content": content,
                            "confidence": 0.8,
                            "quality_score": 0.8,
                            "metadata": {
                                "model": model_name,
                                "provider": "groq",
                                "model_id": replacement_model,  # 실제 사용된 모델
                                "original_model_id": model_config.model_id,  # 원래 요청한 모델
                                "tokens_used": (
                                    replacement_response.usage.total_tokens
                                    if hasattr(replacement_response, "usage")
                                    else len(content.split())
                                ),
                            },
                        }
                    except Exception as replacement_error:
                        logger.debug(
                            f"Replacement model {replacement_model} failed: {replacement_error}, trying next..."
                        )
                        continue

                # 모든 대체 모델 실패
                logger.error(
                    f"All replacement models failed for unavailable model {model_config.model_id}"
                )
                raise RuntimeError(
                    f"Groq model {model_name} ({model_config.model_id}) is not available and all replacement models failed"
                )

            logger.error(f"Groq API error: {e}")
            raise RuntimeError(f"Groq model {model_name} failed: {e}")


    async def _execute_cerebras_model(
        self, model_name: str, prompt: str, system_message: str = None, **kwargs
    ) -> Dict[str, Any]:
        """Cerebras 모델 실행 (OpenAI 호환 API, base_url만 다름)."""
        if model_name not in self.model_clients:
            raise ValueError(f"Cerebras client not initialized for {model_name}")

        client = self.model_clients[model_name]
        model_config = self.models[model_name]

        # 메시지 구성
        history = kwargs.pop("history_messages", [])
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if history:
            messages.extend(history)
        if not history or (history and history[-1].get("content") != prompt):
            messages.append({"role": "user", "content": prompt})

        max_tokens = kwargs.pop("max_tokens", model_config.max_tokens)

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model_config.model_id,
                    messages=messages,
                    temperature=model_config.temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ),
            )

            content = response.choices[0].message.content
            tool_calls = getattr(response.choices[0].message, "tool_calls", [])

            return {
                "content": content,
                "confidence": 0.8,
                "quality_score": 0.8,
                "metadata": {
                    "model": model_name,
                    "provider": "cerebras",
                    "model_id": model_config.model_id,
                    "tokens_used": (
                        response.usage.total_tokens
                        if hasattr(response, "usage")
                        else len(str(content).split())
                    ),
                    "tool_calls": tool_calls,
                },
            }
        except Exception as e:
            logger.error(f"Cerebras API error: {e}")
            raise RuntimeError(f"Cerebras model {model_name} failed: {e}")


    async def _execute_openai_model(
        self, model_name: str, prompt: str, system_message: str = None, **kwargs
    ) -> Dict[str, Any]:
        """OpenAI/GPT 모델 실행."""
        if model_name not in self.model_clients:
            raise ValueError(f"OpenAI client not initialized for {model_name}")

        client = self.model_clients[model_name]
        model_config = self.models[model_name]

        # Filter kwargs to only valid OpenAI-compatible API parameters to avoid
        # orchestrator-internal keys (task_type, skip_providers, model, etc.) leaking
        # into client.chat.completions.create() and triggering TypeError.
        _OPENAI_API_KWARGS = {
            "messages", "model", "temperature", "max_tokens", "top_p", "stream",
            "tools", "tool_choice", "stop", "frequency_penalty", "presence_penalty",
        }
        api_kwargs = {k: v for k, v in kwargs.items() if k in _OPENAI_API_KWARGS}
        max_tokens = api_kwargs.pop("max_tokens", model_config.max_tokens)
        # 메시지 구성
        history = kwargs.pop("history_messages", [])
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if history:
            messages.extend(history)
        if not history or (history and history[-1].get("content") != prompt):
            messages.append({"role": "user", "content": prompt})

        try:
            # OpenAI API 호출
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(  # noqa: E501
                    model=model_config.model_id,
                    messages=messages,
                    temperature=model_config.temperature,
                    max_tokens=max_tokens,
                    **api_kwargs,
                ),
            )

            content = response.choices[0].message.content
            tool_calls = getattr(response.choices[0].message, "tool_calls", [])

            # tool_calls are Pydantic model instances (ChatCompletionMessageToolCall),
            # not JSON-serializable dicts. Convert before storing in metadata to avoid
            # TypeError when metadata is serialized to JSON (logging/telemetry/API).
            serialized_tool_calls = []
            if tool_calls:
                for tc in tool_calls:
                    serialized_tool_calls.append({
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(getattr(tc, "function", None), "name", None),
                            "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
                        },
                    })
            return {
                "content": content,
                "confidence": 0.8,
                "quality_score": 0.8,
                "metadata": {
                    "model": model_name,
                    "provider": "openai",
                    "model_id": model_config.model_id,
                    "tokens_used": (
                        response.usage.total_tokens
                        if hasattr(response, "usage")
                        else len(str(content).split())
                    ),
                    "tool_calls": serialized_tool_calls,
                },
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI model {model_name} failed: {e}")


    async def _execute_nvidia_model(
        self, model_name: str, prompt: str, system_message: str = None, **kwargs
    ) -> Dict[str, Any]:
        """NVIDIA NIM 모델 실행."""
        if model_name not in self.model_clients:
            raise ValueError(f"NVIDIA NIM client not initialized for {model_name}")

        client = self.model_clients[model_name]
        model_config = self.models[model_name]

        # 메시지 구성
        history = kwargs.pop("history_messages", [])
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if history:
            messages.extend(history)
        if not history or (history and history[-1].get("content") != prompt):
            messages.append({"role": "user", "content": prompt})

        max_tokens = kwargs.pop("max_tokens", model_config.max_tokens)

        try:
            # NVIDIA API 호출 (OpenAI 라이브러리 연동) — 429는 백오프 후 재시도
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                try:
                    response = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: client.chat.completions.create(
                            model=model_config.model_id,
                            messages=messages,
                            temperature=model_config.temperature,
                            max_tokens=max_tokens,
                            **kwargs,
                        ),
                    )
                    break
                except Exception as retry_e:
                    retry_str = str(retry_e).lower()
                    is_rate_limit = "429" in retry_str or "too many requests" in retry_str
                    if is_rate_limit and attempt < max_retries - 1:
                        wait_time = 10 * (attempt + 1)
                        logger.warning(
                            f"NVIDIA NIM 429, retrying in {wait_time}s "
                            f"({attempt + 1}/{max_retries - 1})"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    raise
            
            if response is None:
                logger.error(f"NVIDIA NIM model {model_name} failed to return a response after {max_retries} attempts.")
                raise RuntimeError(f"NVIDIA NIM model {model_name} failed: No response received.")
                
            content = response.choices[0].message.content
            tool_calls = getattr(response.choices[0].message, "tool_calls", [])

            return {
                "content": content,
                "confidence": 0.8,
                "quality_score": 0.8,
                "metadata": {
                    "model": model_name,
                    "provider": "nvidia",
                    "model_id": model_config.model_id,
                    "tokens_used": (
                        response.usage.total_tokens
                        if hasattr(response, "usage")
                        else len(str(content).split())
                    ),
                    "tool_calls": tool_calls,
                },
            }
        except Exception as e:
            logger.error(f"NVIDIA NIM API error: {e}")
            raise RuntimeError(f"NVIDIA NIM model {model_name} failed: {e}")


    async def _execute_langchain_model(
        self, model_name: str, prompt: str, system_message: str = None, **kwargs
    ) -> Dict[str, Any]:
        """LangChain 모델 실행."""
        client = self.model_clients[model_name]

        # 메시지 구성
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))

        # 실행
        response = await client.ainvoke(messages, config=get_langfuse_run_config())

        return {
            "content": response.content,
            "confidence": 0.8,
            "quality_score": 0.8,
            "metadata": {"model": model_name, "response_type": type(response).__name__},
        }


