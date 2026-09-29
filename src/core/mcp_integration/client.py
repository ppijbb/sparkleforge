"""OpenRouter client wrapper with robust retry logic, error normalization, and typing.

Extracted from the monolithic ``src/core/mcp_integration.py`` as part of
issue #508 — splitting the 7,778-line file by concern. This module owns the
OpenRouter client surface so it can be reviewed independently of the
connection/session and tool-discovery logic that remains in
``mcp_integration.py``.
"""

import asyncio
import logging
import random
import time
import httpx
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """OpenRouter API client with support for retries, error normalization, and robust error handling."""

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = "anthropic/claude-3.5-sonnet",
        max_retries: int = 3,
        timeout: float = 30.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate a response from OpenRouter with retry logic and error normalization."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        payload = {
            "model": model,
            "messages": messages,
        }

        last_error: dict | None = None

        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )

                    if response.status_code == 200:
                        try:
                            data = response.json()
                        except Exception as e:
                            last_error = {
                                "source": "json_decode_error",
                                "detail": str(e),
                                "raw": response.text,
                            }
                        else:
                            if "error" in data:
                                last_error = {"source": "api_error", **data["error"]}
                            else:
                                try:
                                    return data["choices"][0]["message"]["content"]
                                except (KeyError, IndexError, TypeError) as e:
                                    last_error = {
                                        "source": "malformed_success_response",
                                        "detail": str(e),
                                        "raw": data,
                                    }
                    else:
                        last_error = {
                            "source": "http_error",
                            "status": response.status_code,
                            "body": response.text,
                        }
                except httpx.RequestError as e:
                    last_error = {
                        "source": "request_error",
                        "detail": str(e),
                    }

                if attempt < max_retries - 1:
                    sleep_time = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(f"OpenRouter API failed after {max_retries} attempts: {last_error}")
