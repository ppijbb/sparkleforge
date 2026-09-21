"""OpenRouter vision call for the CLI UX self-audit.

`sparkleforge run`'s pipeline (used by the daily-roadmap prompt) is text-only
stdin/stdout, so it can't carry the rendered screenshots this audit needs --
this calls OpenRouter directly instead, mirroring the auth/endpoint pattern
already used in src/core/llm_manager/providers.py._execute_openrouter_model.
"""

from __future__ import annotations

import os

import requests

DEFAULT_VISION_MODEL = os.getenv("CLI_UX_AUDIT_VISION_MODEL", "google/gemini-2.5-flash")


def call_vision_judge(prompt_text: str, image_data_urls: list[str], *, model: str = DEFAULT_VISION_MODEL, timeout: int = 120) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")

    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mcp-agent.local",
            "X-Title": "SparkleForge CLI UX Audit",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 4096,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
