"""OpenRouter vision call for the CLI UX self-audit.

`sparkleforge run`'s pipeline (used by the daily-roadmap prompt) is text-only
stdin/stdout, so it can't carry the rendered screenshots this audit needs --
this calls OpenRouter directly instead, mirroring the auth/endpoint pattern
already used in src/core/llm_manager/providers.py._execute_openrouter_model.
"""

from __future__ import annotations

import os
import time

import requests

# :free -- the previous default (google/gemini-2.5-flash) is a paid model this
# project's free-tier OpenRouter key can't pay for (see issue #1761); this one
# is NVIDIA + nano-sized + $0 on OpenRouter, in line with the project's own
# lite-model bet rather than reaching for a bigger paid model.
DEFAULT_VISION_MODEL = os.getenv("CLI_UX_AUDIT_VISION_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")


def call_vision_judge(
    prompt_text: str, image_data_urls: list[str], *, model: str = DEFAULT_VISION_MODEL, timeout: int = 120, max_retries: int = 3
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")

    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mcp-agent.local",
        "X-Title": "SparkleForge CLI UX Audit",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4096,
    }

    last_error: object = None
    for attempt in range(max_retries):
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=timeout
        )
        if response.status_code == 200:
            data = response.json()
            # Free-tier upstream capacity errors (e.g. NVIDIA's ResourceExhausted) come
            # back as HTTP 200 with an "error" body instead of a real non-200 status,
            # so raise_for_status() alone won't catch them -- check the body too.
            if "error" not in data:
                return data["choices"][0]["message"]["content"]
            last_error = data["error"]
        else:
            last_error = {"status": response.status_code, "body": response.text[:200]}

        if attempt < max_retries - 1:
            time.sleep(2**attempt)

    raise RuntimeError(f"OpenRouter vision judge failed after {max_retries} attempts: {last_error}")
