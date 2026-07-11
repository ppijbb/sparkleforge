"""OpenRouter client wrapper (disabled).

Extracted from the monolithic ``src/core/mcp_integration.py`` as part of
issue #494 — splitting the 7,778-line file by concern. This module owns the
OpenRouter client surface so it can be reviewed independently of the
connection/session and tool-discovery logic that remains in
``mcp_integration.py``.

OpenRouter routing is intentionally disabled; the project routes LLM traffic
through ``llm_manager`` (Gemini direct path). This wrapper exists only to
preserve the public import surface for callers that still reference
``OpenRouterClient``.
"""


class OpenRouterClient:
    """(비활성화) OpenRouter 경유는 사용하지 않습니다."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def __aenter__(self):
        raise RuntimeError("OpenRouter is disabled. Use Gemini direct path via llm_manager.")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def generate_response(self, *args, **kwargs):
        raise RuntimeError("OpenRouter is disabled. Use Gemini direct path via llm_manager.")
