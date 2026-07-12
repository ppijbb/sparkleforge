"""Result-shape helpers shared by main.py's CLI dispatch and its handlers.

Extracted from main.py (Anvil Phase Sigma, issue #507) into a leaf module
with zero dependencies so both main.py (which re-exports these for
`from main import cli_result_succeeded, extract_cli_result_content`,
still relied on by tests) and src/cli/main_commands.py can import them
without either importing from the other.
"""
from typing import Any


def extract_cli_result_content(result: Any) -> str:
    """Return the primary human-readable result from supported orchestrator payloads."""
    if not isinstance(result, dict):
        return str(result)

    for key in ("content", "final_report"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value

    for key in ("final_synthesis", "synthesis_results"):
        value = result.get(key)
        if isinstance(value, dict):
            content = value.get("content")
            if isinstance(content, str) and content.strip():
                return content

    return ""


def cli_result_succeeded(result: Any, content: str) -> bool:
    """Infer CLI success for orchestrator payloads that may not expose a success flag."""
    if not isinstance(result, dict):
        return bool(content.strip())
    if result.get("success") is not None:
        return bool(result.get("success"))
    if result.get("error") or result.get("error_message"):
        return False
    return bool(content.strip()) and result.get("current_step") == "completed"
