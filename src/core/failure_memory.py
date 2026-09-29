from collections import deque
from typing import Any, Dict, List

class FailureMemory:
    """Tracks recent tool failures to support anti-stuck-loop reflection and backtracking."""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.failures: deque[Dict[str, Any]] = deque(maxlen=max_history)
        self.consecutive_failures = 0
        self.total_failures = 0

    def record_failure(self, tool_name: str, arguments: Dict[str, Any], error: str) -> None:
        self.failures.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "error": error
        })
        self.consecutive_failures += 1
        self.total_failures += 1

    def record_success(self, tool_name: str) -> None:
        # Reset consecutive failure streak on successful tool call
        self.consecutive_failures = 0

    def reset_streak(self) -> None:
        self.consecutive_failures = 0

    def should_reflect(self) -> bool:
        # After 2 consecutive identical or related tool failures, enter reflection mode
        return self.consecutive_failures >= 2

    def should_rollback(self) -> bool:
        # After 3 consecutive no-progress / failures, perform a working tree rollback
        return self.consecutive_failures >= 3

    def build_reflection_prompt(self) -> str:
        recent = list(self.failures)[-self.consecutive_failures:]
        error_summaries = [f"- Tool `{r['tool_name']}` failed with: {r['error']}" for r in recent]
        
        diagnostic_tools_hint = (
            "Available diagnostic tools for failure triage: `inspect_path`, `list_directory`, "
            "`env_vars`, `list_available_tools`."
        )

        return (
            f"REFLECTION MODE: You have encountered {self.consecutive_failures} consecutive tool failures:\n"
            + "\n".join(error_summaries) + "\n\n"
            + "Diagnose the root cause of these failures. Review the available tools and choose an "
            + "alternative strategy. Do not repeat the exact same failing action.\n"
            + diagnostic_tools_hint
        )
