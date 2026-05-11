"""Computer Use Agent - Anthropic Computer Use API agentic loop.

Drives the Anthropic claude-sonnet-4-5+ model with the computer-use-2024-10-22
beta feature to accomplish GUI tasks described in natural language.

The agent runs a screenshot→reason→act loop:
1. Capture a screenshot via computer_use_server MCP tools
2. Send image + task description to Anthropic
3. Receive tool_use blocks (mouse_click, type, key, screenshot, etc.)
4. Execute actions via computer_use_server MCP tools
5. Repeat until model returns a text result or max iterations is reached

Prerequisites:
    ANTHROPIC_API_KEY environment variable must be set.
    The computer_use_server must be functional (Xvfb + pyautogui installed).
"""

import asyncio
import json
import logging
import os
from typing import Any

from .base_cli_agent import BaseCLIAgent, CLIAgentConfig, CLIExecutionResult

logger = logging.getLogger(__name__)

# Anthropic Computer Use beta identifier
COMPUTER_USE_BETA = "computer-use-2024-10-22"

# Default model that supports Computer Use
DEFAULT_MODEL = os.getenv("COMPUTER_USE_MODEL", "claude-sonnet-4-5")

# Maximum number of screenshot→action iterations before giving up
MAX_ITERATIONS = int(os.getenv("COMPUTER_USE_MAX_ITERATIONS", "20"))

# Display dimensions (must match computer_use_server.py)
_DISPLAY_WIDTH = int(os.getenv("COMPUTER_USE_WIDTH", "1280"))
_DISPLAY_HEIGHT = int(os.getenv("COMPUTER_USE_HEIGHT", "800"))


class ComputerUseAgent(BaseCLIAgent):
    """Anthropic Computer Use API 기반 GUI 자동화 에이전트.

    자연어로 작성된 GUI 작업을 자율적으로 수행합니다.
    내부적으로 computer_use_server MCP 도구(screenshot, mouse_click, type_text 등)를
    사용하여 가상 디스플레이를 제어합니다.

    Usage:
        agent = ComputerUseAgent()
        result = await agent.execute_query(
            "브라우저를 열고 https://example.com 에 접속한 뒤 페이지 제목을 알려줘"
        )
        print(result["response"])
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_iterations: int = MAX_ITERATIONS,
    ):
        config = CLIAgentConfig(
            name="computer_use",
            command="python3",  # placeholder — not used (SDK called directly)
            timeout=600,
            output_format="json",
        )
        super().__init__(config)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.max_iterations = max_iterations
        self._client = None

    # ------------------------------------------------------------------
    # Anthropic client (lazy init)
    # ------------------------------------------------------------------

    def _get_client(self):
        """Return (or lazily create) the Anthropic SDK client."""
        if self._client is not None:
            return self._client
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
            return self._client
        except ImportError:
            raise RuntimeError("anthropic SDK not installed. Run: pip install anthropic>=0.40.0")

    # ------------------------------------------------------------------
    # computer_use_server helpers (direct Python import for efficiency)
    # ------------------------------------------------------------------

    async def _capture_screenshot(self) -> str:
        """Capture screenshot via computer_use_server. Returns base64 PNG string."""
        from src.core.mcp_servers.computer_use_server import (
            ScreenshotInput,
            screenshot,
        )

        result_json = await screenshot(ScreenshotInput(format="png"))
        result = json.loads(result_json)
        if not result.get("success"):
            raise RuntimeError(f"Screenshot failed: {result.get('error')}")
        return result["data"]  # base64 PNG

    async def _execute_action(self, action: str, action_input: dict[str, Any]) -> str:
        """Dispatch an Anthropic computer_use tool action to the MCP server.

        Maps Anthropic Computer Use action types to computer_use_server tools.
        Returns the raw JSON string from the tool.
        """
        import src.core.mcp_servers.computer_use_server as cu

        # --- screenshot ---
        if action == "screenshot":
            return await cu.screenshot(cu.ScreenshotInput())

        # --- mouse ---
        elif action == "mouse_move":
            x, y = action_input["coordinate"]
            return await cu.mouse_move(cu.MouseMoveInput(x=x, y=y))

        elif action == "left_click":
            x, y = action_input["coordinate"]
            return await cu.mouse_click(cu.MouseClickInput(x=x, y=y, button="left"))

        elif action == "right_click":
            x, y = action_input["coordinate"]
            return await cu.mouse_click(cu.MouseClickInput(x=x, y=y, button="right"))

        elif action == "middle_click":
            x, y = action_input["coordinate"]
            return await cu.mouse_click(cu.MouseClickInput(x=x, y=y, button="middle"))

        elif action == "double_click":
            x, y = action_input["coordinate"]
            return await cu.mouse_click(cu.MouseClickInput(x=x, y=y, clicks=2))

        elif action == "left_click_drag":
            # pyautogui drag: move to start, then drag to end
            start = action_input.get("start_coordinate", [0, 0])
            end = action_input["coordinate"]
            await cu.mouse_move(cu.MouseMoveInput(x=start[0], y=start[1]))
            pg = cu._get_pyautogui()
            pg.dragTo(end[0], end[1], duration=0.3, button="left")
            return json.dumps({"success": True, "action": "left_click_drag"})

        elif action == "scroll":
            x, y = action_input["coordinate"]
            direction = action_input.get("direction", "down")
            amount = int(action_input.get("amount", 3))
            clicks = amount if direction == "up" else -amount
            return await cu.mouse_scroll(cu.MouseScrollInput(x=x, y=y, clicks=clicks))

        # --- keyboard ---
        elif action == "type":
            text = action_input.get("text", "")
            return await cu.type_text(cu.TypeTextInput(text=text))

        elif action == "key":
            key = action_input.get("key", "")
            return await cu.key_press(cu.KeyPressInput(key=key))

        # --- cursor position (read-only) ---
        elif action == "cursor_position":
            pg = cu._get_pyautogui()
            pos = pg.position()
            return json.dumps({"success": True, "x": pos.x, "y": pos.y})

        else:
            logger.warning("Unknown computer_use action: %s", action)
            return json.dumps({"success": False, "error": f"Unknown action: {action}"})

    # ------------------------------------------------------------------
    # Main agentic loop
    # ------------------------------------------------------------------

    async def execute_query(self, query: str, **kwargs) -> dict[str, Any]:
        """Execute a GUI task using the Anthropic Computer Use agentic loop.

        Args:
            query: Natural language description of the GUI task to perform.
            **kwargs: Optional overrides:
                - model (str): Override the model
                - max_iterations (int): Override max loop iterations
                - system (str): Additional system prompt context

        Returns:
            dict with keys:
                success (bool), response (str), iterations (int),
                error (str | None)
        """
        model = kwargs.get("model", self.model)
        max_iter = int(kwargs.get("max_iterations", self.max_iterations))
        system_extra = kwargs.get("system", "")

        try:
            client = self._get_client()
        except RuntimeError as exc:
            return {"success": False, "error": str(exc), "response": "", "iterations": 0}

        # Capture initial screenshot
        try:
            initial_screenshot = await self._capture_screenshot()
        except Exception as exc:
            return {
                "success": False,
                "error": f"Initial screenshot failed: {exc}",
                "response": "",
                "iterations": 0,
            }

        # Build the initial message (image + task)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": initial_screenshot,
                        },
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]

        # Computer Use tool definition
        tools = [
            {
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": _DISPLAY_WIDTH,
                "display_height_px": _DISPLAY_HEIGHT,
                "display_number": 99,
            }
        ]

        system_prompt = (
            "You are an AI assistant controlling a virtual Linux desktop. "
            "Use the computer tool to interact with the screen. "
            "After completing the task, describe what you accomplished."
        )
        if system_extra:
            system_prompt += f"\n\n{system_extra}"

        # Agentic loop
        for iteration in range(max_iter):
            logger.debug("Computer Use iteration %d/%d", iteration + 1, max_iter)

            try:
                response = client.beta.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=tools,
                    messages=messages,
                    betas=[COMPUTER_USE_BETA],
                )
            except Exception as exc:
                logger.error("Anthropic API call failed: %s", exc)
                return {
                    "success": False,
                    "error": str(exc),
                    "response": "",
                    "iterations": iteration,
                }

            # If the model finished, extract the final text response
            if response.stop_reason == "end_turn":
                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text
                return {
                    "success": True,
                    "response": text,
                    "iterations": iteration + 1,
                    "error": None,
                }

            # Process tool_use blocks
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name != "computer":
                    continue

                action = block.input.get("action", "")
                logger.debug("Executing computer action: %s", action)

                try:
                    action_result_json = await self._execute_action(action, block.input)
                    action_result = json.loads(action_result_json)
                except Exception as exc:
                    action_result = {"success": False, "error": str(exc)}
                    action_result_json = json.dumps(action_result)

                # For screenshot actions, include the new image in the tool result
                if action == "screenshot" and action_result.get("success"):
                    image_data = action_result.get("data", "")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_data,
                                    },
                                }
                            ],
                        }
                    )
                else:
                    # For other actions, return a text confirmation
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": [
                                {
                                    "type": "text",
                                    "text": action_result_json,
                                }
                            ],
                        }
                    )

                # After non-screenshot actions, capture a new screenshot automatically
                # so the model can see the result of its action
                if action != "screenshot" and action_result.get("success"):
                    try:
                        await asyncio.sleep(0.3)  # brief pause for UI to update
                        new_screenshot = await self._capture_screenshot()
                        # Append screenshot as a follow-up tool result image
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id + "_followup",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": new_screenshot,
                                        },
                                    }
                                ],
                            }
                        )
                    except Exception as exc:
                        logger.debug("Follow-up screenshot failed: %s", exc)

            if not tool_results:
                # No tool calls but also not end_turn — extract any text and stop
                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text
                return {
                    "success": True,
                    "response": text or "(no output)",
                    "iterations": iteration + 1,
                    "error": None,
                }

            # Extend the conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        # Exceeded max iterations
        return {
            "success": False,
            "error": f"Exceeded maximum iterations ({max_iter})",
            "response": "",
            "iterations": max_iter,
        }

    # ------------------------------------------------------------------
    # BaseCLIAgent required methods
    # ------------------------------------------------------------------

    def parse_output(self, result: CLIExecutionResult) -> dict[str, Any]:
        """Not used directly — execute_query handles the full loop."""
        return {"success": result.success, "response": result.output}

    async def health_check(self) -> bool:
        """Check that ANTHROPIC_API_KEY is set and computer_use_server is importable."""
        if not self.api_key:
            logger.warning("ComputerUseAgent: ANTHROPIC_API_KEY not set")
            return False
        try:
            import src.core.mcp_servers.computer_use_server  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> dict[str, Any]:
        """Return agent metadata."""
        return {
            "name": "computer_use",
            "type": "cli_agent",
            "model": self.model,
            "max_iterations": self.max_iterations,
            "display_width": _DISPLAY_WIDTH,
            "display_height": _DISPLAY_HEIGHT,
            "api_key_set": bool(self.api_key),
            "description": (
                "Anthropic Computer Use API agent for controlling a virtual desktop. "
                "Requires ANTHROPIC_API_KEY and Xvfb + pyautogui."
            ),
        }
