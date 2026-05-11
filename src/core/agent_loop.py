import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from src.core.llm_manager import MultiModelOrchestrator, TaskType, ModelResult
from src.core.mcp_integration import get_mcp_hub, UniversalMCPHub

logger = logging.getLogger(__name__)

@dataclass
class IterationBudget:
    """Hermes-style iteration budget tracker."""
    max_iterations: int = 90
    current_iteration: int = 0
    start_time: float = field(default_factory=time.time)

    def consume(self):
        self.current_iteration += 1
        if self.current_iteration > self.max_iterations:
            raise RuntimeError(f"Iteration budget exceeded: {self.current_iteration}/{self.max_iterations}")

    @property
    def remaining(self):
        return self.max_iterations - self.current_iteration

class AgentLoop:
    """SparkleForge Autonomous Agent Loop (Phase 1).
    
    This class implements the 'Tool Calling Until Completion' pattern from Hermes.
    """
    
    def __init__(self, orchestrator: Optional[MultiModelOrchestrator] = None):
        from src.core.llm_manager import MultiModelOrchestrator
        self.orchestrator = orchestrator or MultiModelOrchestrator()
        self.mcp_hub: UniversalMCPHub = get_mcp_hub()
        
        # Phase 3: Context Compression
        from src.core.context_compressor import ContextCompressor
        self.compressor = ContextCompressor(self.orchestrator)
        
        # Phase 6: Persistent Memory
        from src.core.memory.persistent import PersistentMemory
        self.memory = PersistentMemory()
        
    async def run_conversation(
        self, 
        messages: List[Dict[str, Any]], 
        task_type: TaskType = TaskType.RESEARCH,
        max_iterations: int = 20,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs the autonomous loop."""
        from src.core.error_classifier import ErrorClassifier, ErrorCategory
        
        budget = IterationBudget(max_iterations=max_iterations)
        
        # Ensure MCP is initialized
        await self.mcp_hub.initialize_mcp()
        
        # Get tool schemas in OpenAI format
        tools = self._get_openai_tools()
        
        # Phase 6: Load persistent memory
        mem_context = self.memory.get_context_block()
        if mem_context:
            system_message = (system_message or "") + "\n\n" + mem_context
        
        # Prepare history
        history = list(messages)
        retry_count = 0
        max_retries = 3
        
        while budget.remaining > 0:
            budget.consume()
            logger.info(f"[AgentLoop] Iteration {budget.current_iteration}/{max_iterations}")
            
            # Phase 3: Compress context if needed
            history = await self.compressor.compress_if_needed(history)
            
            # Step 1: Call LLM with Resilience (Phase 4)
            try:
                last_msg = history[-1]["content"] if history else ""
                prev_msgs = history[:-1] if len(history) > 1 else []
                
                result: ModelResult = await self.orchestrator.execute_with_model(
                    prompt=last_msg,
                    task_type=task_type,
                    system_message=system_message,
                    history_messages=prev_msgs, 
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None,
                )
                
                # Success - reset retry count
                retry_count = 0
                
            except asyncio.CancelledError:
                try:
                    await self._save_executions()
                except Exception as e:
                    logger.error(f"Failed to save executions during cancellation: {e}")
                raise

            except Exception as e:
                category = ErrorClassifier.classify(e)
                logger.warning(f"[AgentLoop] Error detected: {category.value} - {e}")
                
                if category == ErrorCategory.RETRYABLE and retry_count < max_retries:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.info(f"Retrying in {wait_time}s... ({retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    budget.current_iteration -= 1 # Don't count retry as iteration
                    continue
                elif category == ErrorCategory.CONTEXT_LIMIT:
                    # Force compression and retry
                    logger.info("Context limit hit. Forcing EXTREME compression.")
                    history = await self.compressor.compress_by_summarization(history)
                    continue
                else:
                    # Fatal or max retries
                    return {
                        "content": f"Execution failed due to {category.value}: {str(e)}",
                        "error": str(e),
                        "iterations": budget.current_iteration,
                        "history": history
                    }
            
            content = result.content
            tool_calls = result.metadata.get("tool_calls", [])
            
            # Add Assistant response to history
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            history.append(assistant_msg)
            
            if not tool_calls:
                # No tool calls, we are done
                return {
                    "content": content,
                    "metadata": result.metadata,
                    "iterations": budget.current_iteration,
                    "history": history
                }
            
            # Step 3: Execute tools
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name")
                arguments_str = tool_call.get("function", {}).get("arguments", "{}")
                try:
                    arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                except json.JSONDecodeError:
                    arguments = {}
                
                logger.info(f"[AgentLoop] Executing tool: {tool_name}")
                try:
                    tool_exec_result = await self.mcp_hub.execute_tool(tool_name, arguments)
                except Exception as e:
                    logger.error(f"Tool execution failed: {tool_name} - {e}")
                    tool_exec_result = {"success": False, "error": str(e)}
                
                # Phase 3: Prune large tool outputs
                tool_result_str = json.dumps(tool_exec_result, ensure_ascii=False)
                pruned_result = self.compressor.prune_tool_output(tool_result_str)
                
                # Step 4: Add tool result to history
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": tool_name,
                    "content": pruned_result
                })
        
        return {
            "content": "Iteration budget exceeded.",
            "error": "Max iterations reached",
            "iterations": budget.current_iteration,
            "history": history
        }

    def _get_openai_tools(self) -> List[Dict[str, Any]]:
        """Converts MCP tools to OpenAI tool format."""
        openai_tools = []
        # Accessing tools from registry
        for name, info in self.mcp_hub.registry.tools.items():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info.description,
                    "parameters": info.parameters or {"type": "object", "properties": {}}
                }
            })
        return openai_tools
