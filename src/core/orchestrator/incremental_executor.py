import asyncio
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set

class ParserState(Enum):
    TEXT = 1
    IN_TOOL_TAG = 2
    IN_TOOL_BODY = 3

class IncrementalToolParser:
    def __init__(self):
        self.buffer = ""
        self.state = ParserState.TEXT
        self.current_tool_call: Optional[Dict[str, Any]] = None

    def feed_chunk(self, chunk: str) -> List[Dict[str, Any]]:
        self.buffer += chunk
        results = []
        
        while True:
            if self.state == ParserState.TEXT:
                start_idx = self.buffer.find("<tool_call")
                if start_idx != -1:
                    self.buffer = self.buffer[start_idx:]
                    self.state = ParserState.IN_TOOL_TAG
                else:
                    break
            
            if self.state == ParserState.IN_TOOL_TAG:
                end_tag_idx = self.buffer.find(">")
                if end_tag_idx != -1:
                    tag_content = self.buffer[10:end_tag_idx].strip()
                    self.current_tool_call = {"name": tag_content, "args": ""}
                    self.buffer = self.buffer[end_tag_idx + 1:]
                    self.state = ParserState.IN_TOOL_BODY
                else:
                    break
            
            if self.state == ParserState.IN_TOOL_BODY:
                close_idx = self.buffer.find("</tool_call>")
                if close_idx != -1:
                    self.current_tool_call["args"] = self.buffer[:close_idx].strip()
                    try:
                        self.current_tool_call["args"] = json.loads(self.current_tool_call["args"])
                    except:
                        pass
                    results.append(self.current_tool_call)
                    self.buffer = self.buffer[close_idx + 12:]
                    self.current_tool_call = None
                    self.state = ParserState.TEXT
                else:
                    break
        return results

class SpeculativeToolExecutor:
    def __init__(self, execution_registry):
        self.registry = execution_registry
        self.pending_tasks: Set[asyncio.Task] = set()

    async def _run_tool(self, tool_call: Dict[str, Any]):
        try:
            # Logic to invoke tool via registry
            pass
        except Exception as e:
            pass

    def dispatch(self, tool_call: Dict[str, Any]):
        task = asyncio.create_task(self._run_tool(tool_call))
        self.pending_tasks.add(task)
        task.add_done_callback(self.pending_tasks.discard)

    async def wait_all(self):
        if self.pending_tasks:
            await asyncio.gather(*self.pending_tasks, return_exceptions=True)
            self.pending_tasks.clear()
