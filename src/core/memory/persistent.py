import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class PersistentMemory:
    """Manages persistent memory files (MEMORY.md, USER.md) (Phase 6)."""
    
    def __init__(self, base_path: str = "./"):
        self.base_path = Path(base_path)
        self.memory_file = self.base_path / "MEMORY.md"
        self.user_file = self.base_path / "USER.md"

    def load_memory(self) -> str:
        """Loads project memory."""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def load_user_preferences(self) -> str:
        """Loads user preferences."""
        if self.user_file.exists():
            return self.user_file.read_text(encoding="utf-8")
        return ""

    def update_memory(self, content: str):
        """Updates project memory."""
        self.memory_file.write_text(content, encoding="utf-8")
        logger.info("Updated MEMORY.md")

    def update_user_preferences(self, content: str):
        """Updates user preferences."""
        self.user_file.write_text(content, encoding="utf-8")
        logger.info("Updated USER.md")

    def get_context_block(self) -> str:
        """Returns a formatted block for the prompt."""
        mem = self.load_memory()
        user = self.load_user_preferences()
        
        block = ""
        if mem:
            block += f"\n### Project Memory (MEMORY.md)\n{mem}\n"
        if user:
            block += f"\n### User Preferences (USER.md)\n{user}\n"
            
        return block
