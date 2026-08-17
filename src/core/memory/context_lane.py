import ast
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.core.scheduler import get_scheduler
from src.core.memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)

_REPO_ROOT = str(Path(__file__).resolve().parents[3])


class ContextLane:
    """Consolidates conversational context, system executions logs, and AST-parsed codebase relations."""

    def __init__(
        self,
        memory: Optional[SemanticMemory] = None,
        project_root: str = _REPO_ROOT
    ):
        self.memory = memory or SemanticMemory()
        self.project_root = os.path.abspath(project_root)
        self.scheduler = get_scheduler()
        self._codebase_map: Dict[str, List[str]] = {}

    def update_codebase_map(self) -> Dict[str, List[str]]:
        """Scan src/ folder, parse file imports using AST, and build a dependency relationship map."""
        src_path = os.path.join(self.project_root, "src")
        logger.info(f"ContextLane: Scanning AST code map at: {src_path}")
        
        dependency_map = {}
        if not os.path.exists(src_path):
            logger.warning(f"ContextLane: src folder not found at {src_path}")
            return {}

        try:
            for root, _, files in os.walk(src_path):
                for file in files:
                    if not file.endswith(".py"):
                        continue
                        
                    full_path = os.path.join(root, file)
                    # Convert to dot-notation module path relative to workspace
                    rel_path = os.path.relpath(full_path, self.project_root)
                    module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")
                    
                    imports = self._parse_file_imports(full_path)
                    dependency_map[module_name] = imports
                    
            self._codebase_map = dependency_map
            logger.info(f"ContextLane: Mapped {len(dependency_map)} modules inside codebase")
        except Exception as e:
            logger.error(f"ContextLane: Codebase mapping failed: {e}")

        return dependency_map

    def _parse_file_imports(self, file_path: str) -> List[str]:
        """Extract import module names from a Python source file using Abstract Syntax Trees."""
        imports = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=file_path)
                
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except Exception as e:
            logger.debug(f"ContextLane: AST parse error for {file_path}: {e}")
            
        # Clean external imports, keeping local project imports (starts with src)
        local_imports = [imp for imp in imports if imp.startswith("src")]
        return list(set(local_imports))

    def get_active_context(self, session_id: str, search_query: Optional[str] = None) -> Dict[str, Any]:
        """Query conversational histories, execution audit trails, and code mappings into a single context payload."""
        # 1. Fetch related session memories
        query = search_query or f"session_log_{session_id}"
        related_memories = self.memory.search_memory(query, limit=5)
        
        # 2. Query execution logs
        executions = []
        try:
            exec_history = self.scheduler.get_execution_history(limit=5)
            executions = [
                {
                    "execution_id": e.execution_id,
                    "schedule_id": e.schedule_id,
                    "started_at": e.started_at.isoformat() if e.started_at else None,
                    "status": e.status,
                    "error": e.error
                }
                for e in exec_history
            ]
        except Exception as e:
            logger.warning(f"ContextLane: Failed to retrieve execution history: {e}")

        # 3. Dynamic dependency mapping fallback
        if not self._codebase_map:
            self.update_codebase_map()

        return {
            "session_id": session_id,
            "relevant_memories": [
                {"text": m["text"], "score": m["score"], "metadata": m["metadata"]}
                for m in related_memories
            ],
            "execution_history": executions,
            "codebase_map": self._codebase_map
        }
