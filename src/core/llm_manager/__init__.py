"""LLM manager - package facade.

``src/core/llm_manager.py`` used to be a single 3,104-line module (issue
#582, mirroring the Sigma-1 split of ``mcp_integration.py``). It's now this
package, split by responsibility:

- ``types.py``: ``TaskType``, ``Provider``, ``ModelConfig``, ``ModelResult`` --
  kept tiny and dependency-free since ``TaskType``/``execute_llm_task`` are
  imported by ~30 files across the repo.
- ``connection_pool.py``: ``ConnectionPool``.
- ``performance_tracker.py``: ``ModelPerformanceTracker``.
- ``model_registry.py``: ``ModelRegistryMixin`` -- per-provider model catalog
  loading and client initialization (the single biggest chunk of the
  original ``MultiModelOrchestrator``).
- ``routing.py``: ``RoutingMixin`` -- provider rotation, rate-limit
  bookkeeping, ``select_model``.
- ``providers.py``: ``ProviderAdaptersMixin`` -- per-provider execution
  (Gemini/OpenRouter/Groq/Cerebras/OpenAI/NVIDIA/LangChain).
- ``cascade.py``: ``CascadeMixin`` -- cascade classification, draft-quality
  validation, fallback logic.
- ``orchestrator.py``: ``MultiModelOrchestrator`` -- composes the mixins
  above plus ``execute_with_model``/``weighted_ensemble``.
- ``entry.py``: the module-level ``_llm_orchestrator`` singleton,
  ``get_llm_orchestrator``, ``execute_llm_task``, and CLI-agent support.
  Deliberately kept separate from ``orchestrator.py`` so every
  ``get_llm_orchestrator()`` call site across the repo (~3 direct, ~30
  indirect via ``execute_llm_task``) keeps sharing the exact same instance.

Every name that was importable from ``src.core.llm_manager`` before the
split is re-exported here unchanged, so ``from src.core.llm_manager import
X`` keeps working for all existing call sites without modification.
"""

from src.core.llm_manager.cascade import CascadeMixin
from src.core.llm_manager.connection_pool import ConnectionPool
from src.core.llm_manager.entry import (
    _build_active_skills_system_block,
    _execute_cli_agent_task,
    _is_cli_agent,
    execute_llm_task,
    get_best_model_for_task,
    get_llm_orchestrator,
    get_model_performance_stats,
)
from src.core.llm_manager.model_registry import (
    SAFETY_SETTINGS_BLOCK_NONE,
    ModelRegistryMixin,
    _parse_openrouter_json_response,
)
from src.core.llm_manager.orchestrator import MultiModelOrchestrator
from src.core.llm_manager.performance_tracker import ModelPerformanceTracker
from src.core.llm_manager.providers import ProviderAdaptersMixin
from src.core.llm_manager.routing import RoutingMixin
from src.core.llm_manager.types import ModelConfig, ModelResult, Provider, TaskType

__all__ = [
    "CascadeMixin",
    "ConnectionPool",
    "ModelConfig",
    "ModelPerformanceTracker",
    "ModelRegistryMixin",
    "ModelResult",
    "MultiModelOrchestrator",
    "Provider",
    "ProviderAdaptersMixin",
    "RoutingMixin",
    "SAFETY_SETTINGS_BLOCK_NONE",
    "TaskType",
    "_build_active_skills_system_block",
    "_execute_cli_agent_task",
    "_is_cli_agent",
    "_parse_openrouter_json_response",
    "execute_llm_task",
    "get_best_model_for_task",
    "get_llm_orchestrator",
    "get_model_performance_stats",
]
