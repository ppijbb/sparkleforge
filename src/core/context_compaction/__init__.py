"""
Context Compaction Module

컨텍스트 압축 시스템 (AgentPG 패턴 참고).
85% 임계값에서 자동 압축, Hybrid Strategy (Prune + Summarize).
"""

from src.core.context_compaction.manager import (
    CompactionManager,
    CompactionConfig,
    get_compaction_manager,
)
from src.core.context_compaction.strategies import (
    CompactionStrategy,
    PruneStrategy,
    SummarizeStrategy,
    HybridStrategy,
)

__all__ = [
    "CompactionManager",
    "CompactionConfig",
    "get_compaction_manager",
    "CompactionStrategy",
    "PruneStrategy",
    "SummarizeStrategy",
    "HybridStrategy",
]

