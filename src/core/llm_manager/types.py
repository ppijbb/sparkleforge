"""Shared type/config/result dataclasses for the LLM manager package.

Split out of the former monolithic llm_manager.py (issue #582, mirroring the
Sigma-1 split of mcp_integration.py). Kept deliberately tiny and dependency-free
since TaskType/execute_llm_task are imported by ~30 files across the repo --
importing TaskType should not pull in the whole provider/orchestrator stack.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import time


class TaskType(Enum):
    """작업 유형."""

    PLANNING = "planning"
    DEEP_REASONING = "deep_reasoning"
    VERIFICATION = "verification"
    GENERATION = "generation"
    COMPRESSION = "compression"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    CREATIVE = "creative"
    MEMORY_EXTRACTION = "memory_extraction"  # 메모리 추출 (백서 요구사항)
    MEMORY_CONSOLIDATION = "memory_consolidation"  # 메모리 통합 (백서 요구사항)


class Provider(Enum):
    """LLM 제공자."""

    GOOGLE = "google"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    OPENAI = "openai"
    LOCAL = "local"
    NVIDIA = "nvidia"


@dataclass
class ModelConfig:
    """모델 설정."""

    name: str
    provider: str
    model_id: str
    temperature: float
    max_tokens: int
    cost_per_token: float
    speed_rating: float  # 1-10, 높을수록 빠름
    quality_rating: float  # 1-10, 높을수록 품질 좋음
    capabilities: List[TaskType]
    # Provider-side tokens-per-minute (or context window) cap, when known.
    # None = unknown/unbounded -- the cascade won't skip the model on size
    # grounds. Set this for models with a known small org-level TPM limit
    # (e.g. a free tier) so the cascade can skip them for oversized requests
    # instead of hitting a deterministic 413 (#1339).
    context_limit_tokens: Optional[int] = None
    # Wall-clock timestamp (seconds since epoch) at which the learned
    # ``context_limit_tokens`` was last observed from a provider 413 response.
    # Used to re-validate the learned limit after a configurable interval so a
    # transient TPM throttle cannot permanently shrink the model pool (#1349).
    context_limit_learned_at: Optional[float] = None

@dataclass
class ModelResult:
    """모델 실행 결과."""

    content: str
    model_used: str
    execution_time: float
    confidence: float
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)


