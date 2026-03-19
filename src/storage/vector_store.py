"""Vector store types for research memory."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class ResearchMemory:
    """Single research result record for storage and retrieval."""

    research_id: str
    user_id: str
    topic: str
    timestamp: datetime
    embedding: List[float]
    metadata: Dict[str, Any]
    results: Dict[str, Any]
    content: str
    summary: str
    keywords: List[str]
    confidence_score: float = 0.0
    source_count: int = 0
    verification_status: str = "unverified"
    similarity_score: float = 0.0  # set when returned from search_similar_research

    def __post_init__(self) -> None:
        if self.embedding is None:
            object.__setattr__(self, "embedding", [])
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        if self.results is None:
            object.__setattr__(self, "results", {})
        if self.keywords is None:
            object.__setattr__(self, "keywords", [])
