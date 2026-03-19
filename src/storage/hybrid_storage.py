"""Hybrid storage for research memory: in-memory + optional file persistence."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import UTC, datetime

from src.storage.vector_store import ResearchMemory

logger = logging.getLogger(__name__)


class HybridStorage:
    """Stores and retrieves research results with optional similarity search."""

    def __init__(self, storage_path: str | None = None) -> None:
        self._storage_path = Path(
            storage_path or "./storage/research_memory"
        )
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._memory: List[ResearchMemory] = []
        self._index_path = self._storage_path / "index.jsonl"
        self._load_index()

    def _load_index(self) -> None:
        """Load persisted records from disk."""
        if not self._index_path.exists():
            return
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        ts = data.get("timestamp")
                        if isinstance(ts, str):
                            data["timestamp"] = datetime.fromisoformat(
                                ts.replace("Z", "+00:00")
                            )
                        self._memory.append(ResearchMemory(**data))
                    except Exception as e:
                        logger.debug("Skip invalid index line: %s", e)
        except Exception as e:
            logger.warning("Could not load research index: %s", e)

    def _persist_one(self, m: ResearchMemory) -> None:
        """Append one record to index file."""
        try:
            data = {
                "research_id": m.research_id,
                "user_id": m.user_id,
                "topic": m.topic,
                "timestamp": m.timestamp.isoformat(),
                "embedding": m.embedding,
                "metadata": m.metadata,
                "results": m.results,
                "content": m.content,
                "summary": m.summary,
                "keywords": m.keywords,
                "confidence_score": m.confidence_score,
                "source_count": m.source_count,
                "verification_status": m.verification_status,
            }
            with open(self._index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Could not persist research memory: %s", e)

    async def store_research(
        self,
        research_id: str,
        user_id: str,
        topic: str,
        content: str,
        results: Dict[str, Any],
        metadata: Dict[str, Any],
        summary: str = "",
        keywords: List[str] | None = None,
    ) -> bool:
        """Store one research result."""
        try:
            keywords = keywords or []
            memory = ResearchMemory(
                research_id=research_id,
                user_id=user_id,
                topic=topic,
                timestamp=datetime.now(UTC),
                embedding=[],
                metadata=dict(metadata or {}, topic=topic),
                results=results or {},
                content=content or "",
                summary=summary or "",
                keywords=keywords,
                confidence_score=float(
                    (results or {}).get("confidence", 0.0)
                ),
                source_count=len(
                    (metadata or {}).get("execution_results", [])
                ),
                verification_status=(
                    (metadata or {}).get("verification_status")
                    or (results or {}).get("verification_status", "unverified")
                ),
            )
            self._memory.append(memory)
            self._persist_one(memory)
            return True
        except Exception as e:
            logger.error("store_research failed: %s", e)
            return False

    async def search_similar_research(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.3,
    ) -> List[ResearchMemory]:
        """Return research items similar to query (keyword/topic overlap)."""
        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))
        if not query_words:
            query_words = {query_lower}

        scored: List[Tuple[float, ResearchMemory]] = []
        for m in self._memory:
            if m.user_id != user_id:
                continue
            topic_lower = (m.topic or "").lower()
            summary_lower = (m.summary or "").lower()
            text_words = set(re.findall(r"\w+", topic_lower + " " + summary_lower))
            if not text_words:
                continue
            overlap = len(query_words & text_words) / max(
                len(query_words | text_words), 1
            )
            if overlap >= similarity_threshold:
                scored.append((overlap, m))

        scored.sort(key=lambda x: (-x[0], -x[1].timestamp.timestamp()))
        out: List[ResearchMemory] = []
        for sim, m in scored[:limit]:
            m.similarity_score = sim
            out.append(m)
        return out

    async def get_user_research_history(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[ResearchMemory]:
        """Return most recent research for user."""
        by_user = [m for m in self._memory if m.user_id == user_id]
        by_user.sort(key=lambda m: m.timestamp.timestamp(), reverse=True)
        return by_user[:limit]
