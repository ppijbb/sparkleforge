"""Skill Tree - Per-agent hierarchical skill management and retrieval.

Provides SkillTreeNode/SkillTree, SkillPerformanceTracker, HotSkillCache,
and SkillRetriever (BM25 + Vector similarity + FlashRank reranking) for task-specific skill selection.
"""

import json
import logging
import math
import re
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.core.skills_manager import SkillManager

logger = logging.getLogger(__name__)


# Lazy import to avoid circular dependency; SkillMatch used for return type
def _get_skill_match_type():
    from src.core.skills_selector import SkillMatch

    return SkillMatch


@dataclass
class SkillPerformanceMetrics:
    """Per-skill quantitative performance metrics."""

    skill_id: str
    total_uses: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    last_used_at: str = ""
    hot_score: float = 0.0

    MAX_QUALITY_HISTORY = 50

    def record_use(
        self, success: bool, latency_ms: float = 0.0, quality_score: float = 1.0
    ) -> None:
        """Record one skill use and update EMA-style metrics."""
        self.total_uses += 1
        if success:
            self.success_count += 1
        self.success_rate = self.success_count / self.total_uses
        if latency_ms >= 0:
            self.avg_latency_ms = (
                self.avg_latency_ms * (self.total_uses - 1) + latency_ms
            ) / self.total_uses
        self.quality_scores.append(quality_score)
        if len(self.quality_scores) > self.MAX_QUALITY_HISTORY:
            self.quality_scores = self.quality_scores[-self.MAX_QUALITY_HISTORY :]
        self.last_used_at = datetime.now(UTC).isoformat()


@dataclass
class SkillTreeNode:
    """Single node in the skill tree (category or leaf skill)."""

    node_id: str
    skill_id: str | None = None
    category: str = ""
    category_path: List[str] = field(default_factory=list)
    children: Dict[str, "SkillTreeNode"] = field(default_factory=dict)
    performance: SkillPerformanceMetrics | None = None
    depth: int = 0

    def is_leaf(self) -> bool:
        return self.skill_id is not None

    def all_skill_ids(self) -> List[str]:
        if self.is_leaf():
            return [self.skill_id] if self.skill_id else []
        ids = []
        for child in self.children.values():
            ids.extend(child.all_skill_ids())
        return ids


class SkillTree:
    """Per-agent hierarchical skill tree with performance-aware access."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._root: Dict[str, SkillTreeNode] = {}
        self._skill_to_node: Dict[str, SkillTreeNode] = {}
        self._performance: Dict[str, SkillPerformanceMetrics] = {}

    def add_skill(self, skill_id: str, category_path: List[str] | None = None) -> None:
        if not category_path:
            category_path = ["general"]
        key = "/".join(category_path)
        node_id = f"{key}/{skill_id}"
        if skill_id in self._skill_to_node:
            return
        perf = self._performance.get(skill_id)
        if perf is None:
            perf = SkillPerformanceMetrics(skill_id=skill_id)
            self._performance[skill_id] = perf
        node = SkillTreeNode(
            node_id=node_id,
            skill_id=skill_id,
            category=category_path[-1] if category_path else "general",
            category_path=list(category_path),
            children={},
            performance=perf,
            depth=len(category_path),
        )
        self._skill_to_node[skill_id] = node
        current = self._root
        for part in category_path[:-1]:
            if part not in current:
                current[part] = SkillTreeNode(
                    node_id=part,
                    skill_id=None,
                    category=part,
                    children={},
                    performance=None,
                    depth=len(current),
                )
            current = current[part].children
        leaf_key = category_path[-1]
        if leaf_key not in current:
            current[leaf_key] = {}
        current[leaf_key][skill_id] = node

    def remove_skill(self, skill_id: str) -> bool:
        if skill_id not in self._skill_to_node:
            return False
        del self._skill_to_node[skill_id]
        self._performance.pop(skill_id, None)
        self._remove_skill_from_tree(self._root, skill_id)
        return True

    def _remove_skill_from_tree(self, node: Any, skill_id: str) -> bool:
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if isinstance(v, SkillTreeNode) and v.skill_id == skill_id:
                    del node[k]
                    return True
                if self._remove_skill_from_tree(v, skill_id):
                    return True
        return False

    def get_skills_by_category(self, category: str) -> List[str]:
        ids = []
        for node in self._skill_to_node.values():
            if node.category == category:
                ids.append(node.skill_id)
        return [s for s in ids if s]

    def _hot_score(self, perf: SkillPerformanceMetrics | None) -> float:
        if not perf:
            return 0.0
        if perf.hot_score > 0:
            return perf.hot_score
        usage = math.log(1 + perf.total_uses) / 10.0
        return perf.success_rate * 0.6 + min(usage, 0.4)

    def get_hot_skills(self, top_k: int = 10) -> List[str]:
        with_perf = [
            (sid, self._hot_score(node.performance))
            for sid, node in self._skill_to_node.items()
            if node.skill_id
        ]
        with_perf.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in with_perf[:top_k]]

    def update_performance(
        self, skill_id: str, success: bool, latency_ms: float = 0.0, quality: float = 1.0
    ) -> None:
        if skill_id not in self._performance:
            self._performance[skill_id] = SkillPerformanceMetrics(skill_id=skill_id)
        self._performance[skill_id].record_use(success, latency_ms, quality)
        self._performance[skill_id].hot_score = self._hot_score(self._performance[skill_id])
        node = self._skill_to_node.get(skill_id)
        if node and node.performance:
            node.performance = self._performance[skill_id]

    def prune_low_performers(self, threshold: float = 0.3, min_uses: int = 3) -> int:
        to_remove = [
            sid
            for sid, perf in self._performance.items()
            if perf.total_uses >= min_uses and perf.success_rate < threshold
        ]
        for sid in to_remove:
            self.remove_skill(sid)
        return len(to_remove)

    def serialize(self) -> Dict[str, Any]:
        skills_list = []
        for sid, node in self._skill_to_node.items():
            if not node.skill_id:
                continue
            cat_path = node.category_path if node.category_path else [node.category]
            skills_list.append({"skill_id": sid, "category_path": cat_path})
        perf_list = [asdict(p) for p in self._performance.values()]
        return {
            "agent_id": self.agent_id,
            "skills": skills_list,
            "performance": perf_list,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "SkillTree":
        tree = cls(data.get("agent_id", ""))
        for p in data.get("performance", []):
            sid = p.get("skill_id", "")
            if sid:
                tree._performance[sid] = SkillPerformanceMetrics(
                    skill_id=sid,
                    total_uses=p.get("total_uses", 0),
                    success_count=p.get("success_count", 0),
                    success_rate=p.get("success_rate", 0.0),
                    avg_latency_ms=p.get("avg_latency_ms", 0.0),
                    quality_scores=p.get("quality_scores", []),
                    last_used_at=p.get("last_used_at", ""),
                    hot_score=p.get("hot_score", 0.0),
                )
        for item in data.get("skills", []):
            sid = item.get("skill_id", "")
            cat_path = item.get("category_path", ["general"])
            if sid:
                tree.add_skill(sid, cat_path)
                node = tree._skill_to_node.get(sid)
                if node and sid in tree._performance:
                    node.performance = tree._performance[sid]
        return tree


class SkillPerformanceTracker:
    """Global per-skill performance tracking with hot_score and persistence."""

    ALPHA = 0.5
    BETA = 0.3
    GAMMA = 0.2
    RECENCY_HALFLIFE_HOURS = 24.0

    def __init__(self, store_path: Path | None = None) -> None:
        if store_path is None:
            root = Path(__file__).resolve().parent.parent.parent
            store_path = root / ".sparkleforge" / "skill_performance.json"
        self.store_path = Path(store_path)
        self._metrics: Dict[str, SkillPerformanceMetrics] = {}
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            with open(self.store_path, encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("skills", []):
                sid = item.get("skill_id", "")
                if sid:
                    self._metrics[sid] = SkillPerformanceMetrics(
                        skill_id=sid,
                        total_uses=item.get("total_uses", 0),
                        success_count=item.get("success_count", 0),
                        success_rate=item.get("success_rate", 0.0),
                        avg_latency_ms=item.get("avg_latency_ms", 0.0),
                        quality_scores=item.get("quality_scores", []),
                        last_used_at=item.get("last_used_at", ""),
                        hot_score=item.get("hot_score", 0.0),
                    )
        except Exception as e:
            logger.warning("SkillPerformanceTracker load failed: %s", e)

    def save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            skills = [asdict(m) for m in self._metrics.values()]
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"skills": skills, "updated_at": datetime.now(UTC).isoformat()},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            logger.warning("SkillPerformanceTracker save failed: %s", e)

    def record(
        self,
        skill_id: str,
        success: bool,
        latency_ms: float = 0.0,
        quality_score: float = 1.0,
    ) -> None:
        if skill_id not in self._metrics:
            self._metrics[skill_id] = SkillPerformanceMetrics(skill_id=skill_id)
        self._metrics[skill_id].record_use(success, latency_ms, quality_score)
        self._metrics[skill_id].hot_score = self.compute_hot_score(skill_id)
        self.save()

    def compute_hot_score(self, skill_id: str) -> float:
        m = self._metrics.get(skill_id)
        if not m:
            return 0.0
        recency = 0.0
        if m.last_used_at:
            try:
                dt = datetime.fromisoformat(m.last_used_at.replace("Z", "+00:00"))
                age_hours = (datetime.now(UTC) - dt).total_seconds() / 3600.0
                recency = math.exp(-age_hours * (math.log(2) / self.RECENCY_HALFLIFE_HOURS))
            except Exception:
                recency = 0.5
        usage_term = math.log(1 + m.total_uses) / math.log(1 + max(m.total_uses, 1))
        return self.ALPHA * m.success_rate + self.BETA * usage_term + self.GAMMA * recency

    def get_top_skills(self, category: str | None = None, top_k: int = 10) -> List[str]:
        for m in self._metrics.values():
            m.hot_score = self.compute_hot_score(m.skill_id)
        items = list(self._metrics.items())
        if category:
            items = [
                (sid, m)
                for sid, m in items
                if category.lower() in (m.skill_id + str(getattr(m, "category", ""))).lower()
            ]
        items.sort(key=lambda x: x[1].hot_score, reverse=True)
        return [sid for sid, _ in items[:top_k]]

    def get_metrics(self, skill_id: str) -> SkillPerformanceMetrics | None:
        return self._metrics.get(skill_id)


class HotSkillCache:
    """Pre-loaded cache of high-performing skills (LRU + hot_score)."""

    def __init__(
        self,
        skill_manager: "SkillManager",
        max_size: int = 50,
        refresh_interval: float = 300.0,
    ) -> None:
        self.skill_manager = skill_manager
        self.max_size = max_size
        self.refresh_interval = refresh_interval
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._last_refresh: float = 0.0
        self._tracker: SkillPerformanceTracker | None = None

    def refresh(self, tracker: SkillPerformanceTracker) -> None:
        self._tracker = tracker
        now = time.time()
        if now - self._last_refresh < self.refresh_interval and self._cache:
            return
        self._last_refresh = now
        top_ids = tracker.get_top_skills(top_k=self.max_size)
        new_cache: OrderedDict[str, Any] = OrderedDict()
        for sid in top_ids:
            skill = self.skill_manager.load_skill(sid)
            if skill and len(new_cache) < self.max_size:
                new_cache[sid] = skill
        self._cache = new_cache
        logger.debug("HotSkillCache refreshed with %d skills", len(self._cache))

    def get(self, skill_id: str) -> Any:
        if skill_id in self._cache:
            self._cache.move_to_end(skill_id)
            return self._cache[skill_id]
        skill = self.skill_manager.load_skill(skill_id)
        if skill and len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        if skill:
            self._cache[skill_id] = skill
            self._cache.move_to_end(skill_id)
        return skill

    def get_top_cached(self, category: str | None = None, top_k: int = 5) -> List[Any]:
        if category:
            skills = [
                s
                for s in self._cache.values()
                if getattr(s, "metadata", None) and getattr(s.metadata, "category", "") == category
            ]
        else:
            skills = list(self._cache.values())
        return skills[:top_k]


def prepare_skill_document(skill: Any) -> str:
    """임베딩 생성을 위한 스킬 문서 포맷팅."""
    parts = [
        f"Skill: {getattr(skill, 'skill_id', '')}",
        f"Description: {getattr(skill, 'description', '') or ''}",
        f"Tags: {', '.join(getattr(skill, 'tags', []) or [])}",
        f"Capabilities: {', '.join(getattr(skill, 'capabilities', []) or [])}",
    ]
    return "\n".join(parts)


class SkillRetriever:
    """Hybrid skill retrieval: hot cache -> BM25 + Vector similarity -> FlashRank rerank."""

    def __init__(
        self,
        skill_manager: "SkillManager",
        hot_cache: HotSkillCache | None = None,
        tracker: SkillPerformanceTracker | None = None,
        flashrank_model: str = "ms-marco-TinyBERT-L-2-v2",
        embedding_provider: Any = None,
        hybrid_alpha: float = 0.3,
    ) -> None:
        self.skill_manager = skill_manager
        self.hot_cache = hot_cache or HotSkillCache(skill_manager)
        self.tracker = tracker or SkillPerformanceTracker()
        self._ranker = None
        self._flashrank_model = flashrank_model
        self._keyword_map = self._build_keyword_map()
        self._embedding_provider = embedding_provider
        self._hybrid_alpha = hybrid_alpha
        self._skill_vectors: Dict[str, Any] = {}
        self._vectors_built = False
        self._mechanism_db = UniversalMechanismDatabase()

    def _build_keyword_map(self) -> Dict[str, List[str]]:
        return {
            "research_planner": [
                "plan",
                "planning",
                "strategy",
                "objective",
                "goal",
                "research plan",
                "계획",
                "전략",
                "목표",
            ],
            "research_executor": [
                "search",
                "find",
                "gather",
                "execute",
                "research",
                "investigate",
                "검색",
                "수집",
                "실행",
                "연구",
                "조사",
            ],
            "evaluator": [
                "verify",
                "validate",
                "check",
                "evaluate",
                "assess",
                "quality",
                "검증",
                "평가",
                "확인",
                "품질",
            ],
            "synthesizer": [
                "synthesize",
                "summarize",
                "report",
                "generate",
                "create",
                "final",
                "종합",
                "요약",
                "리포트",
                "생성",
                "최종",
            ],
        }

    def _get_ranker(self) -> Any:
        if self._ranker is None:
            try:
                from flashrank import Ranker

                self._ranker = Ranker(model_name=self._flashrank_model, log_level="WARNING")
            except Exception as e:
                logger.warning("FlashRank Ranker unavailable: %s", e)
        return self._ranker

    def _get_embedding_provider(self) -> Any:
        """임베딩 제공자 지연 로딩 (환경변수로 Gemini 전환 가능)."""
        if self._embedding_provider is None:
            try:
                import os

                from src.core.memory_embeddings import get_embedding_provider

                provider_type = os.getenv("SPARKLEFORGE_EMBEDDING_PROVIDER", "local")
                self._embedding_provider = get_embedding_provider(provider_type)
            except Exception as e:
                logger.warning("Embedding provider unavailable: %s", e)
                self._embedding_provider = None
        return self._embedding_provider

    async def _build_skill_vectors(self, all_metadata: List[Any]) -> bool:
        """활성화된 스킬에 대한 임베딩 벡터 캐싱."""
        provider = self._get_embedding_provider()
        if provider is None:
            return False
        try:
            import numpy as np

            stale = set(self._skill_vectors.keys()) != {m.skill_id for m in all_metadata}
            if self._vectors_built and not stale:
                return True
            docs = [prepare_skill_document(m) for m in all_metadata]
            if not docs:
                self._skill_vectors = {}
                self._vectors_built = True
                return True
            embeddings = await provider.embed_documents(docs)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            self._skill_vectors = {
                m.skill_id: embeddings[i] for i, m in enumerate(all_metadata)
            }
            self._vectors_built = True
            logger.debug("Built skill vectors for %d skills", len(self._skill_vectors))
            return True
        except Exception as e:
            logger.warning("Failed to build skill vectors, fallback to BM25: %s", e)
            self._skill_vectors = {}
            self._vectors_built = False
            return False

    async def _vector_scores(self, query: str, all_metadata: List[Any]) -> Dict[str, float]:
        """쿼리와 각 스킬 간 코사인 유사도 점수 반환."""
        if not await self._build_skill_vectors(all_metadata):
            return {}
        try:
            import numpy as np

            provider = self._get_embedding_provider()
            query_vec = np.asarray(await provider.embed_query(query), dtype=np.float32)
            scores: Dict[str, float] = {}
            for meta in all_metadata:
                vec = self._skill_vectors.get(meta.skill_id)
                if vec is None:
                    continue
                q_norm = np.linalg.norm(query_vec)
                v_norm = np.linalg.norm(vec)
                if q_norm == 0 or v_norm == 0:
                    scores[meta.skill_id] = 0.0
                else:
                    scores[meta.skill_id] = float(
                        np.dot(query_vec.flatten(), vec.flatten()) / (q_norm * v_norm)
                    )
            return scores
        except Exception as e:
            logger.warning("Vector scoring failed: %s", e)
            return {}

    def _bm25_style_score(self, query: str, metadata: Any) -> float:
        query_lower = query.lower()
        words = set(re.findall(r"\b\w+\b", query_lower))
        desc = (getattr(metadata, "description", "") or "").lower()
        tags = getattr(metadata, "tags", []) or []
        caps = getattr(metadata, "capabilities", []) or []
        text = desc + " " + " ".join(tags).lower() + " " + " ".join(caps).lower()
        text_words = set(re.findall(r"\b\w+\b", text))
        overlap = len(words & text_words)
        return overlap / max(len(words), 1)

    def _mechanism_boost(self, query: str, metadata: Any) -> tuple[float, List[str]]:
        score, reasons = self._mechanism_db.match(query, metadata)
        return score, reasons

    async def retrieve(
        self,
        query: str,
        agent_skill_tree: SkillTree | None = None,
        top_k: int = 5,
    ) -> List[Any]:
        SkillMatch = _get_skill_match_type()
        candidates: List[tuple] = []
        all_metadata = self.skill_manager.get_all_skills(enabled_only=True)
        if not all_metadata:
            return []

        vector_scores = await self._vector_scores(query, all_metadata)

        if agent_skill_tree:
            hot_ids = agent_skill_tree.get_hot_skills(top_k=top_k * 2)
            for sid in hot_ids:
                meta = self.skill_manager.get_skill_by_id(sid)
                if meta:
                    score = self._bm25_style_score(query, meta)
                    vscore = vector_scores.get(sid, 0.0)
                    score = self._hybrid_alpha * score + (1.0 - self._hybrid_alpha) * vscore
                    candidates.append((sid, score, meta, ["agent_tree_hot"]))

        for meta in all_metadata:
            if any(c[0] == meta.skill_id for c in candidates):
                continue
            score = self._bm25_style_score(query, meta)
            vscore = vector_scores.get(meta.skill_id, 0.0)
            hybrid = self._hybrid_alpha * score + (1.0 - self._hybrid_alpha) * vscore
            mech_score, mech_reasons = self._mechanism_boost(query, meta)
            if mech_score > 0:
                hybrid += 0.15 * mech_score
                reasons = ["bm25"] if score > 0 else []
                if vscore > 0:
                    reasons.append("vector")
                reasons.extend(mech_reasons)
                candidates.append((meta.skill_id, hybrid, meta, reasons))
                continue
            if hybrid > 0:
                reasons = ["bm25"] if score > 0 else []
                if vscore > 0:
                    reasons.append("vector")
                candidates.append((meta.skill_id, hybrid, meta, reasons))

        if not candidates:
            candidates = [(m.skill_id, 0.1, m, ["fallback"]) for m in all_metadata[: top_k * 2]]

        candidates.sort(key=lambda x: x[1], reverse=True)
        to_rerank = candidates[: min(20, len(candidates))]

        ranker = self._get_ranker()
        if ranker:
            try:
                from flashrank import RerankRequest

                passages = [
                    {
                        "id": sid,
                        "text": (meta.description or "")
                        + " "
                        + " ".join(meta.tags or [])
                        + " "
                        + " ".join(meta.capabilities or []),
                    }
                    for sid, _, meta, _ in to_rerank
                ]
                req = RerankRequest(query=query, passages=passages)
                reranked = ranker.rerank(req)
                id_to_cand = {c[0]: c for c in to_rerank}
                ordered = []
                for r in reranked[:top_k]:
                    cand = id_to_cand.get(r["id"])
                    if cand:
                        ordered.append(
                            (cand[0], float(r["score"]), cand[2], cand[3] + ["flashrank"])
                        )
                if ordered:
                    to_rerank = ordered
            except Exception as e:
                logger.debug("FlashRank rerank failed, using BM25 order: %s", e)

        result = []
        for item in to_rerank[:top_k]:
            sid, score, meta, reasons = item
            result.append(SkillMatch(skill_id=sid, score=score, reasons=reasons, metadata=meta))
        return result


@dataclass
class MechanismPattern:
    """A universal operational pattern reusable across creative synthesis."""

    pattern_id: str
    name: str
    category: str
    description: str
    keywords: List[str] = field(default_factory=list)
    applicability: List[str] = field(default_factory=list)


class UniversalMechanismDatabase:
    """Structured database of universal operational patterns.

    Supports automated pattern retrieval during creative synthesis by matching
    task/skill descriptions against canonical mechanisms (rate limiters,
    backpressure, distributed consensus, biological homeostasis, etc.).
    """

    def __init__(self) -> None:
        self._patterns: List[MechanismPattern] = self._seed_patterns()
        self._index: Dict[str, MechanismPattern] = {p.pattern_id: p for p in self._patterns}

    def _seed_patterns(self) -> List[MechanismPattern]:
        return [
            MechanismPattern(
                pattern_id="rate_limiter",
                name="Rate Limiter",
                category="flow_control",
                description="Bound request throughput to protect downstream systems.",
                keywords=["rate", "limit", "throttle", "quota", "throughput", "backpressure"],
                applicability=["api", "queue", "scheduler", "ingestion"],
            ),
            MechanismPattern(
                pattern_id="backpressure",
                name="Backpressure",
                category="flow_control",
                description="Signal upstream producers to slow down when consumers are saturated.",
                keywords=["backpressure", "pressure", "overflow", "buffer", "saturation"],
                applicability=["stream", "pipeline", "queue", "actor"],
            ),
            MechanismPattern(
                pattern_id="circuit_breaker",
                name="Circuit Breaker",
                category="resilience",
                description="Fail fast and stop calling a failing dependency until it recovers.",
                keywords=["circuit", "breaker", "fail", "retry", "resilience", "timeout"],
                applicability=["rpc", "api", "service", "network"],
            ),
            MechanismPattern(
                pattern_id="distributed_consensus",
                name="Distributed Consensus",
                category="coordination",
                description="Agree on shared state across unreliable participants.",
                keywords=["consensus", "raft", "paxos", "quorum", "vote", "agreement"],
                applicability=["cluster", "replication", "leader", "election"],
            ),
            MechanismPattern(
                pattern_id="homeostasis",
                name="Biological Homeostasis",
                category="control",
                description="Negative feedback loops maintain internal equilibrium against perturbations.",
                keywords=["homeostasis", "feedback", "equilibrium", "regulate", "balance", "steady"],
                applicability=["control", "regulation", "adaptive", "thermostat"],
            ),
            MechanismPattern(
                pattern_id="exponential_backoff",
                name="Exponential Backoff",
                category="resilience",
                description="Increase wait time between retries to avoid overload storms.",
                keywords=["backoff", "retry", "jitter", "exponential", "wait"],
                applicability=["retry", "network", "rpc", "queue"],
            ),
            MechanismPattern(
                pattern_id="bulkhead",
                name="Bulkhead Isolation",
                category="resilience",
                description="Partition resources so a failure in one compartment does not sink the whole system.",
                keywords=["bulkhead", "isolation", "partition", "compartment", "pool"],
                applicability=["pool", "thread", "resource", "tenant"],
            ),
            MechanismPattern(
                pattern_id="lease_renewal",
                name="Lease Renewal",
                category="coordination",
                description="Time-bound ownership that must be renewed or it expires.",
                keywords=["lease", "renew", "ttl", "expire", "lock", "heartbeat"],
                applicability=["lock", "session", "leader", "coordination"],
            ),
        ]

    def list_patterns(self) -> List[MechanismPattern]:
        return list(self._patterns)

    def get(self, pattern_id: str) -> MechanismPattern | None:
        return self._index.get(pattern_id)

    def search(self, query: str, top_k: int = 5) -> List[tuple[MechanismPattern, float]]:
        query_lower = query.lower()
        query_words = set(re.findall(r"\b\w+\b", query_lower))
        scored: List[tuple[MechanismPattern, float]] = []
        for pattern in self._patterns:
            corpus = " ".join(
                [pattern.name, pattern.category, pattern.description]
                + pattern.keywords
                + pattern.applicability
            ).lower()
            corpus_words = set(re.findall(r"\b\w+\b", corpus))
            overlap = len(query_words & corpus_words)
            keyword_hits = sum(1 for kw in pattern.keywords if kw in query_lower)
            score = overlap / max(len(query_words), 1) + 0.25 * keyword_hits
            if score > 0:
                scored.append((pattern, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def match(self, query: str, metadata: Any) -> tuple[float, List[str]]:
        """Return (boost_score, reasons) for a skill against universal mechanisms."""
        results = self.search(query, top_k=3)
        if not results:
            return 0.0, []
        desc = (getattr(metadata, "description", "") or "").lower()
        tags = [t.lower() for t in (getattr(metadata, "tags", []) or [])]
        caps = [c.lower() for c in (getattr(metadata, "capabilities", []) or [])]
        skill_corpus = " ".join([desc] + tags + caps)
        matched: List[str] = []
        total = 0.0
        for pattern, score in results:
            applicability_hits = sum(
                1 for a in pattern.applicability if a in skill_corpus
            )
            if applicability_hits > 0 or pattern.category in skill_corpus:
                total += min(score, 1.0)
                matched.append(f"mechanism:{pattern.pattern_id}")
        return min(total, 1.0), matched


def get_skill_performance_tracker(store_path: Path | None = None) -> SkillPerformanceTracker:
    """Return global SkillPerformanceTracker instance."""
    global _skill_performance_tracker
    if "_skill_performance_tracker" not in globals():
        _skill_performance_tracker = SkillPerformanceTracker(store_path)
    return _skill_performance_tracker
