"""Reasoning Memory Core Engine

Google Research ReasoningBank 시스템에서 영감을 받은 추론 메모리 코어.
에이전트의 성공/실패 궤적에서 추출된 '이유와 인사이트'를 임베딩과 함께 저장하고 검색합니다.
"""

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from src.core.memory_embeddings import cosine_similarity_search, get_embedding_provider

logger = logging.getLogger(__name__)


@dataclass
class ReasoningMemoryItem:
    """단일 추론 메모리 아이템.

    ReasoningBank의 아이디어를 바탕으로 설계되었습니다. 단순한 사실이 아닌,
    '이유', '어떻게', '왜 성공/실패했는지'에 대한 인사이트를 담습니다.
    """

    memory_id: str
    title: str  # Memory Item 제목
    description: str  # 언제 사용할지 설명
    content: str  # 추론 인사이트 본문 (1-5문장)
    trajectory_status: str  # "success" | "fail" | "parallel_contrast"
    task_query: str  # 원본 task 쿼리 (출처)
    domain: str  # 도메인 태그 (예: "web_search", "coding")
    created_at: str  # 생성 일시 (ISO 포맷)
    embedding: List[float] = field(default_factory=list)  # 캐시된 임베딩
    usage_count: int = 0  # 검색/사용 횟수
    effectiveness_score: float = 0.5  # 피드백에 기반한 유효성 점수

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningMemoryItem":
        return cls(
            memory_id=data.get("memory_id", str(uuid.uuid4())),
            title=data.get("title", "Untitled Memory"),
            description=data.get("description", ""),
            content=data.get("content", ""),
            trajectory_status=data.get("trajectory_status", "unknown"),
            task_query=data.get("task_query", ""),
            domain=data.get("domain", "general"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            embedding=data.get("embedding", []),
            usage_count=data.get("usage_count", 0),
            effectiveness_score=data.get("effectiveness_score", 0.5),
        )


class ReasoningMemoryBank:
    """추론 메모리 영구 저장소 및 검색 엔진.

    JSONL 파일 기반으로 메모리를 영구 저장하고 (ReasoningBank 방식),
    임베딩을 통한 코사인 유사도 검색을 지원합니다.
    """

    def __init__(
        self, storage_dir: str = ".data/reasoning_memory", embedding_provider: str = "local"
    ):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.file_path = os.path.join(self.storage_dir, "memories.jsonl")

        # 임베딩 프로바이더 지연 로드 (초기화 속도 최적화)
        self._provider_type = embedding_provider
        self._provider = None

        self.memories: List[ReasoningMemoryItem] = []
        self._load_memories()

    @property
    def provider(self):
        if self._provider is None:
            self._provider = get_embedding_provider(self._provider_type)
        return self._provider

    def _load_memories(self):
        """저장된 메모리를 메모리로 로드합니다."""
        if not os.path.exists(self.file_path):
            open(self.file_path, "w").close()
            return

        try:
            with open(self.file_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    self.memories.append(ReasoningMemoryItem.from_dict(data))
            logger.info(f"Loaded {len(self.memories)} reasoning memories.")
        except Exception as e:
            logger.error(f"Failed to load reasoning memories: {e}")

    def _save_memory(self, item: ReasoningMemoryItem):
        """단일 메모리를 JSONL 파일에 추가(Append)합니다."""
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.memories.append(item)
        except Exception as e:
            logger.error(f"Failed to save reasoning memory: {e}")

    def _flush_all(self):
        """현재 인메모리 상태를 JSONL 파일에 원자적으로 동기화합니다."""
        temp_path = f"{self.file_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            for item in self.memories:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, self.file_path)

    async def store_memory(self, item: ReasoningMemoryItem) -> bool:
        """새로운 추론 메모리를 저장합니다. 필요시 임베딩을 자동 생성합니다."""
        if not item.embedding:
            # 설명과 내용을 합쳐 임베딩 생성 (검색 정확도 향상)
            text_to_embed = (
                f"Title: {item.title}\nDescription: {item.description}\nContent: {item.content}"
            )
            try:
                emb = await self.provider.embed_query(text_to_embed)
                item.embedding = emb[0].tolist()
            except Exception as e:
                logger.error(f"Failed to generate embedding for memory {item.memory_id}: {e}")
                return False

        self._save_memory(item)
        logger.info(f"Stored reasoning memory: {item.title}")
        return True

    async def store_batch(self, items: List[ReasoningMemoryItem]) -> int:
        """여러 메모리를 일괄 저장합니다."""
        success_count = 0
        for item in items:
            if await self.store_memory(item):
                success_count += 1
        return success_count

    async def select_reasoning_memory(
        self, query: str, n: int = 3, domain_filter: str = None
    ) -> List[ReasoningMemoryItem]:
        """현재 쿼리와 가장 관련성이 높은 추론 메모리를 검색합니다.

        ReasoningBank의 screening 로직과 유사하게 동작합니다.
        """
        if not self.memories:
            return []

        # 임베딩이 있는 메모리만 필터링
        valid_memories = [m for m in self.memories if m.embedding]

        if domain_filter:
            valid_memories = [m for m in valid_memories if m.domain == domain_filter]

        if not valid_memories:
            return []

        # 쿼리 임베딩 생성 (ReasoningBank처럼 "instruction-aware" 프롬프트 추가)
        task = "Given the prior reasoning memories, your task is to analyze a current query's intent and select relevant prior experiences that could help resolve it."
        instruction_query = f"Instruct: {task}\nQuery: {query}"

        try:
            q_vec = await self.provider.embed_query(instruction_query)

            # 검색 대상 임베딩 매트릭스 구성
            corpus_embeddings = np.array([m.embedding for m in valid_memories], dtype=np.float32)

            # 유사도 검색
            indices, scores = cosine_similarity_search(q_vec, corpus_embeddings, top_k=n)

            # 결과 구성
            results = []
            for idx in indices:
                mem = valid_memories[idx]
                mem.usage_count += 1
                results.append(mem)

            return results

        except Exception as e:
            logger.error(f"Failed to search reasoning memories: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        status_counts = {}
        domain_counts = {}

        for m in self.memories:
            status_counts[m.trajectory_status] = status_counts.get(m.trajectory_status, 0) + 1
            domain_counts[m.domain] = domain_counts.get(m.domain, 0) + 1

        return {
            "total_memories": len(self.memories),
            "by_status": status_counts,
            "by_domain": domain_counts,
        }


# 전역 인스턴스 (싱글톤)
_reasoning_memory_bank = None


def get_reasoning_memory_bank(storage_dir: str = ".data/reasoning_memory") -> ReasoningMemoryBank:
    """전역 Reasoning Memory Bank 인스턴스 반환."""
    global _reasoning_memory_bank
    if _reasoning_memory_bank is None:
        _reasoning_memory_bank = ReasoningMemoryBank(storage_dir=storage_dir)
    return _reasoning_memory_bank
