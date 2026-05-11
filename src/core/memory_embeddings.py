"""Memory Embeddings

Reasoning Memory 검색을 위한 임베딩 인프라.
로컬 sentence-transformers 모델과 Gemini API 모델을 지원합니다.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2 정규화"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return np.where(norm == 0, x, x / norm)


def cosine_similarity_search(
    query_embedding: np.ndarray, corpus_embeddings: np.ndarray, top_k: int = 5
) -> Tuple[List[int], List[float]]:
    """유사도 기반 검색.

    Args:
        query_embedding: 쿼리 임베딩 (1, D)
        corpus_embeddings: 검색 대상 임베딩 (N, D)
        top_k: 반환할 상위 k개

    Returns:
        (인덱스 리스트, 점수 리스트)
    """
    if len(corpus_embeddings) == 0:
        return [], []

    # 정규화
    q_vec = l2_normalize(query_embedding)
    c_vecs = l2_normalize(corpus_embeddings)

    # 코사인 유사도 계산
    scores = np.dot(c_vecs, q_vec.T).squeeze(axis=1) * 100.0

    # 상위 k개 선택
    top_k_indices = np.argsort(scores)[::-1][:top_k]
    top_k_scores = scores[top_k_indices]

    return top_k_indices.tolist(), top_k_scores.tolist()


class EmbeddingProvider(ABC):
    """임베딩 제공자 인터페이스."""

    @abstractmethod
    async def embed_query(self, text: str) -> np.ndarray:
        """단일 쿼리 임베딩 반환."""

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> np.ndarray:
        """여러 문서 임베딩 반환."""

    @property
    @abstractmethod
    def dimensionality(self) -> int:
        """임베딩 차원 반환."""


class LocalEmbeddingProvider(EmbeddingProvider):
    """로컬 임베딩 제공자 (sentence-transformers 기반).

    기본 모델: all-MiniLM-L6-v2 (차원: 384, 작고 빠름)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dim = 384  # all-MiniLM-L6-v2 기본 차원

    def _get_model(self):
        """지연 로딩: 처음 필요할 때 모델 로드"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading local embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                # 모델 차원 확인
                self._dim = self._model.get_sentence_embedding_dimension()
            except ImportError:
                logger.warning(
                    "sentence_transformers is not installed. Using dummy embeddings for testing."
                )
                self._model = "dummy"
        return self._model

    async def embed_query(self, text: str) -> np.ndarray:
        """단일 쿼리 임베딩."""
        return await self.embed_documents([text])

    async def embed_documents(self, texts: List[str]) -> np.ndarray:
        """여러 문서 임베딩."""
        model = self._get_model()
        if model == "dummy":
            # 테스트용 더미 임베딩 반환
            logger.debug(f"Generating {len(texts)} dummy embeddings.")
            return np.random.rand(len(texts), self._dim).astype(np.float32)

        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings

    @property
    def dimensionality(self) -> int:
        return self._dim


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini API 기반 임베딩 제공자."""

    def __init__(self, model_name: str = "models/embedding-001", dimensionality: int = 768):
        self.model_name = model_name
        self._dim = dimensionality
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai

                # API 키는 환경변수에서 자동으로 읽거나 설정되어 있어야 함
                if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
                    logger.warning("Gemini API key not found. Embeddings might fail.")
                self._client = genai
            except ImportError:
                logger.error("google-generativeai is not installed.")
                raise
        return self._client

    async def embed_query(self, text: str) -> np.ndarray:
        """단일 쿼리 임베딩."""
        client = self._get_client()
        try:
            result = client.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query",
            )
            return np.array([result["embedding"]], dtype=np.float32)
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            # Fallback
            return np.zeros((1, self._dim), dtype=np.float32)

    async def embed_documents(self, texts: List[str]) -> np.ndarray:
        """여러 문서 임베딩."""
        client = self._get_client()
        try:
            result = client.embed_content(
                model=self.model_name,
                content=texts,
                task_type="retrieval_document",
            )
            embeddings = [item for item in result["embedding"]]
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Gemini batch embedding failed: {e}")
            return np.zeros((len(texts), self._dim), dtype=np.float32)

    @property
    def dimensionality(self) -> int:
        return self._dim


def get_embedding_provider(provider_type: str = "local") -> EmbeddingProvider:
    """설정에 따른 임베딩 제공자 팩토리 함수."""
    if provider_type.lower() == "gemini":
        return GeminiEmbeddingProvider()
    else:
        return LocalEmbeddingProvider()
