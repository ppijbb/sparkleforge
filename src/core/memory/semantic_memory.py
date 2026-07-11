import hashlib
import json
import logging
import os
import sqlite3
import math
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

EMBEDDING_DIM = 256
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "for",
        "of", "to", "in", "on", "at", "by", "with", "from", "as", "is",
        "are", "was", "were", "be", "been", "being", "this", "that", "these",
        "those", "it", "its", "i", "you", "he", "she", "we", "they", "them",
        "do", "does", "did", "not", "no", "so", "than", "too", "very", "can",
        "will", "just", "into", "about", "up", "out", "over", "after",
    }
)


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, and drop stopwords/short tokens."""
    tokens = []
    for raw in text.split():
        token = raw.strip(".,!?\"'()[]{}:;").lower()
        if not token:
            continue
        if len(token) < 2:
            continue
        if token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _hash_to_bucket(token: str, salt: str, dim: int) -> int:
    """Deterministically project a token+salt pair into a dimension index."""
    digest = hashlib.md5(f"{salt}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % dim


def _signed_hash(token: str) -> int:
    """Return +1/-1 based on a signed hash of the token to reduce collisions."""
    digest = hashlib.md5(f"sign:{token}".encode("utf-8")).digest()
    return 1 if (digest[0] & 1) == 0 else -1


def generate_pseudo_embedding(text: str) -> List[float]:
    """Generate a deterministic normalized bag-of-words embedding for text.

    Uses the feature-hashing trick with signed dimensions and sublinear TF
    weighting so that overlapping vocabulary produces similar vectors and
    disjoint vocabulary produces near-orthogonal vectors. The result is a
    unit vector suitable for cosine similarity search.
    """
    tokens = _tokenize(text)
    if not tokens:
        return [0.0] * EMBEDDING_DIM

    term_counts: Dict[str, int] = {}
    for token in tokens:
        term_counts[token] = term_counts.get(token, 0) + 1

    vector = [0.0] * EMBEDDING_DIM
    for token, count in term_counts.items():
        # Sublinear TF dampening reduces the impact of repeated words.
        weight = 1.0 + math.log(count)
        sign = _signed_hash(token)
        # Project each token into a small number of dimensions (2 hashes) to
        # approximate the dense overlap behavior of real embeddings.
        for salt in ("dim1", "dim2"):
            idx = _hash_to_bucket(token, salt, EMBEDDING_DIM)
            vector[idx] += sign * weight

    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0:
        return [0.0] * EMBEDDING_DIM
    return [x / magnitude for x in vector]


def generate_pseudo_embedding_legacy(text: str) -> List[float]:
    """Backward-compatible alias for the original MD5 pseudo-embedding.

    Kept for callers that explicitly want the legacy behavior; new code should
    use `generate_pseudo_embedding`.
    """
    words = [w.strip(".,!?\"'()[]{}").lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return [0.0] * 128
    accumulated_vector = [0.0] * 128
    for word in words:
        for i in range(128):
            seed = f"word_{word}_{i}"
            h = hashlib.md5(seed.encode("utf-8")).hexdigest()
            val = int(h, 16) % 2000 - 1000
            accumulated_vector[i] += float(val)
    magnitude = sum(x ** 2 for x in accumulated_vector) ** 0.5
    if magnitude == 0:
        return [0.0] * 128
    return [x / magnitude for x in accumulated_vector]


def calculate_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two unit vectors (dot product)."""
    if len(v1) != len(v2):
        return 0.0
    return sum(a * b for a, b in zip(v1, v2))


class SemanticMemory:
    """Provides semantic vector storage and similarity searches backed by local SQLite caching.

    Embeddings are produced by `generate_pseudo_embedding`, a local
    dependency-free bag-of-words hashing embedding. It is not a trained
    sentence-embedding model, but unlike the previous MD5-per-word stub it
    preserves vocabulary overlap, so semantically related texts (sharing
    words or near-synonyms that tokenize identically) score higher than
    unrelated texts.
    """

    def __init__(self, db_path: str = "data/semantic_memory.db"):
        self.db_path = db_path
        # Ensure data directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        self._init_db()

    def _init_db(self):
        """Create standard SQLite storage layout."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT NOT NULL,
                        text TEXT NOT NULL,
                        metadata_json TEXT,
                        embedding_json TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"SemanticMemory: Failed to initialize SQLite store: {e}")

    def add_memory(self, key: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Embed text and insert it into the semantic memory database."""
        metadata = metadata or {}
        embedding = generate_pseudo_embedding(text)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO memories (key, text, metadata_json, embedding_json) VALUES (?, ?, ?, ?)",
                    (
                        key,
                        text,
                        json.dumps(metadata),
                        json.dumps(embedding)
                    )
                )
                conn.commit()
            logger.info(f"SemanticMemory: Added memory under key '{key}'")
            return True
        except Exception as e:
            logger.error(f"SemanticMemory: Failed to add memory: {e}")
            return False

    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Calculate similarity metrics against queries and return top matches."""
        query_vector = generate_pseudo_embedding(query)
        results = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT key, text, metadata_json, embedding_json FROM memories")
                rows = cursor.fetchall()
                
                for key, text, meta_json, emb_json in rows:
                    try:
                        emb = json.loads(emb_json)
                        meta = json.loads(meta_json)
                        similarity = calculate_cosine_similarity(query_vector, emb)
                        results.append({
                            "key": key,
                            "text": text,
                            "metadata": meta,
                            "score": float(similarity)
                        })
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
        except Exception as e:
            logger.error(f"SemanticMemory: Failed to search database: {e}")
            return []

        # Sort descending by similarity score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def clear(self):
        """Reset the SQLite memories table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM memories")
                conn.commit()
            logger.info("SemanticMemory: Cleared database table.")
        except Exception as e:
            logger.error(f"SemanticMemory: Failed to clear database table: {e}")
