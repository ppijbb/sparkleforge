import hashlib
import json
import logging
import os
import re
import time
import sqlite3
import math
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

EMBEDDING_DIM = 256
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "for",
        "about", "above", "after", "again", "all", "am", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below",
        "between", "both", "but", "by", "can", "did", "do", "does", "doing",
        "down", "during", "each", "few", "for", "from", "further", "had",
        "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "into", "is", "it", "its", "itself",
        "of", "to", "in", "on", "at", "by", "with", "from", "as", "is",
        "are", "was", "were", "be", "been", "being", "this", "that", "these",
        "those", "it", "its", "i", "you", "he", "she", "we", "they", "them",
        "do", "does", "did", "not", "no", "so", "than", "too", "very", "can",
        "will", "just", "into", "about", "up", "out", "over", "after",
    }
)


# Four-layer long-term memory taxonomy (Working, Episodic, Semantic, Procedural)
MEMORY_LAYERS = ("working", "episodic", "semantic", "procedural")


def _normalize_layer(layer: Optional[str]) -> str:
    """Normalize an optional memory-layer hint to a known taxonomy value."""
    if not layer:
        return "semantic"
    normalized = layer.strip().lower()
    if normalized in MEMORY_LAYERS:
        return normalized
    return "semantic"


def _parse_timestamp(value: Any) -> Optional[float]:
    """Best-effort coercion of a metadata value into a unix timestamp."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


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
                        embedding_json TEXT,
                        layer TEXT NOT NULL DEFAULT 'semantic',
                        created_at REAL,
                        source TEXT,
                        entity TEXT,
                        relation TEXT,
                        target TEXT,
                        valid_from REAL,
                        valid_to REAL
                    )
                """)
                # Backfill schema for pre-existing databases created before the
                # Temporal GraphRAG columns existed.
                existing = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(memories)").fetchall()
                }
                for column, ddl in (
                    ("layer", "ALTER TABLE memories ADD COLUMN layer TEXT NOT NULL DEFAULT 'semantic'"),
                    ("created_at", "ALTER TABLE memories ADD COLUMN created_at REAL"),
                    ("source", "ALTER TABLE memories ADD COLUMN source TEXT"),
                    ("entity", "ALTER TABLE memories ADD COLUMN entity TEXT"),
                    ("relation", "ALTER TABLE memories ADD COLUMN relation TEXT"),
                    ("target", "ALTER TABLE memories ADD COLUMN target TEXT"),
                    ("valid_from", "ALTER TABLE memories ADD COLUMN valid_from REAL"),
                    ("valid_to", "ALTER TABLE memories ADD COLUMN valid_to REAL"),
                ):
                    if column not in existing:
                        try:
                            conn.execute(ddl)
                        except sqlite3.OperationalError:
                            pass
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_layer ON memories(layer)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(entity)"
                )
                conn.commit()
        except Exception as e:
            logger.error(f"SemanticMemory: Failed to initialize SQLite store: {e}")

    def add_memory(self, key: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Embed text and insert it into the semantic memory database."""
        metadata = metadata or {}
        embedding = generate_pseudo_embedding(text)

        layer = _normalize_layer(metadata.get("layer"))
        created_at = _parse_timestamp(metadata.get("created_at")) or time.time()
        source = metadata.get("source")
        entity = metadata.get("entity")
        relation = metadata.get("relation")
        target = metadata.get("target")
        valid_from = _parse_timestamp(metadata.get("valid_from"))
        valid_to = _parse_timestamp(metadata.get("valid_to"))
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO memories (
                        key, text, metadata_json, embedding_json, layer,
                        created_at, source, entity, relation, target,
                        valid_from, valid_to
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        key,
                        text,
                        json.dumps(metadata),
                        json.dumps(embedding),
                        layer,
                        created_at,
                        source,
                        entity,
                        relation,
                        target,
                        valid_from,
                        valid_to,
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
                cursor.execute(
                    """SELECT key, text, metadata_json, embedding_json, layer,
                              created_at, source, entity, relation, target,
                              valid_from, valid_to
                       FROM memories"""
                )
                rows = cursor.fetchall()
                
                for key, text, meta_json, emb_json, layer, created_at, source, entity, relation, target, valid_from, valid_to in rows:
                    try:
                        emb = json.loads(emb_json)
                        meta = json.loads(meta_json)
                        similarity = calculate_cosine_similarity(query_vector, emb)
                        results.append({
                            "key": key,
                            "text": text,
                            "metadata": meta,
                            "score": float(similarity),
                            "layer": layer,
                            "created_at": created_at,
                            "source": source,
                            "entity": entity,
                            "relation": relation,
                            "target": target,
                            "valid_from": valid_from,
                            "valid_to": valid_to,
                        })
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
        except Exception as e:
            logger.error(f"SemanticMemory: Failed to search database: {e}")
            return []

        # Sort descending by similarity score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def deep_query(
        self,
        query: str,
        limit: int = 5,
        layer: Optional[str] = None,
        entity: Optional[str] = None,
        relation: Optional[str] = None,
        target: Optional[str] = None,
        min_age_seconds: Optional[float] = None,
        as_of: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Temporal GraphRAG deep retrieval across the four-layer memory store.

        Combines vector similarity with structured graph filters (entity,
        relation, target), temporal validity windows (valid_from/valid_to),
        and a minimum-age filter so long-context recall can target memories
        older than a threshold (e.g. >1 week) without losing causal order.
        """
        query_vector = generate_pseudo_embedding(query)
        results: List[Dict[str, Any]] = []

        clauses: List[str] = []
        params: List[Any] = []

        normalized_layer = _normalize_layer(layer) if layer else None
        if normalized_layer:
            clauses.append("layer = ?")
            params.append(normalized_layer)
        if entity:
            clauses.append("entity = ?")
            params.append(entity)
        if relation:
            clauses.append("relation = ?")
            params.append(relation)
        if target:
            clauses.append("target = ?")
            params.append(target)
        if min_age_seconds is not None:
            cutoff = time.time() - float(min_age_seconds)
            clauses.append("(created_at IS NOT NULL AND created_at <= ?)")
            params.append(cutoff)
        if as_of is not None:
            clauses.append(
                "(valid_from IS NULL OR valid_from <= ?) AND "
                "(valid_to IS NULL OR valid_to >= ?)"
            )
            params.extend([as_of, as_of])

        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""SELECT key, text, metadata_json, embedding_json, layer,
                              created_at, source, entity, relation, target,
                              valid_from, valid_to
                       FROM memories{where_sql}""",
                    params,
                )
                rows = cursor.fetchall()

                for key, text, meta_json, emb_json, mem_layer, created_at, source, mem_entity, mem_relation, mem_target, valid_from, valid_to in rows:
                    try:
                        emb = json.loads(emb_json)
                        meta = json.loads(meta_json)
                        similarity = calculate_cosine_similarity(query_vector, emb)
                        results.append({
                            "key": key,
                            "text": text,
                            "metadata": meta,
                            "score": float(similarity),
                            "layer": mem_layer,
                            "created_at": created_at,
                            "source": source,
                            "entity": mem_entity,
                            "relation": mem_relation,
                            "target": mem_target,
                            "valid_from": valid_from,
                            "valid_to": valid_to,
                        })
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
        except Exception as e:
            logger.error(f"SemanticMemory: Failed to run deep query: {e}")
            return []

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def temporal_neighbors(
        self,
        entity: str,
        limit: int = 10,
        direction: str = "outgoing",
    ) -> List[Dict[str, Any]]:
        """Return graph edges adjacent to an entity, ordered by valid_from.

        `direction` may be "outgoing" (entity as subject), "incoming"
        (entity as object/target), or "both".
        """
        if direction not in ("outgoing", "incoming", "both"):
            direction = "both"

        clauses: List[str] = []
        params: List[Any] = []
        if direction == "outgoing":
            clauses.append("entity = ?")
            params.append(entity)
        elif direction == "incoming":
            clauses.append("target = ?")
            params.append(entity)
        else:
            clauses.append("(entity = ? OR target = ?)")
            params.extend([entity, entity])

        where_sql = " WHERE " + " AND ".join(clauses)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""SELECT key, text, metadata_json, layer, created_at,
                              source, entity, relation, target, valid_from, valid_to
                       FROM memories{where_sql}
                       ORDER BY COALESCE(valid_from, created_at) ASC""",
                    params,
                )
                rows = cursor.fetchall()
        except Exception as e:
            logger.error(f"SemanticMemory: Failed to query temporal neighbors: {e}")
            return []

        neighbors: List[Dict[str, Any]] = []
        for key, text, meta_json, layer, created_at, source, mem_entity, relation_name, mem_target, valid_from, valid_to in rows:
            try:
                meta = json.loads(meta_json)
            except (TypeError, ValueError, json.JSONDecodeError):
                meta = {}
            neighbors.append({
                "key": key,
                "text": text,
                "metadata": meta,
                "layer": layer,
                "created_at": created_at,
                "source": source,
                "entity": mem_entity,
                "relation": relation_name,
                "target": mem_target,
                "valid_from": valid_from,
                "valid_to": valid_to,
            })
        return neighbors[:limit]

    def clear(self):
        """Reset the SQLite memories table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM memories")
                conn.commit()
            logger.info("SemanticMemory: Cleared database table.")
        except Exception as e:
            logger.error(f"SemanticMemory: Failed to clear database table: {e}")
