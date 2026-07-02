import hashlib
import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_pseudo_embedding(text: str) -> List[float]:
    """Generate a deterministic 128-dimension normalized pseudo-embedding vector for text."""
    # Clean and split into individual words
    words = [w.strip(".,!?\"'()[]{}").lower() for w in text.split()]
    words = [w for w in words if w]
    
    if not words:
        return [0.0] * 128
        
    accumulated_vector = [0.0] * 128
    for word in words:
        # Build a deterministic 128-dimension vector for each single word
        for i in range(128):
            seed = f"word_{word}_{i}"
            h = hashlib.md5(seed.encode("utf-8")).hexdigest()
            val = int(h, 16) % 2000 - 1000
            accumulated_vector[i] += float(val)
            
    # Normalize to unit length
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
    """Provides semantic vector storage and similarity searches backed by local SQLite caching."""

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
