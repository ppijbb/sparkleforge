"""ContentStore — FTS5 BM25-based knowledge base for context-mode.

Chunks markdown content by headings (keeping code blocks intact),
stores in SQLite FTS5, and retrieves via BM25-ranked search.
Ported from reference/claude-context-mode src/store.ts.
"""

import logging
import math
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# FTS5 may not be available on all systems
try:
    _conn = sqlite3.connect(":memory:")
    _conn.execute("CREATE VIRTUAL TABLE _fts5_test USING fts5(x)")
    _conn.close()
    FTS5_AVAILABLE = True
except sqlite3.OperationalError:
    FTS5_AVAILABLE = False

STOPWORDS = frozenset([
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
    "new", "now", "old", "see", "way", "who", "did", "get", "got", "let",
    "say", "she", "too", "use", "will", "with", "this", "that", "from",
    "they", "been", "have", "many", "some", "them", "than", "each", "make",
    "like", "just", "over", "such", "take", "into", "year", "your", "good",
    "could", "would", "about", "which", "their", "there", "other", "after",
    "should", "through", "also", "more", "most", "only", "very", "when",
    "what", "then", "these", "those", "being", "does", "done", "both",
    "same", "still", "while", "where", "here", "were", "much",
    "update", "updates", "updated", "deps", "dev", "tests", "test",
    "add", "added", "fix", "fixed", "run", "running", "using",
])


def _sanitize_query(query: str) -> str:
    """Sanitize query for FTS5 MATCH (Porter)."""
    words = re.sub(r"['\"(){}[\]*:^~]", " ", query).split()
    words = [w for w in words if w and w.upper() not in ("AND", "OR", "NOT", "NEAR")]
    if not words:
        return '""'
    return " OR ".join(f'"{w}"' for w in words)


def _sanitize_trigram_query(query: str) -> str:
    """Sanitize for trigram MATCH."""
    cleaned = re.sub(r"[\"'(){}[\]*:^~]", "", query).strip()
    if len(cleaned) < 3:
        return ""
    words = [w for w in cleaned.split() if len(w) >= 3]
    if not words:
        return ""
    return " OR ".join(f'"{w}"' for w in words)


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i]
        for j in range(1, len(b) + 1):
            curr.append(
                prev[j - 1] if a[i - 1] == b[j - 1]
                else 1 + min(prev[j], curr[j - 1], prev[j - 1])
            )
        prev = curr
    return prev[len(b)]


def _max_edit_distance(word_length: int) -> int:
    if word_length <= 4:
        return 1
    if word_length <= 12:
        return 2
    return 3


_store_instance: Optional["ContentStore"] = None


def get_store() -> "ContentStore":
    """Lazy singleton ContentStore (shared by interceptor, server, context_engineer)."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ContentStore()
    return _store_instance


def cleanup_stale_dbs() -> int:
    """Remove stale DB files from previous sessions (process no longer exists)."""
    tmpdir = tempfile.gettempdir()
    cleaned = 0
    try:
        for name in os.listdir(tmpdir):
            m = re.match(r"^context-mode-(\d+)\.db$", name)
            if not m:
                continue
            pid = int(m.group(1))
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, 0)
            except (OSError, ProcessLookupError):
                base = os.path.join(tmpdir, name)
                for suffix in ["", "-wal", "-shm"]:
                    try:
                        os.unlink(base + suffix)
                    except FileNotFoundError:
                        pass
                cleaned += 1
    except OSError:
        pass
    return cleaned


class IndexResult:
    """Result of indexing content."""

    def __init__(self, source_id: int, label: str, total_chunks: int, code_chunks: int):
        self.source_id = source_id
        self.label = label
        self.total_chunks = total_chunks
        self.code_chunks = code_chunks


class SearchResult:
    """Single search hit."""

    def __init__(
        self,
        title: str,
        content: str,
        source: str,
        rank: float,
        content_type: str,
        match_layer: Optional[str] = None,
        highlighted: Optional[str] = None,
    ):
        self.title = title
        self.content = content
        self.source = source
        self.rank = rank
        self.content_type = content_type
        self.match_layer = match_layer
        self.highlighted = highlighted


class ContentStore:
    """FTS5 BM25 knowledge base for context-mode."""

    def __init__(self, db_path: Optional[str] = None):
        if not FTS5_AVAILABLE:
            raise RuntimeError("SQLite FTS5 is not available on this system")
        self._db_path = db_path or os.path.join(
            tempfile.gettempdir(), f"context-mode-{os.getpid()}.db"
        )
        self._conn = sqlite3.connect(self._db_path, timeout=5.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                code_chunk_count INTEGER NOT NULL DEFAULT 0,
                indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                title,
                content,
                source_id,
                content_type,
                tokenize='porter unicode61'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_trigram USING fts5(
                title,
                content,
                source_id,
                content_type,
                tokenize='trigram'
            );

            CREATE TABLE IF NOT EXISTS vocabulary (
                word TEXT PRIMARY KEY
            );
        """)
        self._conn.commit()

    def cleanup(self) -> None:
        """Delete this session's DB files. Call on process exit."""
        try:
            self._conn.close()
        except Exception:
            pass
        if self._db_path == ":memory:":
            return
        for suffix in ["", "-wal", "-shm"]:
            try:
                os.unlink(self._db_path + suffix)
            except FileNotFoundError:
                pass

    def index(
        self,
        content: Optional[str] = None,
        path: Optional[str] = None,
        source: Optional[str] = None,
    ) -> IndexResult:
        """Index markdown or text. Provide content or path."""
        if not content and not path:
            raise ValueError("Either content or path must be provided")
        text = content if content is not None else Path(path).read_text(encoding="utf-8", errors="replace")
        label = source or path or "untitled"
        chunks = self._chunk_markdown(text)
        if not chunks:
            cur = self._conn.execute(
                "INSERT INTO sources (label, chunk_count, code_chunk_count) VALUES (?, 0, 0)",
                (label,),
            )
            return IndexResult(
                source_id=cur.lastrowid or 0,
                label=label,
                total_chunks=0,
                code_chunks=0,
            )
        code_chunks = sum(1 for c in chunks if c[2])
        cur = self._conn.execute(
            "INSERT INTO sources (label, chunk_count, code_chunk_count) VALUES (?, ?, ?)",
            (label, len(chunks), code_chunks),
        )
        source_id = cur.lastrowid or 0
        for title, cont, has_code in chunks:
            ct = "code" if has_code else "prose"
            self._conn.execute(
                "INSERT INTO chunks (title, content, source_id, content_type) VALUES (?, ?, ?, ?)",
                (title, cont, source_id, ct),
            )
            self._conn.execute(
                "INSERT INTO chunks_trigram (title, content, source_id, content_type) VALUES (?, ?, ?, ?)",
                (title, cont, source_id, ct),
            )
        self._conn.commit()
        self._extract_and_store_vocabulary(text)
        return IndexResult(
            source_id=source_id,
            label=label,
            total_chunks=len(chunks),
            code_chunks=code_chunks,
        )

    def index_plain_text(
        self,
        content: str,
        source: str,
        lines_per_chunk: int = 20,
    ) -> IndexResult:
        """Index plain text by fixed-size line groups."""
        if not content or not content.strip():
            cur = self._conn.execute(
                "INSERT INTO sources (label, chunk_count, code_chunk_count) VALUES (?, 0, 0)",
                (source,),
            )
            return IndexResult(
                source_id=cur.lastrowid or 0,
                label=source,
                total_chunks=0,
                code_chunks=0,
            )
        chunks = self._chunk_plain_text(content, lines_per_chunk)
        cur = self._conn.execute(
            "INSERT INTO sources (label, chunk_count, code_chunk_count) VALUES (?, ?, 0)",
            (source, len(chunks)),
        )
        source_id = cur.lastrowid or 0
        for title, cont in chunks:
            self._conn.execute(
                "INSERT INTO chunks (title, content, source_id, content_type) VALUES (?, ?, ?, ?)",
                (title, cont, source_id, "prose"),
            )
            self._conn.execute(
                "INSERT INTO chunks_trigram (title, content, source_id, content_type) VALUES (?, ?, ?, ?)",
                (title, cont, source_id, "prose"),
            )
        self._conn.commit()
        self._extract_and_store_vocabulary(content)
        return IndexResult(
            source_id=source_id,
            label=source,
            total_chunks=len(chunks),
            code_chunks=0,
        )

    def search(self, query: str, limit: int = 3, source: Optional[str] = None) -> List[SearchResult]:
        """Porter FTS5 search."""
        sanitized = _sanitize_query(query)
        if source:
            rows = self._conn.execute(
                """
                SELECT chunks.title, chunks.content, chunks.content_type, sources.label,
                       bm25(chunks, 2.0, 1.0, 0, 0) AS rank,
                       highlight(chunks, 1, char(2), char(3)) AS highlighted
                FROM chunks
                JOIN sources ON sources.id = chunks.source_id
                WHERE chunks MATCH ? AND sources.label LIKE ?
                ORDER BY rank
                LIMIT ?
                """,
                (sanitized, f"%{source}%", limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT chunks.title, chunks.content, chunks.content_type, sources.label,
                       bm25(chunks, 2.0, 1.0, 0, 0) AS rank,
                       highlight(chunks, 1, char(2), char(3)) AS highlighted
                FROM chunks
                JOIN sources ON sources.id = chunks.source_id
                WHERE chunks MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (sanitized, limit),
            ).fetchall()
        return [
            SearchResult(
                title=r[0] or "",
                content=r[1] or "",
                source=r[3] or "",
                rank=r[4] or 0.0,
                content_type=r[2] or "prose",
                highlighted=r[5] if len(r) > 5 else None,
            )
            for r in rows
        ]

    def search_trigram(
        self, query: str, limit: int = 3, source: Optional[str] = None
    ) -> List[SearchResult]:
        """Trigram FTS5 search."""
        sanitized = _sanitize_trigram_query(query)
        if not sanitized:
            return []
        if source:
            rows = self._conn.execute(
                """
                SELECT chunks_trigram.title, chunks_trigram.content, chunks_trigram.content_type,
                       sources.label, bm25(chunks_trigram, 2.0, 1.0, 0, 0) AS rank,
                       highlight(chunks_trigram, 1, char(2), char(3)) AS highlighted
                FROM chunks_trigram
                JOIN sources ON sources.id = chunks_trigram.source_id
                WHERE chunks_trigram MATCH ? AND sources.label LIKE ?
                ORDER BY rank LIMIT ?
                """,
                (sanitized, f"%{source}%", limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT chunks_trigram.title, chunks_trigram.content, chunks_trigram.content_type,
                       sources.label, bm25(chunks_trigram, 2.0, 1.0, 0, 0) AS rank,
                       highlight(chunks_trigram, 1, char(2), char(3)) AS highlighted
                FROM chunks_trigram
                JOIN sources ON sources.id = chunks_trigram.source_id
                WHERE chunks_trigram MATCH ? ORDER BY rank LIMIT ?
                """,
                (sanitized, limit),
            ).fetchall()
        return [
            SearchResult(
                title=r[0] or "",
                content=r[1] or "",
                source=r[3] or "",
                rank=r[4] or 0.0,
                content_type=r[2] or "prose",
                highlighted=r[5] if len(r) > 5 else None,
            )
            for r in rows
        ]

    def _fuzzy_correct(self, query: str) -> Optional[str]:
        """Levenshtein-based correction from vocabulary."""
        word = query.lower().strip()
        if len(word) < 3:
            return None
        max_dist = _max_edit_distance(len(word))
        rows = self._conn.execute(
            "SELECT word FROM vocabulary WHERE length(word) BETWEEN ? AND ?",
            (len(word) - max_dist, len(word) + max_dist),
        ).fetchall()
        best_word: Optional[str] = None
        best_dist = max_dist + 1
        for (candidate,) in rows:
            if candidate == word:
                return None
            d = _levenshtein(word, candidate)
            if d < best_dist:
                best_dist = d
                best_word = candidate
        return best_word if best_dist <= max_dist else None

    def search_with_fallback(
        self, query: str, limit: int = 3, source: Optional[str] = None
    ) -> List[SearchResult]:
        """Porter -> Trigram -> Fuzzy correction."""
        results = self.search(query, limit, source)
        if results:
            for r in results:
                r.match_layer = "porter"
            return results
        results = self.search_trigram(query, limit, source)
        if results:
            for r in results:
                r.match_layer = "trigram"
            return results
        words = [w for w in query.lower().strip().split() if len(w) >= 3]
        original = " ".join(words)
        corrected = " ".join(self._fuzzy_correct(w) or w for w in words)
        if corrected != original:
            results = self.search(corrected, limit, source)
            if results:
                for r in results:
                    r.match_layer = "fuzzy"
                return results
            results = self.search_trigram(corrected, limit, source)
            if results:
                for r in results:
                    r.match_layer = "fuzzy"
                return results
        return []

    def list_sources(self) -> List[Dict[str, Any]]:
        """List indexed sources."""
        rows = self._conn.execute(
            "SELECT label, chunk_count FROM sources ORDER BY id DESC"
        ).fetchall()
        return [{"label": r[0], "chunkCount": r[1]} for r in rows]

    def get_chunks_by_source(self, source_id: int) -> List[SearchResult]:
        """Get all chunks for a source (no FTS5 MATCH)."""
        rows = self._conn.execute(
            """
            SELECT c.title, c.content, c.content_type, s.label
            FROM chunks c
            JOIN sources s ON s.id = c.source_id
            WHERE c.source_id = ?
            ORDER BY c.rowid
            """,
            (source_id,),
        ).fetchall()
        return [
            SearchResult(
                title=r[0] or "",
                content=r[1] or "",
                source=r[3] or "",
                rank=0.0,
                content_type=r[2] or "prose",
            )
            for r in rows
        ]

    def get_distinctive_terms(self, source_id: int, max_terms: int = 40) -> List[str]:
        """Distinctive terms for the LLM (searchable vocabulary hints)."""
        row = self._conn.execute(
            "SELECT chunk_count FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        if not row or row[0] < 3:
            return []
        total_chunks = row[0]
        min_app, max_app = 2, max(3, int(total_chunks * 0.4))
        doc_freq: Dict[str, int] = {}
        for (content,) in self._conn.execute(
            "SELECT content FROM chunks WHERE source_id = ?", (source_id,)
        ):
            words = set(
                w for w in re.findall(r"[\w\-_]+", content.lower())
                if len(w) >= 3 and w not in STOPWORDS
            )
            for w in words:
                doc_freq[w] = doc_freq.get(w, 0) + 1
        filtered = [
            (word, count)
            for word, count in doc_freq.items()
            if min_app <= count <= max_app
        ]

        def score(item: Tuple[str, int]) -> float:
            word, count = item
            idf = math.log(total_chunks / count) if count else 0.0
            len_bonus = min(len(word) / 20, 0.5)
            ident_bonus = 1.5 if "_" in word else (0.8 if len(word) >= 12 else 0)
            return idf + len_bonus + ident_bonus

        filtered.sort(key=score, reverse=True)
        return [w for w, _ in filtered[:max_terms]]

    def _extract_and_store_vocabulary(self, content: str) -> None:
        words = re.findall(r"[\w\-_]+", content.lower())
        words = [w for w in words if len(w) >= 3 and w not in STOPWORDS]
        seen = set(words)
        self._conn.executemany(
            "INSERT OR IGNORE INTO vocabulary (word) VALUES (?)",
            [(w,) for w in seen],
        )
        self._conn.commit()

    def _chunk_markdown(self, text: str) -> List[Tuple[str, str, bool]]:
        """(title, content, has_code)."""
        chunks: List[Tuple[str, str, bool]] = []
        lines = text.split("\n")
        heading_stack: List[Tuple[int, str]] = []
        current_content: List[str] = []
        current_heading = ""

        def flush() -> None:
            joined = "\n".join(current_content).strip()
            if not joined:
                return
            title = self._build_title(heading_stack, current_heading)
            has_code = any(re.match(r"^`{3,}", line) for line in current_content)
            chunks.append((title, joined, has_code))
            current_content.clear()

        i = 0
        while i < len(lines):
            line = lines[i]
            if re.match(r"^[-_*]{3,}\s*$", line):
                flush()
                i += 1
                continue
            hm = re.match(r"^(#{1,4})\s+(.+)$", line)
            if hm:
                flush()
                level, heading = len(hm.group(1)), hm.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading))
                current_heading = heading
                current_content.append(line)
                i += 1
                continue
            cm = re.match(r"^(`{3,})(.*)?$", line)
            if cm:
                fence = cm.group(1)
                code_lines = [line]
                i += 1
                while i < len(lines):
                    code_lines.append(lines[i])
                    if lines[i].strip() == fence:
                        i += 1
                        break
                    i += 1
                current_content.extend(code_lines)
                continue
            current_content.append(line)
            i += 1
        flush()
        return chunks

    def _chunk_plain_text(
        self, text: str, lines_per_chunk: int
    ) -> List[Tuple[str, str]]:
        sections = re.split(r"\n\s*\n", text)
        if (
            3 <= len(sections) <= 200
            and all(len(s.encode("utf-8")) < 5000 for s in sections)
        ):
            out = []
            for i, sec in enumerate(sections):
                trimmed = sec.strip()
                if not trimmed:
                    continue
                first = trimmed.split("\n")[0][:80]
                out.append((first or f"Section {i+1}", trimmed))
            return out
        lines = text.split("\n")
        if len(lines) <= lines_per_chunk:
            return [("Output", text)]
        overlap = 2
        step = max(lines_per_chunk - overlap, 1)
        out = []
        for i in range(0, len(lines), step):
            slice_lines = lines[i : i + lines_per_chunk]
            if not slice_lines:
                break
            first = (slice_lines[0].strip()[:80]) if slice_lines else f"Lines {i+1}-{i+len(slice_lines)}"
            out.append((first or f"Lines {i+1}-{i+len(slice_lines)}", "\n".join(slice_lines)))
        return out

    @staticmethod
    def _build_title(heading_stack: List[Tuple[int, str]], current_heading: str) -> str:
        if not heading_stack:
            return current_heading or "Untitled"
        return " > ".join(h[1] for h in heading_stack)
