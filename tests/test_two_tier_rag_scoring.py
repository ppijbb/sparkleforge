"""#1548: semantic (embedding) retrieval and time-decay scoring in two_tier_rag.

MemoryEntry.embedding used to be a dead field (never populated or read) and
created_at/last_accessed were never used in any scoring path. These tests
cover the fix: embeddings populated on add(), blended into both tiers'
search scores, staleness decay applied to importance, and full backward
compatibility when no embedding provider is available.
"""

from datetime import datetime, timedelta

import pytest

from src.core.two_tier_rag import (
    MemoryType,
    STALENESS_HALF_LIFE_DAYS,
    STALENESS_THRESHOLD,
    TierOneCache,
    TierTwoStore,
    TwoTierRAGSystem,
    _cosine_similarity,
    _time_decay_factor,
)


class _FakeEmbeddingProvider:
    """Maps exact input text to a pre-set vector; unknown text -> orthogonal."""

    def __init__(self, vectors: dict):
        self.vectors = vectors

    @property
    def is_available(self) -> bool:
        return True

    async def embed_documents(self, texts):
        import numpy as np

        return np.array(
            [self.vectors.get(t, [0.0, 1.0]) for t in texts], dtype=float
        )


class _UnavailableEmbeddingProvider:
    @property
    def is_available(self) -> bool:
        return False

    async def embed_documents(self, texts):  # pragma: no cover - must never be called
        raise AssertionError("embed_documents should not be called when unavailable")


def _entry(**overrides):
    from src.core.two_tier_rag import MemoryEntry

    defaults = dict(
        entry_id=overrides.pop("entry_id", "e1"),
        memory_type=MemoryType.FACT,
        content="content",
        keywords=["content"],
        importance=0.8,
    )
    defaults.update(overrides)
    return MemoryEntry(**defaults)


def test_cosine_similarity_identical_vectors_is_one():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_negative_is_floored_at_zero():
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0


def test_cosine_similarity_mismatched_or_empty_is_zero():
    assert _cosine_similarity([], [1.0]) == 0.0
    assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0


def test_time_decay_factor_is_one_when_just_created():
    entry = _entry()
    assert _time_decay_factor(entry) == pytest.approx(1.0, abs=1e-6)


def test_time_decay_factor_halves_at_one_half_life():
    entry = _entry(created_at=datetime.now() - timedelta(days=STALENESS_HALF_LIFE_DAYS))
    assert _time_decay_factor(entry) == pytest.approx(0.5, abs=0.01)


def test_time_decay_factor_uses_last_accessed_over_created_at():
    entry = _entry(
        created_at=datetime.now() - timedelta(days=1000),
        last_accessed=datetime.now(),
    )
    assert _time_decay_factor(entry) == pytest.approx(1.0, abs=1e-6)


def test_add_leaves_embedding_none_when_provider_disabled():
    rag = TwoTierRAGSystem(embedding_provider=False)
    entry = rag.add("hello world", MemoryType.FACT)
    assert entry.embedding is None


def test_add_leaves_embedding_none_when_provider_unavailable():
    rag = TwoTierRAGSystem(embedding_provider=_UnavailableEmbeddingProvider())
    entry = rag.add("hello world", MemoryType.FACT)
    assert entry.embedding is None


def test_add_populates_embedding_when_provider_available():
    provider = _FakeEmbeddingProvider({"hello world": [1.0, 0.0]})
    rag = TwoTierRAGSystem(embedding_provider=provider)

    entry = rag.add("hello world", MemoryType.FACT)

    assert entry.embedding == [1.0, 0.0]


def test_tier_one_search_finds_semantically_similar_entry_with_no_keyword_overlap():
    """The core bug: a fact phrased differently than the query keywords used
    to be invisible to search even when it's the most relevant one stored."""
    cache = TierOneCache(max_entries=10)
    cat_entry = _entry(
        entry_id="cat",
        content="The feline was resting on the mat",
        keywords=["feline", "resting", "mat"],
        embedding=[1.0, 0.0],
    )
    unrelated_entry = _entry(
        entry_id="unrelated",
        content="Stock market futures fell overnight",
        keywords=["stock", "market", "futures"],
        embedding=[0.0, 1.0],
    )
    cache.add(cat_entry)
    cache.add(unrelated_entry)

    # Query keywords ("where", "cat", "sleeping") share nothing with either
    # entry's keywords -- pure keyword search would return zero results.
    results = cache.search(
        ["where", "cat", "sleeping"], query_embedding=[1.0, 0.0]
    )

    assert len(results) == 1
    assert results[0][0].entry_id == "cat"


def test_tier_one_search_still_works_keyword_only_without_embedding():
    cache = TierOneCache(max_entries=10)
    cache.add(_entry(entry_id="e1", keywords=["python", "programming"]))

    results = cache.search(["python"])

    assert len(results) == 1
    assert results[0][0].entry_id == "e1"


def test_tier_two_search_blends_embedding_similarity_for_keyword_candidates():
    store = TierTwoStore()
    close_entry = _entry(
        entry_id="close",
        content="machine learning basics",
        keywords=["machine", "learning", "basics"],
        embedding=[1.0, 0.0],
        importance=0.5,
    )
    far_entry = _entry(
        entry_id="far",
        content="machine repair shop",
        keywords=["machine", "repair", "shop"],
        embedding=[0.0, 1.0],
        importance=0.5,
    )
    store.add(close_entry)
    store.add(far_entry)

    # Both share the "machine" keyword (equal keyword_score), but only
    # "close" is embedding-similar to the query.
    results = store.search(["machine"], query_embedding=[1.0, 0.0])

    assert results[0][0].entry_id == "close"


def test_tier_two_search_applies_time_decay_to_importance():
    store = TierTwoStore()
    fresh = _entry(entry_id="fresh", keywords=["topic"], importance=0.5)
    stale = _entry(
        entry_id="stale",
        keywords=["topic"],
        importance=0.9,
        created_at=datetime.now() - timedelta(days=STALENESS_HALF_LIFE_DAYS * 10),
    )
    store.add(fresh)
    store.add(stale)

    results = store.search(["topic"])
    scores = {entry.entry_id: score for entry, score in results}

    # Despite lower static importance, "fresh" should out-rank "stale" once
    # decay has all but zeroed the older entry's importance contribution.
    assert scores["fresh"] > scores["stale"]


def test_get_stale_entries_flags_old_untouched_facts():
    store = TierTwoStore()
    old = _entry(
        entry_id="old",
        importance=0.9,
        created_at=datetime.now() - timedelta(days=STALENESS_HALF_LIFE_DAYS * 10),
    )
    recent = _entry(entry_id="recent", importance=0.9)
    store.add(old)
    store.add(recent)

    stale_ids = {e.entry_id for e in store.get_stale_entries()}

    assert stale_ids == {"old"}


def test_get_stale_entries_ranks_by_original_importance():
    store = TierTwoStore()
    long_ago = datetime.now() - timedelta(days=STALENESS_HALF_LIFE_DAYS * 10)
    high = _entry(entry_id="high", importance=0.9, created_at=long_ago)
    low = _entry(entry_id="low", importance=0.4, created_at=long_ago)
    store.add(low)
    store.add(high)

    stale = store.get_stale_entries()

    assert [e.entry_id for e in stale] == ["high", "low"]


def test_two_tier_system_get_stale_entries_delegates_to_tier2():
    rag = TwoTierRAGSystem(embedding_provider=False)
    rag.tier2.add(
        _entry(
            entry_id="old",
            importance=0.9,
            created_at=datetime.now() - timedelta(days=STALENESS_HALF_LIFE_DAYS * 10),
        )
    )

    stale = rag.get_stale_entries()

    assert [e.entry_id for e in stale] == ["old"]


def test_tier_one_load_orders_by_decayed_importance_not_static():
    cache = TierOneCache(max_entries=10)
    stale_high_importance = _entry(
        entry_id="stale",
        importance=0.95,
        created_at=datetime.now() - timedelta(days=STALENESS_HALF_LIFE_DAYS * 10),
    )
    fresh_lower_importance = _entry(entry_id="fresh", importance=0.6)

    cache.load([stale_high_importance, fresh_lower_importance])

    assert cache.entry_order[0] == "fresh"
