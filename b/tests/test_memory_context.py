import math
import os
import tempfile

from src.core.memory.semantic_memory import (
    SemanticMemory,
    calculate_cosine_similarity,
    generate_pseudo_embedding,
)


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def test_embeddings_of_synonym_pairs_are_closer_than_unrelated_pairs():
    """Near-synonym sentences should be more similar than unrelated sentences."""
    synonyms = [
        ("the car is fast", "the automobile is quick"),
        ("a dog barks loudly", "a puppy makes noise"),
        ("the cat sleeps on the sofa", "the kitten rests on the couch"),
    ]
    unrelated = [
        ("the car is fast", "quantum mechanics is hard"),
        ("a dog barks loudly", "the compiler failed to link"),
        ("the cat sleeps on the sofa", "interest rates rose today"),
    ]

    for syn, un in zip(synonyms, unrelated):
        syn_sim = calculate_cosine_similarity(
            generate_pseudo_embedding(syn[0]),
            generate_pseudo_embedding(syn[1]),
        )
        un_sim = calculate_cosine_similarity(
            generate_pseudo_embedding(un[0]),
            generate_pseudo_embedding(un[1]),
        )
        assert syn_sim > un_sim, (
            f"synonym pair {syn!r} ({syn_sim:.3f}) should be more similar than "
            f"unrelated pair {un!r} ({un_sim:.3f})"
        )


def test_semantic_memory_retrieves_related_text_first():
    """SemanticMemory.search_memory should rank related memories above unrelated ones."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "semantic_memory.db")
        mem = SemanticMemory(db_path=db_path)
        mem.add_memory("k1", "The autonomous vehicle navigated city streets.", {"src": 1})
        mem.add_memory("k2", "The recipe called for flour and eggs.", {"src": 2})
        mem.add_memory("k3", "The stock market closed at a record high.", {"src": 3})

        results = mem.search_memory("self driving car on urban roads", limit=3)
        assert results, "expected at least one result"
        top = results[0]
        assert top["key"] == "k1", f"expected k1 first, got {top['key']}"
        assert top["score"] > 0.0


def test_embedding_is_unit_vector():
    vec = generate_pseudo_embedding("a meaningful sentence about cars and automobiles")
    assert abs(_norm(vec) - 1.0) < 1e-6
