"""#1548: EmbeddingProvider.is_available, so callers (e.g. two_tier_rag.py)
can tell a real embedding from LocalEmbeddingProvider's dummy random-vector
fallback and avoid polluting similarity scoring with noise.
"""

from src.core.memory_embeddings import GeminiEmbeddingProvider, LocalEmbeddingProvider


def test_local_provider_is_available_when_sentence_transformers_installed(monkeypatch):
    provider = LocalEmbeddingProvider()
    monkeypatch.setattr(provider, "_get_model", lambda: object())

    assert provider.is_available is True


def test_local_provider_is_unavailable_in_dummy_fallback_mode(monkeypatch):
    provider = LocalEmbeddingProvider()
    monkeypatch.setattr(provider, "_get_model", lambda: "dummy")

    assert provider.is_available is False


def test_gemini_provider_defaults_to_available():
    provider = GeminiEmbeddingProvider()
    assert provider.is_available is True
