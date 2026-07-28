"""Tests for the Cross-Domain Isomorphism Extractor Engine (issue #922)."""

from src.core.isomorphism_extractor import CrossDomainIsomorphismExtractor


def test_extract_returns_empty_for_empty_request():
    extractor = CrossDomainIsomorphismExtractor()
    assert extractor.extract("") == []


def test_extract_maps_immune_system_domain():
    extractor = CrossDomainIsomorphismExtractor()
    request = "Build a security threat detector with anomaly memory and intrusion response."
    results = extractor.extract(request)
    assert results
    assert results[0].source_domain == "immune_system"
    assert "detector" in results[0].mapping


def test_extract_maps_compiler_passes_domain():
    extractor = CrossDomainIsomorphismExtractor()
    request = "Design a compile pipeline with parser, optimizer pass, and codegen emitter."
    results = extractor.extract(request)
    assert any(iso.source_domain == "compiler_passes" for iso in results)


def test_extract_maps_economic_equilibrium_domain():
    extractor = CrossDomainIsomorphismExtractor()
    request = "Model market supply, demand, and equilibrium pricing allocation."
    results = extractor.extract(request)
    assert any(iso.source_domain == "economic_equilibrium" for iso in results)


def test_to_dict_serializes_isomorphisms():
    extractor = CrossDomainIsomorphismExtractor()
    request = "Build a security anomaly detector with threat memory."
    results = extractor.extract(request)
    serialized = extractor.to_dict(results)
    assert serialized
    assert "topology" in serialized[0]
    assert "mapping" in serialized[0]
    assert isinstance(serialized[0]["confidence"], float)


def test_extract_no_match_for_unrelated_request():
    extractor = CrossDomainIsomorphismExtractor()
    request = "Write a poem about the ocean."
    assert extractor.extract(request) == []
