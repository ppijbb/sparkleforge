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


def _leaf_nodes(node):
    if not node["children"]:
        return [node]
    leaves = []
    for child in node["children"]:
        leaves.extend(_leaf_nodes(child))
    return leaves


def test_decompose_empty_request_returns_empty_statement():
    extractor = CrossDomainIsomorphismExtractor()
    result = extractor.decompose("")
    assert result["statement"] == ""
    assert result["primitive"] is None
    assert result["children"] == []


def test_decompose_matches_primitive_axiom_in_an_atomic_clause():
    extractor = CrossDomainIsomorphismExtractor()
    # A compound statement ("and") is split first, so the primitive match
    # shows up on one of its (atomic) children, not necessarily the root.
    result = extractor.decompose("Model information and entropy in a closed system.")
    matched_names = {leaf["primitive"].name for leaf in _leaf_nodes(result) if leaf["primitive"]}
    assert matched_names & {"information", "entropy"}


def test_decompose_recurses_into_sub_problems():
    extractor = CrossDomainIsomorphismExtractor()
    result = extractor.decompose("Allocate energy across computation and storage.")
    assert result["children"]
    assert any(child["primitive"] is not None for child in result["children"])


def test_decompose_does_not_accept_a_compound_statement_as_a_single_primitive():
    """#940: a statement mentioning a primitive in passing (e.g. "energy") must
    still be split into its other sub-problems, not accepted as one terminal
    "energy" leaf that discards "computation" and "storage"."""
    extractor = CrossDomainIsomorphismExtractor()
    result = extractor.decompose("Allocate energy across computation and storage.")
    assert result["primitive"] is None
    assert result["rationale"] == "Decomposed into sub-problems."


def test_decompose_respects_max_depth():
    extractor = CrossDomainIsomorphismExtractor()
    result = extractor.decompose("A very abstract open-ended challenge with no primitives.", max_depth=2)
    assert result["depth"] <= 2
    assert all(leaf["depth"] <= 2 for leaf in _leaf_nodes(result))


def test_decompose_does_not_loop_forever_on_repeated_words():
    extractor = CrossDomainIsomorphismExtractor()
    result = extractor.decompose("information information information information information")
    for leaf in _leaf_nodes(result):
        assert leaf["primitive"] is not None or leaf["rationale"] in (
            "Cycle detected; stopping recursion.",
            "Atomic statement with no primitive match.",
            "Reached maximum decomposition depth without a primitive match.",
        )


def test_decompose_matches_whole_words_not_substrings():
    """#940: "energy" must not match inside unrelated words like "synergy"."""
    extractor = CrossDomainIsomorphismExtractor()
    result = extractor.decompose("Improve synergy between teams.")
    assert all(leaf["primitive"] is None for leaf in _leaf_nodes(result))
