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


def test_recombine_does_not_mutate_shared_source_topologies():
    """#925: _interleave must copy nodes, not alias the module-level source
    topologies -- otherwise mutating a hybrid's nodes corrupts every other
    extractor instance's view of that source domain."""
    extractor = CrossDomainIsomorphismExtractor()
    original_snapshot = [
        (node.role, list(node.inputs), list(node.outputs))
        for node in extractor.topologies["compiler_passes"]
    ]

    hybrid = extractor.recombine("compiler_passes", "immune_system")
    assert hybrid is not None
    for node in hybrid.topology:
        node.inputs.append("mutated-in-place")
        node.outputs.append("mutated-in-place")

    after_snapshot = [
        (node.role, list(node.inputs), list(node.outputs))
        for node in extractor.topologies["compiler_passes"]
    ]
    assert after_snapshot == original_snapshot


def test_mutate_drop_role_rewires_dangling_inputs():
    """#925: dropping a role must not leave a downstream node requiring an
    input that nothing in the mutated loop still produces."""
    extractor = CrossDomainIsomorphismExtractor()
    mechanism = extractor.topologies["compiler_passes"]

    optimizer = next(n for n in mechanism if n.role == "optimizer")
    emitter = next(n for n in mechanism if n.role == "emitter")
    assert any(out in emitter.inputs for out in optimizer.outputs), (
        "test assumes emitter depends on an optimizer output in the current fixture"
    )

    mutated = extractor.mutate(mechanism, drop_role="optimizer")

    available_outputs = {out for n in mutated for out in n.outputs}
    for node in mutated:
        for dep in node.inputs:
            assert dep in available_outputs, (
                f"{node.role} still requires '{dep}' but no remaining node produces it"
            )
