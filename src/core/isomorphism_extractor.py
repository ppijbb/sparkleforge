"""Cross-Domain Isomorphism Extractor Engine (issue #922).

Extracts abstract operational topologies from non-obvious domains (biological
immune systems, compiler passes, economic equilibrium, etc.) and maps them onto
open engineering and research problems. Also supports algorithmic mutation and
recombination of distinct source mechanisms into hybrid control loops (issue #925).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TopologyNode:
    """A single role within an abstract operational topology."""

    role: str
    responsibility: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass
class Isomorphism:
    """A mapping from a source-domain topology onto a target problem."""

    source_domain: str
    target_domain: str
    topology: List[TopologyNode]
    mapping: Dict[str, str]
    rationale: str
    confidence: float = 0.0

    def control_loop(self) -> List[str]:
        """Return the ordered roles that form this topology's control loop."""
        return [node.role for node in self.topology]

    def signature(self) -> Tuple[str, ...]:
        return tuple(self.control_loop())

_SOURCE_TOPOLOGIES: Dict[str, List[TopologyNode]] = {
    "immune_system": [
        TopologyNode(
            role="detector",
            responsibility="recognize non-self patterns",
            inputs=["raw_signals"],
            outputs=["threat_candidates"],
        ),
        TopologyNode(
            role="effector",
            responsibility="neutralize confirmed threats",
            inputs=["threat_candidates"],
            outputs=["neutralized_threats"],
        ),
        TopologyNode(
            role="memory",
            responsibility="retain learned signatures for fast recall",
            inputs=["neutralized_threats"],
            outputs=["memory_signatures"],
        ),
    ],
    "compiler_passes": [
        TopologyNode(
            role="parser",
            responsibility="convert raw input into structured IR",
            inputs=["source_text"],
            outputs=["intermediate_representation"],
        ),
        TopologyNode(
            role="optimizer",
            responsibility="transform IR toward a goal function",
            inputs=["intermediate_representation"],
            outputs=["optimized_ir"],
        ),
        TopologyNode(
            role="emitter",
            responsibility="materialize optimized IR into target artifacts",
            inputs=["optimized_ir"],
            outputs=["target_artifacts"],
        ),
    ],
    "economic_equilibrium": [
        TopologyNode(
            role="supplier",
            responsibility="produce resources in response to demand signals",
            inputs=["demand_signals"],
            outputs=["supply"],
        ),
        TopologyNode(
            role="consumer",
            responsibility="consume resources and emit demand signals",
            inputs=["supply"],
            outputs=["demand_signals"],
        ),
        TopologyNode(
            role="market",
            responsibility="clear prices to balance supply and demand",
            inputs=["supply", "demand_signals"],
            outputs=["equilibrium_prices"],
        ),
    ],
}


@dataclass
class HybridMechanism:
    """A hybrid architectural pattern produced by recombining two mechanisms."""

    name: str
    parent_a: str
    parent_b: str
    topology: List[TopologyNode]
    recombination: str
    rationale: str
    confidence: float = 0.0

    def control_loop(self) -> List[str]:
        return [node.role for node in self.topology]

    def signature(self) -> Tuple[str, ...]:
        return tuple(self.control_loop())


def _copy_node(node: TopologyNode) -> TopologyNode:
    return TopologyNode(
        role=node.role,
        responsibility=node.responsibility,
        inputs=list(node.inputs),
        outputs=list(node.outputs),
    )


def _interleave(left: List[TopologyNode], right: List[TopologyNode]) -> List[TopologyNode]:
    """Interleave two control loops into a single hybrid loop.

    Copies each node rather than reusing the ones from `self.topologies` --
    those are the shared, module-level source topologies every extractor
    instance reads from, so mutating a returned hybrid's nodes in place
    would corrupt them for every other caller.
    """
    hybrid: List[TopologyNode] = []
    for idx in range(max(len(left), len(right))):
        if idx < len(left):
            hybrid.append(_copy_node(left[idx]))
        if idx < len(right):
            hybrid.append(_copy_node(right[idx]))
    return hybrid


_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "immune_system": ["security", "threat", "anomaly", "intrusion", "malware", "vulnerability"],
    "compiler_passes": ["compile", "build", "transpile", "codegen", "optimization pass", "ir"],
    "economic_equilibrium": ["market", "pricing", "supply", "demand", "equilibrium", "allocation"],
}


@dataclass
class PrimitiveAxiom:
    """A fundamental computational or physical primitive axiom."""

    name: str
    category: str
    description: str
    constraints: List[str] = field(default_factory=list)


_PRIMITIVE_AXIOMS: List[PrimitiveAxiom] = [
    PrimitiveAxiom(
        name="information",
        category="computational",
        description="A distinguishable state that reduces uncertainty.",
        constraints=["finite representation", "lossless under transformation"],
    ),
    PrimitiveAxiom(
        name="computation",
        category="computational",
        description="A deterministic or stochastic mapping between states.",
        constraints=["terminates in bounded steps", "preserves invariants"],
    ),
    PrimitiveAxiom(
        name="energy",
        category="physical",
        description="The capacity to perform work or induce state change.",
        constraints=["conserved in closed systems", "dissipated under irreversible operations"],
    ),
    PrimitiveAxiom(
        name="entropy",
        category="physical",
        description="A measure of disorder or uncertainty in a system.",
        constraints=["non-decreasing in isolated systems", "reducible via measurement"],
    ),
]


class CrossDomainIsomorphismExtractor:
    """Extract abstract topologies and map them onto a target problem."""

    def __init__(self, topologies: Optional[Dict[str, List[TopologyNode]]] = None) -> None:
        self.topologies = topologies or _SOURCE_TOPOLOGIES

    def _score_domain(self, request: str, domain: str) -> int:
        keywords = _DOMAIN_KEYWORDS.get(domain, [])
        text = request.lower()
        return sum(text.count(keyword) for keyword in keywords)

    def _target_topology(self, request: str) -> List[TopologyNode]:
        """Derive a coarse target-problem topology from the request text."""
        steps: List[str] = re.findall(r"(?:^|\n)\s*(?:\d+\.|[-*])\s+(.+)", request)
        if not steps:
            steps = [phrase.strip() for phrase in re.split(r"[.;]\s+", request) if phrase.strip()]
        nodes: List[TopologyNode] = []
        for idx, step in enumerate(steps[:6]):
            nodes.append(
                TopologyNode(
                    role=f"step_{idx + 1}",
                    responsibility=step[:160],
                    inputs=[] if idx == 0 else [f"step_{idx}"],
                    outputs=[] if idx == len(steps[:6]) - 1 else [f"step_{idx + 2}"],
                )
            )
        if not nodes:
            nodes = [
                TopologyNode(
                    role="problem",
                    responsibility=request[:160] or "unspecified problem",
                    inputs=["request"],
                    outputs=["solution"],
                )
            ]
        return nodes

    def extract(self, request: str) -> List[Isomorphism]:
        """Return ranked isomorphisms between source domains and the request."""
        if not request:
            return []
        target = self._target_topology(request)
        scored = sorted(
            ((self._score_domain(request, domain), domain) for domain in self.topologies),
            key=lambda item: item[0],
            reverse=True,
        )
        results: List[Isomorphism] = []
        for score, domain in scored:
            if score <= 0:
                continue
            source = self.topologies[domain]
            mapping = {
                source_node.role: target_node.role
                for source_node, target_node in zip(source, target)
            }
            results.append(
                Isomorphism(
                    source_domain=domain,
                    target_domain="request",
                    topology=source,
                    mapping=mapping,
                    rationale=(
                        f"Request references {domain} signals ({score} keyword hits); "
 f"mapping {len(mapping)} source roles onto target steps."
                    ),
                    confidence=min(1.0, score / 10.0),
                )
            )
        return results

    def _decompose_node(
        self,
        statement: str,
        depth: int,
        max_depth: int,
        visited: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Recursively decompose a statement toward primitive axioms."""
        if visited is None:
            visited = set()
        statement_key = statement.strip().lower()
        if statement_key in visited:
            return {
                "statement": statement,
                "primitive": None,
                "rationale": "Cycle detected; stopping recursion.",
                "children": [],
                "depth": depth,
            }
        visited.add(statement_key)

        # A compound statement must be split before it's tested against a
        # primitive axiom -- otherwise a statement that merely *mentions* a
        # primitive in passing (e.g. "Allocate energy across computation and
        # storage") is accepted as a terminal "energy" leaf and its other
        # sub-problems (computation, storage) are never explored. Only an
        # atomic (unsplittable) clause is eligible to match directly.
        if depth < max_depth:
            sub_problems = self._split_statement(statement)
            if sub_problems:
                children = [
                    self._decompose_node(sub, depth + 1, max_depth, visited.copy())
                    for sub in sub_problems
                ]
                return {
                    "statement": statement,
                    "primitive": None,
                    "rationale": "Decomposed into sub-problems.",
                    "children": children,
                    "depth": depth,
                }

        matched = self._match_primitive(statement)
        return {
            "statement": statement,
            "primitive": matched,
            "rationale": (
                "Matched a primitive axiom."
                if matched is not None
                else "Reached maximum decomposition depth without a primitive match."
                if depth >= max_depth
                else "Atomic statement with no primitive match."
            ),
            "children": [],
            "depth": depth,
        }

    def _match_primitive(self, statement: str) -> Optional[PrimitiveAxiom]:
        """Return the first primitive axiom referenced by the statement."""
        text = statement.lower()
        for axiom in _PRIMITIVE_AXIOMS:
            if re.search(rf"\b{re.escape(axiom.name)}\b", text) or re.search(
                rf"\b{re.escape(axiom.category)}\b", text
            ):
                return axiom
        return None

    def _split_statement(self, statement: str) -> List[str]:
        """Split a statement into candidate sub-problems.

        Returns an empty list when the statement is atomic: no delimiter
        found, and too short to halve into two distinct sub-problems. The
        caller relies on that to recognize a statement it cannot decompose
        any further, rather than recursing into an unchanged copy of the
        same text.
        """
        clauses = re.split(r"\s*(?:;|,|\.|\band\b|\bor\b|->|=>|therefore)\s*", statement)
        sub_problems = [clause.strip() for clause in clauses if clause.strip()]
        if len(sub_problems) > 1:
            return sub_problems[:4]
        tokens = statement.split()
        if len(tokens) > 4:
            mid = len(tokens) // 2
            return [" ".join(tokens[:mid]), " ".join(tokens[mid:])]
        return []

    def decompose(self, request: str, max_depth: int = 4) -> Dict[str, Any]:
        """Recursively decompose an open-ended challenge to primitive axioms."""
        if not request:
            return {
                "statement": "",
                "primitive": None,
                "rationale": "Empty request.",
                "children": [],
                "depth": 0,
            }
        return self._decompose_node(request, 0, max_depth)

    def mutate(
        self,
        mechanism: List[TopologyNode],
        *,
        insert: Optional[TopologyNode] = None,
        drop_role: Optional[str] = None,
        swap_roles: Optional[Tuple[str, str]] = None,
    ) -> List[TopologyNode]:
        """Return a mutated copy of a control loop with a single edit applied."""
        mutated = [_copy_node(n) for n in mechanism]
        if drop_role is not None:
            mutated = [n for n in mutated if n.role != drop_role]
            # Dropping a role can strand a downstream node's input on an
            # output nothing left in the loop produces (e.g. dropping
            # "optimizer" leaves "emitter" still requiring "optimized_ir").
            # Reconcile against what the mutated loop can actually produce
            # rather than silently keeping an unsatisfiable dependency.
            available_outputs = {out for n in mutated for out in n.outputs}
            for n in mutated:
                n.inputs = [i for i in n.inputs if i in available_outputs]
        if swap_roles is not None:
            a, b = swap_roles
            idx_a = next((i for i, n in enumerate(mutated) if n.role == a), None)
            idx_b = next((i for i, n in enumerate(mutated) if n.role == b), None)
            if idx_a is not None and idx_b is not None:
                mutated[idx_a], mutated[idx_b] = mutated[idx_b], mutated[idx_a]
        if insert is not None:
            mutated.append(insert)
        return mutated

    def recombine(
        self,
        domain_a: str,
        domain_b: str,
        *,
        strategy: str = "interleave",
    ) -> Optional[HybridMechanism]:
        """Algorithmically recombine two distinct source mechanisms into a hybrid."""
        if domain_a not in self.topologies or domain_b not in self.topologies:
            return None
        if domain_a == domain_b:
            return None
        left = self.topologies[domain_a]
        right = self.topologies[domain_b]
        if strategy == "interleave":
            hybrid_topology = _interleave(left, right)
            recombination = f"interleave({domain_a}, {domain_b})"
        else:
            return None
        return HybridMechanism(
            name=f"{domain_a}::{domain_b}",
            parent_a=domain_a,
            parent_b=domain_b,
            topology=hybrid_topology,
            recombination=recombination,
            rationale=(
                f"Recombined {len(left)} {domain_a} roles with {len(right)} {domain_b} roles "
                f"using {strategy} to yield a {len(hybrid_topology)}-step hybrid control loop."
            ),
            confidence=min(1.0, (len(left) + len(right)) / 12.0),
        )

    def to_dict(self, isomorphisms: List[Isomorphism]) -> List[Dict[str, Any]]:
        return [
            {
                "source_domain": iso.source_domain,
                "target_domain": iso.target_domain,
                "topology": [
                    {
                        "role": node.role,
                        "responsibility": node.responsibility,
                        "inputs": node.inputs,
                        "outputs": node.outputs,
                    }
                    for node in iso.topology
                ],
                "mapping": iso.mapping,
                "rationale": iso.rationale,
                "confidence": iso.confidence,
            }
            for iso in isomorphisms
        ]

    def hybrid_to_dict(self, hybrid: HybridMechanism) -> Dict[str, Any]:
        return {
            "name": hybrid.name,
            "parent_a": hybrid.parent_a,
            "parent_b": hybrid.parent_b,
            "topology": [
                {
                    "role": node.role,
                    "responsibility": node.responsibility,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                }
                for node in hybrid.topology
            ],
            "recombination": hybrid.recombination,
            "rationale": hybrid.rationale,
            "confidence": hybrid.confidence,
        }
