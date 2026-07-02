import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from src.core.memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)

# Basic regex rules to extract entity relations (triples) from sentences
# Matches patterns like "X is/calls/registers/imports/manages Y"
RELATION_PATTERNS = [
    r"\b([a-zA-Z0-9_\-\.]+)\s+(registers|imports|calls|manages|sets|creates|triggers|initiates|updates)\s+(?:the\s+|a\s+)?([a-zA-Z0-9_\-\.\s]+?)(?:\.|\s|$)",
    r"\b([a-zA-Z0-9_\-\.]+)\s+(is(?:\s+a)?)\s+(?:the\s+|a\s+)?([a-zA-Z0-9_\-\.\s]+?)(?:\.|\s|$)"
]


class HistoryAnalyzer:
    """Extracts entity knowledge graphs from session context, querying adjacencies and searching semantically."""

    def __init__(
        self, 
        memory: Optional[SemanticMemory] = None, 
        graph_path: str = "data/knowledge_graph.json"
    ):
        self.memory = memory or SemanticMemory()
        self.graph_path = graph_path
        
        # In-memory graph structure
        self.nodes: Set[str] = set()
        self.edges: List[Dict[str, str]] = []
        
        self._load_graph()

    def _load_graph(self):
        """Load the knowledge graph from disk if it exists."""
        if not os.path.exists(self.graph_path):
            return

        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.nodes = set(data.get("nodes", []))
                self.edges = data.get("edges", [])
        except Exception as e:
            logger.error(f"HistoryAnalyzer: Failed to load knowledge graph: {e}")

    def _save_graph(self):
        """Save the current knowledge graph to disk."""
        try:
            graph_dir = os.path.dirname(self.graph_path)
            if graph_dir:
                os.makedirs(graph_dir, exist_ok=True)
                
            data = {
                "nodes": list(self.nodes),
                "edges": self.edges
            }
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"HistoryAnalyzer: Failed to save knowledge graph: {e}")

    def extract_entities_and_relations(self, text: str) -> List[Dict[str, str]]:
        """Parse entity-relation triples from conversational texts or system logs."""
        triples = []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        
        for sentence in sentences:
            sentence_clean = sentence.strip()
            for pattern in RELATION_PATTERNS:
                matches = re.findall(pattern, sentence_clean, re.IGNORECASE)
                for subject, relation, obj in matches:
                    subj_clean = subject.strip()
                    rel_clean = relation.strip().lower()
                    obj_clean = obj.strip()
                    
                    triple = {
                        "subject": subj_clean,
                        "relation": rel_clean,
                        "object": obj_clean
                    }
                    triples.append(triple)
                    
                    # Update knowledge graph
                    self.nodes.add(subj_clean)
                    self.nodes.add(obj_clean)
                    
                    # Add edge if not already present
                    edge_exists = any(
                        e["source"] == subj_clean and e["target"] == obj_clean and e["relation"] == rel_clean
                        for e in self.edges
                    )
                    if not edge_exists:
                        self.edges.append({
                            "source": subj_clean,
                            "target": obj_clean,
                            "relation": rel_clean
                        })
                        
        if triples:
            self._save_graph()
            
        return triples

    def get_related_nodes(self, node: str) -> List[str]:
        """Navigate adjacency edges of the knowledge graph and return related nodes."""
        node_clean = node.strip()
        related = set()
        
        for edge in self.edges:
            if edge["source"].lower() == node_clean.lower():
                related.add(edge["target"])
            elif edge["target"].lower() == node_clean.lower():
                related.add(edge["source"])
                
        return list(related)

    def add_session_history(self, session_id: str, utterance: str, role: str = "user") -> bool:
        """Store conversational utterance in semantic memory, indexing it under the session key."""
        metadata = {"session_id": session_id, "role": role}
        ok = self.memory.add_memory(key=f"session_log_{session_id}", text=utterance, metadata=metadata)
        
        # Dynamically extract graph triples from history
        self.extract_entities_and_relations(utterance)
        return ok

    def search_session_history(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search past conversation items semantically."""
        return self.memory.search_memory(query, limit=limit)
