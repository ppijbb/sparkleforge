import os
import tempfile
import pytest
from src.core.bootstrap_graph import BootstrapGraph
from src.core.memory.semantic_memory import SemanticMemory, generate_pseudo_embedding, calculate_cosine_similarity
from src.core.memory.history_analyzer import HistoryAnalyzer
from src.core.memory.context_lane import ContextLane


def test_cosine_similarity():
    v1 = generate_pseudo_embedding("hello world")
    v2 = generate_pseudo_embedding("hello world")
    v3 = generate_pseudo_embedding("completely different text about compiling packages")
    
    sim_same = calculate_cosine_similarity(v1, v2)
    sim_diff = calculate_cosine_similarity(v1, v3)
    
    # Same texts must yield similarity close to 1.0 (unit vector dot product)
    assert abs(sim_same - 1.0) < 1e-5
    # Different texts should have lower similarity score
    assert sim_diff < sim_same


def test_semantic_memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_mem.db")
        mem = SemanticMemory(db_path=db_path)
        
        mem.add_memory("git", "perform git pull from remote origin", {"category": "vcs"})
        mem.add_memory("py", "compile python code for test script", {"category": "build"})
        
        # Search git query
        res = mem.search_memory("git checkout master")
        assert len(res) == 2
        # Git-related text should rank higher due to word matches boosting vector
        assert res[0]["key"] == "git"
        assert res[0]["metadata"]["category"] == "vcs"
        
        mem.clear()
        res_empty = mem.search_memory("git")
        assert len(res_empty) == 0


def test_history_analyzer_knowledge_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_mem.db")
        graph_path = os.path.join(tmpdir, "test_graph.json")
        
        mem = SemanticMemory(db_path=db_path)
        analyzer = HistoryAnalyzer(memory=mem, graph_path=graph_path)
        
        text = "bootstrap_graph.py registers memory_context. Context_lane imports semantic_memory."
        triples = analyzer.extract_entities_and_relations(text)
        
        assert len(triples) >= 2
        
        # Verify first triple
        t1 = triples[0]
        assert t1["subject"].lower() == "bootstrap_graph.py"
        assert t1["relation"] == "registers"
        assert t1["object"].lower() == "memory_context"
        
        # Verify adjacency navigation
        related = analyzer.get_related_nodes("bootstrap_graph.py")
        assert "memory_context" in related

        # Test session logs integration
        analyzer.add_session_history("session_123", "User updates codebase_map", "user")
        related_user = analyzer.get_related_nodes("User")
        assert "codebase_map" in related_user


def test_context_lane_codebase_mapping():
    # We run against the real workspace primary directory to test actual AST parsing
    project_root = "/home/user/workspace/mcp_agent/primary/SparkleForge"
    lane = ContextLane(project_root=project_root)
    
    code_map = lane.update_codebase_map()
    assert len(code_map) > 0
    
    # Check if a custom module like src.core.automation.automation_engine is tracked
    target_mod = "src.core.automation.automation_engine"
    assert target_mod in code_map
    # AutomationEngine imports get_scheduler, EventBus, etc. (starts with src)
    imports = code_map[target_mod]
    assert any(imp.startswith("src.core") for imp in imports)


@pytest.mark.asyncio
async def test_context_lane_active_context():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_mem.db")
        mem = SemanticMemory(db_path=db_path)
        
        # Seed memories
        mem.add_memory("s_1", "we are implementing Phase E memory system", {"session_id": "session_abc"})
        
        project_root = "/home/user/workspace/mcp_agent/primary/SparkleForge"
        lane = ContextLane(memory=mem, project_root=project_root)
        
        context = lane.get_active_context("session_abc")
        
        assert context["session_id"] == "session_abc"
        assert len(context["relevant_memories"]) > 0
        assert "Phase E" in context["relevant_memories"][0]["text"]
        assert "codebase_map" in context
        assert "src.core.automation.automation_engine" in context["codebase_map"]


@pytest.mark.asyncio
async def test_memory_context_bootstrap():
    graph = BootstrapGraph()
    res = await graph.run()
    assert res.ok is True
    
    stages = [s.name for s in res.stages]
    assert "memory_context" in stages
    
    stage_res = next(s for s in res.stages if s.name == "memory_context")
    assert stage_res.ok is True
    assert stage_res.payload["initialized"] is True
    
    assert isinstance(stage_res.payload["semantic_memory"], SemanticMemory)
    assert isinstance(stage_res.payload["history_analyzer"], HistoryAnalyzer)
    assert isinstance(stage_res.payload["context_lane"], ContextLane)
