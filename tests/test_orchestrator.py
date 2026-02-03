"""
Integration test for Multi-Agent Orchestration System

메모리 시스템과 오케스트레이션 기능을 테스트
"""

import asyncio
import pytest
import sys
from pathlib import Path
import chromadb
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.shared_memory import SharedMemory, MemoryScope, init_shared_memory
from src.core.agent_orchestrator import AgentOrchestrator, AgentState


class TestSharedMemory:
    """Test shared memory system."""
    
    def test_write_read_memory(self):
        """Test basic memory write and read."""
        memory = SharedMemory(storage_path="./test_storage", enable_chromadb=False)
        
        # Write memory
        success = memory.write(
            key="test_key",
            value="test_value",
            scope=MemoryScope.GLOBAL
        )
        assert success is True
        
        # Read memory
        value = memory.read(key="test_key", scope=MemoryScope.GLOBAL)
        assert value == "test_value"
    
    def test_session_memory(self):
        """Test session-scoped memory."""
        memory = SharedMemory(storage_path="./test_storage", enable_chromadb=False)
        
        # Write to session
        memory.write(
            key="session_key",
            value="session_value",
            scope=MemoryScope.SESSION,
            session_id="test_session"
        )
        
        # Read from session
        value = memory.read(
            key="session_key",
            scope=MemoryScope.SESSION,
            session_id="test_session"
        )
        assert value == "session_value"
    
    def test_agent_memory(self):
        """Test agent-scoped memory."""
        memory = SharedMemory(storage_path="./test_storage", enable_chromadb=False)
        
        # Write to agent
        memory.write(
            key="agent_key",
            value="agent_value",
            scope=MemoryScope.AGENT,
            session_id="test_session",
            agent_id="test_agent"
        )
        
        # Read from agent
        value = memory.read(
            key="agent_key",
            scope=MemoryScope.AGENT,
            session_id="test_session",
            agent_id="test_agent"
        )
        assert value == "agent_value"
    
    def test_search_memory(self):
        """Test memory search functionality."""
        memory = SharedMemory(storage_path="./test_storage", enable_chromadb=False)
        
        # Write multiple memories
        memory.write(key="python_code", value="def hello(): pass", scope=MemoryScope.GLOBAL)
        memory.write(key="test_data", value="research findings", scope=MemoryScope.GLOBAL)
        memory.write(key="analysis", value="data analysis results", scope=MemoryScope.GLOBAL)
        
        # Search
        results = memory.search("python", limit=10, scope=MemoryScope.GLOBAL)
        assert len(results) > 0
        assert any("python" in str(r).lower() for r in results)


class TestAgentOrchestrator:
    """Test agent orchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized."""
        orchestrator = AgentOrchestrator(config=None)
        assert orchestrator is not None
        assert orchestrator.graph is not None
        assert orchestrator.planner is not None
        assert orchestrator.executor is not None
        assert orchestrator.verifier is not None
        assert orchestrator.generator is not None
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self):
        """Test full workflow execution."""
        orchestrator = AgentOrchestrator(config=None)
        
        user_query = "Test research query"
        
        # Execute workflow
        result = await orchestrator.execute(user_query)
        
        # Verify result
        assert result is not None
        assert result['user_query'] == user_query
        assert result['research_plan'] is not None
        assert result['final_report'] is not None
        assert len(result['research_results']) > 0
        assert len(result['verified_results']) > 0
    
    @pytest.mark.asyncio
    async def test_stream_workflow(self):
        """Test workflow streaming."""
        orchestrator = AgentOrchestrator(config=None)
        
        user_query = "Streaming test query"
        
        # Stream workflow
        events = []
        async for event in orchestrator.stream(user_query):
            events.append(event)
        
        # Verify events
        assert len(events) > 0
        assert any('planner' in str(event) or 'executor' in str(event) for event in events)


class TestMultiAgentIntegration:
    """Integration tests for multi-agent system."""
    
    @pytest.mark.asyncio
    async def test_memory_orchestrator_integration(self):
        """Test memory and orchestrator integration."""
        # Initialize memory
        memory = init_shared_memory(
            storage_path="./test_storage",
            enable_chromadb=False
        )
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator(config=None)
        
        # Execute workflow
        result = await orchestrator.execute("Integration test query")
        
        # Verify memory was used
        session_id = result.get('session_id')
        plan = memory.read(
            key=f"plan_{session_id}",
            scope=MemoryScope.SESSION,
            session_id=session_id
        )
        assert plan is not None
        
        # Verify shared memory contains results
        results = memory.search("test", scope=MemoryScope.SESSION)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


