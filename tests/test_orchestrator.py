"""
Integration test for Multi-Agent Orchestration System

메모리 시스템과 오케스트레이션 기능을 테스트
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Add project root to path so `src` is importable without PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.researcher_config import load_config_from_env

load_config_from_env()

from src.core.agent_orchestrator import AgentOrchestrator, AgentState
from src.core.shared_memory import MemoryScope, SharedMemory, init_shared_memory


@pytest.fixture
def temp_storage_path(tmp_path):
    """Fixture to provide a clean temporary storage path."""
    return str(tmp_path / "test_storage")



class TestSharedMemory:
    """Test shared memory system."""

    def test_write_read_memory(self, temp_storage_path):
        """Test basic memory write and read."""
        memory = SharedMemory(storage_path=temp_storage_path, enable_chromadb=False)

        # Write memory
        success = memory.write(
            key="test_key", value="test_value", scope=MemoryScope.GLOBAL
        )
        assert success is True

        # Read memory
        value = memory.read(key="test_key", scope=MemoryScope.GLOBAL)
        assert value == "test_value"

    def test_session_memory(self, temp_storage_path):
        """Test session-scoped memory."""
        memory = SharedMemory(storage_path=temp_storage_path, enable_chromadb=False)

        # Write to session
        memory.write(
            key="session_key",
            value="session_value",
            scope=MemoryScope.SESSION,
            session_id="test_session",
        )

        # Read from session
        value = memory.read(
            key="session_key", scope=MemoryScope.SESSION, session_id="test_session"
        )
        assert value == "session_value"

    def test_agent_memory(self, temp_storage_path):
        """Test agent-scoped memory."""
        memory = SharedMemory(storage_path=temp_storage_path, enable_chromadb=False)

        # Write to agent
        memory.write(
            key="agent_key",
            value="agent_value",
            scope=MemoryScope.AGENT,
            session_id="test_session",
            agent_id="test_agent",
        )

        # Read from agent
        value = memory.read(
            key="agent_key",
            scope=MemoryScope.AGENT,
            session_id="test_session",
            agent_id="test_agent",
        )
        assert value == "agent_value"

    def test_search_memory(self, temp_storage_path):
        """Test memory search functionality."""
        memory = SharedMemory(storage_path=temp_storage_path, enable_chromadb=False)

        # Write multiple memories
        memory.write(
            key="python_code", value="def hello(): pass", scope=MemoryScope.GLOBAL
        )
        memory.write(
            key="test_data", value="research findings", scope=MemoryScope.GLOBAL
        )
        memory.write(
            key="analysis", value="data analysis results", scope=MemoryScope.GLOBAL
        )

        # Search
        results = memory.search("python", limit=10, scope=MemoryScope.GLOBAL)
        assert len(results) > 0
        assert any("python" in str(r).lower() for r in results)


class TestAgentOrchestrator:
    """Test agent orchestrator (AgentHarness-based)."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized."""
        orchestrator = AgentOrchestrator(config=None)
        assert orchestrator is not None
        assert orchestrator.harness is not None

    @pytest.mark.asyncio
    async def test_execute_workflow(self):
        """Test full workflow execution returns expected keys."""
        orchestrator = AgentOrchestrator(config=None)

        user_query = "Test research query"

        # Execute workflow
        result = await orchestrator.execute(user_query)

        # Verify result has expected structure (AgentHarness-based API)
        assert result is not None
        assert "final_report" in result
        assert "session_id" in result
        assert "success" in result

    @pytest.mark.asyncio
    async def test_stream_workflow(self):
        """Test that execute returns a result dict (streaming via harness)."""
        orchestrator = AgentOrchestrator(config=None)

        user_query = "Streaming test query"

        # AgentHarness doesn't expose streaming directly; execute returns a dict
        result = await orchestrator.execute(user_query)
        assert result is not None
        assert isinstance(result, dict)


class TestMultiAgentIntegration:
    """Integration tests for multi-agent system."""

    @pytest.mark.asyncio
    async def test_memory_orchestrator_integration(self, temp_storage_path):
        """Test memory and orchestrator integration."""
        # Initialize memory
        memory = init_shared_memory(
            storage_path=temp_storage_path, enable_chromadb=False
        )

        # Initialize orchestrator
        orchestrator = AgentOrchestrator(config=None)

        # Execute workflow
        result = await orchestrator.execute("Integration test query")

        # Verify session_id is returned
        session_id = result.get("session_id")
        assert session_id is not None

        # Verify result has expected structure
        assert "final_report" in result or "results" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
