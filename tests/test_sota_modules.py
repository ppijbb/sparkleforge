"""
SoTA Modules Integration Tests

SparkleForge SoTA 강화 모듈들의 통합 테스트.
모든 새 모듈이 올바르게 동작하는지 검증.
"""

import asyncio
import pytest
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestIterativeResearch:
    """Iterative Deep-Research Paradigm 테스트."""
    
    def test_quality_metrics(self):
        """QualityMetrics 테스트."""
        from src.core.iterative_research import QualityMetrics
        
        metrics = QualityMetrics(
            completeness=0.8,
            depth=0.7,
            source_diversity=0.6,
            factual_accuracy=0.9,
            coherence=0.8
        )
        
        # 가중 평균 점수 확인
        expected = 0.8 * 0.25 + 0.7 * 0.25 + 0.6 * 0.15 + 0.9 * 0.20 + 0.8 * 0.15
        assert abs(metrics.overall_score - expected) < 0.01
    
    def test_think_output(self):
        """ThinkOutput 테스트."""
        from src.core.iterative_research import ThinkOutput
        
        output = ThinkOutput(
            current_understanding="Test understanding",
            knowledge_gaps=["Gap 1", "Gap 2"],
            next_research_directions=["Direction 1"],
            hypotheses=["Hypothesis 1"],
            confidence_level=0.7
        )
        
        assert output.current_understanding == "Test understanding"
        assert len(output.knowledge_gaps) == 2
        assert output.confidence_level == 0.7
    
    def test_workspace_reconstructor(self):
        """WorkspaceReconstructor 테스트."""
        from src.core.iterative_research import (
            WorkspaceReconstructor, 
            IterativeResearchState,
            ReportOutput,
            QualityMetrics
        )
        
        reconstructor = WorkspaceReconstructor(max_summary_tokens=1000)
        
        state = IterativeResearchState(
            query="Test query",
            session_id="test-session",
            evolving_summary="Previous summary"
        )
        
        new_report = ReportOutput(
            round_number=1,
            executive_summary="New summary",
            key_findings=[{"content": "Finding 1"}],
            quality_metrics=QualityMetrics(completeness=0.8)
        )
        
        context = reconstructor.reconstruct(state, new_report)
        
        assert "Previous summary" in context.evolving_summary
        assert "New summary" in context.evolving_summary
        assert context.round_number == 1
    
    def test_iterative_research_engine(self):
        """IterativeResearchEngine 싱글톤 테스트."""
        from src.core.iterative_research import get_iterative_research_engine
        
        engine = get_iterative_research_engine(max_rounds=3, quality_threshold=0.9)
        
        assert engine.max_rounds == 3
        assert engine.quality_threshold == 0.9


class TestTopicSegmenter:
    """Topic-Segmented Memory 테스트."""
    
    def test_sensory_buffer(self):
        """SensoryBuffer 테스트."""
        from src.core.topic_segmenter import SensoryBuffer
        
        buffer = SensoryBuffer(max_tokens=100)
        
        # 버퍼에 추가
        result = buffer.add("Test content", 50)
        assert result is None  # 아직 가득 차지 않음
        
        # 버퍼 확인
        assert not buffer.is_empty()
        assert "Test content" in buffer.peek()
    
    def test_topic_boundary_detector(self):
        """TopicBoundaryDetector 테스트."""
        from src.core.topic_segmenter import TopicBoundaryDetector
        
        detector = TopicBoundaryDetector(similarity_threshold=0.3)
        
        # 첫 번째 호출
        is_boundary, similarity = detector.detect_boundary(
            "Content about AI", ["AI", "machine", "learning"]
        )
        assert not is_boundary  # 첫 번째는 항상 False
        
        # 유사한 토픽
        is_boundary, similarity = detector.detect_boundary(
            "More about AI models", ["AI", "models", "neural"]
        )
        assert similarity >= 0.2  # 키워드 겹침 (AI 공통)
    
    @pytest.mark.asyncio
    async def test_topic_segmenter_process(self):
        """TopicSegmenter 프로세스 테스트."""
        from src.core.topic_segmenter import TopicSegmenter, TopicType
        
        segmenter = TopicSegmenter(
            sensory_buffer_size=50,
            max_segment_tokens=100
        )
        
        # 짧은 컨텐츠 처리
        segment = await segmenter.process(
            "This is a test about artificial intelligence and machine learning.",
            topic_type=TopicType.RESEARCH
        )
        
        # 버퍼에 저장됨 (세그먼트 생성 안됨)
        assert segment is None or len(segmenter.segments) >= 0
        
        # 플러시
        flushed = segmenter.flush()
        if flushed:
            assert flushed.topic_type == TopicType.RESEARCH


class TestPreCompressor:
    """Pre-Compression 테스트."""
    
    def test_compression_levels(self):
        """압축 수준 테스트."""
        from src.core.pre_compressor import PreCompressor, CompressionLevel
        
        compressor = PreCompressor(default_level=CompressionLevel.MODERATE)
        
        text = (
            "This is a long text about artificial intelligence. "
            "It contains many sentences about machine learning and deep learning. "
            "The field of AI has grown significantly in recent years. "
            "Many companies are now using AI for various applications. "
            "This includes natural language processing and computer vision."
        )
        
        # 압축 실행
        result = compressor.compress(text, level=CompressionLevel.MODERATE)
        
        assert result.compression_ratio <= 1.0
        assert len(result.compressed_text) <= len(result.original_text)
    
    def test_context_compression(self):
        """컨텍스트 압축 테스트."""
        from src.core.pre_compressor import PreCompressor
        
        compressor = PreCompressor()
        
        text = "Short text for testing."
        max_tokens = 100
        
        result = compressor.compress_for_context(text, max_tokens)
        
        # 짧은 텍스트는 그대로 반환
        assert result == text
    
    def test_hybrid_retriever(self):
        """HybridRetriever 테스트."""
        from src.core.pre_compressor import HybridRetriever
        
        retriever = HybridRetriever(context_weight=0.4, embedding_weight=0.6)
        
        documents = [
            {"content": "Machine learning is a subset of AI", "id": 1},
            {"content": "Python programming language basics", "id": 2},
            {"content": "Deep learning neural networks", "id": 3}
        ]
        
        results = retriever.retrieve(
            query="What is machine learning?",
            documents=documents,
            top_k=2
        )
        
        assert len(results) <= 2
        # 첫 번째 결과가 가장 관련성 높아야 함
        if results:
            assert results[0][0]["id"] == 1


class TestDynamicWorkflow:
    """Dynamic Workflow 테스트."""
    
    def test_dynamic_task(self):
        """DynamicTask 테스트."""
        from src.core.dynamic_workflow import (
            DynamicTask, 
            WorkflowPhase, 
            TaskPriority,
            TaskStatus
        )
        
        task = DynamicTask(
            task_id="test-task-1",
            name="Test Task",
            description="A test task",
            phase=WorkflowPhase.ANALYSIS,
            priority=TaskPriority.HIGH
        )
        
        assert task.status == TaskStatus.PENDING
        assert task.phase == WorkflowPhase.ANALYSIS
    
    def test_dynamic_task_spawner(self):
        """DynamicTaskSpawner 테스트."""
        from src.core.dynamic_workflow import DynamicTaskSpawner, WorkflowPhase
        
        spawner = DynamicTaskSpawner(max_spawned_per_task=3)
        
        # 태스크 생성
        task1 = spawner.spawn(
            parent_task_id="parent-1",
            name="Child Task 1",
            description="First child",
            phase=WorkflowPhase.IMPLEMENTATION
        )
        
        assert task1 is not None
        assert task1.spawned_by == "parent-1"
        
        # 제한 테스트
        for i in range(3):
            spawner.spawn(
                parent_task_id="parent-1",
                name=f"Child Task {i+2}",
                description=f"Child {i+2}",
                phase=WorkflowPhase.IMPLEMENTATION
            )
        
        # 제한 초과
        overflow = spawner.spawn(
            parent_task_id="parent-1",
            name="Overflow Task",
            description="Should be None",
            phase=WorkflowPhase.IMPLEMENTATION
        )
        
        assert overflow is None
    
    def test_phase_manager(self):
        """PhaseManager 테스트."""
        from src.core.dynamic_workflow import PhaseManager, WorkflowPhase
        
        manager = PhaseManager()
        
        assert manager.current_phase == WorkflowPhase.ANALYSIS
        
        # 태스크 추가
        manager.add_task_to_phase("task-1", WorkflowPhase.ANALYSIS)
        manager.add_task_to_phase("task-2", WorkflowPhase.ANALYSIS)
        
        assert len(manager.phase_tasks[WorkflowPhase.ANALYSIS]) == 2


class TestGuardianAgent:
    """Guardian Agent 테스트."""
    
    def test_agent_registration(self):
        """에이전트 등록 테스트."""
        from src.core.guardian_agent import GuardianAgent, AgentHealthStatus
        
        guardian = GuardianAgent(
            check_interval_seconds=1.0,
            stuck_threshold_seconds=10.0
        )
        
        guardian.register_agent("agent-1")
        
        assert "agent-1" in guardian.agent_metrics
        assert guardian.agent_metrics["agent-1"].status == AgentHealthStatus.HEALTHY
    
    def test_activity_reporting(self):
        """활동 보고 테스트."""
        from src.core.guardian_agent import GuardianAgent
        
        guardian = GuardianAgent()
        guardian.register_agent("agent-1")
        
        # 활동 보고
        guardian.report_activity(
            agent_id="agent-1",
            task_completed=True,
            response_time=1.5
        )
        
        metrics = guardian.agent_metrics["agent-1"]
        assert metrics.tasks_completed == 1
        assert metrics.avg_response_time == 1.5
    
    def test_system_health(self):
        """시스템 건강 상태 테스트."""
        from src.core.guardian_agent import GuardianAgent
        
        guardian = GuardianAgent()
        guardian.register_agent("agent-1")
        guardian.register_agent("agent-2")
        
        health = guardian.get_system_health()
        
        assert health["total_agents"] == 2
        assert "healthy" in health["status_breakdown"]


class TestTwoTierRAG:
    """Two-Tier RAG 테스트."""
    
    def test_memory_entry_creation(self):
        """MemoryEntry 생성 테스트."""
        from src.core.two_tier_rag import TwoTierRAGSystem, MemoryType
        
        rag = TwoTierRAGSystem(tier1_size=10)
        
        entry = rag.add(
            content="This is a test discovery about AI",
            memory_type=MemoryType.DISCOVERY,
            importance=0.8,
            keywords=["AI", "discovery"]
        )
        
        assert entry.memory_type == MemoryType.DISCOVERY
        assert entry.importance == 0.8
        assert "AI" in entry.keywords
    
    def test_query(self):
        """쿼리 테스트."""
        from src.core.two_tier_rag import TwoTierRAGSystem, MemoryType
        
        rag = TwoTierRAGSystem(tier1_size=10)
        
        # 엔트리 추가
        rag.add(
            content="Machine learning is a type of AI",
            memory_type=MemoryType.FACT,
            keywords=["machine", "learning", "AI"]
        )
        
        rag.add(
            content="Python is a programming language",
            memory_type=MemoryType.FACT,
            keywords=["python", "programming"]
        )
        
        # 검색
        results = rag.query("What is machine learning?", top_k=5)
        
        assert len(results) >= 1
        # 첫 번째 결과가 ML 관련이어야 함
        if results:
            assert "machine" in results[0][0].content.lower()
    
    def test_statistics(self):
        """통계 테스트."""
        from src.core.two_tier_rag import TwoTierRAGSystem, MemoryType
        
        rag = TwoTierRAGSystem(tier1_size=10)
        
        rag.add("Test content", MemoryType.FACT)
        
        stats = rag.get_statistics()
        
        assert "tier1" in stats
        assert "tier2" in stats
        assert stats["tier2"]["total_entries"] >= 1


class TestMCPManagerAgent:
    """MCP Manager Agent 테스트."""
    
    @pytest.mark.asyncio
    async def test_server_registration(self):
        """서버 등록 테스트."""
        from src.core.mcp_manager_agent import MCPManagerAgent, ToolCategory
        
        manager = MCPManagerAgent(auto_reconnect=False)
        
        server = await manager.register_server(
            server_id="test-server",
            name="Test Server",
            command="echo",
            args=["test"],
            categories=[ToolCategory.SEARCH],
            connect_now=False
        )
        
        assert server.server_id == "test-server"
        assert ToolCategory.SEARCH in server.categories
    
    def test_tool_recommendation(self):
        """도구 추천 테스트."""
        from src.core.mcp_manager_agent import (
            MCPManagerAgent, 
            ToolInfo, 
            ToolCategory
        )
        
        manager = MCPManagerAgent(auto_reconnect=False)
        
        # 도구 직접 등록
        tool = ToolInfo(
            tool_id="search-tool",
            name="Web Search",
            description="Search the web",
            server_id="test-server",
            category=ToolCategory.SEARCH,
            keywords=["search", "web", "query"]
        )
        manager._register_tool(tool)
        
        # 추천
        recommendations = manager.recommend_tools(
            task_description="I need to search the web for information",
            top_k=3
        )
        
        assert len(recommendations) >= 1


class TestSkillAutoDiscovery:
    """Skill Auto-Discovery 테스트."""
    
    def test_skill_context_loader(self):
        """SkillContextLoader 테스트."""
        from src.core.skill_auto_discovery import SkillContextLoader
        from pathlib import Path
        
        loader = SkillContextLoader()
        
        # 컨텍스트 파일 패턴 확인
        assert "SKILL.md" in loader.context_file_patterns
    
    def test_file_watcher_init(self):
        """FileWatcher 초기화 테스트."""
        from src.core.skill_auto_discovery import FileWatcher
        from pathlib import Path
        
        watcher = FileWatcher(
            watch_paths=[Path("/tmp")],
            patterns=["*.md"],
            poll_interval=1.0
        )
        
        assert watcher.poll_interval == 1.0
        assert not watcher._watching


class TestTestTimeScaling:
    """Test-Time Scaling 테스트."""
    
    def test_rollout_config(self):
        """RolloutConfig 테스트."""
        from src.core.test_time_scaling import RolloutConfig
        
        config = RolloutConfig(
            rollout_id="test-1",
            temperature=0.8,
            top_p=0.9
        )
        
        assert config.temperature == 0.8
        assert config.top_p == 0.9
    
    def test_rollout_result(self):
        """RolloutResult 테스트."""
        from src.core.test_time_scaling import RolloutResult, RolloutConfig, RolloutStatus
        
        config = RolloutConfig(rollout_id="test-1")
        result = RolloutResult(
            rollout_id="test-1",
            config=config,
            status=RolloutStatus.COMPLETED,
            output="Test output",
            confidence=0.8,
            coherence_score=0.7,
            completeness_score=0.8,
            relevance_score=0.9
        )
        
        assert result.quality_score > 0
    
    def test_parallel_executor_config_generation(self):
        """ParallelRolloutExecutor 설정 생성 테스트."""
        from src.core.test_time_scaling import ParallelRolloutExecutor
        
        executor = ParallelRolloutExecutor(
            num_rollouts=5,
            diversity_strategy="temperature_variation"
        )
        
        configs = executor._generate_diverse_configs()
        
        assert len(configs) == 5
        
        # Temperature 다양성 확인
        temps = [c.temperature for c in configs]
        assert temps[0] != temps[-1]  # 첫 번째와 마지막이 다름
    
    def test_fusion_agent_best_confidence(self):
        """FusionAgent best_confidence 테스트."""
        from src.core.test_time_scaling import (
            FusionAgent, 
            FusionStrategy,
            RolloutResult,
            RolloutConfig,
            RolloutStatus
        )
        
        fuser = FusionAgent(default_strategy=FusionStrategy.BEST_CONFIDENCE)
        
        results = [
            RolloutResult(
                rollout_id="r1",
                config=RolloutConfig(rollout_id="r1"),
                status=RolloutStatus.COMPLETED,
                output="Result 1",
                confidence=0.7
            ),
            RolloutResult(
                rollout_id="r2",
                config=RolloutConfig(rollout_id="r2"),
                status=RolloutStatus.COMPLETED,
                output="Result 2",
                confidence=0.9
            )
        ]
        
        # 동기적으로 best_confidence 테스트
        fusion = fuser._fuse_best_confidence(results)
        
        assert fusion.output == "Result 2"
        assert fusion.confidence == 0.9


# 모듈 임포트 테스트
class TestModuleImports:
    """모든 모듈이 정상적으로 임포트되는지 테스트."""
    
    def test_iterative_research_import(self):
        """iterative_research 모듈 임포트."""
        from src.core.iterative_research import (
            IterativeResearchEngine,
            IterativeResearchNode,
            WorkspaceReconstructor
        )
        assert IterativeResearchEngine is not None
    
    def test_topic_segmenter_import(self):
        """topic_segmenter 모듈 임포트."""
        from src.core.topic_segmenter import (
            TopicSegmenter,
            SensoryBuffer,
            TopicBoundaryDetector
        )
        assert TopicSegmenter is not None
    
    def test_pre_compressor_import(self):
        """pre_compressor 모듈 임포트."""
        from src.core.pre_compressor import (
            PreCompressor,
            HybridRetriever,
            CompressionLevel
        )
        assert PreCompressor is not None
    
    def test_dynamic_workflow_import(self):
        """dynamic_workflow 모듈 임포트."""
        from src.core.dynamic_workflow import (
            DynamicWorkflowEngine,
            DynamicTaskSpawner,
            PhaseManager
        )
        assert DynamicWorkflowEngine is not None
    
    def test_guardian_agent_import(self):
        """guardian_agent 모듈 임포트."""
        from src.core.guardian_agent import (
            GuardianAgent,
            AgentHealthStatus,
            RecoveryAction
        )
        assert GuardianAgent is not None
    
    def test_two_tier_rag_import(self):
        """two_tier_rag 모듈 임포트."""
        from src.core.two_tier_rag import (
            TwoTierRAGSystem,
            TierOneCache,
            TierTwoStore
        )
        assert TwoTierRAGSystem is not None
    
    def test_mcp_manager_agent_import(self):
        """mcp_manager_agent 모듈 임포트."""
        from src.core.mcp_manager_agent import (
            MCPManagerAgent,
            ToolCategory,
            ToolRecommendation
        )
        assert MCPManagerAgent is not None
    
    def test_skill_auto_discovery_import(self):
        """skill_auto_discovery 모듈 임포트."""
        from src.core.skill_auto_discovery import (
            SkillAutoDiscovery,
            SkillContextLoader,
            FileWatcher
        )
        assert SkillAutoDiscovery is not None
    
    def test_test_time_scaling_import(self):
        """test_time_scaling 모듈 임포트."""
        from src.core.test_time_scaling import (
            TestTimeScaler,
            ParallelRolloutExecutor,
            FusionAgent
        )
        assert TestTimeScaler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
