"""
Orchestrator Pipeline Integration

AgentOrchestrator에 DataFlow Pipeline을 통합합니다.
"""

import logging
from typing import Dict, Any, Optional

from ..pipeline.research_pipeline import ResearchPipeline
from ..storage.agent_storage_adapter import StorageFactory, AgentStateStorage

logger = logging.getLogger(__name__)


class OrchestratorPipelineIntegration:
    """
    AgentOrchestrator와 DataFlow Pipeline을 통합하는 클래스.
    """
    
    def __init__(self, use_pipeline: bool = False):
        """
        초기화.
        
        Args:
            use_pipeline: Pipeline 사용 여부 (기본값: False, 기존 방식 사용)
        """
        self.use_pipeline = use_pipeline
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def execute_with_pipeline(
        self,
        agent_state: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Pipeline을 사용하여 연구를 실행합니다.
        
        Args:
            agent_state: AgentState 딕셔너리
            session_id: 세션 ID
            
        Returns:
            실행 결과 딕셔너리
        """
        self.logger.info("Executing research with DataFlow Pipeline")
        
        try:
            # AgentState를 Storage로 변환
            storage = StorageFactory.create_agent_state_storage(agent_state, session_id)
            
            # Research Pipeline 생성 및 실행
            pipeline = ResearchPipeline(
                storage=storage,
                agent_state=agent_state,
                session_id=session_id,
            )
            
            # Pipeline 실행
            pipeline.run_pipeline()
            
            # 결과를 AgentState로 역변환
            updated_state = self._storage_to_agent_state(storage, agent_state)
            
            self.logger.info("Pipeline execution completed")
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            # 실패 시 기존 방식으로 fallback
            self.logger.warning("Falling back to traditional agent execution")
            raise
    
    def _storage_to_agent_state(
        self,
        storage: AgentStateStorage,
        original_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Storage의 데이터를 AgentState로 역변환합니다.
        
        Args:
            storage: AgentStateStorage 인스턴스
            original_state: 원본 AgentState
            
        Returns:
            업데이트된 AgentState
        """
        # Storage에서 최신 데이터 읽기
        df = storage.read("dataframe")
        
        # DataFrame에서 AgentState 필드 추출
        updated_state: Dict[str, Any] = original_state.copy()
        
        # research_results 추출
        if "research_results" in df.columns:
            research_results = []
            for value in df["research_results"]:
                if value is not None:
                    if isinstance(value, list):
                        research_results.extend(value)
                    else:
                        research_results.append(value)
            if research_results:
                updated_state["research_results"] = research_results
        
        # evaluation_results 추출
        if "evaluation_results" in df.columns:
            for value in df["evaluation_results"]:
                if value is not None and isinstance(value, dict):
                    updated_state["evaluation_results"] = [value]
                    break
        
        # final_report 추출
        if "final_report" in df.columns:
            for value in df["final_report"]:
                if value is not None:
                    updated_state["final_report"] = value
                    break
        
        return updated_state
    
    @staticmethod
    def should_use_pipeline(agent_state: Dict[str, Any]) -> bool:
        """
        Pipeline 사용 여부를 결정합니다.
        
        Args:
            agent_state: AgentState 딕셔너리
            
        Returns:
            Pipeline 사용 여부
        """
        # 간단한 휴리스틱: research_tasks가 있고 구조화되어 있으면 Pipeline 사용
        research_tasks = agent_state.get("research_tasks", [])
        
        if not research_tasks:
            return False
        
        # 모든 태스크가 dict 형태인지 확인
        if isinstance(research_tasks, list) and len(research_tasks) > 0:
            return all(isinstance(task, dict) for task in research_tasks)
        
        return False

