"""
Research Pipeline

Research → Evaluation → Synthesis 파이프라인을 정의합니다.
"""

import logging
from typing import Optional

from .agent_pipeline import AgentPipeline
from ..operators.research_operator import ResearchOperator
from ..operators.evaluation_operator import EvaluationOperator
from ..operators.synthesis_operator import SynthesisOperator
from ..storage.agent_storage_adapter import DataFlowStorage, StorageFactory

logger = logging.getLogger(__name__)


class ResearchPipeline(AgentPipeline):
    """
    Research → Evaluation → Synthesis 파이프라인.
    
    Agent 워크플로우를 구조화된 Pipeline으로 변환합니다.
    """
    
    def __init__(
        self,
        storage: Optional[DataFlowStorage] = None,
        agent_state: Optional[dict] = None,
        session_id: Optional[str] = None,
    ):
        """
        초기화.
        
        Args:
            storage: DataFlowStorage 인스턴스 (None이면 agent_state에서 생성)
            agent_state: AgentState 딕셔너리 (storage가 None일 때 사용)
            session_id: 세션 ID
        """
        super().__init__()
        
        # Storage 설정
        if storage is None:
            if agent_state is None:
                raise ValueError("Either storage or agent_state must be provided")
            self.storage = StorageFactory.create_agent_state_storage(agent_state, session_id)
        else:
            self.storage = storage
        
        self.session_id = session_id
        
        # Operators 초기화
        self.research_op = None
        self.evaluation_op = None
        self.synthesis_op = None
    
    def forward(self):
        """
        Pipeline을 정의하고 실행합니다.
        
        이 메서드는 compile() 전에 호출되어 op_runtimes를 수집합니다.
        """
        # Research Operator
        self.research_op = ResearchOperator()
        self.research_op.run(
            storage=self.storage,
            input_key="research_tasks",
            output_key="research_results",
        )
        
        # Evaluation Operator
        self.evaluation_op = EvaluationOperator()
        self.evaluation_op.run(
            storage=self.storage,
            input_key="research_results",
            output_key="evaluation_results",
        )
        
        # Synthesis Operator
        self.synthesis_op = SynthesisOperator()
        self.synthesis_op.run(
            storage=self.storage,
            input_key="evaluation_results",
            output_key="final_report",
        )
    
    def run_pipeline(self, resume_step: int = 0):
        """
        파이프라인을 실행합니다.
        
        Args:
            resume_step: 재개할 단계 (기본값: 0)
        """
        if not self.compiled:
            self.logger.info("Compiling pipeline...")
            self.compile()
        
        self.logger.info("Running research pipeline...")
        self.run(resume_step=resume_step)
        self.logger.info("Research pipeline completed")
    
    def get_final_report(self) -> Optional[str]:
        """
        최종 보고서를 가져옵니다.
        
        Returns:
            최종 보고서 내용
        """
        df = self.storage.read("dataframe")
        
        if "final_report" in df.columns:
            # 첫 번째 non-null 값을 반환
            for value in df["final_report"]:
                if value is not None:
                    return value
        
        return None
    
    def get_research_results(self) -> list:
        """
        연구 결과를 가져옵니다.
        
        Returns:
            연구 결과 목록
        """
        df = self.storage.read("dataframe")
        
        if "research_results" in df.columns:
            results = []
            for value in df["research_results"]:
                if value is not None:
                    if isinstance(value, list):
                        results.extend(value)
                    else:
                        results.append(value)
            return results
        
        return []
    
    def get_evaluation_results(self) -> Optional[dict]:
        """
        평가 결과를 가져옵니다.
        
        Returns:
            평가 결과 딕셔너리
        """
        df = self.storage.read("dataframe")
        
        if "evaluation_results" in df.columns:
            # 첫 번째 non-null 값을 반환
            for value in df["evaluation_results"]:
                if value is not None:
                    return value if isinstance(value, dict) else {}
        
        return None








