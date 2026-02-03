"""
Synthesis Operator

SynthesisAgent의 종합 로직을 Operator로 추출합니다.
"""

import logging
import asyncio
from typing import Any, Dict, Optional, Literal

from .base_operator import SparkleForgeOperatorABC
from ..storage.agent_storage_adapter import DataFlowStorage

logger = logging.getLogger(__name__)


class SynthesisOperator(SparkleForgeOperatorABC):
    """
    종합 실행 Operator.
    
    SynthesisAgent의 로직을 Operator로 변환하여 Pipeline에서 사용할 수 있도록 합니다.
    
    input_key: evaluation_results (평가 결과)
    output_key: final_report (최종 보고서)
    """
    
    def __init__(
        self,
        synthesis_agent=None,
        deliverable_type: str = "detailed_report",
    ):
        """
        초기화.
        
        Args:
            synthesis_agent: SynthesisAgent 인스턴스 (None이면 내부에서 생성)
            deliverable_type: 생성할 결과물 타입
        """
        super().__init__()
        self.synthesis_agent = synthesis_agent
        self.deliverable_type = deliverable_type
        self._agent_initialized = False
    
    def _ensure_agent_initialized(self):
        """SynthesisAgent가 초기화되었는지 확인하고 필요시 초기화합니다."""
        if self._agent_initialized:
            return
        
        if self.synthesis_agent is None:
            try:
                from src.agents.synthesis_agent import SynthesisAgent
                self.synthesis_agent = SynthesisAgent()
                self.logger.info("SynthesisAgent initialized for SynthesisOperator")
            except Exception as e:
                self.logger.warning(f"Failed to initialize SynthesisAgent: {e}")
                self.synthesis_agent = None
        
        self._agent_initialized = True
    
    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "evaluation_results",
        output_key: str = "final_report",
        **kwargs
    ) -> Optional[str]:
        """
        종합을 실행합니다.
        
        Args:
            storage: DataFlowStorage 인스턴스
            input_key: 입력 키 (기본값: "evaluation_results")
            output_key: 출력 키 (기본값: "final_report")
            **kwargs: 추가 파라미터
            
        Returns:
            실행 결과 메시지
        """
        self.logger.info(f"Executing synthesis from '{input_key}' to '{output_key}'")
        
        # SynthesisAgent 초기화 확인
        self._ensure_agent_initialized()
        
        if self.synthesis_agent is None:
            self.logger.error("SynthesisAgent is not available")
            return "SynthesisAgent is not available"
        
        # 입력 데이터 읽기
        df = storage.read("dataframe")
        
        if input_key not in df.columns:
            self.logger.warning(f"Input key '{input_key}' not found. Creating empty results.")
            df[output_key] = None
            storage.write(df)
            return "No evaluation results found"
        
        # 각 행에서 종합 실행
        final_reports = []
        
        for idx, row in df.iterrows():
            evaluation_results = row.get(input_key)
            research_results = row.get("research_results", [])
            original_objectives = row.get("research_tasks", [])
            
            if evaluation_results is None:
                final_reports.append(None)
                continue
            
            # 종합 실행
            try:
                # SynthesisAgent의 synthesize_results 호출
                synthesis_result = asyncio.run(
                    self.synthesis_agent.synthesize_results(
                        execution_results=research_results if isinstance(research_results, list) else [research_results],
                        evaluation_results=evaluation_results if isinstance(evaluation_results, dict) else {},
                        original_objectives=original_objectives if isinstance(original_objectives, list) else [],
                        context=row.to_dict() if hasattr(row, 'to_dict') else {},
                        deliverable_type=self.deliverable_type,
                    )
                )
                
                # 최종 보고서 추출
                final_report = synthesis_result.get("content", synthesis_result.get("final_report", ""))
                final_reports.append(final_report)
            except Exception as e:
                self.logger.error(f"Synthesis failed: {e}")
                final_reports.append(f"Synthesis failed: {str(e)}")
        
        # 결과를 DataFrame에 추가
        df[output_key] = final_reports
        storage.write(df)
        
        non_null_count = sum(1 for r in final_reports if r is not None)
        self.logger.info(f"Synthesized {non_null_count} reports")
        
        return f"Synthesized {non_null_count} reports"








