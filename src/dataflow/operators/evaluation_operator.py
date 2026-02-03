"""
Evaluation Operator

EvaluationAgent의 평가 로직을 Operator로 추출합니다.
"""

import logging
import asyncio
from typing import Any, Dict, Optional, Literal

from .base_operator import SparkleForgeOperatorABC
from ..storage.agent_storage_adapter import DataFlowStorage

logger = logging.getLogger(__name__)


class EvaluationOperator(SparkleForgeOperatorABC):
    """
    평가 실행 Operator.
    
    EvaluationAgent의 로직을 Operator로 변환하여 Pipeline에서 사용할 수 있도록 합니다.
    
    input_key: research_results (연구 결과 목록)
    output_key: evaluation_results (평가 결과)
    """
    
    def __init__(
        self,
        evaluation_agent=None,
        enable_continuous_verification: bool = True,
    ):
        """
        초기화.
        
        Args:
            evaluation_agent: EvaluationAgent 인스턴스 (None이면 내부에서 생성)
            enable_continuous_verification: 연속 검증 활성화 여부
        """
        super().__init__()
        self.evaluation_agent = evaluation_agent
        self.enable_continuous_verification = enable_continuous_verification
        self._agent_initialized = False
    
    def _ensure_agent_initialized(self):
        """EvaluationAgent가 초기화되었는지 확인하고 필요시 초기화합니다."""
        if self._agent_initialized:
            return
        
        if self.evaluation_agent is None:
            try:
                from src.agents.evaluation_agent import EvaluationAgent
                self.evaluation_agent = EvaluationAgent()
                self.logger.info("EvaluationAgent initialized for EvaluationOperator")
            except Exception as e:
                self.logger.warning(f"Failed to initialize EvaluationAgent: {e}")
                self.evaluation_agent = None
        
        self._agent_initialized = True
    
    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "research_results",
        output_key: str = "evaluation_results",
        **kwargs
    ) -> Optional[str]:
        """
        평가를 실행합니다.
        
        Args:
            storage: DataFlowStorage 인스턴스
            input_key: 입력 키 (기본값: "research_results")
            output_key: 출력 키 (기본값: "evaluation_results")
            **kwargs: 추가 파라미터
            
        Returns:
            실행 결과 메시지
        """
        self.logger.info(f"Executing evaluation from '{input_key}' to '{output_key}'")
        
        # EvaluationAgent 초기화 확인
        self._ensure_agent_initialized()
        
        if self.evaluation_agent is None:
            self.logger.error("EvaluationAgent is not available")
            return "EvaluationAgent is not available"
        
        # 입력 데이터 읽기
        df = storage.read("dataframe")
        
        if input_key not in df.columns:
            self.logger.warning(f"Input key '{input_key}' not found. Creating empty results.")
            df[output_key] = None
            storage.write(df)
            return "No research results found"
        
        # 각 행에서 평가 실행
        evaluation_results = []
        
        for idx, row in df.iterrows():
            results = row.get(input_key, [])
            
            if not isinstance(results, list):
                results = [results] if results else []
            
            if not results:
                evaluation_results.append(None)
                continue
            
            # 평가 실행
            try:
                # EvaluationAgent의 evaluate_results 호출
                evaluation_result = asyncio.run(
                    self.evaluation_agent.evaluate_results(
                        execution_results=results,
                        original_objectives=row.get("research_tasks", []),
                        context=row.to_dict() if hasattr(row, 'to_dict') else {},
                    )
                )
                evaluation_results.append(evaluation_result)
            except Exception as e:
                self.logger.error(f"Evaluation failed: {e}")
                evaluation_results.append({
                    "status": "failed",
                    "error": str(e),
                })
        
        # 결과를 DataFrame에 추가
        df[output_key] = evaluation_results
        storage.write(df)
        
        non_null_count = sum(1 for r in evaluation_results if r is not None)
        self.logger.info(f"Evaluated {non_null_count} results")
        
        return f"Evaluated {non_null_count} results"








