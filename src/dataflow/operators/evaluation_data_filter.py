"""
Evaluation Data Filter Operator

평가 결과를 필터링하고 정제합니다.
"""

import logging
import pandas as pd
from typing import Any, Dict, Optional, Literal, Callable, List, TYPE_CHECKING

from .base_operator import SparkleForgeOperatorABC
from ..storage.agent_storage_adapter import DataFlowStorage

logger = logging.getLogger(__name__)


class EvaluationDataFilter(SparkleForgeOperatorABC):
    """
    평가 결과를 필터링하고 정제하는 Operator.
    
    input_key: research_results (연구 결과 목록)
    output_key: filtered_results (필터링된 결과 목록)
    """
    
    def __init__(
        self,
        filter_rules: Optional[List[Callable]] = None,
        min_quality_score: float = 0.5,
        required_fields: Optional[List[str]] = None,
    ):
        """
        초기화.
        
        Args:
            filter_rules: 커스텀 필터 규칙 함수 목록
            min_quality_score: 최소 품질 점수
            required_fields: 필수 필드 목록
        """
        super().__init__()
        self.filter_rules = filter_rules or []
        self.min_quality_score = min_quality_score
        self.required_fields = required_fields or ["content"]
    
    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "research_results",
        output_key: str = "filtered_results",
        **kwargs
    ) -> Optional[str]:
        """
        평가 데이터를 필터링합니다.
        
        Args:
            storage: DataFlowStorage 인스턴스
            input_key: 입력 키 (기본값: "research_results")
            output_key: 출력 키 (기본값: "filtered_results")
            **kwargs: 추가 파라미터
            
        Returns:
            실행 결과 메시지
        """
        self.logger.info(f"Filtering evaluation data from '{input_key}' to '{output_key}'")
        
        # 입력 데이터 읽기
        df = storage.read("dataframe")
        
        if input_key not in df.columns:
            self.logger.warning(f"Input key '{input_key}' not found. Creating empty results.")
            df[output_key] = None
            storage.write(df)
            return "No research results found"
        
        # 각 행에서 결과 필터링
        filtered_results = []
        total_count = 0
        
        for idx, row in df.iterrows():
            results = row.get(input_key, [])
            
            if not isinstance(results, list):
                results = [results] if results else []
            
            total_count += len(results)
            
            # 각 결과에 대해 필터링 적용
            for result in results:
                if not isinstance(result, dict):
                    continue
                
                # 필수 필드 검증
                if not all(field in result for field in self.required_fields):
                    continue
                
                # 품질 점수 검증
                quality_score = result.get("quality_score", result.get("score", 1.0))
                if isinstance(quality_score, (int, float)) and quality_score < self.min_quality_score:
                    continue
                
                # 커스텀 필터 규칙 적용
                passed = True
                for filter_rule in self.filter_rules:
                    if callable(filter_rule):
                        try:
                            if not filter_rule(result):
                                passed = False
                                break
                        except Exception as e:
                            self.logger.warning(f"Filter rule failed: {e}")
                            passed = False
                            break
                
                if passed:
                    filtered_results.append(result)
        
        # 결과를 DataFrame에 추가
        if filtered_results:
            df[output_key] = None
            for idx in range(len(df)):
                if idx < len(filtered_results):
                    if df.loc[idx, output_key] is None:
                        df.loc[idx, output_key] = []
                    if not isinstance(df.loc[idx, output_key], list):
                        df.loc[idx, output_key] = [df.loc[idx, output_key]]
                    df.loc[idx, output_key].append(filtered_results[idx])
        else:
            df[output_key] = None
        
        storage.write(df)
        
        filtered_count = len(filtered_results)
        self.logger.info(
            f"Filtered {filtered_count} results from {total_count} total "
            f"({filtered_count/total_count*100:.1f}% pass rate)"
        )
        
        return f"Filtered {filtered_count} results from {total_count} total"

