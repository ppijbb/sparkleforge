"""
Synthesis Data Transformer Operator

종합 데이터를 변환합니다.
"""

import logging
import pandas as pd
from typing import Any, Dict, Optional, Literal, List

from .base_operator import SparkleForgeOperatorABC
from ..storage.agent_storage_adapter import DataFlowStorage

logger = logging.getLogger(__name__)


class SynthesisDataTransformer(SparkleForgeOperatorABC):
    """
    종합 데이터를 변환하는 Operator.
    
    input_key: evaluation_results 또는 filtered_results
    output_key: synthesized_content (종합된 콘텐츠)
    """
    
    def __init__(
        self,
        synthesis_strategy: str = "merge",
        include_metadata: bool = True,
        max_content_length: Optional[int] = None,
    ):
        """
        초기화.
        
        Args:
            synthesis_strategy: 종합 전략 ("merge", "summarize", "extract")
            include_metadata: 메타데이터 포함 여부
            max_content_length: 최대 콘텐츠 길이 (None이면 제한 없음)
        """
        super().__init__()
        self.synthesis_strategy = synthesis_strategy
        self.include_metadata = include_metadata
        self.max_content_length = max_content_length
    
    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "evaluation_results",
        output_key: str = "synthesized_content",
        **kwargs
    ) -> Optional[str]:
        """
        종합 데이터를 변환합니다.
        
        Args:
            storage: DataFlowStorage 인스턴스
            input_key: 입력 키 (기본값: "evaluation_results")
            output_key: 출력 키 (기본값: "synthesized_content")
            **kwargs: 추가 파라미터
            
        Returns:
            실행 결과 메시지
        """
        self.logger.info(f"Transforming synthesis data from '{input_key}' to '{output_key}'")
        
        # 입력 데이터 읽기
        df = storage.read("dataframe")
        
        if input_key not in df.columns:
            self.logger.warning(f"Input key '{input_key}' not found. Creating empty results.")
            df[output_key] = None
            storage.write(df)
            return "No evaluation results found"
        
        # 각 행에서 데이터 변환
        synthesized_contents: List[Optional[str]] = []
        
        for idx, row in df.iterrows():
            results = row.get(input_key, [])
            
            if not isinstance(results, list):
                results = [results] if results else []
            
            if not results:
                synthesized_contents.append(None)
                continue
            
            # 종합 전략에 따라 변환
            if self.synthesis_strategy == "merge":
                content = self._merge_results(results)
            elif self.synthesis_strategy == "summarize":
                content = self._summarize_results(results)
            elif self.synthesis_strategy == "extract":
                content = self._extract_key_points(results)
            else:
                content = self._merge_results(results)
            
            # 길이 제한 적용
            if self.max_content_length and content:
                if len(content) > self.max_content_length:
                    content = content[:self.max_content_length] + "..."
            
            synthesized_contents.append(content)
        
        # 결과를 DataFrame에 추가
        df[output_key] = synthesized_contents
        storage.write(df)
        
        non_null_count = sum(1 for c in synthesized_contents if c is not None)
        self.logger.info(f"Transformed {non_null_count} synthesis contents")
        
        return f"Transformed {non_null_count} synthesis contents"
    
    def _merge_results(self, results: List[Dict[str, Any]]) -> str:
        """결과를 병합합니다."""
        contents = []
        
        for result in results:
            if isinstance(result, dict):
                content = result.get("content", result.get("text", ""))
                if content:
                    if self.include_metadata:
                        source = result.get("source", result.get("url", "unknown"))
                        contents.append(f"[Source: {source}]\n{content}")
                    else:
                        contents.append(content)
            elif isinstance(result, str):
                contents.append(result)
        
        return "\n\n".join(contents)
    
    def _summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """결과를 요약합니다."""
        # 간단한 요약: 각 결과의 첫 부분만 추출
        summaries = []
        
        for result in results:
            if isinstance(result, dict):
                content = result.get("content", result.get("text", ""))
                if content:
                    # 첫 200자만 추출
                    summary = content[:200] + "..." if len(content) > 200 else content
                    summaries.append(summary)
            elif isinstance(result, str):
                summary = result[:200] + "..." if len(result) > 200 else result
                summaries.append(summary)
        
        return "\n".join(summaries)
    
    def _extract_key_points(self, results: List[Dict[str, Any]]) -> str:
        """핵심 포인트를 추출합니다."""
        key_points = []
        
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                content = result.get("content", result.get("text", ""))
                title = result.get("title", result.get("source", f"Point {i}"))
                if content:
                    key_points.append(f"{i}. {title}: {content[:100]}...")
            elif isinstance(result, str):
                key_points.append(f"{i}. {result[:100]}...")
        
        return "\n".join(key_points)

