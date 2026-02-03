"""
Context Data Enricher Operator

컨텍스트 데이터를 보강합니다.
"""

import logging
import pandas as pd
from typing import Any, Dict, Optional, Literal, List

from .base_operator import SparkleForgeOperatorABC
from ..storage.agent_storage_adapter import DataFlowStorage

logger = logging.getLogger(__name__)


class ContextDataEnricher(SparkleForgeOperatorABC):
    """
    컨텍스트 데이터를 보강하는 Operator.
    
    input_key: synthesized_content 또는 다른 데이터
    output_key: enriched_content (보강된 콘텐츠)
    """
    
    def __init__(
        self,
        enrichment_fields: Optional[List[str]] = None,
        add_timestamps: bool = True,
        add_sources: bool = True,
    ):
        """
        초기화.
        
        Args:
            enrichment_fields: 보강할 필드 목록
            add_timestamps: 타임스탬프 추가 여부
            add_sources: 소스 정보 추가 여부
        """
        super().__init__()
        self.enrichment_fields = enrichment_fields or ["metadata", "quality_score"]
        self.add_timestamps = add_timestamps
        self.add_sources = add_sources
    
    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "synthesized_content",
        output_key: str = "enriched_content",
        **kwargs
    ) -> Optional[str]:
        """
        컨텍스트 데이터를 보강합니다.
        
        Args:
            storage: DataFlowStorage 인스턴스
            input_key: 입력 키 (기본값: "synthesized_content")
            output_key: 출력 키 (기본값: "enriched_content")
            **kwargs: 추가 파라미터
            
        Returns:
            실행 결과 메시지
        """
        self.logger.info(f"Enriching context data from '{input_key}' to '{output_key}'")
        
        # 입력 데이터 읽기
        df = storage.read("dataframe")
        
        if input_key not in df.columns:
            self.logger.warning(f"Input key '{input_key}' not found. Creating empty results.")
            df[output_key] = None
            storage.write(df)
            return "No input data found"
        
        # 각 행에서 데이터 보강
        enriched_contents: List[Optional[str]] = []
        
        for idx, row in df.iterrows():
            content = row.get(input_key)
            
            if content is None:
                enriched_contents.append(None)
                continue
            
            # 보강된 콘텐츠 생성
            enriched = self._enrich_content(content, row)
            enriched_contents.append(enriched)
        
        # 결과를 DataFrame에 추가
        df[output_key] = enriched_contents
        storage.write(df)
        
        non_null_count = sum(1 for c in enriched_contents if c is not None)
        self.logger.info(f"Enriched {non_null_count} contents")
        
        return f"Enriched {non_null_count} contents"
    
    def _enrich_content(self, content: Any, row: pd.Series) -> str:
        """콘텐츠를 보강합니다."""
        from datetime import datetime
        
        if isinstance(content, dict):
            # dict인 경우 문자열로 변환
            text = content.get("content", content.get("text", str(content)))
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)
        
        # 보강 정보 추가
        enrichment_parts = []
        
        if self.add_timestamps:
            timestamp = datetime.now().isoformat()
            enrichment_parts.append(f"[Timestamp: {timestamp}]")
        
        if self.add_sources and isinstance(content, dict):
            source = content.get("source", content.get("url", "unknown"))
            enrichment_parts.append(f"[Source: {source}]")
        
        # 메타데이터 추가
        for field in self.enrichment_fields:
            if field in row and row[field] is not None:
                enrichment_parts.append(f"[{field}: {row[field]}]")
        
        # 보강 정보를 콘텐츠 앞에 추가
        if enrichment_parts:
            enriched = "\n".join(enrichment_parts) + "\n\n" + text
        else:
            enriched = text
        
        return enriched

