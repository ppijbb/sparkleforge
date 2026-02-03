"""
Research Data Extractor Operator

연구 결과에서 구조화된 데이터를 추출합니다.
"""

import logging
import pandas as pd
from typing import Any, Dict, Optional, Literal, List

from .base_operator import SparkleForgeOperatorABC
from ..storage.agent_storage_adapter import DataFlowStorage

logger = logging.getLogger(__name__)


class ResearchDataExtractor(SparkleForgeOperatorABC):
    """
    연구 결과에서 구조화된 데이터를 추출하는 Operator.
    
    input_key: research_tasks (연구 태스크 목록)
    output_key: research_results (연구 결과 목록)
    """
    
    def __init__(self, extract_fields: Optional[List[str]] = None):
        """
        초기화.
        
        Args:
            extract_fields: 추출할 필드 목록 (None이면 모든 필드 추출)
        """
        super().__init__()
        self.extract_fields = extract_fields or [
            "title",
            "url",
            "snippet",
            "content",
            "source",
            "timestamp",
        ]
    
    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "research_tasks",
        output_key: str = "research_results",
        **kwargs
    ) -> Optional[str]:
        """
        연구 데이터를 추출합니다.
        
        Args:
            storage: DataFlowStorage 인스턴스
            input_key: 입력 키 (기본값: "research_tasks")
            output_key: 출력 키 (기본값: "research_results")
            **kwargs: 추가 파라미터
            
        Returns:
            실행 결과 메시지
        """
        self.logger.info(f"Extracting research data from '{input_key}' to '{output_key}'")
        
        # 입력 데이터 읽기
        df = storage.read("dataframe")
        
        # research_tasks 컬럼에서 데이터 추출
        if input_key not in df.columns:
            self.logger.warning(f"Input key '{input_key}' not found. Creating empty results.")
            df[output_key] = None
            storage.write(df)
            return "No research tasks found"
        
        # 각 행에서 research_tasks 추출 및 변환
        extracted_results = []
        
        for idx, row in df.iterrows():
            tasks = row.get(input_key, [])
            
            if not isinstance(tasks, list):
                tasks = [tasks] if tasks else []
            
            # 각 태스크에서 결과 추출
            for task in tasks:
                if isinstance(task, dict):
                    # 태스크가 dict인 경우, 필요한 필드만 추출
                    result = {}
                    for field in self.extract_fields:
                        if field in task:
                            result[field] = task[field]
                        elif field == "content" and "result" in task:
                            result["content"] = task["result"]
                        elif field == "source" and "url" in task:
                            result["source"] = task["url"]
                    
                    # 기본 필드 추가
                    if "task_id" in task:
                        result["task_id"] = task["task_id"]
                    if "status" in task:
                        result["status"] = task["status"]
                    
                    extracted_results.append(result)
                elif isinstance(task, str):
                    # 태스크가 문자열인 경우, 간단한 결과 생성
                    extracted_results.append({
                        "content": task,
                        "source": "unknown",
                    })
        
        # 결과를 DataFrame에 추가
        if extracted_results:
            # 결과를 리스트로 저장
            df[output_key] = None
            for idx in range(len(df)):
                if idx < len(extracted_results):
                    # 각 행에 해당하는 결과 할당
                    if df.loc[idx, output_key] is None:
                        df.loc[idx, output_key] = []
                    if not isinstance(df.loc[idx, output_key], list):
                        df.loc[idx, output_key] = [df.loc[idx, output_key]]
                    df.loc[idx, output_key].append(extracted_results[idx])
        else:
            df[output_key] = None
        
        storage.write(df)
        
        self.logger.info(f"Extracted {len(extracted_results)} research results")
        
        return f"Extracted {len(extracted_results)} research results"

