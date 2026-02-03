"""
Batch Wrapper for sparkleforge DataFlow Integration

대용량 연구 결과를 배치로 처리합니다.
"""

import logging
import pandas as pd
from typing import Any, Dict, Generic, TypeVar, ParamSpec, Callable, Union

from ..operators.base_operator import SparkleForgeOperatorABC
from ..storage.agent_storage_adapter import DataFlowStorage

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class BatchWrapper(Generic[P, R]):
    """Batch wrapper for operators."""
    """
    Operator를 배치 처리로 래핑합니다.
    
    DataFlow의 BatchWrapper를 참고하여 sparkleforge에 맞게 구현했습니다.
    """
    
    def __init__(
        self,
        operator: SparkleForgeOperatorABC,
        batch_size: int = 100,
    ):
        """
        초기화.
        
        Args:
            operator: 배치 처리할 Operator
            batch_size: 배치 크기
        """
        self._operator = operator
        self._batch_size = batch_size
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(
        self,
        storage: DataFlowStorage,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> R:
        """
        배치 처리로 Operator를 실행합니다.
        
        Args:
            storage: DataFlowStorage 인스턴스
            *args: Operator 실행 파라미터
            **kwargs: Operator 실행 파라미터
            
        Returns:
            실행 결과
        """
        # 전체 데이터 읽기
        whole_dataframe = storage.read("dataframe")
        
        if len(whole_dataframe) == 0:
            self.logger.warning("No data to process")
            return ""
        
        # 배치 수 계산
        num_batches = (len(whole_dataframe) + self._batch_size - 1) // self._batch_size
        
        self.logger.info(
            f"Total {len(whole_dataframe)} items, will process in {num_batches} batches of size {self._batch_size}."
        )
        
        # 각 배치 처리
        for batch_num in range(num_batches):
            start_index = batch_num * self._batch_size
            end_index = min((batch_num + 1) * self._batch_size, len(whole_dataframe))
            batch_df = whole_dataframe.iloc[start_index:end_index].copy()
            
            # 배치용 임시 Storage 생성
            from ..storage.agent_storage_adapter import AgentStateStorage
            batch_storage = AgentStateStorage({}, session_id=None)
            batch_storage.write(batch_df)
            
            # Operator 실행
            self.logger.info(f"Running batch {batch_num + 1}/{num_batches} with {len(batch_df)} items...")
            try:
                self._operator.run(batch_storage, *args, **kwargs)
                
                # 결과 읽기
                res_df = batch_storage.read("dataframe")
                
                # 새 컬럼 추가
                new_cols = [c for c in res_df.columns if c not in whole_dataframe.columns]
                for c in new_cols:
                    whole_dataframe[c] = pd.NA
                
                # 결과를 원본 DataFrame에 병합
                whole_dataframe.loc[res_df.index, res_df.columns] = res_df
                
            except Exception as e:
                self.logger.error(f"Batch {batch_num + 1} failed: {e}")
                raise
        
        # 최종 결과 저장
        storage.write(whole_dataframe)
        
        self.logger.info("Batch processing completed")
        return ""  # type: ignore[return-value]

