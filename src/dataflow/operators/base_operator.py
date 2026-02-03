"""
Base Operator for sparkleforge DataFlow Integration

DataFlow의 OperatorABC를 확장하여 sparkleforge에 특화된 기본 Operator를 제공합니다.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Literal, List

# DataFlow Operator 인터페이스 (로컬 구현)
try:
    from dataflow.core.operator import OperatorABC as DataFlowOperatorABC  # type: ignore
except ImportError:
    # DataFlow가 설치되지 않은 경우를 위한 기본 인터페이스
    from abc import ABC, abstractmethod
    
    class DataFlowOperatorABC(ABC):  # type: ignore
        def __init__(self):
            self.logger = logging.getLogger(__name__)
        
        @abstractmethod
        def run(self) -> None:
            pass

# DataFlowStorage는 agent_storage_adapter에서 import
from ..storage.agent_storage_adapter import DataFlowStorage  # type: ignore

logger = logging.getLogger(__name__)


class SparkleForgeOperatorABC(DataFlowOperatorABC):
    """
    sparkleforge용 기본 Operator 클래스.
    
    DataFlow의 OperatorABC를 확장하여 sparkleforge에 특화된 기능을 제공합니다.
    """
    
    def __init__(self):
        """초기화."""
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def run(self, storage: DataFlowStorage, **kwargs) -> Optional[str]:
        """
        Operator를 실행합니다.
        
        Args:
            storage: DataFlowStorage 인스턴스
            **kwargs: 추가 파라미터 (input_key, output_key 등)
            
        Returns:
            실행 결과 메시지 (선택사항)
        """
        pass
    
    def validate_input_keys(self, storage: DataFlowStorage, required_keys: List[str]) -> bool:
        """
        입력 키를 검증합니다.
        
        Args:
            storage: DataFlowStorage 인스턴스
            required_keys: 필수 입력 키 목록
            
        Returns:
            검증 성공 여부
            
        Raises:
            ValueError: 필수 키가 없는 경우
        """
        available_keys = storage.get_keys_from_dataframe()
        missing_keys = [key for key in required_keys if key not in available_keys]
        
        if missing_keys:
            raise ValueError(
                f"Missing required input keys: {missing_keys}. "
                f"Available keys: {available_keys}"
            )
        
        return True
    
    def get_input_data(self, storage: DataFlowStorage, input_key: str) -> Any:
        """
        입력 데이터를 가져옵니다.
        
        Args:
            storage: DataFlowStorage 인스턴스
            input_key: 입력 키
            
        Returns:
            입력 데이터
        """
        df = storage.read("dataframe")
        
        if input_key not in df.columns:
            raise ValueError(f"Input key '{input_key}' not found in storage")
        
        return df[input_key]
    
    def set_output_data(self, storage: DataFlowStorage, output_key: str, data: Any) -> None:
        """
        출력 데이터를 설정합니다.
        
        Args:
            storage: DataFlowStorage 인스턴스
            output_key: 출력 키
            data: 출력 데이터
        """
        df = storage.read("dataframe")
        
        # 데이터를 DataFrame에 추가
        if isinstance(data, list):
            # 리스트인 경우 새 컬럼으로 추가
            if len(data) == len(df):
                df[output_key] = data
            else:
                # 길이가 다른 경우 새 행으로 추가하거나 병합
                self.logger.warning(
                    f"Data length ({len(data)}) doesn't match DataFrame length ({len(df)}). "
                    f"Creating new DataFrame with output key."
                )
                new_df = df.copy()
                new_df[output_key] = data[:len(df)] if len(data) > len(df) else data + [None] * (len(df) - len(data))
                storage.write(new_df)
                return
        else:
            # 단일 값인 경우 모든 행에 동일한 값 설정
            df[output_key] = data
        
        storage.write(df)

