"""
Agent Pipeline for sparkleforge

Agent 워크플로우를 Pipeline으로 변환하여 구조화된 데이터 처리를 제공합니다.
"""

import logging
import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from collections import Counter

from ..operators.base_operator import SparkleForgeOperatorABC
from ..storage.agent_storage_adapter import DataFlowStorage

logger = logging.getLogger(__name__)


class KeyNode:
    """데이터 흐름 그래프의 키 노드."""
    
    def __init__(self, key_para_name: str, key: str):
        """
        초기화.
        
        Args:
            key_para_name: 키 파라미터 이름 (예: "input_key", "output_key")
            key: 키 값 (예: "research_results")
        """
        self.key_para_name = key_para_name
        self.key = key
        self.index: Optional[int] = None
        self.ptr: List['KeyNode'] = []  # 연결된 다른 KeyNode들
    
    def set_index(self, index: int):
        """인덱스를 설정합니다."""
        self.index = index
    
    def __str__(self):
        ptr_str = ", ".join([f"{p.key}(idx={p.index})" for p in self.ptr])
        return f"KeyNode(key_para_name={self.key_para_name}, key={self.key}, index={self.index}, ptr_keys={ptr_str})"
    
    def __repr__(self):
        return self.__str__()


class OperatorNode:
    """데이터 흐름 그래프의 Operator 노드."""
    
    def __init__(
        self,
        op_obj: Optional[SparkleForgeOperatorABC] = None,
        op_name: str = None,
        storage: Optional[DataFlowStorage] = None,
        **kwargs
    ):
        """
        초기화.
        
        Args:
            op_obj: Operator 객체
            op_name: Operator 이름
            storage: DataFlowStorage 인스턴스
            **kwargs: Operator 실행 파라미터
        """
        self.op_obj = op_obj
        self.op_name = op_name
        self.storage = storage
        self.kwargs = kwargs
        
        # 입력/출력 키 초기화
        self.input_keys: List[str] = []
        self.input_key_nodes: Dict[str, KeyNode] = {}
        self.output_keys: List[str] = []
        self.output_keys_nodes: Dict[str, KeyNode] = {}
        
        self._get_keys_from_kwargs()
    
    def _get_keys_from_kwargs(self):
        """kwargs에서 input_key, output_key를 추출합니다."""
        for k, v in self.kwargs.items():
            if k.startswith("input_") and isinstance(v, str):
                self.input_keys.append(v)
                self.input_key_nodes[v] = KeyNode(k, v)
            elif k.startswith("output_") and isinstance(v, str):
                self.output_keys.append(v)
                self.output_keys_nodes[v] = KeyNode(k, v)
    
    def init_output_keys_nodes(self, keys: List[str]):
        """출력 키 노드를 초기화합니다."""
        for key in keys:
            self.output_keys.append(key)
            self.output_keys_nodes[key] = KeyNode(key, key)
    
    def init_input_keys_nodes(self, keys: List[str]):
        """입력 키 노드를 초기화합니다."""
        for key in keys:
            self.input_keys.append(key)
            self.input_key_nodes[key] = KeyNode(key, key)
    
    def __str__(self):
        op_class_name = self.op_obj.__class__.__name__ if self.op_obj else "None"
        return f"OperatorNode(op_name={self.op_name}, op_class={op_class_name}, input_keys={self.input_keys}, output_keys={self.output_keys})"
    
    def __repr__(self):
        return self.__str__()


class OPRuntime:
    """Operator 실행 정보."""
    
    def __init__(
        self,
        operator: SparkleForgeOperatorABC,
        operator_name: str,
        args: Dict[str, Any]
    ):
        """
        초기화.
        
        Args:
            operator: Operator 객체
            operator_name: Operator 이름
            args: 실행 파라미터
        """
        self.op = operator
        self.op_name = operator_name
        self.kwargs = args
    
    def __repr__(self):
        return f"OPRuntime(operator={repr(self.op)}, op_name={self.op_name}, args={self.kwargs})"


class AgentPipeline(ABC):
    """
    Agent 워크플로우를 Pipeline으로 변환하는 기본 클래스.
    
    DataFlow의 PipelineABC를 참고하여 sparkleforge에 맞게 구현했습니다.
    """
    
    def __init__(self):
        """초기화."""
        self.op_runtimes: List[OPRuntime] = []
        self.compiled = False
        self.accumulated_keys: List[List[str]] = []  # 각 operator 전의 키 목록
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.op_nodes_list: List[OperatorNode] = []
        self.final_keys: List[str] = []
        self.last_modified_index_of_keys: Dict[str, List[int]] = {}
    
    @abstractmethod
    def forward(self):
        """
        Pipeline을 실행합니다.
        
        하위 클래스에서 구현해야 합니다.
        """
        pass
    
    def compile(self):
        """
        Pipeline을 컴파일하고 데이터 흐름을 검증합니다.
        """
        if self.compiled:
            self.logger.warning("Pipeline is already compiled")
            return
        
        self.logger.info("Compiling pipeline...")
        
        # forward()를 호출하여 op_runtimes 수집
        self.forward()
        
        # 컴파일된 forward로 교체
        self.forward = self._compiled_forward
        
        self.logger.info(
            f"Compiling pipeline and validating key integrity "
            f"across {len(self.op_runtimes)} operator runtimes."
        )
        
        # Operator 노드 그래프 구축
        self._build_operator_nodes_graph()
        
        self.compiled = True
        self.logger.info("Pipeline compilation completed")
    
    def _build_operator_nodes_graph(self):
        """Operator 노드 그래프를 구축합니다."""
        # op_runtimes에서 OperatorNode 생성
        for op_runtime in self.op_runtimes:
            storage_obj = op_runtime.kwargs.pop("storage", None)
            
            if storage_obj is None:
                raise ValueError(
                    f"Storage must be provided for operator '{op_runtime.op_name}'. "
                    f"Add 'storage' parameter to operator.run() call."
                )
            
            if not isinstance(storage_obj, DataFlowStorage):
                raise TypeError(
                    f"Storage must be a DataFlowStorage instance, "
                    f"but got {type(storage_obj)} for operator '{op_runtime.op_name}'"
                )
            
            # OperatorNode 생성
            op_node = OperatorNode(
                op_obj=op_runtime.op,
                op_name=op_runtime.op_name,
                storage=storage_obj,
                **op_runtime.kwargs
            )
            
            self.op_nodes_list.append(op_node)
        
        # 첫 번째 storage에서 초기 키 가져오기
        first_op = self.op_nodes_list[0] if self.op_nodes_list else None
        if first_op and first_op.storage:
            iter_storage_keys = first_op.storage.get_keys_from_dataframe()
        else:
            iter_storage_keys = []
        
        self.accumulated_keys.append(copy.deepcopy(iter_storage_keys))
        
        # 각 operator에 대해 키 검증 및 누적
        error_msg = []
        for op_node in self.op_nodes_list:
            # 입력 키 검증
            for input_key in op_node.input_keys:
                if input_key not in self.accumulated_keys[-1]:
                    error_msg.append({
                        "input_key": input_key,
                        "op_name": op_node.op_name,
                        "class_name": op_node.op_obj.__class__.__name__ if op_node.op_obj else "None",
                        "key_para_name": op_node.input_key_nodes[input_key].key_para_name
                    })
            
            # 출력 키 추가
            for output_key in op_node.output_keys:
                if output_key not in iter_storage_keys:
                    iter_storage_keys.append(output_key)
            
            self.accumulated_keys.append(copy.deepcopy(iter_storage_keys))
        
        # 에러가 있으면 예외 발생
        if error_msg:
            details = "\n".join(
                f"- Input key '{e['input_key']}' in `{e['op_name']}` "
                f"(class <{e['class_name']}>) does not match any output keys "
                f"from previous operators or dataset keys. "
                f"Check parameter '{e['key_para_name']}' in the `{e['op_name']}.run()`."
                for e in error_msg
            )
            msg = f"Key Matching Error in following Operators during pipeline.compile():\n{details}"
            self.logger.error(msg)
            raise KeyError(msg)
        
        self.final_keys = copy.deepcopy(iter_storage_keys)
        
        # 입력/출력 데이터셋 노드 추가
        self.input_dataset_node = OperatorNode(
            None,
            "DATASET-INPUT",
            None,
        )
        self.input_dataset_node.init_output_keys_nodes(self.accumulated_keys[0])
        self.op_nodes_list.insert(0, self.input_dataset_node)
        
        self.output_dataset_node = OperatorNode(
            None,
            "DATASET-OUTPUT",
            None,
        )
        self.output_dataset_node.init_input_keys_nodes(self.final_keys)
        self.op_nodes_list.append(self.output_dataset_node)
        
        # 키 수정 인덱스 추적 초기화
        for key in self.final_keys:
            self.last_modified_index_of_keys[key] = []
        
        # 키 노드 간 연결 구축
        for idx, i_op in enumerate(self.op_nodes_list):
            # 입력 키 처리
            for input_key in i_op.input_keys:
                current_keynode = i_op.input_key_nodes[input_key]
                current_keynode.set_index(idx)
                
                if len(self.last_modified_index_of_keys[input_key]) > 0:
                    last_modified_idx = self.last_modified_index_of_keys[input_key][-1]
                    last_modified_keynode = self.op_nodes_list[last_modified_idx].output_keys_nodes[input_key]
                    # 양방향 포인터 설정
                    last_modified_keynode.ptr.append(current_keynode)
                    current_keynode.ptr.append(last_modified_keynode)
            
            # 출력 키 처리
            for output_key in i_op.output_keys:
                current_keynode = i_op.output_keys_nodes[output_key]
                current_keynode.set_index(idx)
                self.last_modified_index_of_keys[output_key].append(idx)
        
        self.logger.debug(f"Built operator nodes graph with {len(self.op_nodes_list)} nodes")
        self.logger.debug(f"Accumulated keys: {self.accumulated_keys}")
    
    def _compiled_forward(self, resume_step: int = 0):
        """
        컴파일된 Pipeline을 실행합니다.
        
        Args:
            resume_step: 재개할 단계 (기본값: 0)
        """
        self.logger.info(f"Running compiled pipeline (resume_step={resume_step})")
        
        for idx, op_node in enumerate(self.op_nodes_list):
            # resume_step 이전은 건너뛰기
            if idx - 1 < resume_step:  # -1은 INPUT-DATA 노드 때문
                continue
            
            self.logger.debug(f"Running operator {idx}: {op_node.op_name}")
            
            # Operator 실행
            if op_node.op_obj is not None:
                try:
                    op_node.op_obj.run(
                        storage=op_node.storage,
                        **op_node.kwargs
                    )
                except Exception as e:
                    self.logger.error(f"Error running operator {op_node.op_name}: {e}")
                    raise
        
        self.logger.info("Pipeline execution completed")
    
    def run(self, resume_step: int = 0):
        """
        Pipeline을 실행합니다.
        
        Args:
            resume_step: 재개할 단계 (기본값: 0)
        """
        if not self.compiled:
            self.logger.warning("Pipeline is not compiled. Compiling now...")
            self.compile()
        
        self._compiled_forward(resume_step)








