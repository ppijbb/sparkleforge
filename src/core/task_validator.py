#!/usr/bin/env python3
"""
Task Validator - 작업 검증 강화 시스템

사전 검증, 실행 중 검증, 결과 검증을 통해 작업 성공률을 향상시킵니다.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """검증 결과."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    confidence: float  # 0.0-1.0
    metadata: Dict[str, Any]


class TaskValidator:
    """작업 검증 시스템."""
    
    def __init__(self):
        """초기화."""
        self.validation_history: List[Dict[str, Any]] = []
    
    async def validate_task_before_execution(
        self,
        task: Dict[str, Any],
        task_id: str,
        task_queue: Any = None,
        available_tools: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        사전 검증: 작업 실행 전 검증.
        
        검증 항목:
        1. 필수 필드 확인
        2. 의존성 확인
        3. 리소스 가용성 확인
        4. 파라미터 검증
        """
        errors = []
        warnings = []
        confidence = 1.0
        metadata = {}
        
        # 1. 필수 필드 확인
        required_fields = ['task_id', 'name']
        for field in required_fields:
            if field not in task or not task[field]:
                errors.append(f"Missing required field: {field}")
                confidence -= 0.2
        
        # task_id가 없으면 task_id 생성 시도
        if 'task_id' not in task or not task.get('task_id'):
            task['task_id'] = task_id
            warnings.append(f"task_id was missing, using provided task_id: {task_id}")
        
        # 2. 작업 이름 검증
        task_name = task.get('name', '')
        if not task_name or len(task_name.strip()) == 0:
            errors.append("Task name is empty or invalid")
            confidence -= 0.3
        elif len(task_name) > 500:
            warnings.append(f"Task name is very long ({len(task_name)} chars), may cause issues")
            confidence -= 0.05
        
        # 3. 의존성 확인
        dependencies = task.get('dependencies', [])
        if dependencies and task_queue:
            for dep_id in dependencies:
                if not task_queue.is_completed(dep_id):
                    errors.append(f"Dependency task '{dep_id}' is not completed")
                    confidence -= 0.15
                else:
                    metadata[f"dependency_{dep_id}"] = "completed"
        
        # 4. 작업 타입 검증
        task_type = task.get('task_type', 'general').lower()
        valid_task_types = ['search', 'fetch', 'data', 'code', 'academic', 'business', 'general']
        if task_type not in valid_task_types:
            warnings.append(f"Unknown task_type '{task_type}', using 'general'")
            task['task_type'] = 'general'
            confidence -= 0.05
        
        # 5. 리소스 가용성 확인 (도구)
        if available_tools is not None:
            tool_category = self._get_tool_category_for_task(task)
            if not available_tools:
                errors.append(f"No available tools for category '{tool_category}'")
                confidence -= 0.3
            else:
                metadata['available_tools'] = len(available_tools)
                metadata['tool_category'] = tool_category
        
        # 6. 파라미터 검증
        param_validation = self._validate_task_parameters(task)
        if not param_validation['is_valid']:
            errors.extend(param_validation['errors'])
            warnings.extend(param_validation['warnings'])
            confidence -= param_validation['confidence_penalty']
        
        # 7. 쿼리/입력 검증
        query = task.get('query', task.get('description', ''))
        if not query or len(query.strip()) == 0:
            # query가 없어도 name이 있으면 경고만
            if task_name:
                warnings.append("No query provided, using task name as query")
                task['query'] = task_name
            else:
                errors.append("No query or task name provided")
                confidence -= 0.2
        
        # 쿼리 길이 검증
        if query and len(query) > 1000:
            warnings.append(f"Query is very long ({len(query)} chars), may cause performance issues")
            confidence -= 0.05
        
        # 8. max_results 검증
        max_results = task.get('max_results', 10)
        if not isinstance(max_results, int) or max_results < 1:
            warnings.append(f"Invalid max_results ({max_results}), using default 10")
            task['max_results'] = 10
            confidence -= 0.05
        elif max_results > 100:
            warnings.append(f"max_results is very high ({max_results}), may cause performance issues")
            confidence -= 0.05
        
        confidence = max(0.0, min(1.0, confidence))
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=confidence,
            metadata=metadata
        )
        
        # 검증 이력 기록
        self.validation_history.append({
            'task_id': task_id,
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'pre_execution',
            'result': result
        })
        
        if errors:
            logger.warning(f"Task {task_id} pre-execution validation failed: {errors}")
        if warnings:
            logger.debug(f"Task {task_id} pre-execution validation warnings: {warnings}")
        
        return result
    
    async def validate_task_during_execution(
        self,
        task_id: str,
        intermediate_result: Optional[Dict[str, Any]],
        task: Dict[str, Any],
        execution_time: float,
        max_execution_time: float = 300.0
    ) -> ValidationResult:
        """
        실행 중 검증: 작업 실행 중간 결과 검증.
        
        검증 항목:
        1. 실행 시간 모니터링
        2. 중간 결과 유효성
        3. 이상 징후 감지
        """
        errors = []
        warnings = []
        confidence = 1.0
        metadata = {
            'execution_time': execution_time,
            'has_intermediate_result': intermediate_result is not None
        }
        
        # 1. 실행 시간 검증
        if execution_time > max_execution_time:
            errors.append(f"Execution time ({execution_time:.2f}s) exceeds max ({max_execution_time}s)")
            confidence -= 0.3
        elif execution_time > max_execution_time * 0.8:
            warnings.append(f"Execution time ({execution_time:.2f}s) is approaching limit")
            confidence -= 0.1
        
        # 2. 중간 결과 검증
        if intermediate_result:
            if not isinstance(intermediate_result, dict):
                errors.append("Intermediate result is not a dictionary")
                confidence -= 0.2
            elif 'success' not in intermediate_result:
                warnings.append("Intermediate result missing 'success' field")
                confidence -= 0.05
            
            # 중간 결과의 데이터 유효성
            if intermediate_result.get('success', False):
                data = intermediate_result.get('data')
                if data is None:
                    warnings.append("Intermediate result has success=True but data is None")
                    confidence -= 0.1
                elif isinstance(data, (list, dict)) and len(data) == 0:
                    warnings.append("Intermediate result data is empty")
                    confidence -= 0.05
            else:
                error_msg = intermediate_result.get('error', 'Unknown error')
                warnings.append(f"Intermediate result indicates failure: {error_msg}")
                confidence -= 0.15
        
        confidence = max(0.0, min(1.0, confidence))
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=confidence,
            metadata=metadata
        )
        
        # 검증 이력 기록
        self.validation_history.append({
            'task_id': task_id,
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'during_execution',
            'result': result
        })
        
        return result
    
    async def validate_task_result(
        self,
        tool_result: Dict[str, Any],
        task: Dict[str, Any]
    ) -> ValidationResult:
        """
        결과 검증: 도구 실행 결과 검증 (강화된 버전).
        
        검증 항목:
        1. 기본 결과 검증
        2. 작업 유형별 맞춤 검증
        3. 품질 점수 기반 검증
        4. 데이터 일관성 검증
        """
        errors = []
        warnings = []
        confidence = 1.0
        metadata = {}
        
        # 1. 기본 결과 검증
        if not isinstance(tool_result, dict):
            errors.append("Tool result is not a dictionary")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                confidence=0.0,
                metadata={}
            )
        
        success = tool_result.get("success", False)
        if not success:
            error_msg = tool_result.get("error", "Unknown error")
            errors.append(f"Tool execution failed: {error_msg}")
            confidence = 0.0
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                confidence=confidence,
                metadata={'error': error_msg}
            )
        
        data = tool_result.get("data")
        metadata['has_data'] = data is not None
        
        # 2. 데이터 존재 검증
        if data is None:
            errors.append("Tool result has no data")
            confidence -= 0.5
        elif isinstance(data, (list, dict)) and len(data) == 0:
            warnings.append("Tool result data is empty")
            confidence -= 0.2
            metadata['data_type'] = type(data).__name__
            metadata['data_length'] = 0
        
        # 3. 작업 유형별 맞춤 검증
        task_type = task.get('task_type', 'general').lower()
        
        if task_type == 'search' or 'search' in task.get('name', '').lower():
            # 검색 결과 검증
            if isinstance(data, list):
                if len(data) == 0:
                    warnings.append("Search returned no results")
                    confidence -= 0.1
                else:
                    # 검색 결과 항목 검증
                    valid_items = 0
                    for item in data:
                        if isinstance(item, dict):
                            if 'title' in item or 'url' in item or 'content' in item:
                                valid_items += 1
                    
                    if valid_items == 0:
                        warnings.append("Search results have no valid items (missing title/url/content)")
                        confidence -= 0.2
                    elif valid_items < len(data) * 0.5:
                        warnings.append(f"Only {valid_items}/{len(data)} search results are valid")
                        confidence -= 0.1
                    
                    metadata['total_results'] = len(data)
                    metadata['valid_results'] = valid_items
        
        elif task_type == 'fetch' or 'fetch' in task.get('name', '').lower():
            # 페치 결과 검증
            if isinstance(data, dict):
                if 'content' not in data and 'text' not in data and 'html' not in data:
                    warnings.append("Fetch result missing content/text/html fields")
                    confidence -= 0.1
            elif isinstance(data, str):
                if len(data.strip()) == 0:
                    warnings.append("Fetched content is empty")
                    confidence -= 0.2
                elif len(data) < 50:
                    warnings.append("Fetched content is very short")
                    confidence -= 0.1
        
        elif task_type == 'data' or 'data' in task.get('name', '').lower():
            # 데이터 작업 검증
            if isinstance(data, (list, dict)):
                if len(data) == 0:
                    warnings.append("Data operation returned empty result")
                    confidence -= 0.1
        
        # 4. 품질 점수 기반 검증
        quality_score = tool_result.get("quality_score", None)
        if quality_score is not None:
            if isinstance(quality_score, (int, float)):
                if quality_score < 0.5:
                    warnings.append(f"Low quality score: {quality_score:.2f}")
                    confidence -= 0.2
                elif quality_score < 0.7:
                    warnings.append(f"Moderate quality score: {quality_score:.2f}")
                    confidence -= 0.1
                metadata['quality_score'] = quality_score
        
        confidence_score = tool_result.get("confidence", 1.0)
        if isinstance(confidence_score, (int, float)):
            if confidence_score < 0.5:
                warnings.append(f"Low confidence score: {confidence_score:.2f}")
                confidence -= 0.2
            elif confidence_score < 0.7:
                warnings.append(f"Moderate confidence score: {confidence_score:.2f}")
                confidence -= 0.1
            metadata['confidence_score'] = confidence_score
        
        # 5. 실행 시간 검증
        execution_time = tool_result.get("execution_time", 0.0)
        if execution_time > 60.0:
            warnings.append(f"Long execution time: {execution_time:.2f}s")
            confidence -= 0.05
        metadata['execution_time'] = execution_time
        
        # 6. 데이터 일관성 검증
        if isinstance(data, list) and len(data) > 0:
            # 리스트 항목들이 일관된 구조를 가지는지 확인
            first_item_type = type(data[0]).__name__
            consistent_types = sum(1 for item in data if type(item).__name__ == first_item_type)
            if consistent_types < len(data) * 0.8:
                warnings.append(f"Inconsistent data types in result list ({consistent_types}/{len(data)} consistent)")
                confidence -= 0.1
        
        confidence = max(0.0, min(1.0, confidence))
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=confidence,
            metadata=metadata
        )
        
        return result
    
    def _get_tool_category_for_task(self, task: Dict[str, Any]) -> str:
        """작업 유형에 따른 도구 카테고리 반환."""
        task_type = task.get('task_type', 'general').lower()
        task_name = task.get('name', '').lower()
        
        if 'search' in task_type or 'search' in task_name:
            return 'search'
        elif 'fetch' in task_type or 'fetch' in task_name or 'url' in task_name:
            return 'data'
        elif 'code' in task_type or 'code' in task_name:
            return 'code'
        elif 'academic' in task_type or 'academic' in task_name or 'paper' in task_name:
            return 'academic'
        elif 'business' in task_type or 'business' in task_name:
            return 'business'
        else:
            return 'general'
    
    def _validate_task_parameters(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """작업 파라미터 검증."""
        errors = []
        warnings = []
        confidence_penalty = 0.0
        
        # URL 검증 (있는 경우)
        url = task.get('url', '')
        if url:
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            if not url_pattern.match(url):
                errors.append(f"Invalid URL format: {url}")
                confidence_penalty += 0.2
        
        # 타임아웃 검증
        timeout = task.get('timeout', 30)
        if isinstance(timeout, (int, float)):
            if timeout < 1:
                errors.append(f"Timeout too short: {timeout}s")
                confidence_penalty += 0.2
            elif timeout > 300:
                warnings.append(f"Timeout very long: {timeout}s")
                confidence_penalty += 0.05
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'confidence_penalty': confidence_penalty
        }
    
    def get_validation_history(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """검증 이력 조회."""
        if task_id:
            return [v for v in self.validation_history if v.get('task_id') == task_id]
        return self.validation_history.copy()

